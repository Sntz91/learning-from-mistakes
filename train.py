import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from models import DenseNet, ViT, FocalLoss, BranchLoss
from dataset import CTDataset
from functools import partial

RUN_ROOT = Path('runs/new')
TASKS = ['diagnostic', 'second_opinion', 'monitor']

DEVICE = 'cuda'
WORKERS = 16
BATCH_SIZE = 256
EPOCHS = 100
SENSITIVITY = 0.85 # to pick thresholds
DATASET_NAME = 'rsna' # will be saved in prediction csv files

SETTINGS = {
    'diagnostic': dict(
        model=ViT,
        train='rsna/match-sinoct-train', 
        val='rsna/match-sinoct-val', 
        test='rsna/no-match-sinoct-test', 
        seed=42,   
        patience=12,
    ),
    'second_opinion': dict(
        model=partial(DenseNet, heads=1),
        train='rsna/no-match-sinoct-train', 
        val='rsna/no-match-sinoct-val', 
        test='rsna/no-match-sinoct-test', 
        seed=4242, 
        patience=7,
    ),
    'monitor': dict(
        model=partial(DenseNet, heads=2),
        train='rsna/no-match-sinoct-train', 
        val='rsna/no-match-sinoct-val', 
        test='rsna/no-match-sinoct-test', 
        seed=42,   
        patience=7,
    ),
}

def seed_all(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

def predictions_path(run, split):
    return run / f"predictions-{split.replace('/', '-')}.csv"

def make_dataset(task, split, augment=False):
    if task != 'monitor':
        return CTDataset(split, train=augment)
    diagnostic = RUN_ROOT / 'diagnostic'
    threshold = yaml.safe_load((diagnostic / 'thresholds.yaml').read_text())['slice'][0]
    return CTDataset(split, train=augment, primary=predictions_path(diagnostic, split), threshold=threshold)

def make_loader(dataset, seed, shuffle=False, persistent=False):
    return DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=WORKERS, pin_memory=True,
        shuffle=shuffle, persistent_workers=persistent and WORKERS > 0,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )

@torch.no_grad()
def run_inference(model, loader):
    model.eval()
    probs, targets = [], []
    for images, target, _ in loader:
        with torch.amp.autocast('cuda', enabled=True):
            probs.append(torch.sigmoid(model(images.to(DEVICE))).float().cpu().numpy())
        targets.append(target.numpy())
    return np.concatenate(probs), np.concatenate(targets)

def validation_auroc(probs, targets):
    """ Mean AUROC over heads. Target -1 ignored (monitor chooses head) """
    aurocs = []
    for head in range(probs.shape[1]):
        keep = targets[:, head] >= 0
        aurocs.append(roc_auc_score(targets[keep, head], probs[keep, head]))
    return float(np.mean(aurocs))

def predict(model, dataset, task, seed):
    probs, _ = run_inference(model, make_loader(dataset, seed))
    # label = dataset.label.astype(int)
    frame = pd.DataFrame({
        'sop_uid': dataset.sop_uid, 
        'series_id': dataset.series_id,
        'dataset': DATASET_NAME,
    })
    if task == 'monitor':
        positive = dataset.primary >= dataset.threshold
        wrong = positive != dataset.label.astype(bool)
        frame['diagnostic_positive'] = positive
        frame['label_error'] = wrong.astype(int)
        frame['error_type'] = np.select([wrong & positive, wrong & ~positive], ['fp', 'fn'], default='none')
        frame['score_fp'] = probs[:, 0]
        frame['score_fn'] = probs[:, 1]
    else:
        frame['label_ich'] = dataset.label.astype(int)
        frame['prob_ich'] = probs[:, 0]
    return frame

def threshold_at_sensitivity(labels, probabilities):
    _, tpr, thresholds = roc_curve(labels, probabilities, drop_intermediate=False)
    return float(np.clip(thresholds[np.flatnonzero(tpr >= SENSITIVITY)[0]], 0, 1))

def scan_labels_and_scores(frame, K=3):
    scans = frame.groupby(['dataset', 'series_id'], sort=False)
    labels = scans.label_ich.max().to_numpy(int)
    scores = scans.prob_ich.apply(lambda s: np.sort(s.to_numpy())[-K:].mean()).to_numpy()
    return labels, scores

def train(task):
    cfg = SETTINGS[task]
    run = RUN_ROOT / task
    if (run / 'best_model.pt').exists():
        raise FileExistsError(f'{run} already contains a model. Delete it to retrain.')
    if task == 'monitor':
        assert (RUN_ROOT / 'diagnostic' / 'thresholds.yaml').exists(), 'Train the diagnostic model first.'
    run.mkdir(parents=True, exist_ok=True)
    seed_all(cfg['seed'])

    train_ds = make_dataset(task, cfg['train'], augment=True)
    val_ds = make_dataset(task, cfg['val'])

    model = cfg['model']().to(DEVICE)
    if task == 'monitor':
        positive = train_ds.primary >= train_ds.threshold
        error = positive != train_ds.label.astype(bool)
        weights = [(~error & positive).sum() / (error & positive).sum(),
                   (~error & ~positive).sum() / (error & ~positive).sum()]
        loss_fn = BranchLoss(torch.tensor(weights, dtype=torch.float32)).to(DEVICE)
    elif task == 'second_opinion':
        loss_fn = FocalLoss()
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    train_loader = make_loader(train_ds, cfg['seed'], shuffle=True, persistent=True)
    val_loader = make_loader(val_ds, cfg['seed'], persistent=True)
    
    # Start Training
    best, stale = -1.0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for images, targets, _ in train_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=True):
                loss = loss_fn(model(images), targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        score = validation_auroc(*run_inference(model, val_loader))
        print(f'[{task}] epoch {epoch}: validation AUROC {score:.4f}')
        if score > best:
            best, stale = score, 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, run / 'best_model.pt')
        else:
            stale += 1
            if stale >= cfg['patience']:
                break

    # Predictions with best checkpoints
    checkpoint = torch.load(run / 'best_model.pt', map_location = DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    if task == 'diagnostic':
        # We need the training predictions for trainign the monitor!
        splits = {s[k] for s in SETTINGS.values() for k in ('train', 'val', 'test')}
    else:
        splits = {cfg['val'], cfg['test']}

    for split in splits:
        predictions = predict(model, make_dataset(task, split), task, cfg['seed'])
        predictions.to_csv(predictions_path(run, split), index=False)

    # Save thresholds
    if task != 'monitor':
        frame = pd.read_csv(predictions_path(run, cfg['val']))
        scan_labels, scan_scores = scan_labels_and_scores(frame)
        thresholds = {
            'slice': [threshold_at_sensitivity(frame.label_ich, frame.prob_ich)],
            'scan': [threshold_at_sensitivity(scan_labels, scan_scores)],
        }
        (run / 'thresholds.yaml').write_text(yaml.safe_dump(thresholds))

    print(f'[{task}] saved {run}')

def main():
    for task in TASKS:
        train(task)

if __name__ == '__main__':
    main()
