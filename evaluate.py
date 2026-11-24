from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import json
from pprint import pprint

from sklearn.metrics import average_precision_score, roc_auc_score


RUN = Path('runs/paper')
DIAGNOSTIC_RUN = RUN / 'diagnostic'
SECOND_OPINION_RUN = RUN / 'second_opinion'
MONITOR_RUN = RUN / 'monitor'

TEST = 'rsna-no-match-sinoct-test'
OUTPUT = Path('results/paper')
BUDGET = 0.20
KEYS = ['sop_uid', 'series_id', 'dataset']

# --- Helpers to load / print --- #
def load(run):
    """ Load predictions parquet file. """
    return pd.read_csv(run / f'predictions-{TEST}.csv').sort_values(KEYS).reset_index(drop=True)

def round_floats(obj, ndigits=3):
    if isinstance(obj, dict):
        return {k: round_floats(v) for k, v in obj.items()}
    if isinstance(obj, float):
        return round(obj, ndigits)
    return obj


# --- Diagnostic Metrics --- #
def ada_ece(prob, label, K=10):
    """ Adaptive ECE: K bins with equal numbers of samples. """
    bin_boundaries = np.percentile(prob, np.linspace(0, 100, K + 1))
    ece = 0.0
    for i in range(K):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (prob >= lower) & (prob < upper if i < K - 1 else prob <= upper)
        if in_bin.any():
            ece += in_bin.mean() * abs(label[in_bin].mean() - prob[in_bin].mean())
    return float(ece)

def diagnostic_metrics(prob, label, threshold):
    """ Calculate several metrics of the diagnostic model. """
    predicted = prob >= threshold
    tp = int((predicted & (label==1)).sum())
    fp = int((predicted & (label==0)).sum())
    tn = int((~predicted & (label==0)).sum())
    fn = int((~predicted & (label==1)).sum())
    prevalence = label.mean()
    brier = np.mean((prob - label) ** 2)

    return {
        'samples': len(label),
        'positive': int(label.sum()),
        'negative': int((1-label).sum()),
        'auroc': roc_auc_score(label, prob),
        'auprc': average_precision_score(label, prob),
        'ada_ece': ada_ece(prob, label),
        'brier_skill_score': float(1 - brier / (prevalence * (1 - prevalence))),
        'threshold': threshold,
        'sensitivity': tp / (tp + fn),
        'specificity': tn / (tn + fp),
        'failure_rate': (fp + fn) / len(label),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    }

def scan_metrics(frame, threshold, K=3):
    """ Scan-level score -> mean of top K slice probabilities """
    scans = frame.groupby('series_id', sort=False)
    labels = scans.label_ich.max().to_numpy(int)
    scores = scans.prob_ich.apply(lambda s: np.sort(s.to_numpy())[-K:].mean()).to_numpy() 
    return diagnostic_metrics(scores, labels, threshold)


# --- Failure Metrics --- #
def decision_distance(probs, threshold):
    """ 1 at threshold annealing 0 at prob=0 or prob=1 --> Higher risk if closer at threshold. """
    return np.minimum(probs / threshold, (1-probs) / (1-threshold))

def caught(risk, error, budget=BUDGET):
    """ Nr of errors caught when reviewing top-budget fraction of a queue. """
    k = round(len(risk) * budget)
    top = np.argsort(-risk, kind='stable')[:k]
    return error[top].sum()

def queue_metrics(risk, error):
    """ How well risk ranks errors inside review queue. """
    return {
        'failure_auroc': roc_auc_score(error, risk),
        'failure_auprc': average_precision_score(error, risk),
        f'error_detection_at_{int(BUDGET * 100)}pct': float(caught(risk, error) / error.sum())
    }

def evaluate_risk(risk, fp, fn, pred_pos, pred_neg):
    """ Three queues: All diagnostic predictions, predicted-positive slices & predicted-negative slices. """
    errors = fp | fn
    n_caught = caught(risk[pred_pos], fp[pred_pos]) + caught(risk[pred_neg], fn[pred_neg])
    return {
        'all': {
            'failure_auroc': roc_auc_score(errors, risk),
            'failure_auprc': average_precision_score(errors, risk),
            # 'all' pools 20% of each queue seperately! (vs: 20% of the combined queue) -> we preserve prevalence
            f'error_detection_at_{int(BUDGET * 100)}pct': float(n_caught / errors.sum())
        },
        'fp_queue': queue_metrics(risk[pred_pos], fp[pred_pos]),
        'fn_queue': queue_metrics(risk[pred_neg], fn[pred_neg]),
    }

# --- Helper for having one csv for later plotting --- 
def write_results_csv():
    diagnostic = json.loads((OUTPUT / "diagnostic_metrics.json").read_text())
    results = json.loads((OUTPUT / "failure_detection.json").read_text())
    rows = []

    for level, metrics in diagnostic.items():
        for metric, value in metrics.items():
            rows.append(("diagnostic", "primary", level, metric, value))

    for group, value in results["queue_slices"].items():
        rows.append(("monitoring", "queue", group, "slices", value))
    for group, value in results["diagnostic_errors"].items():
        rows.append(("monitoring", "primary", group, "errors", value))

    for method, groups in results.items():
        if method in ("queue_slices", "diagnostic_errors"):
            continue
        for group, metrics in groups.items():
            for metric, value in metrics.items():
                rows.append(("monitoring", method, group, metric, value))

    pd.DataFrame(
        rows, columns=["section", "method", "group", "metric", "value"]
    ).to_csv(OUTPUT / "results.csv", index=False)


def main():
    primary = load(DIAGNOSTIC_RUN)
    monitor = load(MONITOR_RUN)
    second = load(SECOND_OPINION_RUN)

    assert primary[KEYS].equals(monitor[KEYS]), 'monitor predictions are not aligned with the diagnostic run'
    assert primary[KEYS].equals(second[KEYS]), 'second opinion predictions are not aligned with the diagnostic run'

    tau_diag = yaml.safe_load((DIAGNOSTIC_RUN / 'thresholds.yaml').read_text())
    tau_diag_slice, tau_diag_scan = tau_diag['slice'][0], tau_diag['scan'][0]
    p_diag = primary.prob_ich.to_numpy(float)
    y_diag = primary.label_ich.to_numpy(int)

    # Diagnostic Model
    diagnostic = {
        'slice': diagnostic_metrics(p_diag, y_diag, tau_diag_slice),
        'scan': scan_metrics(primary, tau_diag_scan),
    }
    print('Diagnostic metrics')
    print('--'*14)
    pprint(round_floats(diagnostic))

    # Review queues and errors
    pred_pos = p_diag >= tau_diag_slice
    pred_neg = ~pred_pos
    fp_diag = pred_pos & (y_diag==0)
    fn_diag = pred_neg & (y_diag==1)
    errors_diag = fp_diag | fn_diag
    
    # Risk scores
    risks = {
        'decision_distance': decision_distance(p_diag, tau_diag_slice),
        'visual_monitor': np.where(pred_pos, monitor.score_fp.to_numpy(float), monitor.score_fn.to_numpy(float)),
        'second_opinion': np.where(pred_pos, 1 - second.prob_ich.to_numpy(float), second.prob_ich.to_numpy(float)),
        'upperbound': errors_diag.astype(float), # perfect ranking: theoretical upperbound
    }

    results = {
        'queue_slices': {
            'all': len(y_diag), 'positive': int(pred_pos.sum()), 'negative': int(pred_neg.sum()),
        },
        'diagnostic_errors': {
            'all': int(errors_diag.sum()), 'fp': int(fp_diag.sum()), 'fn': int(fn_diag.sum()),
        },
        **{name: evaluate_risk(risk, fp_diag, fn_diag, pred_pos, pred_neg) for name, risk in risks.items()},
    }

    print('\n', 'Monitor Metrics')
    print('--'*14)
    pprint(round_floats(results))

    # Write files
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'diagnostic_metrics.json').write_text(json.dumps(round_floats(diagnostic), indent=2))
    (OUTPUT / 'failure_detection.json').write_text(json.dumps(round_floats(results), indent=2))

    write_results_csv()


if __name__=='__main__':
    main()
