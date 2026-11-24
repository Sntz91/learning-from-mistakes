import io
import os 
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

DATA = Path('data/rsna-processed')
WINDOWS = ((40, 80), (80, 200), (40, 380))
SPLITS = Path('splits')
LABELS = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']


class Transform:
    def __init__(self, train=False):
        self.spatial = (T.Compose([
            T.RandomResizedCrop((224, 224), scale=(0.9, 1.0),
                                interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=(-30, 30)),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        ]) if train else T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC, antialias=True))
        self.normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __call__(self, image):
        channels = [np.clip((image.astype(np.float32) - (c - w / 2)) / w, 0, 1) for c, w in WINDOWS]
        return self.normalize(self.spatial(torch.from_numpy(np.stack(channels)).float()))


class CTDataset(Dataset):
    def __init__(self, split, train=False, primary=None, threshold=None):
        index = pd.read_parquet(DATA / 'index.parquet')
        ids = pd.read_parquet(SPLITS / f'{split}.parquet').sop_uid
        rows = index[index.sop_uid.isin(ids)].reset_index(drop=True)

        # Precompute everything as numpy arrays: much faster than rows.iloc[i] per item
        self.sop_uid = rows.sop_uid.to_numpy()
        self.series_id = rows.series_id.to_numpy()
        self.tar = rows.tar.to_numpy()
        self.size = rows['size'].to_numpy(int)
        self.offset = rows.data_offset.to_numpy(int)
        self.label = rows[[f'label_{n}' for n in LABELS]].any(axis=1).to_numpy(np.float32)

        self.primary = None
        if primary is not None:
            assert threshold is not None, 'threshold is required when primary is given'
            prob = rows.sop_uid.map(pd.read_csv(primary).set_index('sop_uid').prob_ich)
            assert prob.notna().all(), 'some slices are missing from the primary predictions'
            self.primary = prob.to_numpy(np.float32)
            self.threshold = np.float32(threshold)

        self.transform = Transform(train)
        self.fds = {}

    def __len__(self):
        return len(self.sop_uid)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["fds"] = {}
        return state

    def __getitem__(self, i):
        path = DATA / self.tar[i]
        if path not in self.fds:
            self.fds[path] = os.open(path, os.O_RDONLY)
        image = np.load(io.BytesIO(os.pread(self.fds[path], self.size[i], self.offset[i])))

        if self.primary is None:
            target = torch.tensor([self.label[i]])
        else:
            positive = bool(self.primary[i] >= self.threshold)
            error = float(positive != bool(self.label[i]))
            target = torch.tensor([error, -1.0] if positive else [-1.0, error])

        return self.transform(image), target, self.sop_uid[i]
