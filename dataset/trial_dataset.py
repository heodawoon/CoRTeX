import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .labels import LABEL_COLS


class TrialDataset(Dataset):
    def __init__(self, bold_path, manifest_path, indices):
        npz = np.load(bold_path)
        self.bold = npz['bold'].astype(np.float32)
        manifest = pd.read_csv(manifest_path)
        self.labels = manifest[LABEL_COLS].values.astype(np.float32)
        self.idx = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        return (
            torch.from_numpy(self.bold[j]),
            torch.from_numpy(self.labels[j]),
            j,
        )
