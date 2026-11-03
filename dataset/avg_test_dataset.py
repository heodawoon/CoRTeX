import numpy as np
import torch
from torch.utils.data import Dataset


class AvgTrialBOLDTestDataset(Dataset):
    def __init__(self, bold_avg_path, labels_avg_path):
        npz = np.load(bold_avg_path)
        self.bold = npz['bold'].astype(np.float32)
        self.coco_ids = npz['coco_ids']
        self.labels = (np.load(labels_avg_path) >= 0.5).astype(np.float32)

    def __len__(self):
        return self.bold.shape[0]

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.bold[i]),
            torch.from_numpy(self.labels[i]),
            int(self.coco_ids[i]),
        )
