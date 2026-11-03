import numpy as np
import torch
import torch.nn.functional as F


def compute_pos_weight(Y):
    freq = np.asarray(Y).mean(axis=0) + 1e-6
    arr = np.ascontiguousarray((1.0 / np.sqrt(freq)).astype(np.float32))
    return torch.from_numpy(arr)


def multilabel_bce(logits, labels, pos_weight=None):
    loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    return {'loss': loss}
