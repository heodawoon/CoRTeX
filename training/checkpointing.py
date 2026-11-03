import json
import math
from pathlib import Path

import torch


def is_better(metric, new, cur):
    if metric == 'loss':
        return new < cur
    return new > cur


class Checkpointer:
    def __init__(self, out_dir, metrics=('macro_auc', 'micro_auc', 'acc', 'loss')):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = metrics
        self.best = {
            m: {'value': math.inf if m == 'loss' else -math.inf, 'epoch': -1}
            for m in metrics
        }

    def write(self, state, path):
        tmp = path.with_suffix(path.suffix + '.tmp')
        torch.save(state, tmp)
        tmp.replace(path)

    def save_last(self, state):
        self.write(state, self.out_dir / 'last.pt')

    def maybe_save(self, epoch, metrics, state):
        improved = []
        for m in self.metrics:
            if m not in metrics:
                continue
            if is_better(m, metrics[m], self.best[m]['value']):
                self.best[m] = {'value': float(metrics[m]), 'epoch': int(epoch)}
                self.write(state, self.out_dir / ('best_' + m + '.pt'))
                improved.append(m)
        (self.out_dir / 'best_summary.json').write_text(json.dumps(self.best, indent=2))
        return improved

    @staticmethod
    def load(ckpt_path, map_location='cpu'):
        return torch.load(ckpt_path, map_location=map_location, weights_only=False)
