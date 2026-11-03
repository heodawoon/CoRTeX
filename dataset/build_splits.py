import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .labels import LABEL_COLS


def one_split(train_all, y_strat, val_ratio):
    try:
        tr, va = train_test_split(
            train_all, test_size=val_ratio,
            random_state=0, stratify=y_strat,
        )
        return tr, va, 'full_stratified'
    except ValueError:
        try:
            tr, va = train_test_split(
                train_all, test_size=val_ratio,
                random_state=0,
                stratify=(np.asarray(y_strat).astype(str).astype('U1')),
            )
            return tr, va, 'coarse_stratified'
        except ValueError:
            tr, va = train_test_split(
                train_all, test_size=val_ratio,
                random_state=0, stratify=None,
            )
            return tr, va, 'random'


def build_split_for_subject(manifest_path, out_dir, val_ratio=0.04):
    manifest = pd.read_csv(manifest_path)
    test_mask = manifest['is_shared1000'].astype(bool).values
    test_idx  = np.where(test_mask)[0]
    train_all = np.where(~test_mask)[0]

    y_strat = (manifest.loc[train_all, LABEL_COLS].astype(int).astype(str)
               .agg(''.join, axis=1).values)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'test_idx.npy', test_idx.astype(np.int64))

    tr, va, strat = one_split(train_all, y_strat, val_ratio)
    np.save(out_dir / 'train_idx.npy', np.asarray(tr, dtype=np.int64))
    np.save(out_dir / 'val_idx.npy',   np.asarray(va, dtype=np.int64))

    (out_dir / 'split_metadata.json').write_text(json.dumps({
        'total_trials': int(len(manifest)),
        'n_test':       int(len(test_idx)),
        'n_train':      int(len(tr)),
        'n_val':        int(len(va)),
        'val_ratio':    val_ratio,
        'stratified':   strat,
    }, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--val_ratio', type=float, default=0.04)
    args = ap.parse_args()

    build_split_for_subject(args.manifest, args.out_dir, val_ratio=args.val_ratio)
    with open(Path(args.out_dir) / 'split_metadata.json') as f:
        meta = json.load(f)
    print('n_train=' + str(meta['n_train']) + ' n_val=' + str(meta['n_val'])
          + ' n_test=' + str(meta['n_test']) + ' (' + meta['stratified'] + ')')


if __name__ == '__main__':
    main()
