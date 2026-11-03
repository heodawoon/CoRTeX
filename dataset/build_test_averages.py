import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import LABEL_COLS


def image_mean_rows(idx_rows, X):
    out = np.stack([X[rows].mean(axis=0) for rows in idx_rows], axis=0)
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--split_dir', required=True,
                    help='directory containing test_idx.npy')
    ap.add_argument('--bold_path', required=True,
                    help='path to subj{XX}_bold_R360_T{T}.npz')
    ap.add_argument('--out_bold', required=True,
                    help='output averaged BOLD .npz path')
    ap.add_argument('--out_labels', required=True,
                    help='output averaged labels .npy path')
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    test_idx = np.load(Path(args.split_dir) / 'test_idx.npy')

    sub = manifest.iloc[test_idx].copy()
    sub['_orig_row'] = test_idx
    order = sub.sort_values(['coco_id', 'rep_index']).reset_index(drop=True)
    groups = order.groupby('coco_id', sort=True)
    coco_ids = np.asarray(sorted(groups.groups.keys()), dtype=np.int64)
    idx_rows = [order.loc[groups.get_group(c).index, '_orig_row'].values
                 for c in coco_ids]
    trial_counts = np.asarray([len(r) for r in idx_rows], dtype=np.int32)
    print('test unique coco_ids: ' + str(len(coco_ids)))

    labels_trial = manifest[LABEL_COLS].values.astype(np.float32)
    labels_avg = np.stack(
        [labels_trial[rows].mean(axis=0) for rows in idx_rows],
        axis=0,
    ).astype(np.float32)

    out_labels = Path(args.out_labels)
    out_labels.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_labels, labels_avg)

    bold = np.load(args.bold_path)['bold'].astype(np.float32)
    bold_avg = image_mean_rows(idx_rows, bold)

    out_bold = Path(args.out_bold)
    out_bold.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_bold,
        bold=bold_avg, coco_ids=coco_ids, trial_counts=trial_counts,
    )
    print('saved ' + str(out_bold) + ' shape=' + str(bold_avg.shape))


if __name__ == '__main__':
    main()
