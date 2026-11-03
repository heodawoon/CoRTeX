import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DX_MAP = {'Patient': 1, 'Control': 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--min_T', type=int, default=150)
    ap.add_argument('--n_folds', type=int, default=5)
    ap.add_argument('--val_frac', type=float, default=0.1)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    rows, excluded = [], []
    for npz_path in sorted(root.glob('sub-*.npz')):
        try:
            d = np.load(npz_path, allow_pickle=True)
        except Exception as e:
            excluded.append({'path': str(npz_path), 'reason': 'load_error: ' + str(e)})
            continue
        bold = d['bold']
        T = int(bold.shape[1])
        bad = ''
        if T < args.min_T:
            bad = 'T<min'
        elif not np.isfinite(bold).all():
            bad = 'NaN/Inf in bold'
        elif (bold.var(axis=1) < 1e-12).any():
            bad = 'dead ROI'
        sid = npz_path.stem.split('_')[0]
        dx_str = str(d['dx'])
        dx = DX_MAP.get(dx_str, None)
        if dx is None:
            bad = bad or ('unknown dx=' + repr(dx_str))
        row = {
            'sid': sid, 'site': 'COBRE',
            'dx': dx,
            'dx_str': dx_str,
            'age':    int(d['age'])    if d['age'].dtype.kind in 'iuf' else -1,
            'gender': str(d['gender']),
            'T': T,
            'npz_path': str(npz_path),
        }
        if bad:
            row['exclude_reason'] = bad
            excluded.append(row)
            continue
        rows.append(row)

    print('kept ' + str(len(rows)) + ', excluded ' + str(len(excluded)))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'manifest.csv', index=False)
    pd.DataFrame(excluded).to_csv(out_dir / 'excluded_subjects.csv', index=False)

    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=0)
    splits = {}
    for fold, (tv_idx, te_idx) in enumerate(skf.split(df, df['dx'])):
        tv = df.iloc[tv_idx].reset_index(drop=True)
        tr_idx, va_idx = next(sss.split(tv, tv['dx']))
        splits['fold_' + str(fold)] = dict(
            train=tv.iloc[tr_idx]['sid'].tolist(),
            val=tv.iloc[va_idx]['sid'].tolist(),
            test=df.iloc[te_idx]['sid'].tolist(),
        )
    out_split = out_dir / 'splits_5fold_dx.json'
    out_split.write_text(json.dumps(splits, indent=2))
    print('saved ' + str(out_split))


if __name__ == '__main__':
    main()
