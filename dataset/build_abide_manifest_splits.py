import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def is_subject_bad(npz, min_T=60):
    bold = npz['bold_raw']
    if bold.shape[1] < min_T:
        return True, 'T<min'
    if not np.isfinite(bold).all():
        return True, 'NaN/Inf in bold_raw'
    roi_var = bold.var(axis=1)
    n_dead = int((roi_var < 1e-8).sum())
    if n_dead > 0:
        return True, str(n_dead) + ' dead ROIs'
    return False, 'OK'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--min_T', type=int, default=60)
    ap.add_argument('--n_folds', type=int, default=5)
    ap.add_argument('--val_frac', type=float, default=0.1)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    excluded = []
    for site_dir in sorted(root.iterdir()):
        if not site_dir.is_dir():
            continue
        for npz_path in sorted(site_dir.glob('Schaefer200_*.npz')):
            try:
                d = np.load(npz_path, allow_pickle=True)
            except Exception as e:
                excluded.append(dict(path=str(npz_path), reason='load_error: ' + str(e)))
                continue
            bad, reason = is_subject_bad(d, min_T=args.min_T)
            dx_raw = int(d['dx'])
            dx = 1 if dx_raw == 1 else 0
            row = dict(
                sid=str(d['sid']),
                site=str(d['site']),
                site2=str(d['site2']),
                dx=dx,
                age=float(d['age']),
                gender=int(d['gender']),
                T=int(d['bold_raw'].shape[1]),
                npz_path=str(npz_path),
            )
            if bad:
                row['exclude_reason'] = reason
                excluded.append(row)
                continue
            rows.append(row)

    print('kept ' + str(len(rows)) + ', excluded ' + str(len(excluded)))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'manifest.csv', index=False)
    pd.DataFrame(excluded).to_csv(out_dir / 'excluded_subjects.csv', index=False)

    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    df['strat_key'] = df['site'].astype(str) + '_dx' + df['dx'].astype(str)
    counts = df['strat_key'].value_counts()
    tiny = counts[counts < args.n_folds].index.tolist()
    if tiny:
        df.loc[df['strat_key'].isin(tiny), 'strat_key'] = (
            'OTHER_dx' + df.loc[df['strat_key'].isin(tiny), 'dx'].astype(str)
        )

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=0)

    splits = {}
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(df, df['strat_key'])):
        tv = df.iloc[train_val_idx].reset_index(drop=True)
        train_idx, val_idx = next(sss.split(tv, tv['strat_key']))
        splits['fold_' + str(fold)] = dict(
            train=tv.iloc[train_idx]['sid'].tolist(),
            val=tv.iloc[val_idx]['sid'].tolist(),
            test=df.iloc[test_idx]['sid'].tolist(),
        )

    out_split = out_dir / 'splits_5fold_site_dx.json'
    out_split.write_text(json.dumps(splits, indent=2))
    print('saved ' + str(out_split))


if __name__ == '__main__':
    main()
