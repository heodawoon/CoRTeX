import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample as fft_resample


ABIDEI_TR = {
    'CALTECH': 2.0,
    'CMU': 2.0,
    'KKI': 2.5,
    'LEUVEN_1': 1.667,
    'LEUVEN_2': 1.667,
    'MAX_MUN': 3.0,
    'NYU': 2.0,
    'OHSU': 2.5,
    'OLIN': 1.5,
    'PITT': 1.5,
    'SBL': 2.2,
    'SDSU': 2.0,
    'STANFORD': 2.0,
    'TRINITY': 2.0,
    'UCLA_1': 3.0,
    'UCLA_2': 3.0,
    'UM_1': 2.0,
    'UM_2': 2.0,
    'USM': 2.0,
    'YALE': 2.0,
}

DEFAULT_TR_TARGET = 2.0


def check_bad_rois(bold, eps=1e-8):
    all_zero = np.abs(bold).sum(axis=1) < eps
    has_nan = np.isnan(bold).any(axis=1)
    has_inf = np.isinf(bold).any(axis=1)
    bad_any = all_zero | has_nan | has_inf
    return {
        'zero_roi': int(all_zero.sum()),
        'nan_roi': int(has_nan.sum()),
        'inf_roi': int(has_inf.sum()),
        'bad_total': int(bad_any.sum()),
    }


def resample_to_tr(bold, tr_orig, tr_target=DEFAULT_TR_TARGET):
    if abs(tr_orig - tr_target) < 1e-3:
        return bold.astype(np.float32, copy=False)
    T_old = bold.shape[-1]
    T_new = int(round(T_old * tr_orig / tr_target))
    out = fft_resample(bold, T_new, axis=-1)
    return out.astype(np.float32)


def build_manifest(root, atlas, tr_target, bad_roi_threshold, out_dir):
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    atlas_dir = Path(root) / atlas

    kept = []
    excluded = []

    for site_dir in sorted(p for p in atlas_dir.iterdir() if p.is_dir()):
        for f in sorted(site_dir.glob('*.npz')):
            x = np.load(f, allow_pickle=True)
            bold = x['bold_znorm']
            site_key = str(x['site'])
            tr_orig = ABIDEI_TR.get(site_key)
            bad = check_bad_rois(bold)

            if tr_orig is None:
                excluded.append({
                    'path': str(f), 'reason': 'unknown_site=' + site_key,
                    'site': site_key, **bad,
                })
                continue

            if bad['bad_total'] >= bad_roi_threshold:
                excluded.append({
                    'path': str(f), 'reason': 'bad_rois',
                    'site': site_key, **bad,
                })
                continue

            T_orig = int(bold.shape[1])
            T_resamp = int(round(T_orig * tr_orig / tr_target))
            kept.append({
                'path': str(f),
                'sid': str(x['sid']),
                'site': site_key,
                'site2': str(x['site2']),
                'dx': int(x['dx']),
                'age': float(x['age']),
                'gender': int(x['gender']),
                'tr_orig': tr_orig,
                'T_orig': T_orig,
                'T_resamp': T_resamp,
                **bad,
            })

    df = pd.DataFrame(kept)
    manifest_path = out_dir_p / (atlas + '_manifest.csv')
    excluded_path = out_dir_p / (atlas + '_excluded.txt')
    df.to_csv(manifest_path, index=False)

    with open(excluded_path, 'w') as fp:
        for e in excluded:
            fp.write(e.get('path', '?') + '\t' + e.get('reason', '?')
                     + '\t' + str(e.get('site', '?'))
                     + '\t' + str(e.get('bad_total', '-')) + '\n')

    print('kept ' + str(len(kept)) + ', excluded ' + str(len(excluded)))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--atlas', default='Schaefer200')
    ap.add_argument('--tr_target', type=float, default=DEFAULT_TR_TARGET)
    ap.add_argument('--bad_threshold', type=int, default=1)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    build_manifest(args.root, args.atlas, args.tr_target,
                   args.bad_threshold, args.out_dir)


if __name__ == '__main__':
    main()
