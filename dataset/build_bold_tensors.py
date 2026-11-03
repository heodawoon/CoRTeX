import argparse
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
import torch


T_ANCHORS = {
    3:  ( 1,  3),
    4:  ( 1,  4),
    5:  ( 0,  4),
    6:  ( 0,  5),
    7:  (-1,  5),
    8:  (-1,  6),
    9:  (-2,  6),
    10: (-2,  7),
    11: ( 0, 10),
    12: ( 0, 11),
    13: (-1, 11),
    14: (-1, 12),
    15: (-2, 12),
}

ALL_T = tuple(sorted(T_ANCHORS.keys()))


def build_combined_360_atlas(subj, nsd_root):
    roi_dir = Path(nsd_root) / ('nsddata/ppdata/subj' + ('%02d' % subj) + '/func1pt8mm/roi')
    lh = nib.load(str(roi_dir / 'lh.HCP_MMP1.nii.gz')).get_fdata().astype(np.int32)
    rh = nib.load(str(roi_dir / 'rh.HCP_MMP1.nii.gz')).get_fdata().astype(np.int32)
    lh_pos = np.where(lh > 0, lh, 0)
    rh_pos = np.where(rh > 0, rh + 180, 0)
    return lh_pos + rh_pos


def precompute_roi_indices(atlas):
    lbl = atlas.reshape(-1).astype(np.int64)
    counts = np.bincount(lbl, minlength=361).astype(np.float32)
    counts = np.clip(counts, 1.0, None)
    return {
        'labels': torch.from_numpy(lbl),
        'counts': torch.from_numpy(counts),
    }


def accumulate_roi_mean(bold_XYZT, cache):
    Vflat = int(np.prod(bold_XYZT.shape[:-1]))
    T = bold_XYZT.shape[-1]
    lbl = cache['labels']
    counts = cache['counts']
    b = torch.from_numpy(bold_XYZT.astype(np.float32).reshape(Vflat, T))
    out = torch.zeros(361, T, dtype=torch.float32)
    out.scatter_add_(0, lbl.unsqueeze(-1).expand(-1, T), b)
    out = out / counts.unsqueeze(-1)
    return out[1:].numpy()


def zscore_per_roi(x):
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return ((x - mu) / sd).astype(np.float32)


def extract_per_run(subj, nsd_root, out_root):
    atlas = build_combined_360_atlas(subj, nsd_root)
    cache = precompute_roi_indices(atlas)
    ts_dir = Path(nsd_root) / ('nsddata_timeseries/ppdata/subj' + ('%02d' % subj) + '/func1pt8mm/timeseries')
    files = sorted(ts_dir.glob('timeseries_session*_run*.nii.gz'))

    out_dir = Path(out_root) / ('subj' + ('%02d' % subj))
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for i, f in enumerate(files):
        name = f.name
        sess = int(name[len('timeseries_session'):len('timeseries_session') + 2])
        run  = int(name.split('_run')[1].split('.')[0])
        out_p = out_dir / ('session' + ('%02d' % sess) + '_run' + ('%02d' % run) + '.npz')
        if out_p.exists():
            continue
        img = nib.load(str(f))
        arr = np.asarray(img.dataobj)
        arr = arr[..., :-1]
        roi = accumulate_roi_mean(arr, cache)
        z   = zscore_per_roi(roi)
        np.savez_compressed(out_p, bold_znorm=z)
        if (i + 1) % 20 == 0 or i == len(files) - 1:
            print('  ' + str(i+1) + '/' + str(len(files)) + ' (' + ('%.0f' % (time.time()-t0)) + 's)')


def build_trial_tensors(subj, manifest_path, per_run_dir, out_dir, t_values):
    manifest = pd.read_csv(manifest_path)
    manifest = manifest.sort_values('bold_tensor_idx').reset_index(drop=True)
    manifest['_run_key'] = list(zip(manifest['session'], manifest['run']))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_cache = {}
    for key in manifest['_run_key'].unique():
        sess, run = key
        p = Path(per_run_dir) / ('subj' + ('%02d' % subj)) / ('session' + ('%02d' % sess) + '_run' + ('%02d' % run) + '.npz')
        run_cache[key] = np.load(p)['bold_znorm']

    for T in t_values:
        a, b = T_ANCHORS[T]
        N = len(manifest)
        tensor = np.zeros((N, 360, T), dtype=np.float32)
        n_dropped = 0
        for i, row in manifest.iterrows():
            onset = int(row['onset_TR'])
            ws = onset + a
            we = onset + b
            run_ts = run_cache[(row['session'], row['run'])]
            if ws < 0 or we >= run_ts.shape[-1]:
                n_dropped += 1
                continue
            tensor[i] = run_ts[:, ws:we + 1]
        out_p = out_dir / ('subj' + ('%02d' % subj) + '_bold_R360_T' + str(T) + '.npz')
        np.savez_compressed(out_p, bold=tensor, window_t=T,
                             anchor_start=a, anchor_end=b)
        print('T=' + str(T) + ' shape=' + str(tensor.shape)
              + ' dropped=' + str(n_dropped) + ' -> ' + str(out_p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subj', type=int, required=True)
    ap.add_argument('--nsd_root', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--per_run_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--ts', type=int, nargs='+', default=list(ALL_T))
    ap.add_argument('--skip_runs', action='store_true')
    args = ap.parse_args()

    for T in args.ts:
        assert T in T_ANCHORS

    if not args.skip_runs:
        extract_per_run(args.subj, args.nsd_root, args.per_run_dir)
    build_trial_tensors(
        subj=args.subj,
        manifest_path=args.manifest,
        per_run_dir=args.per_run_dir,
        out_dir=args.out_dir,
        t_values=tuple(sorted(args.ts)),
    )


if __name__ == '__main__':
    main()
