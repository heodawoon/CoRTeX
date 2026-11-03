import argparse
from pathlib import Path

import numpy as np
import nibabel as nib


def extract_centroids_from_dir(roi_dir):
    roi_dir = Path(roi_dir)
    lh_img = nib.load(str(roi_dir / 'lh.HCP_MMP1.nii.gz'))
    rh_img = nib.load(str(roi_dir / 'rh.HCP_MMP1.nii.gz'))
    lh = lh_img.get_fdata().astype(np.int32)
    rh = rh_img.get_fdata().astype(np.int32)
    affine = lh_img.affine

    centroids_vox = np.full((360, 3), np.nan, dtype=np.float64)
    voxel_counts  = np.zeros(360, dtype=np.int64)
    for i in range(1, 181):
        v = np.argwhere(lh == i)
        if v.size == 0:
            continue
        centroids_vox[i - 1] = v.mean(axis=0)
        voxel_counts[i - 1]  = len(v)
    for i in range(1, 181):
        v = np.argwhere(rh == i)
        if v.size == 0:
            continue
        centroids_vox[i - 1 + 180] = v.mean(axis=0)
        voxel_counts[i - 1 + 180]  = len(v)

    valid = ~np.isnan(centroids_vox[:, 0])
    centroids_world = np.full_like(centroids_vox, np.nan)
    hom = np.c_[centroids_vox[valid], np.ones(valid.sum())]
    centroids_world[valid] = (affine @ hom.T).T[:, :3]
    return centroids_vox, centroids_world, voxel_counts


def normalize_minmax(coords):
    valid = ~np.isnan(coords[:, 0])
    v = coords[valid]
    cmin = v.min(axis=0); cmax = v.max(axis=0)
    denom = np.where((cmax - cmin) == 0, 1.0, cmax - cmin)
    out = np.zeros_like(coords)
    out[valid] = 2 * (v - cmin) / denom - 1
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roi_dir', required=True,
                    help='dir containing lh.HCP_MMP1.nii.gz, rh.HCP_MMP1.nii.gz')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--subj', required=True, help='2-digit subject id, e.g. 01')
    args = ap.parse_args()

    cv, cw, counts = extract_centroids_from_dir(args.roi_dir)
    cv_n = normalize_minmax(cv)
    cw_n = normalize_minmax(cw)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.subj
    np.save(out / ('subj' + tag + '_roi_centroids_voxel.npy'),           cv)
    np.save(out / ('subj' + tag + '_roi_centroids_voxel_normalized.npy'), cv_n)
    np.save(out / ('subj' + tag + '_roi_centroids_world_mm.npy'),         cw)
    np.save(out / ('subj' + tag + '_roi_centroids_world_normalized.npy'), cw_n)
    np.save(out / ('subj' + tag + '_roi_voxel_counts.npy'),               counts)
    print('saved centroids -> ' + str(out))


if __name__ == '__main__':
    main()
