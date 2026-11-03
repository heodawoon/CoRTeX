import numpy as np
import torch
from torch.utils.data import Dataset


SITE_TR = {
    'CALTECH':2.0,  'CMU':2.0,   'KKI':2.5,      'LEUVEN_1':1.667,
    'LEUVEN_2':1.667,'MAX_MUN':3.0,'NYU':2.0,   'OHSU':2.5,
    'OLIN':1.5,      'PITT':1.5,  'SBL':2.2,    'SDSU':2.0,
    'STANFORD':2.0,  'TRINITY':2.0,'UCLA_1':3.0,'UCLA_2':3.0,
    'UM_1':2.0,      'UM_2':2.0,  'USM':2.0,    'YALE':2.0,
}
TR_TARGET = 2.0
DEFAULT_WINDOW = 98


def fft_resample(x, T_new, axis=-1):
    from scipy.signal import resample
    return resample(x, T_new, axis=axis)


def normalize_bold(bold, variant):
    if variant == 'raw':
        return bold.astype(np.float32)
    if variant == 'raw_minmax':
        mn = bold.min(axis=1, keepdims=True)
        mx = bold.max(axis=1, keepdims=True)
        rng = (mx - mn)
        rng[rng < 1e-8] = 1.0
        return ((bold - mn) / rng).astype(np.float32)
    if variant == 'znorm':
        return bold.astype(np.float32)
    raise ValueError('unknown variant: ' + repr(variant))


def load_subject_bold(npz_path, site, variant, target_tr=TR_TARGET,
                      bold_keys=None, resample=True):
    if bold_keys is None:
        bold_keys = {'znorm': 'bold_znorm', 'raw': 'bold_raw',
                     'raw_minmax': 'bold_raw'}
    npz = np.load(npz_path, allow_pickle=True)
    key = bold_keys.get(variant, 'bold_raw')
    bold = npz[key]
    bold = normalize_bold(bold.astype(np.float32), variant)

    if resample:
        site_key = site.upper().replace('-', '_')
        tr_orig = SITE_TR.get(site_key, target_tr)
        if abs(tr_orig - target_tr) > 1e-3:
            T_new = int(round(bold.shape[-1] * tr_orig / target_tr))
            bold = fft_resample(bold, T_new, axis=-1)
    return bold.astype(np.float32)


def make_segments(bold_RT, window=DEFAULT_WINDOW):
    T = bold_RT.shape[-1]
    segs = []
    for i in range(0, T - window + 1, window):
        segs.append(bold_RT[:, i:i + window])
    return segs


class ABIDESegmentDataset(Dataset):
    def __init__(self, sids, manifest, bold_variant='znorm',
                 window=DEFAULT_WINDOW, bold_keys=None, resample=True):
        self.window = window
        self.bold_variant = bold_variant
        self.items = []

        def norm(s):
            s = str(s)
            return s if not s.isdigit() else s.zfill(7)
        sid_to_idx = {norm(s): i for i, s in enumerate(manifest['sid'].tolist())}
        for sid in sids:
            sid_str = norm(sid)
            if sid_str not in sid_to_idx:
                continue
            i = sid_to_idx[sid_str]
            row = manifest.iloc[i]
            bold = load_subject_bold(row['npz_path'], row['site'], bold_variant,
                                     bold_keys=bold_keys, resample=resample)
            for seg in make_segments(bold, window=window):
                self.items.append((seg, int(row['dx']), i))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        seg, y, sid_idx = self.items[k]
        return (
            torch.from_numpy(seg).float(),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(sid_idx, dtype=torch.long),
        )
