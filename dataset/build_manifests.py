import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import SUPERCATS


WINDOW_PRE_MAX  = 3
WINDOW_POST_MAX = 12


def find_refined_labels(nsd_root):
    hits = list(Path(nsd_root).glob('Refined_all_image_labels*.json'))
    if not hits:
        raise FileNotFoundError('Refined_all_image_labels*.json not found under ' + str(nsd_root))
    return hits[0]


def load_coco_supercat_mapping(nsd_root):
    with open(Path(nsd_root) / 'nsddata_stimuli/stimuli/nsd/annotations/instances_train2017.json') as f:
        coco = json.load(f)
    return {c['name']: c['supercategory'] for c in coco['categories']}


def build_manifest(subj, nsd_root):
    nsd_root = Path(nsd_root)
    stim = pd.read_csv(nsd_root / 'nsddata/experiments/nsd/nsd_stim_info_merged.csv')
    trial_to_meta = {}
    for rep in (0, 1, 2):
        col = 'subject' + str(subj) + '_rep' + str(rep)
        sub = stim.loc[stim[col] > 0, ['nsdId', 'cocoId', 'shared1000', col]]
        for _, r in sub.iterrows():
            t = int(r[col])
            trial_to_meta[t] = (int(r['nsdId']), rep,
                                 int(r['cocoId']), bool(r['shared1000']))

    refined = json.loads(find_refined_labels(nsd_root).read_text())
    name_to_super = load_coco_supercat_mapping(nsd_root)

    dsg_dir = nsd_root / ('nsddata_timeseries/ppdata/subj' + ('%02d' % subj) + '/func1pt8mm/design')
    design_files = sorted(glob.glob(str(dsg_dir) + '/design_session*_run*.tsv'))

    rows = []
    counter = 0
    for dp in design_files:
        name = Path(dp).name
        sess = int(name[len('design_session'):len('design_session') + 2])
        run  = int(name.split('_run')[1].split('.')[0])
        dsg = np.loadtxt(dp).astype(int)
        T_run = len(dsg)
        onsets = np.where(dsg > 0)[0]

        for trial_in_run, onset_tr in enumerate(onsets):
            counter += 1
            stim_73k = int(dsg[onset_tr])
            nsd_id_0 = stim_73k - 1
            meta = trial_to_meta.get(counter)
            if meta is None:
                continue
            nsd_csv, rep, coco_id, is_shared = meta

            r_entry = refined.get(str(nsd_id_0), {})
            text_map = r_entry.get('text_label_mapping', {})
            cat_names = set(text_map.keys())
            things = cat_names & set(name_to_super.keys())
            supers_present = {name_to_super[n] for n in things}

            win_valid = ((onset_tr - WINDOW_PRE_MAX) >= 0 and
                          (onset_tr + WINDOW_POST_MAX) <= T_run - 1)

            row = dict(
                subj=subj, session=sess, run=run,
                trial_in_run=trial_in_run,
                subj_trial_idx=counter,
                onset_TR=int(onset_tr),
                window_valid=bool(win_valid),
                nsd_73k_id=stim_73k, nsd_id_0based=nsd_id_0,
                coco_id=coco_id, is_shared1000=is_shared, rep_index=rep,
                n_things=len(things),
                supercats=','.join(sorted(supers_present)),
            )
            for s in SUPERCATS:
                row['lbl_' + s] = 1 if s in supers_present else 0
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(['session', 'run', 'trial_in_run']).reset_index(drop=True)
    df['bold_tensor_idx'] = np.arange(len(df))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subj', type=int, required=True)
    ap.add_argument('--nsd_root', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    df = build_manifest(args.subj, args.nsd_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print('trials: ' + str(len(df)) + ' -> ' + str(out))


if __name__ == '__main__':
    main()
