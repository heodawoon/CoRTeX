import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.abide_segments import ABIDESegmentDataset, DEFAULT_WINDOW
from models                 import CORTEXConfig, build_cortex
from .config_utils  import load_config, apply_overrides
from .metrics       import binary_subject_metrics
from .checkpointing import Checkpointer
from .scheduler     import cosine_with_warmup


def build_cortex_for_cobre(cfg, centroids=None):
    m = cfg['model']; tr = cfg['train']
    rc = CORTEXConfig(
        R=m.get('R', 200),
        T=tr.get('T', DEFAULT_WINDOW),
        M=tr.get('M', 20),
        position_mode=m.get('position_mode', 'none'),
        attn_kind=m.get('attn_kind', 'standard'),
        head=m.get('head', 'mlp'),
        d_attn=m.get('d_attn', 64),
        nerf_n_freqs=m.get('nerf_n_freqs', 6),
        prp_d_hidden=m.get('prp_d_hidden', 16),
        d_hidden_rsr=m.get('d_hidden_rsr', 1024),
        rsti_sharpness=m.get('rsti_sharpness', 10.0),
        rsti_init_logits_std=m.get('rsti_init_logits_std', 1.0),
        rsti_diversity_weight=m.get('rsti_diversity_weight', 0.01),
        hebbian_alpha_init=m.get('hebbian_alpha_init', 0.5),
        hebbian_rate_init_std=m.get('hebbian_rate_init_std', 0.5),
        hebbian_H_clip=m.get('hebbian_H_clip', 5.0),
        n_classes=2,
        cls_e2e_filters=m.get('cls_e2e_filters', 8),
        cls_e2n_filters=m.get('cls_e2n_filters', 16),
        cls_fc_hidden=m.get('cls_fc_hidden', 64),
        mlp_head_hidden=tuple(m.get('mlp_head_hidden', (1024, 256))),
        mlp_head_upper_tri=m.get('mlp_head_upper_tri', True),
        mlp_head_activation=m.get('mlp_head_activation', 'relu'),
        prp_dropout=cfg['train'].get('prp_dropout', 0.3),
        rsti_dropout=cfg['train'].get('rsti_dropout', 0.1),
        head_dropout=cfg['train'].get('head_dropout', 0.3),
    )
    return build_cortex(rc, centroids=centroids)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ys, sids = [], [], []
    for x, y, sid in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        logits = out['logits'] if isinstance(out, dict) else out
        if logits.ndim == 2 and logits.shape[1] == 2:
            p = F.softmax(logits, dim=-1)[:, 1].cpu()
        else:
            p = torch.sigmoid(logits).cpu()
        probs.append(p); ys.append(y); sids.append(sid)
    return (torch.cat(probs).numpy(),
            torch.cat(ys).numpy(),
            torch.cat(sids).numpy())


def aggregate_subject_level(probs, ys, sids):
    unique = np.unique(sids)
    p_subj = np.array([probs[sids == s].mean() for s in unique], dtype=np.float32)
    y_subj = np.array([ys[sids == s][0]            for s in unique], dtype=np.int64)
    return p_subj, y_subj


def save_predictions(out_dir, suffix, y_proba, y_true, sid_idx, level):
    np.savez_compressed(
        out_dir / ('predictions' + suffix + '.npz'),
        y_proba=y_proba.astype(np.float32),
        y_pred=(y_proba >= 0.5).astype(np.int8),
        y_true=y_true.astype(np.int8),
        sid_idx=sid_idx.astype(np.int64),
    )
    (out_dir / ('predictions_manifest' + suffix + '.json')).write_text(json.dumps({
        'level': level,
        'n_samples': int(y_proba.shape[0]),
        'n_classes': 2,
        'threshold': 0.5,
    }, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--split_json', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--variant', default='znorm')
    ap.add_argument('--override', nargs='*', default=[])
    ap.add_argument('--output', required=True)
    ap.add_argument('--centroids', default='',
                    help='path to ROI centroid .npy (required for nerf modes)')
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)
    cfg.setdefault('run', {})
    cfg['run'].update(fold=args.fold, config_name=Path(args.config).stem,
                       variant=args.variant)

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    manifest = pd.read_csv(args.manifest)
    splits   = json.loads(Path(args.split_json).read_text())
    fold_key = 'fold_' + str(args.fold) if ('fold_' + str(args.fold)) in splits else 'fold' + str(args.fold)
    fold = splits[fold_key]
    train_sids = fold['train']; val_sids = fold['val']; test_sids = fold['test']

    window    = cfg['train'].get('T', DEFAULT_WINDOW)
    bold_keys = cfg['data'].get('bold_keys', None)
    resample  = bool(cfg['data'].get('resample', True))

    tr_ds = ABIDESegmentDataset(train_sids, manifest, args.variant, window,
                                  bold_keys=bold_keys, resample=resample)
    va_ds = ABIDESegmentDataset(val_sids,   manifest, args.variant, window,
                                  bold_keys=bold_keys, resample=resample)
    te_ds = ABIDESegmentDataset(test_sids,  manifest, args.variant, window,
                                  bold_keys=bold_keys, resample=resample)

    bs = cfg['train']['batch_size']
    kw = dict(num_workers=cfg['train'].get('num_workers', 2), pin_memory=True)
    tr_ld = DataLoader(tr_ds, batch_size=bs, shuffle=True,  drop_last=True, **kw)
    va_ld = DataLoader(va_ds, batch_size=bs, shuffle=False, **kw)
    te_ld = DataLoader(te_ds, batch_size=bs, shuffle=False, **kw)
    
    centroids = None
    if cfg['model'].get('position_mode') in ('nerf', 'nerf_plus_onehot'):
        assert args.centroids, '--centroids required for nerf position modes'
        centroids = torch.from_numpy(np.load(args.centroids)).float()
        assert centroids.shape == (cfg['model'].get('R', 200), 3)
        
    model = build_cortex_for_cobre(cfg, centroids).to(device)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(cfg, indent=2))

    print('train=' + str(len(tr_ds)) + ' val=' + str(len(va_ds))
          + ' test=' + str(len(te_ds)) + ' params=' + str(sum(p.numel() for p in model.parameters())),
          flush=True)

    opt = torch.optim.AdamW(model.parameters(),
                             lr=cfg['train']['lr'],
                             weight_decay=cfg['train'].get('weight_decay', 1e-5))
    n_ep = cfg['train']['epochs']
    steps_per_epoch = max(len(tr_ld), 1)
    warmup_steps = int(cfg['train'].get('warmup_epochs', 0)) * steps_per_epoch
    sched = cosine_with_warmup(opt,
        total_steps=n_ep * steps_per_epoch,
        warmup_steps=warmup_steps,
        min_lr_ratio=cfg['train'].get('min_lr_ratio', 0.05))

    ck = Checkpointer(out, metrics=('auc_subject',))

    best_val_auc = -1.0
    val_log = []
    eval_test_each_epoch = bool(cfg['train'].get('eval_test_each_epoch', False))
    es_patience = int(cfg['train'].get('early_stopping_patience', 0))
    es_counter = 0
    es_stopped_epoch = None

    for epoch in range(n_ep):
        model.train(); t0 = time.time()
        tot, n = 0.0, 0
        for x, y, _ in tr_ld:
            x = x.to(device); y = y.to(device)
            mout = model(x)
            logits = mout['logits'] if isinstance(mout, dict) else mout
            loss = F.cross_entropy(logits, y)
            div = model.diversity_loss() if hasattr(model, 'diversity_loss') else None
            if div is not None:
                loss = loss + div
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            cfg['train'].get('grad_clip', 1.0))
            opt.step(); sched.step()
            tot += float(loss) * x.size(0); n += x.size(0)
        tr_loss = tot / max(n, 1)

        vp, vy, vsid = predict(model, va_ld, device)
        p_subj, y_subj = aggregate_subject_level(vp, vy, vsid)
        va = binary_subject_metrics(y_subj, p_subj)
        row = {'epoch': epoch, 'val_subj_auc': va['auc'],
               'val_subj_bacc': va['balanced_acc'],
               'val_subj_acc':  va['acc'],
               'tr_loss':       tr_loss,
               'time_s':        time.time() - t0}

        if eval_test_each_epoch:
            tp_e, ty_e, tsid_e = predict(model, te_ld, device)
            p_te, y_te = aggregate_subject_level(tp_e, ty_e, tsid_e)
            te = binary_subject_metrics(y_te, p_te)
            row.update({
                'test_subj_auc':  te['auc'],
                'test_subj_bacc': te['balanced_acc'],
                'test_subj_acc':  te['acc'],
                'test_subj_sens': te['sens'],
                'test_subj_spec': te['spec'],
                'test_subj_f1':   te['f1'],
            })
        val_log.append(row)
        msg = ('ep' + str(epoch) + ' tr_loss=' + ('%.4f' % tr_loss)
               + ' val_auc=' + ('%.4f' % va['auc'])
               + ' val_bacc=' + ('%.4f' % va['balanced_acc']))
        if eval_test_each_epoch:
            msg += ' test_auc=' + ('%.4f' % row['test_subj_auc'])
        msg += ' (' + ('%.1f' % (time.time()-t0)) + 's)'
        print(msg, flush=True)

        ck.save_last({'epoch': epoch, 'model': model.state_dict(),
                       'optim': opt.state_dict(), 'sched': sched.state_dict(),
                       'cfg': cfg, 'val': va})
        if va['auc'] > best_val_auc + 1e-6:
            best_val_auc = va['auc']
            es_counter = 0
            ck.maybe_save(epoch, {'auc_subject': va['auc']},
                {'epoch': epoch, 'model': model.state_dict(),
                 'optim': opt.state_dict(), 'sched': sched.state_dict(),
                 'cfg': cfg, 'val': va})
        else:
            es_counter += 1
            if es_patience > 0 and es_counter >= es_patience:
                es_stopped_epoch = epoch
                print('early stop at ep' + str(epoch), flush=True)
                break

    pd.DataFrame(val_log).to_csv(out / 'val_log.csv', index=False)
    if es_stopped_epoch is not None:
        (out / 'early_stop.json').write_text(json.dumps({
            'stopped_at_epoch': es_stopped_epoch,
            'best_val_auc': best_val_auc,
            'patience': es_patience,
        }, indent=2))

    p = out / 'best_auc_subject.pt'
    if p.exists():
        ck_state = Checkpointer.load(p, map_location=device)
        model.load_state_dict(ck_state['model'])

    tp, ty, tsid = predict(model, te_ld, device)
    p_subj, y_subj = aggregate_subject_level(tp, ty, tsid)
    sids_subj = np.unique(tsid)

    test_subject = binary_subject_metrics(y_subj, p_subj)
    test_segment = binary_subject_metrics(ty, tp)

    (out / 'test_metrics.json').write_text(json.dumps({
        'config_name': cfg['run']['config_name'],
        'variant': args.variant, 'fold': args.fold,
        'best_val_auc_subject': best_val_auc,
        'test_subject': test_subject,
        'test_segment': test_segment,
    }, indent=2))

    save_predictions(out, '_test_subject', p_subj, y_subj, sids_subj, 'subject')
    save_predictions(out, '_test_segment', tp,     ty,     tsid,      'segment')

    print('done fold' + str(args.fold) + ' subj_auc=' + ('%.3f' % test_subject['auc']), flush=True)


if __name__ == '__main__':
    main()
