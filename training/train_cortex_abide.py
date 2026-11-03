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
from models import CORTEXConfig, build_cortex
from .config_utils  import load_config, apply_overrides
from .metrics       import binary_subject_metrics
from .checkpointing import Checkpointer
from .scheduler     import cosine_with_warmup


def build_cortex_for_abide(cfg, centroids=None):
    m, tr = cfg['model'], cfg['train']
    R = m.get('R', 200)
    T = tr.get('T', DEFAULT_WINDOW)
    rc = CORTEXConfig(
        R=R, T=T, M=tr.get('M', 20),
        position_mode=m.get('position_mode', 'none'),
        attn_kind=m.get('attn_kind', 'standard'),
        head=m.get('head', 'mlp'),
        d_attn=m.get('d_attn', 64), nerf_n_freqs=m.get('nerf_n_freqs', 6),
        prp_d_hidden=m.get('prp_d_hidden', 16),
        d_hidden_rsr=m.get('d_hidden_rsr', 1024),
        rsti_sharpness=m.get('rsti_sharpness', 10.0),
        rsti_init_logits_std=m.get('rsti_init_logits_std', 1.0),
        rsti_diversity_weight=m.get('rsti_diversity_weight', 0.01),
        hebbian_alpha_init=m.get('hebbian_alpha_init', 0.5),
        hebbian_rate_init_std=m.get('hebbian_rate_init_std', 0.5),
        hebbian_H_clip=m.get('hebbian_H_clip', 5.0),
        n_classes=2,
        prp_dropout=tr.get('prp_dropout', 0.3),
        rsti_dropout=tr.get('rsti_dropout', 0.1),
        head_dropout=tr.get('head_dropout', 0.3),
    )
    return build_cortex(rc, centroids=centroids)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ys, sids = [], [], []
    for seg, y, sid in loader:
        seg = seg.to(device, non_blocking=True)
        out = model(seg)
        logits = out['logits'] if isinstance(out, dict) else out
        if logits.ndim == 2 and logits.shape[1] == 2:
            p = F.softmax(logits, dim=-1)[:, 1].cpu()
        else:
            p = torch.sigmoid(logits).cpu()
            if p.ndim > 1:
                p = p.mean(dim=-1)
        probs.append(p); ys.append(y); sids.append(sid)
    return (torch.cat(probs).numpy(),
            torch.cat(ys).numpy(),
            torch.cat(sids).numpy())


def aggregate_subject_level(probs, ys, sids):
    unique = np.unique(sids)
    p_subj = np.array([probs[sids == s].mean() for s in unique], dtype=np.float32)
    y_subj = np.array([ys[sids == s][0]            for s in unique], dtype=np.int64)
    return p_subj, y_subj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--split_json', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--override', nargs='*', default=[])
    ap.add_argument('--output', required=True)
    ap.add_argument('--centroids', default='',
                    help='path to ROI centroid .npy (required for nerf modes)')
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)
    cfg.setdefault('run', {})
    cfg['run']['fold'] = args.fold
    cfg['run']['config_name'] = Path(args.config).stem

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    manifest = pd.read_csv(args.manifest)
    splits   = json.loads(Path(args.split_json).read_text())
    fold_key = 'fold_' + str(args.fold) if ('fold_' + str(args.fold)) in splits else 'fold' + str(args.fold)
    fold = splits[fold_key]

    train_sids = list(fold['train'])
    val_sids   = list(fold['val'])
    test_sids  = list(fold['test'])

    window  = cfg['train'].get('T', DEFAULT_WINDOW)
    variant = cfg['data'].get('bold_variant', 'znorm')
    tr_ds = ABIDESegmentDataset(train_sids, manifest, variant, window)
    va_ds = ABIDESegmentDataset(val_sids,   manifest, variant, window)
    te_ds = ABIDESegmentDataset(test_sids,  manifest, variant, window)
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
    
    model = build_cortex_for_abide(cfg, centroids).to(device)

    print('train=' + str(len(tr_ds)) + ' val=' + str(len(va_ds))
          + ' test=' + str(len(te_ds)) + ' params=' + str(sum(p.numel() for p in model.parameters())),
          flush=True)

    opt = torch.optim.AdamW(model.parameters(),
                             lr=cfg['train']['lr'],
                             weight_decay=cfg['train'].get('weight_decay', 1e-4))
    ep  = cfg['train']['epochs']

    steps_per_epoch = max(len(tr_ld), 1)
    n_steps = ep * steps_per_epoch
    warmup_steps = int(cfg['train'].get('warmup_epochs', 0)) * steps_per_epoch
    sched = cosine_with_warmup(opt, total_steps=n_steps,
                                warmup_steps=warmup_steps,
                                min_lr_ratio=cfg['train'].get('min_lr_ratio', 0.05))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(cfg, indent=2))
    ck = Checkpointer(out, metrics=('auc_subject',))

    best_val_auc = -1.0
    for epoch in range(ep):
        model.train(); t0 = time.time()
        tot, n = 0.0, 0
        for seg, y, _ in tr_ld:
            seg = seg.to(device); y = y.to(device)
            out_dict = model(seg)
            logits = out_dict['logits'] if isinstance(out_dict, dict) else out_dict
            loss = F.cross_entropy(logits, y)
            div = model.diversity_loss() if hasattr(model, 'diversity_loss') else None
            if div is not None:
                loss = loss + div
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            cfg['train'].get('grad_clip', 1.0))
            opt.step(); sched.step()
            tot += float(loss) * seg.size(0); n += seg.size(0)

        vp, vy, vsid = predict(model, va_ld, device)
        p_subj, y_subj = aggregate_subject_level(vp, vy, vsid)
        va = binary_subject_metrics(y_subj, p_subj)
        print('ep' + str(epoch) + ' tr_loss=' + ('%.4f' % (tot/max(n,1)))
              + ' val_auc=' + ('%.4f' % va['auc'])
              + ' val_bacc=' + ('%.4f' % va['balanced_acc'])
              + ' (' + ('%.1f' % (time.time()-t0)) + 's)', flush=True)

        ck.save_last({'epoch': epoch, 'model': model.state_dict(),
                       'optim': opt.state_dict(), 'sched': sched.state_dict(),
                       'cfg': cfg, 'val': va})
        if va['auc'] > best_val_auc:
            best_val_auc = va['auc']
            ck.maybe_save(epoch, {'auc_subject': va['auc']},
                           {'epoch': epoch, 'model': model.state_dict(),
                            'optim': opt.state_dict(), 'sched': sched.state_dict(),
                            'cfg': cfg, 'val': va})

    p = out / 'best_auc_subject.pt'
    if p.exists():
        ck_state = Checkpointer.load(p, map_location=device)
        model.load_state_dict(ck_state['model'])
    tp, ty, tsid = predict(model, te_ld, device)
    p_subj, y_subj = aggregate_subject_level(tp, ty, tsid)
    test_subject = binary_subject_metrics(y_subj, p_subj)
    test_segment = binary_subject_metrics(ty, tp)

    (out / 'test_metrics.json').write_text(json.dumps({
        'config_name': cfg['run']['config_name'],
        'fold': args.fold,
        'best_val_auc_subject': best_val_auc,
        'test_subject': test_subject,
        'test_segment': test_segment,
    }, indent=2))
    print('done fold' + str(args.fold) + ' subj_auc=' + ('%.3f' % test_subject['auc']))


if __name__ == '__main__':
    main()
