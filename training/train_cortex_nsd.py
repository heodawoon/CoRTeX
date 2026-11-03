import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset.trial_dataset    import TrialDataset
from dataset.avg_test_dataset import AvgTrialBOLDTestDataset
from dataset.labels           import LABEL_COLS, N_CLASSES
from models          import CORTEXConfig, build_cortex
from .config_utils   import load_config, apply_overrides
from .losses         import multilabel_bce, compute_pos_weight
from .metrics        import multilabel_metrics
from .checkpointing  import Checkpointer
from .scheduler      import cosine_with_warmup


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, ys = [], []
    for bold, y, _ in loader:
        bold = bold.to(device, non_blocking=True)
        p = torch.sigmoid(model(bold)['logits']).cpu()
        probs.append(p); ys.append(y)
    p = torch.cat(probs).numpy(); y = torch.cat(ys).numpy()
    return multilabel_metrics(y, p), p, y


@torch.no_grad()
def evaluate_loss(model, loader, device, pos_w):
    model.eval()
    tot, n = 0.0, 0
    for bold, y, _ in loader:
        bold = bold.to(device, non_blocking=True)
        y    = y.to(device, non_blocking=True)
        d = multilabel_bce(model(bold)['logits'], y, pos_weight=pos_w)
        tot += float(d['loss']) * bold.size(0); n += bold.size(0)
    return tot / max(n, 1)


def build_model_from_cfg(cfg, centroids):
    m = cfg['model']
    rcfg = CORTEXConfig(
        R=m.get('R', 360), T=cfg['train']['T'], M=cfg['train']['M'],
        position_mode=m['position_mode'], attn_kind=m['attn_kind'],
        head=m.get('head', 'brainnet'),
        d_attn=m.get('d_attn', 64), nerf_n_freqs=m.get('nerf_n_freqs', 6),
        include_bold_in_feat=m.get('include_bold_in_feat', True),
        prp_d_hidden=m.get('prp_d_hidden', 16),
        d_hidden_rsr=m.get('d_hidden_rsr', 1024),
        rsti_sharpness=m.get('rsti_sharpness', 10.0),
        rsti_init_logits_std=m.get('rsti_init_logits_std', 1.0),
        rsti_diversity_weight=m.get('rsti_diversity_weight', 0.01),
        hebbian_alpha_init=m.get('hebbian_alpha_init', 0.5),
        hebbian_rate_init_std=m.get('hebbian_rate_init_std', 0.5),
        hebbian_H_clip=m.get('hebbian_H_clip', 5.0),
        n_classes=cfg['data'].get('n_classes', N_CLASSES),
        cls_e2e_filters=m.get('cls_e2e_filters', 8),
        cls_e2n_filters=m.get('cls_e2n_filters', 16),
        cls_fc_hidden=m.get('cls_fc_hidden', 64),
        prp_dropout=cfg['train'].get('prp_dropout', 0.3),
        rsti_dropout=cfg['train'].get('rsti_dropout', 0.1),
        head_dropout=cfg['train'].get('head_dropout', 0.3),
        init_scale=m.get('init_scale', 0.1),
    )
    return build_cortex(rcfg, centroids=centroids), rcfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--subj', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--bold_path', required=True,
                    help='path to subj{XX}_bold_R360_T{T}.npz')
    ap.add_argument('--bold_avg_path', required=True,
                    help='path to subj{XX}_bold_R360_T{T}_testavg.npz')
    ap.add_argument('--manifest', required=True,
                    help='path to subj{XX}_trial_manifest.csv')
    ap.add_argument('--split_dir', required=True,
                    help='directory with train_idx.npy / val_idx.npy / test_idx.npy / test_avg_labels.npy')
    ap.add_argument('--centroids', default='',
                    help='path to ROI centroid .npy (required for nerf modes)')
    ap.add_argument('--output', required=True)
    ap.add_argument('--override', nargs='*', default=[])
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)
    cfg.setdefault('run', {})
    cfg['run']['subj'] = args.subj
    cfg['run']['config_name'] = Path(args.config).stem
    subj = args.subj

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / 'config.json').write_text(json.dumps(cfg, indent=2))

    T = cfg['train']['T']
    split_dir = Path(args.split_dir)
    tr_idx = np.load(split_dir / 'train_idx.npy')
    va_idx = np.load(split_dir / 'val_idx.npy')
    te_idx = np.load(split_dir / 'test_idx.npy')

    tr_ds = TrialDataset(args.bold_path, args.manifest, tr_idx)
    va_ds = TrialDataset(args.bold_path, args.manifest, va_idx)
    te_ds_raw = TrialDataset(args.bold_path, args.manifest, te_idx)
    te_ds_avg = AvgTrialBOLDTestDataset(args.bold_avg_path,
                                         split_dir / 'test_avg_labels.npy')

    bs = cfg['train']['batch_size']
    kw = dict(num_workers=cfg['train'].get('num_workers', 2), pin_memory=True)
    tr_ld = DataLoader(tr_ds, batch_size=bs, shuffle=True,  drop_last=True, **kw)
    va_ld = DataLoader(va_ds, batch_size=bs, shuffle=False, **kw)
    te_raw_ld = DataLoader(te_ds_raw, batch_size=bs, shuffle=False, **kw)
    te_avg_ld = DataLoader(te_ds_avg, batch_size=bs, shuffle=False, **kw)

    pos_mode = cfg['model']['position_mode']
    centroids = None
    if pos_mode in ('nerf', 'nerf_plus_onehot'):
        assert args.centroids, '--centroids required for nerf position modes'
        centroids = torch.from_numpy(np.load(args.centroids)).float()
        assert centroids.shape == (cfg['model'].get('R', 360), 3)

    model, rcfg = build_model_from_cfg(cfg, centroids)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print('train=' + str(len(tr_idx)) + ' val=' + str(len(va_idx))
          + ' test_raw=' + str(len(te_ds_raw)) + ' test_avg=' + str(len(te_ds_avg))
          + ' params=' + str(n_params), flush=True)

    lr  = cfg['train']['lr']
    wd  = cfg['train'].get('weight_decay', 1e-4)
    ep  = cfg['train']['epochs']
    hebb_lr_mult = float(cfg['train'].get('hebbian_lr_mult', 1.0))
    if hebb_lr_mult != 1.0:
        hebb_params = [p for n, p in model.named_parameters() if 'hebbian' in n and p.requires_grad]
        other_params = [p for n, p in model.named_parameters() if 'hebbian' not in n and p.requires_grad]
        opt = torch.optim.AdamW([
            {'params': other_params, 'lr': lr,                'weight_decay': wd},
            {'params': hebb_params,  'lr': lr * hebb_lr_mult, 'weight_decay': wd},
        ])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    steps_per_epoch = max(len(tr_ld), 1)
    n_steps = ep * steps_per_epoch
    warmup_steps = int(cfg['train'].get('warmup_epochs', 0)) * steps_per_epoch
    sched = cosine_with_warmup(opt, total_steps=n_steps,
                                warmup_steps=warmup_steps,
                                min_lr_ratio=cfg['train'].get('min_lr_ratio', 0.05))

    Y_train = pd.read_csv(args.manifest)[LABEL_COLS].values[tr_idx].astype(np.float32)
    pos_w = compute_pos_weight(Y_train).to(device)

    ck = Checkpointer(out_root)

    train_log_rows = []
    val_log_rows = []
    test_log_rows = []
    es_patience = int(cfg['train'].get('early_stop_patience', 0))
    es_warmup   = int(cfg['train'].get('early_stop_warmup', 0))
    es_best     = -1.0
    es_wait     = 0
    eval_test_each_epoch = bool(cfg['train'].get('eval_test_each_epoch', False))
    for epoch in range(ep):
        model.train(); t_ep = time.time()
        ep_loss, n = 0.0, 0
        for bold, y, _ in tr_ld:
            bold = bold.to(device, non_blocking=True)
            y    = y.to(device, non_blocking=True)
            d = multilabel_bce(model(bold)['logits'], y, pos_weight=pos_w)
            loss = d['loss']
            div = model.diversity_loss() if hasattr(model, 'diversity_loss') else None
            if div is not None:
                loss = loss + div
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            cfg['train'].get('grad_clip', 1.0))
            opt.step(); sched.step()
            ep_loss += float(d['loss']) * bold.size(0); n += bold.size(0)
        tr_loss = ep_loss / max(n, 1)

        va_m, _, _ = evaluate(model, va_ld, device)
        va_loss    = evaluate_loss(model, va_ld, device, pos_w)

        train_log_rows.append({'epoch': epoch, 'loss': tr_loss,
                                'lr': opt.param_groups[0]['lr'],
                                'time_s': time.time() - t_ep})
        val_log_rows.append({
            'epoch': epoch, 'loss': va_loss,
            'macro_auc': va_m['macro_auc'], 'micro_auc': va_m['micro_auc'],
            'acc_macro': va_m['acc_macro'], 'balanced_acc': va_m['balanced_acc'],
            'macro_f1':  va_m['macro_f1'],
        })
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optim': opt.state_dict(),
            'sched': sched.state_dict(),
            'best':  ck.best,
            'cfg':   cfg,
        }
        ck.save_last(state)
        ck.maybe_save(epoch, {
            'macro_auc': va_m['macro_auc'],
            'micro_auc': va_m['micro_auc'],
            'acc':       va_m['acc_macro'],
            'loss':      va_loss,
        }, state)
        if eval_test_each_epoch:
            te_avg_m, _, _ = evaluate(model, te_avg_ld, device)
            test_log_rows.append({
                'epoch': epoch,
                'macro_auc': te_avg_m['macro_auc'],
                'micro_auc': te_avg_m['micro_auc'],
                'acc_macro': te_avg_m['acc_macro'],
                'balanced_acc': te_avg_m['balanced_acc'],
                'macro_f1':    te_avg_m['macro_f1'],
            })
        msg = ('ep' + str(epoch) + ' tr=' + ('%.4f' % tr_loss)
               + ' va_loss=' + ('%.4f' % va_loss)
               + ' va_macro=' + ('%.4f' % va_m['macro_auc'])
               + ' va_micro=' + ('%.4f' % va_m['micro_auc'])
               + ' va_acc='   + ('%.4f' % va_m['acc_macro']))
        if eval_test_each_epoch:
            msg += ' te_macro=' + ('%.4f' % te_avg_m['macro_auc'])
        msg += ' (' + ('%.1f' % (time.time() - t_ep)) + 's)'
        print(msg, flush=True)

        if es_patience > 0:
            if va_m['macro_auc'] > es_best + 1e-6:
                es_best = va_m['macro_auc']
                es_wait = 0
            else:
                if epoch + 1 > es_warmup:
                    es_wait += 1
                    if es_wait >= es_patience:
                        print('early stop at ep' + str(epoch), flush=True)
                        break

    pd.DataFrame(train_log_rows).to_csv(out_root / 'train_log.csv', index=False)
    pd.DataFrame(val_log_rows).to_csv(out_root / 'val_log.csv', index=False)
    if eval_test_each_epoch and test_log_rows:
        pd.DataFrame(test_log_rows).to_csv(out_root / 'test_log_per_epoch.csv', index=False)

    for key in ('macro_auc', 'micro_auc', 'acc', 'loss'):
        p = out_root / ('best_' + key + '.pt')
        if not p.exists():
            continue
        ck_state = Checkpointer.load(p, map_location=device)
        model.load_state_dict(ck_state['model'])
        test_avg, _, _ = evaluate(model, te_avg_ld, device)
        test_raw, _, _ = evaluate(model, te_raw_ld, device)
        (out_root / ('test_metrics_by_best_' + key + '.json')).write_text(json.dumps({
            'config_name': cfg['run']['config_name'],
            'subj': subj,
            'best_on': key,
            'best_val_value': ck.best[key]['value'],
            'best_epoch':     ck.best[key]['epoch'],
            'test':              test_avg,
            'test_non_averaged': test_raw,
        }, indent=2))

    print('done subj' + subj, flush=True)


if __name__ == '__main__':
    main()
