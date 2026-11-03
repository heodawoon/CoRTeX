import numpy as np
from sklearn.metrics import (roc_auc_score, f1_score, average_precision_score,
                              balanced_accuracy_score)


def multilabel_metrics(y_true, probs, thr=0.5):
    pred = (probs >= thr).astype(int)
    per_cat_auc, per_cat_ap, per_cat_f1 = [], [], []
    per_cat_sens, per_cat_spec, per_cat_bal = [], [], []
    for c in range(y_true.shape[1]):
        yt, ps, pd_ = y_true[:, c], probs[:, c], pred[:, c]
        try:
            per_cat_auc.append(roc_auc_score(yt, ps))
        except Exception:
            per_cat_auc.append(float('nan'))
        try:
            per_cat_ap.append(average_precision_score(yt, ps))
        except Exception:
            per_cat_ap.append(float('nan'))
        per_cat_f1.append(f1_score(yt, pd_, zero_division=0))
        tp = ((yt==1)&(pd_==1)).sum(); fn = ((yt==1)&(pd_==0)).sum()
        tn = ((yt==0)&(pd_==0)).sum(); fp = ((yt==0)&(pd_==1)).sum()
        sens = tp / max(tp+fn, 1); spec = tn / max(tn+fp, 1)
        per_cat_sens.append(sens); per_cat_spec.append(spec)
        per_cat_bal.append(0.5*(sens+spec))
    try:
        micro_auc = roc_auc_score(y_true.ravel(), probs.ravel())
    except Exception:
        micro_auc = float('nan')
    return {
        'macro_auc':    float(np.nanmean(per_cat_auc)),
        'micro_auc':    float(micro_auc),
        'macro_ap':     float(np.nanmean(per_cat_ap)),
        'macro_f1':     float(np.mean(per_cat_f1)),
        'micro_f1':     float(f1_score(y_true.ravel(), pred.ravel(), zero_division=0)),
        'balanced_acc': float(np.nanmean(per_cat_bal)),
        'hamming_acc':  float((y_true == pred).mean()),
        'exact_acc':    float((y_true == pred).all(axis=1).mean()),
        'acc_macro':    float(np.mean((y_true == pred).mean(axis=0))),
        'sens_macro':   float(np.nanmean(per_cat_sens)),
        'spec_macro':   float(np.nanmean(per_cat_spec)),
        'per_cat_auc':  [float(x) for x in per_cat_auc],
    }


def binary_subject_metrics(y, probs_pos):
    try:
        auc = roc_auc_score(y, probs_pos)
    except Exception:
        auc = float('nan')
    try:
        ap = average_precision_score(y, probs_pos)
    except Exception:
        ap = float('nan')
    pred = (probs_pos >= 0.5).astype(int)
    return {
        'auc':          float(auc),
        'ap':           float(ap),
        'acc':          float((pred == y).mean()),
        'balanced_acc': float(balanced_accuracy_score(y, pred)),
        'f1':           float(f1_score(y, pred, zero_division=0)),
        'sens':         float(((y == 1) & (pred == 1)).sum() / max((y == 1).sum(), 1)),
        'spec':         float(((y == 0) & (pred == 0)).sum() / max((y == 0).sum(), 1)),
        'n_pos':        int((y == 1).sum()),
        'n_neg':        int((y == 0).sum()),
    }
