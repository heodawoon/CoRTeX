import json
from pathlib import Path

import numpy as np


def save_predictions(out_dir, y_pred_proba, y_true, trial_ids,
                     split='test_avg', thresholds=None,
                     threshold_method='fixed_0.5', suffix=''):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_pred_proba = np.asarray(y_pred_proba, dtype=np.float32)
    y_true       = np.asarray(y_true,       dtype=np.int8)
    trial_ids    = np.asarray(trial_ids,    dtype=np.int64)

    if thresholds is None:
        thresholds = np.full(y_pred_proba.shape[1], 0.5, dtype=np.float32)
    else:
        thresholds = np.asarray(thresholds, dtype=np.float32)

    y_pred_binary = (y_pred_proba >= thresholds[None, :]).astype(np.int8)

    np.savez_compressed(
        out_dir / ('predictions' + suffix + '.npz'),
        y_pred_proba=y_pred_proba,
        y_pred_binary=y_pred_binary,
        y_true=y_true,
        trial_ids=trial_ids,
        thresholds=thresholds,
    )
    (out_dir / ('predictions_manifest' + suffix + '.json')).write_text(json.dumps({
        'split': split,
        'n_samples': int(y_pred_proba.shape[0]),
        'n_classes': int(y_pred_proba.shape[1]),
        'threshold_method': threshold_method,
        'shape': '(' + str(y_pred_proba.shape[0]) + ', ' + str(y_pred_proba.shape[1]) + ')',
    }, indent=2))
