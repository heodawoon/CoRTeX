import re
from pathlib import Path

import yaml


def deep_update(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config(path):
    path = Path(path).resolve()
    cfg = {}
    bases = yaml.safe_load(path.read_text()).get('base', []) or []
    if isinstance(bases, str):
        bases = [bases]
    for b in bases:
        bpath = (path.parent / b).resolve() if not Path(b).is_absolute() else Path(b)
        deep_update(cfg, load_config(bpath))
    data = yaml.safe_load(path.read_text())
    data.pop('base', None)
    deep_update(cfg, data)
    return cfg


def apply_overrides(cfg, overrides):
    sci_re = re.compile(r'^[+-]?\d+(?:\.\d*)?[eE][+-]?\d+$')
    for ov in overrides or []:
        if '=' not in ov:
            raise ValueError('override must be "key=value", got ' + repr(ov))
        k, _, v = ov.partition('=')
        try:
            val = yaml.safe_load(v)
        except Exception:
            val = v
        if isinstance(val, str) and sci_re.match(val):
            val = float(val)
        parts = k.split('.')
        node = cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return cfg
