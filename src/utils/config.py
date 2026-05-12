import argparse
import yaml
from types import SimpleNamespace


def load_config(defaults: dict) -> SimpleNamespace:
    """Load YAML config, then override with any extra CLI args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    for k, v in defaults.items():
        parser.add_argument(f"--{k}", type=type(v) if v is not None else str, default=None)
    args = parser.parse_args()

    cfg = dict(defaults)
    if args.config:
        with open(args.config) as f:
            for k, v in yaml.safe_load(f).items():
                expected = type(defaults[k]) if k in defaults and defaults[k] is not None else None
                cfg[k] = expected(v) if expected is not None else v
    for k in defaults:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    return SimpleNamespace(**cfg)
