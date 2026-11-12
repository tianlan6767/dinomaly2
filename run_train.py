"""按 configs/*.py 训练 Dinomaly2，不改动 dinomaly_2D.py。

示例::

    python run_train.py -c configs/ad_fine_crop_800_5cls.py
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dinomaly_2D import get_logger  # noqa: E402
from train_runner import resolve_item_list, train_from_config  # noqa: E402


def _load_config(path: Path):
    spec = importlib.util.spec_from_file_location("dinomaly2_cfg", str(path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load config: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(description="Dinomaly2 train from configs/*.py")
    parser.add_argument("-c", "--config", required=True, help="e.g. configs/ad_fine_crop_800_5cls.py")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        cfg_path = _ROOT / args.config
    if not cfg_path.is_file():
        raise SystemExit(f"config not found: {args.config}")

    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    cfg = _load_config(cfg_path)
    item_list = resolve_item_list(cfg.data_path, getattr(cfg, "item_list", None))

    logger = get_logger(cfg.save_name, os.path.join(cfg.save_dir, cfg.save_name))
    print_fn = logger.info
    print_fn(f"config: {cfg_path.resolve()}")
    print_fn(f"item_list: {item_list}")
    train_from_config(cfg, item_list, print_fn)


if __name__ == "__main__":
    main()
