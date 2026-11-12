"""将原图 | GT mask | pred mask 横向拼成一张图。

依赖 test.py 数据集评测产出的 test_scores.csv 与 masks/。

示例::

    python stitch_panels.py ^
      --data-root E:/.../ad_fine_crop_800_5cls ^
      --eval-dir runs/fine_crop_800_5cls_eval_5000
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_mask_gray(path: Path | None, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if path is None or not path.is_file():
        return np.zeros((h, w), dtype=np.uint8)
    gray = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return np.zeros((h, w), dtype=np.uint8)
    if gray.shape[:2] != (h, w):
        gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_NEAREST)
    return gray


def _gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def _add_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    bar = np.full((28, out.shape[1], 3), 32, dtype=np.uint8)
    cv2.putText(
        bar,
        text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([bar, out])


def _meta_mask_index(meta: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for phase in ("train", "test"):
        if phase not in meta:
            continue
        for items in meta[phase].values():
            for it in items:
                mp = (it.get("mask_path") or "").strip()
                if mp:
                    out[it["img_path"].replace("\\", "/")] = mp
    return out


def stitch_one(
    data_root: Path,
    eval_dir: Path,
    img_rel: str,
    mask_rel: str,
    gt_rel: str | None,
    gap: int = 6,
) -> np.ndarray:
    img_path = data_root / img_rel.replace("/", "\\")
    pred_path = eval_dir / mask_rel.replace("/", "\\")
    rgb = _load_rgb(img_path)
    h, w = rgb.shape[:2]
    gt_path = data_root / gt_rel.replace("/", "\\") if gt_rel else None
    gt = _load_mask_gray(gt_path, (h, w))
    pred = _load_mask_gray(pred_path, (h, w))

    p0 = _add_label(rgb, "image")
    p1 = _add_label(_gray_to_rgb(gt), "gt")
    p2 = _add_label(_gray_to_rgb(pred), "pred")
    sep = np.full((p0.shape[0], gap, 3), 48, dtype=np.uint8)
    return np.hstack([p0, sep, p1, sep, p2])


def main():
    p = argparse.ArgumentParser(description="Stitch image | gt | pred masks")
    p.add_argument("--data-root", required=True)
    p.add_argument("--eval-dir", required=True, help="test.py output with test_scores.csv")
    p.add_argument("--meta", default="", help="default <data-root>/meta.json")
    p.add_argument("--scores", default="", help="default <eval-dir>/test_scores.csv")
    p.add_argument("--out", default="", help="default <eval-dir>/panels")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--only-ng", action="store_true")
    p.add_argument("--only-ok", action="store_true")
    args = p.parse_args()

    data_root = Path(args.data_root)
    eval_dir = Path(args.eval_dir)
    scores_path = Path(args.scores) if args.scores else eval_dir / "test_scores.csv"
    meta_path = Path(args.meta) if args.meta else data_root / "meta.json"
    out_root = Path(args.out) if args.out else eval_dir / "panels"

    if not scores_path.is_file():
        raise SystemExit(f"missing {scores_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    gt_index = _meta_mask_index(meta)

    rows = list(csv.DictReader(scores_path.open(encoding="utf-8")))
    if args.only_ng:
        rows = [r for r in rows if int(r["anomaly"]) == 1]
    if args.only_ok:
        rows = [r for r in rows if int(r["anomaly"]) == 0]
    if args.limit > 0:
        rows = rows[: args.limit]

    n_ok = 0
    for i, row in enumerate(rows, 1):
        img_rel = row["img_path"].replace("\\", "/")
        pred_rel = row["mask_path"].replace("\\", "/")
        gt_rel = gt_index.get(img_rel)
        panel = stitch_one(data_root, eval_dir, img_rel, pred_rel, gt_rel)
        stem = Path(img_rel).stem
        dest = out_root / Path(img_rel).parent / f"{stem}_panel.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        cv2.imencode(".png", bgr)[1].tofile(str(dest))
        n_ok += 1
        if i % 40 == 0 or i == len(rows):
            print(f"  panels {i}/{len(rows)}", flush=True)

    print(f"wrote {n_ok} panels under {out_root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
