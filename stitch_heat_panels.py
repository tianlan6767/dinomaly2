"""原图 | GT | 热力图/max | 热力图/floor | pred_mask。

方式 1：本图异常图 / 本图最大值，峰值总是红色，只看响应位置。
方式 2：异常图 / pred_mask 的绝对门槛 ng_abs_floor，到红色才够 mask 像素门槛。
pred_mask 仍是整图门控后的二值结果。

示例::

    python stitch_heat_panels.py ^
      --data-root E:/.../ad_fine_crop_800_5cls ^
      --checkpoint saved_results/dinomaly2_fine_crop_800_5cls_in800/model_5000.pth ^
      --calib runs/fine_crop_800_5cls_eval_5000_masks/calibration.json ^
      --image_size 798 --crop_size 798
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stitch_panels import _add_label, _gray_to_rgb, _load_mask_gray, _load_rgb, _meta_mask_index
from test import _forward_amap, _image_score_from_amap, _meta_items, build_model, mask_gated


def _jet(norm: np.ndarray) -> np.ndarray:
    u8 = (np.clip(norm, 0.0, 1.0) * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _annotate_pred(pred: np.ndarray, score: float, thr: float, peak: float) -> np.ndarray:
    rgb = _gray_to_rgb(pred)
    lines = [
        f"score {score:.5f}",
        f"thr   {thr:.5f}",
        f"max   {peak:.5f}",
    ]
    band = np.zeros((22 * len(lines) + 8, rgb.shape[1], 3), dtype=np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(
            band,
            line,
            (8, 20 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return np.vstack([band, rgb])


def _panel(rgb, gt, heat_max, heat_floor, pred_rgb, gap: int = 6) -> np.ndarray:
    extra = pred_rgb.shape[0] - rgb.shape[0]

    def _align(img: np.ndarray) -> np.ndarray:
        if extra <= 0:
            return img
        blank = np.zeros((extra, img.shape[1], 3), dtype=np.uint8)
        return np.vstack([blank, img])

    parts = [
        _add_label(_align(rgb), "image"),
        _add_label(_align(_gray_to_rgb(gt)), "gt"),
        _add_label(_align(heat_max), "heat/max"),
        _add_label(_align(heat_floor), "heat/floor"),
        _add_label(pred_rgb, "pred"),
    ]
    sep = np.full((parts[0].shape[0], gap, 3), 48, dtype=np.uint8)
    out = parts[0]
    for part in parts[1:]:
        out = np.hstack([out, sep, part])
    return out


def main():
    p = argparse.ArgumentParser(description="Stitch image, GT, two heatmaps, pred mask")
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--calib", required=True, help="calibration.json from test.py")
    p.add_argument("--out", default="", help="default <calib dir>/panels_heat")
    p.add_argument("--image_size", type=int, default=798)
    p.add_argument("--crop_size", type=int, default=798)
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--backbone", default="dinov2reg_vit_small_14")
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--la", type=int, default=1)
    p.add_argument("--lc", type=int, default=2)
    p.add_argument("--cr", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--only-ng", action="store_true")
    p.add_argument("--img-thr", type=float, default=-1.0, help="override calib img_thr; pixel floor unchanged")
    args = p.parse_args()

    data_root = Path(args.data_root)
    calib_path = Path(args.calib)
    calib = json.loads(calib_path.read_text(encoding="utf-8"))
    if args.img_thr >= 0:
        calib["img_thr"] = float(args.img_thr)
    floor = float(calib["ng_abs_floor"])
    img_thr = float(calib["img_thr"])
    out_root = Path(args.out) if args.out else calib_path.parent / "panels_heat"

    meta = json.loads((data_root / "meta.json").read_text(encoding="utf-8"))
    gt_index = _meta_mask_index(meta)
    items = _meta_items(meta, "test")
    if args.only_ng:
        items = [it for it in items if int(it.get("anomaly", 0)) == 1]
    if args.limit > 0:
        items = items[: args.limit]

    args.upscale = True
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(
        f"device: {device}  images: {len(items)}  floor: {floor:.6f}  img_thr: {calib['img_thr']:.6f}",
        flush=True,
    )
    model = build_model(args, device)

    for i, it in enumerate(items, 1):
        img_rel = it["img_path"].replace("\\", "/")
        img_path = data_root / img_rel.replace("/", "\\")
        if not img_path.is_file():
            print(f"skip missing {img_path}", flush=True)
            continue
        rgb = _load_rgb(img_path)
        h, w = rgb.shape[:2]
        gt_rel = gt_index.get(img_rel)
        gt_path = data_root / gt_rel.replace("/", "\\") if gt_rel else None
        gt = _load_mask_gray(gt_path, (h, w))

        pil = Image.open(img_path).convert("RGB")
        amap, _ = _forward_amap(model, pil, args, device)
        if amap.shape[:2] != (h, w):
            amap = cv2.resize(amap, (w, h), interpolation=cv2.INTER_LINEAR)
        score = _image_score_from_amap(amap)
        pred = mask_gated(amap, score, calib)
        peak = float(amap.max())
        heat_max = _jet(amap / (peak + 1e-8))
        heat_floor = _jet(amap / floor)
        pred_rgb = _annotate_pred(pred, score, img_thr, peak)

        panel = _panel(rgb, gt, heat_max, heat_floor, pred_rgb)
        dest = out_root / Path(img_rel).parent / f"{Path(img_rel).stem}_panel.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        cv2.imencode(".png", bgr)[1].tofile(str(dest))
        if i % 20 == 0 or i == len(items):
            print(f"  panels {i}/{len(items)}", flush=True)

    print(f"wrote panels under {out_root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
