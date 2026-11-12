"""Dinomaly2 单图 / 测试集推理：输出 mask 与汇总指标（不修改 dinomaly_2D.py）。

单图::

    python test.py ^
      --image E:/path/to/img.png ^
      --checkpoint ./saved_results/dinomaly2_ad_test_5cls/model_5000.pth ^
      --out ./runs/infer_one

测试集（默认 --mask-mode calib。整图 score<=img_thr 则 mask 全黑；NG 为 amap/max>0.5）::

    python test.py ^
      --data-root E:/.../ad_test ^
      --checkpoint ./saved_results/dinomaly2_ad_test_5cls/model_5000.pth ^
      --out ./runs/ad_test_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

_DINO_ROOT = Path(__file__).resolve().parent
if str(_DINO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DINO_ROOT))

from dataset import get_data_transforms  # noqa: E402
from models.uad import Dinomaly  # noqa: E402
from models import vit_encoder  # noqa: E402
from models.vision_transformer import Block as VitBlock, LinearAttention2, Attention  # noqa: E402
from utils import cal_anomaly_maps, get_gaussian_kernel  # noqa: E402


def _fuse_layers(lc: int):
    if lc == 2:
        return [[0, 1, 2, 3], [4, 5, 6, 7]], [[0, 1, 2, 3], [4, 5, 6, 7]]
    raise ValueError(f"unsupported --lc {lc}; training used lc=2")


def build_model(args, device: torch.device) -> Dinomaly:
    encoder_name = args.backbone
    fuse_layer_encoder, fuse_layer_decoder = _fuse_layers(args.lc)

    if "small" not in encoder_name:
        raise ValueError("only dinov2reg_vit_small_14 supported in this script")

    embed_dim, num_heads = 384, 6
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    encoder = vit_encoder.load(encoder_name)

    dropout = args.dropout
    bottleneck = nn.ModuleList([
        nn.Sequential(nn.Linear(embed_dim, 256), nn.Dropout(p=dropout)),
        nn.Sequential(
            nn.Linear(256, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(p=dropout),
        ),
    ])
    decoder = nn.ModuleList()
    for _ in range(8):
        blk = VitBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-8),
            attn=partial(LinearAttention2, eps=1e-8) if args.la else Attention,
        )
        decoder.append(blk)

    model = Dinomaly(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        remove_class_token=False,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
        context_aware_recenter=bool(args.cr),
    )
    model = model.to(device)
    model.init_weights()
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _meta_items(meta: dict, phase: str) -> list[dict]:
    return [it for cls in meta[phase].values() for it in cls]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _print_infer_times(times_ms: list[float]) -> None:
    if not times_ms:
        return
    arr = np.asarray(times_ms, dtype=np.float64)
    warm = arr[1:] if arr.size > 1 else arr
    print(
        f"infer {len(arr)} imgs, skip 1 warmup: "
        f"mean {warm.mean():.1f} ms  median {np.median(warm):.1f} ms  "
        f"p90 {np.percentile(warm, 90):.1f} ms",
        flush=True,
    )


def _image_score_from_amap(amap: np.ndarray) -> float:
    flat = amap.reshape(-1)
    top_k = max(1, int(flat.size * 0.01))
    return float(np.sort(flat)[-top_k:].mean())


def _sample_pixels(amap: np.ndarray, max_n: int = 400_000) -> np.ndarray:
    flat = amap.astype(np.float64).ravel()
    if flat.size <= max_n:
        return flat
    rng = np.random.default_rng(0)
    return rng.choice(flat, size=max_n, replace=False)


def calibrate_gated_from_ok(
    train_ok: list[Path],
    model: Dinomaly,
    args,
    device: torch.device,
) -> tuple[dict, np.ndarray]:
    """与 ader infer_single_image 一致：train OK 上分位数得 img_thr / pix_thr。"""
    pix_chunks: list[np.ndarray] = []
    sp_list: list[float] = []
    q = float(args.mask_calib_percentile)
    margin = float(args.thr_margin)

    for i, p in enumerate(train_ok, 1):
        if not p.is_file():
            continue
        pil = Image.open(p).convert("RGB")
        amap, _ = _forward_amap(model, pil, args, device)
        pix_chunks.append(_sample_pixels(amap))
        sp_list.append(_image_score_from_amap(amap))
        if i % 50 == 0 or i == len(train_ok):
            print(f"  calibrate train_ok {i}/{len(train_ok)}", flush=True)

    if not sp_list:
        raise SystemExit("no train OK images for calibration")

    all_pix = np.concatenate(pix_chunks)
    if all_pix.size > 5_000_000:
        rng = np.random.default_rng(0)
        all_pix = rng.choice(all_pix, size=5_000_000, replace=False)

    sp_arr = np.asarray(sp_list, dtype=np.float64)
    t0_pix = float(np.percentile(all_pix, q))
    t0_sp = float(np.percentile(sp_arr, q))
    pix_thr = t0_pix * margin
    img_thr = t0_sp * margin
    ng_abs_floor = max(pix_thr * 1.6, pix_thr * 1.02)

    calib = {
        "mask_mode": "calib",
        "percentile": q,
        "margin": margin,
        "sp_def": "amap.top1pct_mean",
        "T0_pix": t0_pix,
        "T0_sp": t0_sp,
        "pix_thr": pix_thr,
        "img_thr": img_thr,
        "ng_abs_floor": ng_abs_floor,
        "ng_rel_ratio": float(args.ng_rel_ratio),
        "n_ok": len(sp_list),
        "ok_sp_min": float(sp_arr.min()),
        "ok_sp_max": float(sp_arr.max()),
        "ok_sp_p50": float(np.percentile(sp_arr, 50)),
    }
    return calib, sp_arr


def mask_gated(
    amap: np.ndarray,
    image_score: float,
    calib: dict,
) -> np.ndarray:
    """整图 score<=img_thr → 全黑；NG 用本图最大值归一化，大于 0.5 为白。"""
    out = np.zeros(amap.shape, dtype=np.uint8)
    img_thr = float(calib["img_thr"])
    if float(image_score) <= img_thr:
        return out
    am = amap.astype(np.float32)
    mx = float(am.max()) + 1e-8
    out[(am / mx) > 0.5] = 255
    return out


@torch.no_grad()
def _forward_amap(model: Dinomaly, pil: Image.Image, args, device: torch.device) -> tuple[np.ndarray, tuple[int, int]]:
    orig_w, orig_h = pil.size
    data_transform, _ = get_data_transforms(args.image_size, args.crop_size)
    tensor = data_transform(pil).unsqueeze(0).to(device)

    en, de = model(tensor)
    amap, _ = cal_anomaly_maps(en, de, tensor.shape[-1])
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    amap = gaussian_kernel(amap)
    amap_np = amap[0, 0].float().cpu().numpy()

    if args.upscale:
        amap_full = cv2.resize(amap_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    else:
        amap_full = amap_np
    return amap_full, (orig_w, orig_h)


def _mask_from_amap(amap: np.ndarray, args, thr: float | None = None) -> tuple[np.ndarray, float, float, float]:
    peak = float(amap.max())
    image_score = _image_score_from_amap(amap)
    if thr is None:
        if args.thr_mode == "rel_max":
            thr = peak * args.thr_ratio if peak > 0 else 0.0
        elif args.thr_mode == "abs":
            if args.thr < 0:
                raise SystemExit("--thr-mode abs requires --thr >= 0")
            thr = float(args.thr)
        else:
            raise SystemExit(f"unknown thr_mode: {args.thr_mode}")
    mask = (amap >= thr).astype(np.uint8) * 255
    return mask, image_score, peak, thr


@torch.no_grad()
def infer_one(model: Dinomaly, image_path: Path, args, device: torch.device):
    pil = Image.open(image_path).convert("RGB")
    amap_full, (orig_w, orig_h) = _forward_amap(model, pil, args, device)
    mask, image_score, peak, thr = _mask_from_amap(amap_full, args)
    crop = amap_full.shape[0] if not args.upscale else args.crop_size

    return {
        "amap": amap_full,
        "mask": mask,
        "image_score": image_score,
        "peak": peak,
        "thr": thr,
        "orig_size": (orig_w, orig_h),
        "crop_size": crop,
    }


def _save_mask_only(mask: np.ndarray, image_path: Path, data_root: Path, out_dir: Path) -> Path:
    try:
        rel = image_path.resolve().relative_to(data_root.resolve())
        sub = rel.parent
        stem = rel.stem
    except ValueError:
        sub = Path(".")
        stem = image_path.stem
    dest_dir = out_dir / "masks" / sub
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{stem}_mask.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", mask)
    if not ok:
        raise OSError(f"failed to encode mask: {dest}")
    buf.tofile(str(dest))
    return dest


def run_dataset(model: Dinomaly, args, device: torch.device) -> None:
    data_root = Path(args.data_root)
    meta_path = Path(args.meta) if args.meta else data_root / "meta.json"
    if not meta_path.is_file():
        raise SystemExit(f"meta not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    train_ok = [data_root / it["img_path"] for it in _meta_items(meta, "train") if int(it.get("anomaly", 0)) == 0]
    test_items = [
        (data_root / it["img_path"], int(it.get("anomaly", 0)))
        for it in _meta_items(meta, "test")
    ]
    n_test_ok = sum(1 for _, a in test_items if a == 0)
    n_test_ng = sum(1 for _, a in test_items if a == 1)
    if args.limit > 0:
        test_items = test_items[: args.limit]

    print(
        f"dataset: train_ok={len(train_ok)} test_ok={n_test_ok} test_ng={n_test_ng} "
        f"run_test={len(test_items)}",
        flush=True,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    if args.mask_mode == "calib":
        print("calibrating from train OK (img_thr / pix_thr)...", flush=True)
        calib, train_scores_arr = calibrate_gated_from_ok(train_ok, model, args, device)
        calib_path = out_dir / "calibration.json"
        calib_path.write_text(json.dumps(calib, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"calib p{calib['percentile']} img_thr={calib['img_thr']:.6f} "
            f"pix_thr={calib['pix_thr']:.6f}",
            flush=True,
        )
    else:
        calib = None
        print("scoring train OK (image-level rates only)...", flush=True)
        train_scores = []
        for i, p in enumerate(train_ok, 1):
            if not p.is_file():
                continue
            pil = Image.open(p).convert("RGB")
            amap, _ = _forward_amap(model, pil, args, device)
            train_scores.append(_image_score_from_amap(amap))
            if i % 50 == 0 or i == len(train_ok):
                print(f"  train_ok {i}/{len(train_ok)}", flush=True)
        train_scores_arr = np.asarray(train_scores, dtype=np.float64)

    percentiles = [float(x) for x in args.percentiles.split(",") if x.strip()]
    if args.mask_mode == "calib" and args.mask_calib_percentile not in percentiles:
        percentiles.append(float(args.mask_calib_percentile))
        percentiles = sorted(set(percentiles))
    rates = []
    test_ok_scores: list[float] = []
    test_ng_scores: list[float] = []
    infer_ms: list[float] = []

    for idx, (img_path, anomaly) in enumerate(test_items, 1):
        if not img_path.is_file():
            print(f"skip missing {img_path}", flush=True)
            continue
        pil = Image.open(img_path).convert("RGB")
        _sync(device)
        t0 = time.perf_counter()
        amap, _ = _forward_amap(model, pil, args, device)
        is_ng = anomaly == 1
        image_score = _image_score_from_amap(amap)
        if calib is not None:
            mask = mask_gated(amap, image_score, calib)
            mask_thr_note = f"calib img_thr={calib['img_thr']:.6g}"
            mask_thr_val = float(calib["img_thr"])
        else:
            mask, _, _, thr_rel = _mask_from_amap(amap, args)
            mask_thr_note = args.thr_mode
            mask_thr_val = thr_rel
        _sync(device)
        infer_ms.append((time.perf_counter() - t0) * 1000.0)
        if is_ng:
            test_ng_scores.append(image_score)
        else:
            test_ok_scores.append(image_score)

        peak = float(amap.max())
        mask_path = _save_mask_only(mask, img_path, data_root, out_dir)
        rows.append(
            {
                "img_path": str(img_path.relative_to(data_root)).replace("\\", "/"),
                "anomaly": int(is_ng),
                "image_score": image_score,
                "peak": peak,
                "mask_path": str(mask_path.relative_to(out_dir)).replace("\\", "/"),
                "mask_thr_mode": mask_thr_note,
                "mask_thr": mask_thr_val,
            }
        )
        if args.save_heatmap:
            result = {"amap": amap, "mask": mask, "image_score": image_score, "peak": peak, "thr": mask_thr_val}
            _save_outputs(result, img_path, out_dir / "detail")

        if idx % 20 == 0 or idx == len(test_items):
            print(f"  test {idx}/{len(test_items)}  last {infer_ms[-1]:.1f} ms", flush=True)

    _print_infer_times(infer_ms)

    scores_csv = out_dir / "test_scores.csv"
    with scores_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["img_path"])
        w.writeheader()
        w.writerows(rows)

    test_ok_arr = np.asarray(test_ok_scores, dtype=np.float64)
    test_ng_arr = np.asarray(test_ng_scores, dtype=np.float64)
    for q in percentiles:
        thr = float(np.percentile(train_scores_arr, q))
        ok_hit = int((test_ok_arr > thr).sum())
        ng_hit = int((test_ng_arr > thr).sum())
        rates.append(
            {
                "percentile": q,
                "img_thr": thr,
                "n_test_ok": int(len(test_ok_arr)),
                "n_test_ng": int(len(test_ng_arr)),
                "overkill": ok_hit,
                "detect": ng_hit,
                "overkill_rate": ok_hit / max(len(test_ok_arr), 1),
                "detect_rate": ng_hit / max(len(test_ng_arr), 1),
            }
        )

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_root": str(data_root.resolve()),
        "n_train_ok": len(train_scores_arr),
        "train_score_p50": float(np.percentile(train_scores_arr, 50)),
        "train_score_min": float(train_scores_arr.min()),
        "train_score_max": float(train_scores_arr.max()),
        "n_test_ok": len(test_ok_arr),
        "n_test_ng": len(test_ng_arr),
        "mask_mode": args.mask_mode,
        "mask_calib_percentile": float(args.mask_calib_percentile),
        "rates": rates,
        "scores_csv": str(scores_csv.name),
    }
    if calib is not None:
        summary["calibration"] = calib
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    for row in rates:
        print(
            f"p{row['percentile']:g} thr={row['img_thr']:.6f} "
            f"检测 {row['detect']}/{row['n_test_ng']} ({row['detect_rate']:.1%}) "
            f"过杀 {row['overkill']}/{row['n_test_ok']} ({row['overkill_rate']:.1%})",
            flush=True,
        )
    print(f"wrote {summary_path}", flush=True)
    print(f"masks under {out_dir / 'masks'}", flush=True)


def _save_outputs(result: dict, image_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    amap = result["amap"]
    amap_norm = amap.copy()
    lo, hi = float(amap_norm.min()), float(amap_norm.max())
    if hi > lo:
        amap_norm = (amap_norm - lo) / (hi - lo)
    heat_u8 = (amap_norm * 255.0).clip(0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    cv2.imwrite(str(out_dir / f"{stem}_heatmap.png"), heat_color)
    cv2.imwrite(str(out_dir / f"{stem}_mask.png"), result["mask"])
    np.save(str(out_dir / f"{stem}_amap.npy"), amap.astype(np.float32))

    meta = out_dir / f"{stem}_score.txt"
    meta.write_text(
        "\n".join(
            [
                f"image: {image_path}",
                f"image_score(top1%_mean): {result['image_score']:.6f}",
                f"peak: {result['peak']:.6f}",
                f"thr: {result['thr']:.6f}",
                f"orig_size: {result['orig_size']}",
                f"amap_shape: {amap.shape}",
            ]
        ),
        encoding="utf-8",
    )


def parse_args():
    p = argparse.ArgumentParser(description="Dinomaly2 infer + mask (single or dataset)")
    p.add_argument("--image", default="", help="single image; omit when using --data-root")
    p.add_argument("--data-root", default="", help="dataset root with meta.json (e.g. ad_test)")
    p.add_argument("--meta", default="", help="meta.json path; default <data-root>/meta.json")
    p.add_argument("--checkpoint", required=True, help="model_*.pth state_dict")
    p.add_argument("--out", default="./runs/infer_single", help="output directory")
    p.add_argument(
        "--percentiles",
        default="95,99.5",
        help="image-score thresholds from train OK (dataset mode)",
    )
    p.add_argument("--limit", type=int, default=0, help="max test images (0=all)")
    p.add_argument("--save-heatmap", action="store_true", help="also save heatmap/amap per image (slow, large)")
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--image_size", type=int, default=448)
    p.add_argument("--crop_size", type=int, default=392)
    p.add_argument("--backbone", type=str, default="dinov2reg_vit_small_14")
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--la", type=int, default=1)
    p.add_argument("--lc", type=int, default=2)
    p.add_argument("--cr", type=int, default=1)
    p.add_argument(
        "--mask-mode",
        choices=["calib", "rel_max", "abs"],
        default="calib",
        help="calib: train OK p分位 img_thr/pix_thr + 整图门控（与 ader 产线一致）",
    )
    p.add_argument(
        "--mask-calib-percentile",
        type=float,
        default=99.5,
        help="mask 与 summary 图像阈同源分位，默认 99.5",
    )
    p.add_argument("--thr-margin", type=float, default=0.85, help="标定阈值 × margin")
    p.add_argument("--ng-rel-ratio", type=float, default=0.55, help="NG 图 mask 相对峰值比例")
    p.add_argument(
        "--calib-json",
        default="",
        help="单图模式读已有 calibration.json（mask-mode=calib 时）",
    )
    p.add_argument(
        "--thr-mode",
        choices=["rel_max", "abs"],
        default="rel_max",
        help="仅 mask-mode=rel_max/abs 时用于单图或非 calib 数据集",
    )
    p.add_argument("--thr-ratio", type=float, default=0.5, help="default 0.5 for rel_max")
    p.add_argument("--thr", type=float, default=-1.0, help="absolute threshold when thr-mode=abs")
    p.add_argument(
        "--upscale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resize amap/mask to original image size (default: on)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    model = build_model(args, device)
    print(f"loaded {ckpt}", flush=True)

    if args.data_root:
        run_dataset(model, args, device)
        return

    if not args.image:
        raise SystemExit("provide --image or --data-root")
    image_path = Path(args.image)
    if not image_path.is_file():
        raise SystemExit(f"image not found: {image_path}")

    out_dir = Path(args.out)
    if args.mask_mode == "calib":
        if not args.calib_json:
            raise SystemExit("单图 calib mask 需要 --calib-json（先跑 --data-root 生成）")
        calib = json.loads(Path(args.calib_json).read_text(encoding="utf-8"))
        pil = Image.open(image_path).convert("RGB")
        _sync(device)
        t0 = time.perf_counter()
        amap, (orig_w, orig_h) = _forward_amap(model, pil, args, device)
        image_score = _image_score_from_amap(amap)
        mask = mask_gated(amap, image_score, calib)
        _sync(device)
        one_ms = (time.perf_counter() - t0) * 1000.0
        result = {
            "amap": amap,
            "mask": mask,
            "image_score": image_score,
            "peak": float(amap.max()),
            "thr": float(calib["img_thr"]),
            "orig_size": (orig_w, orig_h),
            "crop_size": args.crop_size,
        }
    else:
        _sync(device)
        t0 = time.perf_counter()
        result = infer_one(model, image_path, args, device)
        _sync(device)
        one_ms = (time.perf_counter() - t0) * 1000.0
    _save_outputs(result, image_path, out_dir)

    print(
        f"image_score={result['image_score']:.6f} peak={result['peak']:.6f} thr={result['thr']:.6f}",
        flush=True,
    )
    print(f"infer {one_ms:.1f} ms", flush=True)
    print(f"saved to {out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
