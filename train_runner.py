"""项目侧训练逻辑：不修改 upstream dinomaly_2D.py，由此模块承载定制项。"""

from __future__ import annotations

import os
from argparse import Namespace
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import ImageFolder

from dataset import MVTecDataset, get_data_transforms
from dinomaly_2D import get_logger, setup_seed
from models import vit_encoder
from models.uad import Dinomaly
from models.vision_transformer import Attention, Block as VitBlock, LinearAttention2
from optimizers import StableAdamW
from utils import (
    WarmupCosineScheduler,
    evaluation_batch,
    global_cosine,
    global_cosine_hm_percent,
)


def _fuse_layers(lc: int):
    table = {
        0: ([[0], [1], [2], [3], [4], [5], [6], [7]], [[0], [1], [2], [3], [4], [5], [6], [7]]),
        1: ([[0, 1, 2, 3, 4, 5, 6, 7]], [[0, 1, 2, 3, 4, 5, 6, 7]]),
        2: ([[0, 1, 2, 3], [4, 5, 6, 7]], [[0, 1, 2, 3], [4, 5, 6, 7]]),
        3: ([[0, 1, 2], [3, 4, 5], [6, 7]], [[0, 1, 2], [3, 4, 5], [6, 7]]),
        4: ([[0, 1], [2, 3], [4, 5], [6, 7]], [[0, 1], [2, 3], [4, 5], [6, 7]]),
        11: ([[7]], [[7]]),
        12: ([[3], [7]], [[3], [7]]),
        14: ([[1], [3], [5], [7]], [[1], [3], [5], [7]]),
    }
    if lc not in table:
        raise ValueError(f"unsupported lc={lc}")
    return table[lc]


def resolve_item_list(data_path: str, item_list) -> list[str]:
    if item_list:
        return list(item_list)
    return sorted(
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d))
    )


def cfg_module_to_args(mod) -> Namespace:
    keys = [
        "data_path", "save_dir", "save_name", "image_size", "crop_size", "batch_size",
        "num_workers", "total_iters", "save_every_iters", "eval_resize_mask",
        "backbone", "dropout", "la", "lc", "ll", "ll_ratio", "ll_factor", "cr",
        "lr_decay_ratio", "cuda", "seed",
    ]
    return Namespace(**{k: getattr(mod, k) for k in keys if hasattr(mod, k)})


def train_from_config(cfg_mod, item_list: list[str], print_fn):
    args = cfg_module_to_args(cfg_mod)
    setup_seed(int(getattr(cfg_mod, "seed", 1)))

    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    print_fn(device)
    print_fn(args)

    batch_size = args.batch_size
    num_workers = int(getattr(args, "num_workers", 0))
    data_transform, gt_transform = get_data_transforms(args.image_size, args.crop_size)

    train_data_list = []
    test_data_list = []
    for cls_idx, item in enumerate(item_list):
        train_path = os.path.join(args.data_path, item, "train")
        test_path = os.path.join(args.data_path, item)
        train_data = ImageFolder(root=train_path, transform=data_transform)
        train_data.classes = item
        train_data.class_to_idx = {item: cls_idx}
        train_data.samples = [(sample[0], cls_idx) for sample in train_data.samples]
        test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
        train_data_list.append(train_data)
        test_data_list.append(test_data)

    train_data = ConcatDataset(train_data_list)
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )

    fuse_layer_encoder, fuse_layer_decoder = _fuse_layers(args.lc)
    encoder_name = args.backbone
    encoder = vit_encoder.load(encoder_name)

    if "small" in encoder_name:
        embed_dim, num_heads = 384, 6
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    elif "base" in encoder_name:
        embed_dim, num_heads = 768, 12
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    elif "large" in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError("backbone must contain small/base/large")

    bottleneck = nn.ModuleList([
        nn.Sequential(nn.Linear(embed_dim, 256), nn.Dropout(p=args.dropout)),
        nn.Sequential(
            nn.Linear(256, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(p=args.dropout),
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
        context_aware_recenter=args.cr,
    ).to(device)
    trainable = nn.ModuleList([bottleneck, decoder])
    model.init_weights()

    optimizer = StableAdamW(
        [
            {"params": bottleneck[0].parameters(), "lr": 2e-4},
            {"params": bottleneck[1].parameters()},
            {"params": decoder.parameters()},
        ],
        lr=2e-3,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        amsgrad=False,
        eps=1e-10,
    )
    lr_scheduler = WarmupCosineScheduler(
        optimizer, final_ratio=args.lr_decay_ratio, total_epochs=args.total_iters, warmup_epochs=100
    )

    print_fn(f"train image number:{len(train_data)}")

    save_every = int(getattr(args, "save_every_iters", 5000))
    resize_mask = int(getattr(args, "eval_resize_mask", 256))

    it = 0
    total_iters = args.total_iters
    for _epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        model.train()
        loss_list = []
        for img, _label in train_dataloader:
            img = img.to(device)
            en, de = model(img)
            p_final = args.ll_ratio
            p = min(p_final * it / 1000, p_final)
            if args.ll:
                loss = global_cosine_hm_percent(en, de, p=p, factor=args.ll_factor)
            else:
                loss = global_cosine(en, de)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)
            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()

            if save_every > 0 and (it + 1) % save_every == 0:
                save_dir = os.path.join(args.save_dir, args.save_name)
                os.makedirs(save_dir, exist_ok=True)
                ckpt_path = os.path.join(save_dir, f"model_{it + 1}.pth")
                torch.save(model.state_dict(), ckpt_path)
                print_fn(f"saved checkpoint {ckpt_path}")

                for item, test_data in zip(item_list, test_data_list):
                    test_dataloader = DataLoader(
                        test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
                    )
                    results = evaluation_batch(
                        model, test_dataloader, device, max_ratio=0.01, resize_mask=resize_mask
                    )
                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
                    print_fn(
                        f"{item}: I-Auroc:{auroc_sp:.4f}, I-AP:{ap_sp:.4f}, I-F1:{f1_sp:.4f}, "
                        f"P-AUROC:{auroc_px:.4f}, P-AP:{ap_px:.4f}, P-F1:{f1_px:.4f}, P-AUPRO:{aupro_px:.4f}"
                    )
                model.train()

            it += 1
            if it == total_iters:
                break
            if (it + 1) % 100 == 0:
                print_fn(f"iter [{it}/{total_iters}], loss:{np.mean(loss_list):.4f}")
                loss_list = []
        if it == total_iters:
            break

    save_dir = os.path.join(args.save_dir, args.save_name)
    os.makedirs(save_dir, exist_ok=True)
    final_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), final_path)
    print_fn(f"saved final checkpoint {final_path}")
