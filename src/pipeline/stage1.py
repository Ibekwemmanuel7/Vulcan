"""
Stage 1 training and full-AOI inference
=======================================

train   Fine-tunes the Attention U-Net on the target mask with a leakage-free
        spatial block split, optional SSL encoder initialisation, AMP on CUDA, and
        early stopping on validation average precision.

        Validation protocol (review finding 5): validation windows are re-planned on
        a non-overlapping lattice inside the validation blocks and the edge overlap is
        trimmed, so every validation cell is counted exactly once. AP, IoU and F1 are
        computed from binned counts over ALL eligible cells (no subsampling). Only
        label 1 (target) and label 0 (confirmed background) cells are scored; 255
        (unknown ground) is excluded.

        If the AOI is too small for a block split, training stops with an error.
        Set stage1.allow_random_split: true to run anyway; the metrics file is then
        stamped "development_only" and must not be used for performance claims.

infer   Block-with-halo sliding-window inference with cosine blending and optional
        D4 test-time augmentation. Writes stage1_features.tif:
            band 1                  target probability
            bands 2 .. 1+L          latent embedding (L = latent_channels)
            bands 2+L .. 1+L+depth  attention-gate values (internal network gates,
                                    not geological evidence)

Both functions accept path overrides so the nested evaluation can train and infer
one model per geographic fold.

    python -m src.pipeline.stage1 train --config configs/geosentry.yaml
    python -m src.pipeline.stage1 infer --config configs/geosentry.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.windows import Window
from torch.utils.data import DataLoader

from src.config import Config, get_logger, load_config, seed_everything
from src.evaluation.folds import check_same_grid
from src.evaluation.metrics import BinnedPR
from src.models.attention_unet import (
    AttentionUNet,
    FocalDiceLoss,
    GeospatialDataset,
    apply_d4_torch,
    build_model,
    invert_d4_torch,
    owned_slices,
    plan_windows,
    spatial_split,
    window_blocks,
)

LOG = get_logger("stage1")


def file_sha256(path: Path, chunk: int = 1 << 22) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# =============================================================================
# Training
# =============================================================================
@torch.no_grad()
def evaluate(model: AttentionUNet, dl: DataLoader, loss_fn: FocalDiceLoss, device: str,
             patch: int, shape) -> Dict[str, float]:
    """Unique-cell validation metrics (each cell once, all eligible cells)."""
    model.eval()
    pr = BinnedPR()
    losses: List[float] = []
    for batch in dl:
        x = batch["image"].to(device)
        y = batch["mask"].to(device)
        logits = model(x)
        loss, _ = loss_fn(logits, y)
        losses.append(float(loss))
        p = torch.sigmoid(logits.squeeze(1)).cpu().numpy()
        yy = y.cpu().numpy()
        for i in range(len(p)):
            rs, cs = owned_slices(int(batch["row"][i]), int(batch["col"][i]), patch, shape)
            pi, yi = p[i][rs, cs], yy[i][rs, cs]
            ok = (yi == 0) | (yi == 1)
            pr.update(pi[ok], yi[ok] == 1)
    out = {"loss": float(np.mean(losses)) if losses else float("nan"), "ap": pr.average_precision()}
    out.update(pr.at_threshold(0.5))
    return out


def train(cfg: Config, device: Optional[str] = None, use_ssl: bool = True,
          target_path: Optional[Path] = None, valid_path: Optional[Path] = None,
          weights_out: Optional[Path] = None, report_tag: str = "stage1", require_ssl: bool = False) -> Path:
    s1 = cfg["stage1"]
    seed = int(cfg["project"]["seed"])
    seed_everything(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    patch = int(s1["patch_size"])
    target_path = Path(target_path or cfg.path("target_mask"))
    valid_path = Path(valid_path or cfg.path("valid_mask"))
    if not target_path.exists():
        raise FileNotFoundError(f"Target mask not found: {target_path}")
    check_same_grid([cfg.path("stack"), valid_path, target_path])
    with rasterio.open(valid_path) as v:
        shape = v.shape

    windows = plan_windows(valid_path, target_path, patch, int(s1["stride"]), float(s1["min_valid_fraction"]))
    if not any(w.has_target for w in windows):
        raise RuntimeError("No training patch contains a target pixel; check target_mask.tif alignment and values.")
    block_px = int(s1["val_block_px"])
    train_w, _, val_blocks = spatial_split(windows, patch, block_px, float(s1["val_fraction"]), seed)
    # Validation: non-overlapping lattice inside validation blocks (unique cells).
    val_w = [w for w in plan_windows(valid_path, target_path, patch, patch, 0.05)
             if window_blocks(w, patch, block_px) <= val_blocks]
    development_only = False
    if not train_w or not val_w:
        if not bool(s1.get("allow_random_split", False)):
            raise RuntimeError(
                f"AOI too small for a {block_px}px block split. Reduce stage1.val_block_px, or set "
                "stage1.allow_random_split: true for a DEVELOPMENT-ONLY run.")
        LOG.warning("DEVELOPMENT-ONLY: random window split; validation metrics are not geographic.")
        development_only = True
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(windows))
        n_val = max(1, int(len(windows) * float(s1["val_fraction"])))
        val_w = [windows[i] for i in perm[:n_val]]
        train_w = [windows[i] for i in perm[n_val:]]
    LOG.info("Windows: %d train (%d with targets), %d val (%d with targets)",
             len(train_w), sum(w.has_target for w in train_w), len(val_w), sum(w.has_target for w in val_w))

    common = dict(stack_path=cfg.path("stack"), valid_mask_path=valid_path, target_mask_path=target_path, patch=patch)
    ds_tr = GeospatialDataset(windows=train_w, augment=True, positive_oversample=int(s1["positive_oversample"]), **common)
    ds_va = GeospatialDataset(windows=val_w, augment=False, **common)
    nw = int(s1["num_workers"])
    bs = int(s1["batch_size"])
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, drop_last=len(ds_tr) > bs, num_workers=nw,
                       pin_memory=device == "cuda", persistent_workers=nw > 0)
    dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=device == "cuda",
                       persistent_workers=nw > 0)

    model = build_model(s1, in_channels=ds_tr.n_channels).to(device)
    ssl_path = cfg.path("ssl_weights")
    freeze_epochs = 0
    init = {"type": "scratch"}
    if use_ssl and ssl_path.exists():
        state = torch.load(ssl_path, map_location=device, weights_only=False)
        if list(state.get("channels", [])) != list(cfg["channels"]):
            raise ValueError("SSL checkpoint channels do not match the config channels")
        loaded, skipped = model.load_encoder(state["encoder"])
        if skipped:
            raise ValueError(f"SSL checkpoint incompatible with Stage 1 architecture ({skipped} tensors skipped)")
        freeze_epochs = int(s1.get("freeze_encoder_epochs", 3))
        init = {"type": "ssl", "path": str(ssl_path), "sha256": file_sha256(ssl_path),
                "ssl_epoch": state.get("epoch"), "ssl_masked_mse": state.get("masked_mse"), "tensors_loaded": loaded}
        LOG.info("Loaded SSL encoder (%d tensors); encoder frozen for %d epochs", loaded, freeze_epochs)
    elif use_ssl and require_ssl:
        raise FileNotFoundError(f"SSL initialisation required but {ssl_path} does not exist; run the ssl step")
    elif use_ssl:
        LOG.warning("No SSL weights at %s; training from scratch", ssl_path)

    loss_fn = FocalDiceLoss(alpha=float(s1["loss"]["focal_alpha"]), gamma=float(s1["loss"]["focal_gamma"]),
                            focal_weight=float(s1["loss"]["focal_weight"]), dice_weight=float(s1["loss"]["dice_weight"]))
    opt = torch.optim.AdamW(model.parameters(), lr=float(s1["lr"]), weight_decay=float(s1["weight_decay"]))
    epochs = int(s1["epochs"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(s1["lr"]), total_steps=epochs * max(1, len(dl_tr)),
                                                pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    reports = cfg.path("reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    best_path = Path(weights_out or cfg.path("stage1_weights"))
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_score, patience = -np.inf, 0
    with open(reports / f"{report_tag}_history.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "train_focal", "train_dice", "val_loss", "val_ap", "val_iou", "val_f1", "sec"])
        for ep in range(1, epochs + 1):
            frozen = ep <= freeze_epochs
            for p in model.encoder_parameters():
                p.requires_grad_(not frozen)
            model.train()
            t0 = time.time()
            tl, tf, td, nb = 0.0, 0.0, 0.0, 0
            for batch in dl_tr:
                x = batch["image"].to(device, non_blocking=True)
                y = batch["mask"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                    logits = model(x)
                loss, parts = loss_fn(logits, y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                tl += float(loss.detach()); tf += parts["focal"]; td += parts["dice"]; nb += 1
            nb = max(nb, 1)
            m = evaluate(model, dl_va, loss_fn, device, patch, shape)
            dt = time.time() - t0
            writer.writerow([ep, tl / nb, tf / nb, td / nb, m["loss"], m["ap"], m["iou"], m["f1"], round(dt, 1)])
            fh.flush()
            LOG.info("Epoch %3d/%d %s train %.4f | val loss %.4f AP %.4f IoU %.4f F1 %.4f (%.1fs)", ep, epochs,
                     "[enc frozen]" if frozen else "            ", tl / nb, m["loss"], m["ap"], m["iou"], m["f1"], dt)
            score = m["ap"] if np.isfinite(m["ap"]) else -m["loss"]
            if score > best_score:
                best_score, patience = score, 0
                torch.save({"state_dict": model.state_dict(), "stage1_cfg": s1, "channels": cfg["channels"],
                            "in_channels": ds_tr.n_channels, "epoch": ep, "val": m, "init": init}, best_path)
            else:
                patience += 1
                if patience >= int(s1["early_stop_patience"]):
                    LOG.info("Early stopping at epoch %d", ep)
                    break
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    report = {"best_epoch": ckpt["epoch"], **ckpt["val"], "development_only": development_only, "init": init,
              "protocol": ("Spatial block validation; unique cells; AP over all label-0 and label-1 cells "
                           "(label 255 excluded). Model selection used this split, so these numbers are "
                           "tuning scores, not an independent test. Use the evaluate step for test metrics.")}
    (reports / f"{report_tag}_metrics.json").write_text(json.dumps(report, indent=2))
    LOG.info("Best model: epoch %d, val AP %.4f -> %s", ckpt["epoch"], ckpt["val"]["ap"], best_path)
    return best_path


# =============================================================================
# Inference
# =============================================================================
def load_stage1(path: Path, device: str, expected_channels: Optional[List[str]] = None) -> AttentionUNet:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if expected_channels is not None and list(ckpt.get("channels", [])) != list(expected_channels):
        raise ValueError(f"Stage 1 checkpoint channels {ckpt.get('channels')} do not match config {expected_channels}")
    model = build_model(ckpt["stage1_cfg"], in_channels=int(ckpt["in_channels"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def _cosine_window(p: int) -> np.ndarray:
    w1 = np.sin(np.linspace(0, np.pi, p + 2)[1:-1]) ** 2
    return np.maximum(np.outer(w1, w1), 1e-3).astype(np.float32)


@torch.no_grad()
def _forward_features(model: AttentionUNet, x: torch.Tensor, tta: bool) -> torch.Tensor:
    """Return (N, 1 + L + depth, H, W): prob, latent, upsampled attention maps (D4-TTA averaged)."""
    codes = range(8) if tta else [0]
    acc = None
    for code in codes:
        xi = apply_d4_torch(x, code)
        logits, z, alphas = model(xi, return_features=True)
        h, w = xi.shape[-2:]
        maps = [F.interpolate(a, size=(h, w), mode="bilinear", align_corners=False) for a in alphas]
        f = torch.cat([torch.sigmoid(logits[:, :1]), z] + maps, dim=1)
        f = invert_d4_torch(f, code)
        acc = f if acc is None else acc + f
    return acc / len(codes)


def predict_block(model: AttentionUNet, img: np.ndarray, patch: int, overlap: int, tta: bool, device: str,
                  batch: int = 8) -> np.ndarray:
    """Sliding-window prediction on one (C, H, W) block with cosine blending."""
    _, H, W = img.shape
    stride = patch - overlap

    def origins(n: int) -> List[int]:
        o = list(range(0, max(n - patch, 0) + 1, stride))
        if o[-1] + patch < n:
            o.append(n - patch)
        return o

    weight = _cosine_window(patch)
    acc = None
    wsum = np.zeros((H, W), dtype=np.float32)
    coords = [(r, c) for r in origins(H) for c in origins(W)]
    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        x = torch.from_numpy(np.stack([img[:, r:r + patch, c:c + patch] for r, c in chunk])).to(device)
        f = _forward_features(model, x, tta).float().cpu().numpy()
        if acc is None:
            acc = np.zeros((f.shape[1], H, W), dtype=np.float32)
        for k, (r, c) in enumerate(chunk):
            acc[:, r:r + patch, c:c + patch] += f[k] * weight
            wsum[r:r + patch, c:c + patch] += weight
    return acc / np.maximum(wsum, 1e-6)


def infer(cfg: Config, device: Optional[str] = None, block: int = 1024,
          weights: Optional[Path] = None, out_path: Optional[Path] = None) -> Path:
    s1 = cfg["stage1"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_stage1(Path(weights or cfg.path("stage1_weights")), device, cfg["channels"])
    check_same_grid([cfg.path("stack"), cfg.path("valid_mask")])
    patch = int(s1["patch_size"])
    overlap = int(s1["inference"]["overlap"])
    tta = bool(s1["inference"]["tta"])
    halo = max(overlap, patch // 4)
    block = max(block, patch)
    L, depth = int(s1["latent_channels"]), int(s1["depth"])
    names = ["stage1_prob"] + [f"latent_{i:02d}" for i in range(L)] + [f"attn_gate_L{i}" for i in range(depth)]

    out_path = Path(out_path or cfg.path("stage1_features"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(cfg.path("stack")) as src, rasterio.open(cfg.path("valid_mask")) as vm:
        prof = src.profile.copy()
        prof.update(count=len(names), dtype="float32", nodata=np.nan, BIGTIFF="IF_SAFER")
        H, W = src.height, src.width
        q = 2 ** depth
        ext = int(np.ceil((block + 2 * halo) / q) * q)
        t0 = time.time()
        with rasterio.open(out_path, "w", **prof) as dst:
            for i, n in enumerate(names, start=1):
                dst.set_band_description(i, n)
            n_blocks = int(np.ceil(H / block) * np.ceil(W / block))
            k = 0
            for r0 in range(0, H, block):
                for c0 in range(0, W, block):
                    k += 1
                    h, w = min(block, H - r0), min(block, W - c0)
                    win_ext = Window(c0 - halo, r0 - halo, ext, ext)
                    img = src.read(window=win_ext, boundless=True, fill_value=np.nan).astype(np.float32)
                    vext = vm.read(1, window=win_ext, boundless=True, fill_value=0).astype(bool)
                    finite = np.isfinite(img).all(axis=0) & vext
                    img[:, ~finite] = 0.0  # same nodata convention as training
                    feats = predict_block(model, np.nan_to_num(img, nan=0.0), patch, overlap, tta, device)
                    feats = feats[:, halo:halo + h, halo:halo + w]
                    feats[:, ~finite[halo:halo + h, halo:halo + w]] = np.nan
                    dst.write(feats.astype(np.float32), window=Window(c0, r0, w, h))
                    LOG.info("  block %d/%d done (%.0fs elapsed)", k, n_blocks, time.time() - t0)
            dst.update_tags(PROJECT="GeoSentry-SOTA", STAGE="1", TTA=str(tta), WEIGHTS=str(weights or cfg.path("stage1_weights")))
    LOG.info("Stage 1 feature raster written: %s", out_path)
    return out_path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Stage 1 Attention U-Net")
    ap.add_argument("action", choices=["train", "infer"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-ssl", action="store_true")
    ap.add_argument("--block", type=int, default=1024)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.action == "train":
        train(cfg, args.device, use_ssl=not args.no_ssl)
    else:
        infer(cfg, args.device, args.block)


if __name__ == "__main__":
    main()
