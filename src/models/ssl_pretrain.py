"""
Self-supervised foundation-encoder pretraining (masked multimodal reconstruction)
================================================================================

Labelled pegmatite / REE outlines cover a tiny fraction of any Nigerian AOI,
but the unlabelled 11-channel stack covers all of it. The encoder is therefore
pretrained without labels using two complementary pretext tasks at once:

1. Spatial masked autoencoding (MAE-style): random 16 x 16 pixel blocks are
   hidden in every channel and must be reconstructed from context. This
   teaches the encoder lithological texture and structural continuity
   (dykes, foliation, shear fabrics).
2. Cross-modal channel dropout: with probability p a whole modality group
   (optical, SAR, spectral indices, airborne geophysics, terrain) is removed
   and must be predicted from the others. This forces the encoder to learn
   the physical coupling between, for example, thorium highs, Al-OH
   alteration and magnetic lineaments, which is exactly the coupling that
   survives under laterite and canopy where optical data alone fails.

The loss is MSE on hidden pixels only. The trained encoder (enc.* and
bottleneck.* weights) is saved and loaded into the Stage 1 segmentation
network before fine-tuning.

    python -m src.models.ssl_pretrain --config configs/geosentry.yaml
"""
from __future__ import annotations

import argparse
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.config import Config, get_logger, load_config, seed_everything
from src.models.attention_unet import (
    IGNORE_INDEX,
    modality_groups,
    GeospatialDataset,
    build_model,
    plan_windows,
)

LOG = get_logger("ssl")


def make_masks(x: torch.Tensor, valid: torch.Tensor, mask_patch: int, mask_ratio: float,
               channel_drop_prob: float, groups: Optional[dict] = None) -> torch.Tensor:
    """
    Returns a float mask M (N, C, H, W): 1 where the input is hidden and must be
    reconstructed. Combines block masking with whole-modality dropout.
    """
    n, c, h, w = x.shape
    gh, gw = h // mask_patch, w // mask_patch
    block = (torch.rand(n, 1, gh, gw, device=x.device) < mask_ratio).float()
    spatial = torch.nn.functional.interpolate(block, scale_factor=mask_patch, mode="nearest")
    m = spatial.expand(n, c, h, w).clone()
    groups = list((groups or modality_groups([f"c{i}" for i in range(c)])).values())
    for i in range(n):
        if torch.rand(1).item() < channel_drop_prob:
            g = groups[int(torch.randint(0, len(groups), (1,)).item())]
            m[i, g] = 1.0
    return m * valid[:, None].float()


def pretrain(cfg: Config, device: Optional[str] = None) -> str:
    s1 = cfg["stage1"]
    ssl = s1["ssl"]
    seed_everything(int(cfg["project"]["seed"]))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    windows = plan_windows(cfg.path("valid_mask"), None, int(s1["patch_size"]), int(s1["stride"]),
                           float(s1["min_valid_fraction"]))
    if not windows:
        raise RuntimeError("No valid patches found for SSL; check valid_mask.tif")
    ds = GeospatialDataset(cfg.path("stack"), cfg.path("valid_mask"), None, windows,
                           patch=int(s1["patch_size"]), augment=True)
    dl = DataLoader(ds, batch_size=int(s1["batch_size"]), shuffle=True, drop_last=len(ds) > int(s1["batch_size"]),
                    num_workers=int(s1["num_workers"]), pin_memory=device == "cuda",
                    persistent_workers=int(s1["num_workers"]) > 0)

    # Reconstruction head outputs all 11 channels; latent kept wide (= base) for capacity.
    model = build_model({**s1, "latent_channels": s1["base_channels"], "dropout": 0.0},
                        in_channels=ds.n_channels, out_channels=ds.n_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(ssl["lr"]), weight_decay=float(s1["weight_decay"]))
    epochs = int(ssl["epochs"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(ssl["lr"]), total_steps=epochs * max(1, len(dl)))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    import rasterio

    with rasterio.open(cfg.path("stack")) as src:
        groups = modality_groups(list(src.descriptions))
    LOG.info("SSL pretraining on %d patches, %d epochs, device=%s, modality groups %s", len(ds), epochs, device,
             {k: len(v) for k, v in groups.items()})
    best = float("inf")
    out_path = cfg.path("ssl_weights")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, epochs + 1):
        model.train()
        t0, total, nb = time.time(), 0.0, 0
        for batch in dl:
            x = batch["image"].to(device, non_blocking=True)
            valid = (batch["mask"] != IGNORE_INDEX).to(device)
            m = make_masks(x, valid, int(ssl["mask_patch"]), float(ssl["mask_ratio"]), float(ssl["channel_drop_prob"]),
                           groups)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                recon = model(x * (1 - m))
            loss = (((recon.float() - x) ** 2) * m).sum() / m.sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += float(loss.detach())
            nb += 1
        mean = total / max(nb, 1)
        LOG.info("SSL epoch %3d/%d  masked-MSE %.5f  (%.1fs)", ep, epochs, mean, time.time() - t0)
        if mean < best:
            best = mean
            torch.save({"encoder": model.encoder_state_dict(), "epoch": ep, "masked_mse": mean,
                        "channels": cfg["channels"]}, out_path)
    LOG.info("Best SSL encoder saved to %s (masked-MSE %.5f)", out_path, best)
    return str(out_path)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Self-supervised encoder pretraining")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    pretrain(load_config(args.config), args.device)


if __name__ == "__main__":
    main()
