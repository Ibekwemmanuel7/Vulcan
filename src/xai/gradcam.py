"""
Explainability diagnostics for Stage 1 (CNN)
============================================

These are diagnostics of what the network uses. They are not independent
geological evidence and do not prove a mineralisation mechanism.

seg_gradcam          Grad-CAM adapted to segmentation: the score is the summed
                     target logit inside a region of interest (ROI).
integrated_gradients Per-channel integrated gradients over the ROI against an
                     explicit, documented baseline:
                       "blur"  Gaussian-blurred copy of the patch (sigma 8 px):
                               keeps regional geology, removes local structure.
                               Physically plausible; the default.
                       "mean"  per-channel mean of valid stack cells (measured,
                               not assumed; zero is NOT the mean of MinMax data).
                       "zero"  all channels 0, kept only for comparison.
                     Completeness is checked: sum(IG) should equal
                     F(x) - F(baseline); the relative error is reported.
modality_ablation    Replace one modality group at a time with the baseline and
                     measure the drop in ROI logit. Robust to the attribution
                     sharing that correlated channels cause in IG and SHAP.

    python -m src.xai.gradcam --config configs/geosentry.yaml --row 1200 --col 3400
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, Optional

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.windows import Window
from scipy import ndimage

from src.config import Config, get_logger, load_config
from src.models.attention_unet import AttentionUNet, modality_groups

LOG = get_logger("xai")


def _roi_score(model: AttentionUNet, x: torch.Tensor, roi: torch.Tensor) -> torch.Tensor:
    return (model(x)[0, 0] * roi.to(x.dtype)).sum()


def seg_gradcam(model: AttentionUNet, x: torch.Tensor, roi: torch.Tensor,
                layer: Optional[torch.nn.Module] = None) -> np.ndarray:
    """(H, W) heat map in [0, 1] for the ROI-summed target logit at `layer` (default bottleneck)."""
    model.eval()
    layer = layer or model.bottleneck
    store: Dict[str, torch.Tensor] = {}
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0]))
    try:
        xx = x.clone().requires_grad_(True)
        score = _roi_score(model, xx, roi)
        model.zero_grad(set_to_none=True)
        score.backward()
        weights = store["grad"].mean(dim=(-2, -1), keepdim=True)
        cam = F.relu((weights * store["act"]).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        return (cam / cam.max().clamp_min(1e-8)).detach().cpu().numpy()
    finally:
        h1.remove(); h2.remove()


def make_baseline(x: torch.Tensor, kind: str, channel_mean: Optional[np.ndarray] = None) -> torch.Tensor:
    if kind == "zero":
        return torch.zeros_like(x)
    if kind == "mean":
        if channel_mean is None:
            raise ValueError("mean baseline needs channel_mean")
        return torch.as_tensor(channel_mean, dtype=x.dtype, device=x.device)[None, :, None, None].expand_as(x).clone()
    if kind == "blur":
        arr = x[0].detach().cpu().numpy()
        blurred = np.stack([ndimage.gaussian_filter(c, 8.0, mode="reflect") for c in arr])
        return torch.as_tensor(blurred, dtype=x.dtype, device=x.device)[None]
    raise ValueError(kind)


def integrated_gradients(model: AttentionUNet, x: torch.Tensor, roi: torch.Tensor, baseline: torch.Tensor,
                         steps: int = 64) -> Dict[str, object]:
    """Per-channel IG (ROI-summed) with a completeness check."""
    model.eval()
    total = torch.zeros_like(x)
    alphas = (torch.arange(steps, dtype=x.dtype) + 0.5) / steps  # midpoint Riemann rule
    for a in alphas:
        xi = (baseline + a * (x - baseline)).requires_grad_(True)
        g, = torch.autograd.grad(_roi_score(model, xi, roi), xi)
        total += g
    ig = ((x - baseline) * total / steps)[0].sum(dim=(-2, -1)).detach().cpu().numpy()
    with torch.no_grad():
        delta = float(_roi_score(model, x, roi) - _roi_score(model, baseline, roi))
    err = abs(ig.sum() - delta) / max(abs(delta), 1e-8)
    return {"per_channel": ig, "f_x_minus_f_baseline": delta, "completeness_rel_error": float(err)}


@torch.no_grad()
def modality_ablation(model: AttentionUNet, x: torch.Tensor, roi: torch.Tensor, baseline: torch.Tensor,
                      channel_names) -> Dict[str, float]:
    """ROI logit drop when each modality group is replaced by the baseline."""
    model.eval()
    ref = float(_roi_score(model, x, roi))
    out = {}
    for name, idx in modality_groups(channel_names).items():
        xa = x.clone()
        xa[:, idx] = baseline[:, idx]
        out[name] = ref - float(_roi_score(model, xa, roi))
    return out


def stack_channel_means(stack_path, valid_path, n: int = 200_000, seed: int = 0) -> np.ndarray:
    """Per-channel mean over a random sample of valid cells."""
    rng = np.random.default_rng(seed)
    with rasterio.open(valid_path) as v:
        valid = v.read(1).astype(bool)
    r, c = np.nonzero(valid)
    pick = rng.choice(len(r), size=min(n, len(r)), replace=False)
    with rasterio.open(stack_path) as s:
        vals = np.array(list(s.sample(zip(*rasterio.transform.xy(s.transform, r[pick], c[pick])))), dtype=np.float64)
    return np.nanmean(vals, axis=0).astype(np.float32)


def explain_location(cfg: Config, row: int, col: int, device: Optional[str] = None, roi_px: int = 16) -> dict:
    from src.pipeline.stage1 import load_stage1

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_stage1(cfg.path("stage1_weights"), device, cfg["channels"])
    patch = int(cfg["stage1"]["patch_size"])
    r0, c0 = row - patch // 2, col - patch // 2
    with rasterio.open(cfg.path("stack")) as src:
        img = src.read(window=Window(c0, r0, patch, patch), boundless=True, fill_value=np.nan).astype(np.float32)
        names = list(src.descriptions)
    x = torch.from_numpy(np.nan_to_num(img))[None].to(device)
    roi = torch.zeros(patch, patch, dtype=torch.bool, device=device)
    h = roi_px // 2
    roi[patch // 2 - h: patch // 2 + h, patch // 2 - h: patch // 2 + h] = True

    means = stack_channel_means(cfg.path("stack"), cfg.path("valid_mask"))
    cam = seg_gradcam(model, x, roi)
    result: Dict[str, object] = {"row": row, "col": col, "roi_px": roi_px, "channels": names,
                                 "note": "Diagnostics of the network, not geological evidence."}
    for kind in ("blur", "mean", "zero"):
        base = make_baseline(x, kind, means)
        ig = integrated_gradients(model, x, roi, base)
        share = np.abs(ig["per_channel"]) / max(np.abs(ig["per_channel"]).sum(), 1e-12)
        result[f"ig_{kind}"] = {
            "completeness_rel_error": ig["completeness_rel_error"],
            "f_x_minus_f_baseline": ig["f_x_minus_f_baseline"],
            "channels": {n: {"signed": float(a), "share_pct": float(100 * s)}
                         for n, a, s in zip(names, ig["per_channel"], share)}}
        result[f"ablation_{kind}"] = modality_ablation(model, x, roi, base, names)

    out_dir = cfg.path("reports_dir"); out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"gradcam_r{row}_c{col}.npy", cam)
    (out_dir / f"stage1_attribution_r{row}_c{col}.json").write_text(json.dumps(result, indent=2))
    top = sorted(result["ig_blur"]["channels"].items(), key=lambda kv: -kv[1]["share_pct"])[:4]
    LOG.info("IG (blur baseline, completeness err %.1e): %s", result["ig_blur"]["completeness_rel_error"],
             ", ".join(f"{n} {v['share_pct']:.0f}%" for n, v in top))
    LOG.info("Modality ablation (blur): %s", {k: round(v, 2) for k, v in result["ablation_blur"].items()})
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Grad-CAM, integrated gradients and modality ablation for Stage 1")
    ap.add_argument("--config", default=None)
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--col", type=int, required=True)
    args = ap.parse_args(argv)
    explain_location(load_config(args.config), args.row, args.col)


if __name__ == "__main__":
    main()
