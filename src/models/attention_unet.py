"""
Stage 1: Spatial-Channel Attention U-Net for pegmatite / REE target segmentation
================================================================================

Contents
--------
GeospatialDataset   Streams 256 x 256 patches from the 11-channel stack with
                    windowed rasterio reads (never loads the full AOI), applies
                    D4 augmentations (flips + 90 degree rotations) identically to
                    image and label, and returns the patch's geographic origin
                    and augmentation code so every prediction can be mapped back
                    to EPSG:32632 exactly.
AttentionGate       Additive soft-attention gate (Oktay et al., 2018) fused with
                    squeeze-and-excitation channel attention. The spatial branch
                    lets the decoder's coarse geological context decide which
                    skip-connection pixels to pass (thin dykes, shear corridors);
                    the channel branch learns to down-weight modalities that
                    carry seasonal canopy or farmland signal at that scale.
AttentionUNet       Encoder-decoder with gated skips, a trained per-pixel latent
                    embedding (exported to Stage 2) and a 1-channel logit head.
FocalDiceLoss       Focal BCE + soft Dice with an ignore index, for target masks
                    where positives are often < 0.5% of pixels.

D4 augmentation and geographic alignment
----------------------------------------
Flips and 90 degree rotations are exact pixel permutations: no interpolation,
no change of pixel size. Each sample records (row, col) of its window in the
stack and a D4 code 0-7. `invert_d4` undoes the permutation so model outputs
land back on the original pixel lattice; the window's affine transform is
`rasterio.windows.transform(window, stack_transform)`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from rasterio.windows import Window
from torch.utils.data import Dataset

IGNORE_INDEX = 255

# Modality of each known channel. Groups are built from the channel names in the
# stack, so projects with fewer channels (e.g. no radiometrics yet) work unchanged.
CHANNEL_MODALITY: Dict[str, str] = {
    "B2_blue": "optical", "B3_green": "optical", "B4_red": "optical", "B8_nir": "optical",
    "S1_VV_dB": "sar", "S1_VH_dB": "sar", "OH_alteration": "spectral_indices", "NDVI": "spectral_indices",
    "MAG_1VD": "geophysics", "RAD_eTh": "geophysics", "DEM_slope": "terrain",
}


def modality_groups(channel_names: Sequence[str]) -> Dict[str, List[int]]:
    """Map modality -> channel indices for the given stack band order."""
    out: Dict[str, List[int]] = {}
    for i, n in enumerate(channel_names):
        out.setdefault(CHANNEL_MODALITY.get(n, n), []).append(i)
    return out


# Default groups for the full 11-channel stack.
MODALITY_GROUPS: Dict[str, List[int]] = modality_groups(list(CHANNEL_MODALITY))


# =============================================================================
# D4 augmentation helpers
# =============================================================================
def apply_d4(arr: np.ndarray, code: int) -> np.ndarray:
    """Apply D4 element `code` (0-7) to the last two axes. 0-3: rot90 k times; 4-7: flip then rot."""
    out = arr
    if code >= 4:
        out = np.flip(out, axis=-1)
    out = np.rot90(out, k=code % 4, axes=(-2, -1))
    return np.ascontiguousarray(out)


def invert_d4(arr: np.ndarray, code: int) -> np.ndarray:
    """Exact inverse of `apply_d4`."""
    out = np.rot90(arr, k=-(code % 4), axes=(-2, -1))
    if code >= 4:
        out = np.flip(out, axis=-1)
    return np.ascontiguousarray(out)


def apply_d4_torch(t: torch.Tensor, code: int) -> torch.Tensor:
    out = torch.flip(t, dims=(-1,)) if code >= 4 else t
    return torch.rot90(out, k=code % 4, dims=(-2, -1))


def invert_d4_torch(t: torch.Tensor, code: int) -> torch.Tensor:
    out = torch.rot90(t, k=-(code % 4), dims=(-2, -1))
    return torch.flip(out, dims=(-1,)) if code >= 4 else out


# =============================================================================
# Dataset
# =============================================================================
def lattice_origins(n: int, patch: int, stride: int) -> List[int]:
    """Patch origins along one axis; the last origin is snapped to the raster edge."""
    if n <= patch:
        return [0]
    o = list(range(0, n - patch + 1, stride))
    if o[-1] != n - patch:
        o.append(n - patch)
    return o


def owned_slices(row: int, col: int, patch: int, shape: Tuple[int, int]) -> Tuple[slice, slice]:
    """
    For windows on a stride == patch lattice, the part of window (row, col) that no
    earlier window covers. Only the edge-snapped last row/column of windows
    overlaps its neighbour; this trims that overlap so every cell is counted once.
    """
    def own(o: int, n: int) -> slice:
        snapped_last = n > patch and o == n - patch and o % patch != 0
        start = (o // patch + 1) * patch - o if snapped_last else 0
        return slice(start, min(patch, n - o))
    return own(row, shape[0]), own(col, shape[1])


@dataclass(frozen=True)
class PatchWindow:
    row: int
    col: int
    has_target: bool


def plan_windows(
    valid_mask_path: Path,
    target_mask_path: Optional[Path],
    patch: int,
    stride: int,
    min_valid_fraction: float,
) -> List[PatchWindow]:
    """
    Enumerate patch origins on a regular lattice and keep those with enough
    valid data. Uses summed-area tables, so the cost is one pass over the masks.
    The last row/column of windows is snapped to the raster edge so no strip
    of the AOI is skipped.
    """
    with rasterio.open(valid_mask_path) as src:
        valid = src.read(1).astype(np.float32)
        H, W = valid.shape
    tgt = None
    if target_mask_path is not None and Path(target_mask_path).exists():
        with rasterio.open(target_mask_path) as src:
            t = src.read(1)
            if t.shape != (H, W):
                raise ValueError("Target mask grid differs from the stack grid; run align_label_raster first.")
            tgt = (t == 1).astype(np.float32)

    def sat(a: np.ndarray) -> np.ndarray:
        s = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
        s[1:, 1:] = a.cumsum(0).cumsum(1)
        return s

    def box(s: np.ndarray, r: int, c: int, h: int, w: int) -> float:
        return s[r + h, c + w] - s[r, c + w] - s[r + h, c] + s[r, c]

    def origins(n: int) -> List[int]:
        return lattice_origins(n, patch, stride)

    sv = sat(valid)
    st = sat(tgt) if tgt is not None else None
    out: List[PatchWindow] = []
    for r in origins(H):
        for c in origins(W):
            h, w = min(patch, H - r), min(patch, W - c)
            frac = box(sv, r, c, h, w) / float(patch * patch)
            if frac < min_valid_fraction:
                continue
            has_t = bool(st is not None and box(st, r, c, h, w) > 0)
            out.append(PatchWindow(r, c, has_t))
    return out


class GeospatialDataset(Dataset):
    """
    Streams (image, mask) patches from the harmonized stack.

    Parameters
    ----------
    stack_path        11-channel GeoTIFF from the harmonizer.
    valid_mask_path   uint8 mask (1 = all channels finite).
    target_mask_path  uint8 labels (1 target, 0 background, 255 ignore). None for SSL.
    windows           Patch origins (from plan_windows / spatial_split).
    augment           Apply random D4 transforms.
    positive_oversample  Each target-bearing window appears this many extra times per epoch.
    """

    def __init__(
        self,
        stack_path: Path,
        valid_mask_path: Path,
        target_mask_path: Optional[Path],
        windows: Sequence[PatchWindow],
        patch: int = 256,
        augment: bool = True,
        positive_oversample: int = 0,
    ):
        self.stack_path = Path(stack_path)
        self.valid_mask_path = Path(valid_mask_path)
        self.target_mask_path = Path(target_mask_path) if target_mask_path else None
        self.patch = patch
        self.augment = augment
        self.windows = list(windows)
        self.index: List[int] = []
        for i, w in enumerate(self.windows):
            self.index.append(i)
            if w.has_target:
                self.index.extend([i] * positive_oversample)
        self._handles: Optional[Tuple[int, dict]] = None
        with rasterio.open(self.stack_path) as src:
            self.n_channels = src.count
            self.transform = src.transform
            self.crs = src.crs

    def __len__(self) -> int:
        return len(self.index)

    def _open(self) -> dict:
        """Open rasterio handles once per worker process (safe with Windows spawn)."""
        pid = os.getpid()
        if self._handles is None or self._handles[0] != pid:
            h = {"stack": rasterio.open(self.stack_path), "valid": rasterio.open(self.valid_mask_path)}
            if self.target_mask_path is not None and self.target_mask_path.exists():
                h["target"] = rasterio.open(self.target_mask_path)
            self._handles = (pid, h)
        return self._handles[1]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = None  # handles cannot be pickled to DataLoader workers
        return state

    def read_window(self, row: int, col: int) -> Tuple[np.ndarray, np.ndarray]:
        h = self._open()
        win = Window(col, row, self.patch, self.patch)
        img = h["stack"].read(window=win, boundless=True, fill_value=np.nan).astype(np.float32)
        valid = h["valid"].read(1, window=win, boundless=True, fill_value=0).astype(bool)
        if "target" in h:
            tgt = h["target"].read(1, window=win, boundless=True, fill_value=IGNORE_INDEX).astype(np.uint8)
        else:
            tgt = np.zeros((self.patch, self.patch), dtype=np.uint8)
        valid &= np.isfinite(img).all(axis=0)
        tgt[~valid] = IGNORE_INDEX
        # Cells outside the valid mask (nodata, or test / dead-zone cells in a
        # geographic fold) carry no information into training: zero the image too.
        img[:, ~valid] = 0.0
        img = np.nan_to_num(img, nan=0.0)
        return img, tgt

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        w = self.windows[self.index[i]]
        img, tgt = self.read_window(w.row, w.col)
        code = int(torch.randint(0, 8, (1,)).item()) if self.augment else 0
        if code:
            img, tgt = apply_d4(img, code), apply_d4(tgt, code)
        return {
            "image": torch.from_numpy(img),
            "mask": torch.from_numpy(tgt.astype(np.int64)),
            "row": torch.tensor(w.row),
            "col": torch.tensor(w.col),
            "d4": torch.tensor(code),
        }


def window_blocks(w: PatchWindow, patch: int, block_px: int) -> set:
    r0, r1 = w.row // block_px, (w.row + patch - 1) // block_px
    c0, c1 = w.col // block_px, (w.col + patch - 1) // block_px
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def spatial_split(
    windows: Sequence[PatchWindow], patch: int, block_px: int, val_fraction: float, seed: int
) -> Tuple[List[PatchWindow], List[PatchWindow], set]:
    """
    Leakage-free train/validation split. The AOI is cut into square blocks;
    a random subset of blocks becomes validation. Validation windows must lie
    fully inside validation blocks; training windows must not touch them.
    Overlapping strided patches therefore never share pixels across the split.
    Returns (train windows, val windows, val block ids).
    """
    rng = np.random.default_rng(seed)

    def blocks_of(w: PatchWindow) -> set:
        return window_blocks(w, patch, block_px)

    all_blocks = sorted({b for w in windows for b in blocks_of(w)})
    rng.shuffle(all_blocks)
    # Prefer at least one target-bearing block in validation when targets exist.
    target_blocks = {b for w in windows if w.has_target for b in blocks_of(w)}
    n_val = max(1, int(round(val_fraction * len(all_blocks))))
    val_blocks = set()
    for b in all_blocks:
        if b in target_blocks and len(val_blocks) < max(1, n_val // 2):
            val_blocks.add(b)
    for b in all_blocks:
        if len(val_blocks) >= n_val:
            break
        val_blocks.add(b)

    train, val = [], []
    for w in windows:
        bl = blocks_of(w)
        if bl <= val_blocks:
            val.append(w)
        elif not (bl & val_blocks):
            train.append(w)
    return train, val, val_blocks


# =============================================================================
# Network
# =============================================================================
def _gn(ch: int) -> nn.GroupNorm:
    """GroupNorm is stable at the small batch sizes typical of 11-channel 256 px patches."""
    groups = 8 if ch % 8 == 0 else (4 if ch % 4 == 0 else 1)
    return nn.GroupNorm(groups, ch)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), _gn(cout), nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), _gn(cout), nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ChannelSE(nn.Module):
    """Squeeze-and-excitation channel attention."""

    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, ch // reduction)
        self.fc = nn.Sequential(nn.Linear(ch, hidden), nn.SiLU(inplace=True), nn.Linear(hidden, ch), nn.Sigmoid())
        self.last_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x.mean(dim=(-2, -1)))
        self.last_weights = w.detach()
        return x * w[:, :, None, None]


class AttentionGate(nn.Module):
    """
    Spatial-channel attention gate on a skip connection.

        alpha = sigmoid( psi( SiLU( W_g * g + W_x * x ) ) )      spatial, in [0, 1]
        out   = SE(x) * alpha                                     channel x spatial

    g is the decoder feature (already upsampled to x's size), x the encoder skip.
    alpha is cached in `last_alpha` for Grad-CAM style inspection and is
    exported as a Stage 2 feature.
    """

    def __init__(self, f_g: int, f_x: int, f_int: int):
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1, bias=False), _gn(f_int))
        self.w_x = nn.Sequential(nn.Conv2d(f_x, f_int, 1, bias=False), _gn(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1), nn.Sigmoid())
        self.se = ChannelSE(f_x)
        self.act = nn.SiLU(inplace=True)
        self.last_alpha: Optional[torch.Tensor] = None

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        alpha = self.psi(self.act(self.w_g(g) + self.w_x(x)))
        self.last_alpha = alpha
        return self.se(x) * alpha


class AttentionUNet(nn.Module):
    """
    Attention U-Net.

    in_channels=11, depth=4, base=32 gives encoder widths 32-64-128-256 and a
    512-wide bottleneck (about 8.9 M parameters). Input height and width must be
    divisible by 2**depth (256 is).

    forward(x)                        -> logits (N, out_channels, H, W)
    forward(x, return_features=True)  -> (logits, latent (N, latent, H, W),
                                          [alpha_level0 ... alpha_level{depth-1}])
    """

    def __init__(self, in_channels: int = 11, out_channels: int = 1, base: int = 32, depth: int = 4,
                 latent_channels: int = 8, dropout: float = 0.1):
        super().__init__()
        self.depth = depth
        widths = [base * 2 ** i for i in range(depth)]
        self.enc = nn.ModuleList()
        cin = in_channels
        for w in widths:
            self.enc.append(ConvBlock(cin, w))
            cin = w
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(widths[-1], widths[-1] * 2, dropout=dropout)

        self.up = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.dec = nn.ModuleList()
        cin = widths[-1] * 2
        for w in reversed(widths):
            self.up.append(nn.ConvTranspose2d(cin, w, 2, stride=2))
            self.gates.append(AttentionGate(f_g=w, f_x=w, f_int=max(8, w // 2)))
            self.dec.append(ConvBlock(2 * w, w))
            cin = w

        self.latent = nn.Sequential(nn.Conv2d(widths[0], latent_channels, 1, bias=False), _gn(latent_channels),
                                    nn.SiLU(inplace=True))
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Conv2d(latent_channels, out_channels, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Bias the head toward the rare class prior (about 1%) for a stable start.
        if self.head.out_channels == 1:
            nn.init.constant_(self.head.bias, -4.6)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        return self.bottleneck(x), skips

    def forward(self, x: torch.Tensor, return_features: bool = False):
        h, w = x.shape[-2:]
        if h % (2 ** self.depth) or w % (2 ** self.depth):
            raise ValueError(f"Input {h}x{w} must be divisible by {2 ** self.depth}")
        x, skips = self.encode(x)
        alphas: List[torch.Tensor] = []
        for up, gate, dec, skip in zip(self.up, self.gates, self.dec, reversed(skips)):
            g = up(x)
            gated = gate(g, skip)
            alphas.append(gate.last_alpha)
            x = dec(torch.cat([gated, g], dim=1))
        z = self.latent(x)
        logits = self.head(self.drop(z))
        if return_features:
            # alphas collected deep -> shallow; return shallow -> deep (level 0 = full res).
            return logits, z, list(reversed(alphas))
        return logits

    def encoder_state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v for k, v in self.state_dict().items() if k.startswith(("enc.", "bottleneck."))}

    def load_encoder(self, state: Dict[str, torch.Tensor]) -> Tuple[int, int]:
        """Load SSL encoder weights where names and shapes match. Returns (loaded, skipped)."""
        own = self.state_dict()
        ok = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        own.update(ok)
        self.load_state_dict(own)
        return len(ok), len(state) - len(ok)

    def encoder_parameters(self):
        for name, p in self.named_parameters():
            if name.startswith(("enc.", "bottleneck.")):
                yield p


# =============================================================================
# Loss
# =============================================================================
class FocalDiceLoss(nn.Module):
    """
    L = w_f * Focal(alpha, gamma) + w_d * (1 - SoftDice)

    Focal term (Lin et al., 2017) down-weights the millions of easy background
    pixels; alpha > 0.5 up-weights the rare target class. The soft Dice term is
    computed over the whole batch (not per image), which keeps its gradient
    defined for the many patches that contain no target at all. Pixels equal to
    `ignore_index` (nodata, cloud, unsurveyed) contribute to neither term.
    """

    def __init__(self, alpha: float = 0.85, gamma: float = 2.0, focal_weight: float = 1.0,
                 dice_weight: float = 1.0, ignore_index: int = IGNORE_INDEX, smooth: float = 1.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
        self.wf, self.wd = focal_weight, dice_weight
        self.ignore_index, self.smooth = ignore_index, smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits = logits.squeeze(1).float()
        # Only 0 / 1 are trainable labels; ignore_index and any stray code are masked out.
        valid = (target == 0) | (target == 1)
        t = torch.where(valid, target, torch.zeros_like(target)).float()
        n_valid = valid.sum().clamp_min(1)

        bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * t + (1 - p) * (1 - t)
        alpha_t = self.alpha * t + (1 - self.alpha) * (1 - t)
        focal = (alpha_t * (1 - p_t).pow(self.gamma) * bce * valid).sum() / n_valid

        pv, tv = p * valid, t * valid
        inter = (pv * tv).sum()
        dice = (2 * inter + self.smooth) / (pv.sum() + tv.sum() + self.smooth)
        loss = self.wf * focal + self.wd * (1 - dice)
        return loss, {"focal": float(focal.detach()), "dice": float(dice.detach())}


def build_model(cfg_stage1: dict, in_channels: int = 11, out_channels: int = 1) -> AttentionUNet:
    return AttentionUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base=int(cfg_stage1["base_channels"]),
        depth=int(cfg_stage1["depth"]),
        latent_channels=int(cfg_stage1["latent_channels"]),
        dropout=float(cfg_stage1["dropout"]),
    )
