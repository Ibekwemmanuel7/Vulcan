"""
Multimodal Spatial Harmonizer
=============================

Takes raw rasters with mismatched CRS, extent and pixel size (Sentinel-2,
Sentinel-1, airborne magnetics, airborne radiometrics, DEM) and produces one
analysis-ready 11-channel GeoTIFF on a common 10 m grid in WGS 84 / UTM 32N
(EPSG:32632).

Processing chain per layer
--------------------------
1. Footprint of every input is projected into EPSG:32632 and the target grid is
   built from their intersection (or union / explicit extent), snapped to 10 m.
2. Each layer is warped with a kind-specific resampler:
     optical, SAR, DEM  -> bilinear
     airborne geophysics -> cubic, then Gaussian de-blocking (scipy.ndimage),
                            because 100-200 m line-spaced surveys produce
                            staircase artefacts when upsampled to 10 m.
3. Derived physics layers are computed on the common grid:
     NDVI            = (B8 - B4) / (B8 + B4)
     OH alteration   = B11 / B12  (Al-OH / Mg-OH absorption near 2.2 um lowers
                       B12, so sericite, muscovite and kaolinite halos around
                       pegmatites raise the ratio)
     MAG 1VD         = |k| * FFT(TMI), with a Butterworth low-pass at the
                       native survey Nyquist so the derivative does not amplify
                       interpolation noise
     Slope           = arctan(|grad(DEM)|) in degrees
4. Scaling, with statistics saved to JSON so later scenes or AOIs are scaled
   identically (pass --reuse-stats):
     optical / SAR / indices / slope -> percentile clip then MinMax to [0, 1]
     airborne geophysics             -> robust Z-score, clipped to +/- 5 sigma

Run from PowerShell in the project root:
    python -m src.utils.harmonizer --config configs/geosentry.yaml
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from scipy import fft as sfft
from scipy import ndimage

from src.config import Config, get_logger, load_config

LOG = get_logger("harmonizer")

# Project CRS. Default WGS 84 / UTM Zone 32N. Only the three UTM zones that
# cover Nigeria are accepted; set project.crs in the config to switch zone.
TARGET_CRS = CRS.from_epsg(32632)
NIGERIA_UTM_ZONES = {32631: (0.0, 6.0), 32632: (6.0, 12.0), 32633: (12.0, 18.0)}


def set_target_crs(crs_like) -> CRS:
    """Validate and activate the project CRS (EPSG:32631, 32632 or 32633)."""
    global TARGET_CRS
    crs = CRS.from_user_input(crs_like)
    if crs.to_epsg() not in NIGERIA_UTM_ZONES:
        raise ValueError(f"project.crs {crs_like} not supported; use EPSG:32631, EPSG:32632 or EPSG:32633")
    TARGET_CRS = crs
    return crs

RESAMPLING_BY_KIND = {
    "optical": Resampling.bilinear,
    "sar": Resampling.bilinear,
    "dem": Resampling.bilinear,
    "geophys": Resampling.cubic,
    "categorical": Resampling.nearest,
}

# Raw layers each output channel needs. Only layers used by the configured
# channels must exist, so a project can run without, for example, radiometrics.
CHANNEL_LAYERS: Dict[str, Tuple[str, ...]] = {
    "B2_blue": ("s2_b02",), "B3_green": ("s2_b03",), "B4_red": ("s2_b04",), "B8_nir": ("s2_b08",),
    "S1_VV_dB": ("s1_vv",), "S1_VH_dB": ("s1_vh",), "OH_alteration": ("s2_b11", "s2_b12"),
    "NDVI": ("s2_b04", "s2_b08"), "MAG_1VD": ("mag",), "RAD_eTh": ("th",), "DEM_slope": ("dem",),
}

# How each output channel is scaled.
MINMAX = "minmax"
ZSCORE = "zscore"


# =============================================================================
# Target grid
# =============================================================================
@dataclass(frozen=True)
class TargetGrid:
    """A north-up raster grid in TARGET_CRS."""

    xmin: float
    ymax: float
    width: int
    height: int
    res: float

    @property
    def transform(self):
        return from_origin(self.xmin, self.ymax, self.res, self.res)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (self.xmin, self.ymax - self.height * self.res, self.xmin + self.width * self.res, self.ymax)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)


def _native_res_m(src: rasterio.io.DatasetReader) -> float:
    """Approximate native pixel size in metres (handles geographic CRS)."""
    res = float(np.mean(np.abs(src.res)))
    if src.crs is not None and src.crs.is_geographic:
        lat = 0.5 * (src.bounds.top + src.bounds.bottom)
        res *= 111_320.0 * np.cos(np.deg2rad(lat))
    return res


def build_target_grid(layer_paths: Dict[str, Path], res: float, extent: str | list) -> TargetGrid:
    """Build a snapped 10 m grid from the projected footprints of all inputs."""
    if isinstance(extent, (list, tuple)):
        xmin, ymin, xmax, ymax = map(float, extent)
    else:
        boxes = []
        for name, p in layer_paths.items():
            with rasterio.open(p) as src:
                if src.crs is None:
                    raise ValueError(f"Layer '{name}' ({p}) has no CRS; assign one before harmonizing.")
                boxes.append(transform_bounds(src.crs, TARGET_CRS, *src.bounds, densify_pts=21))
        arr = np.array(boxes)
        if extent == "intersection":
            xmin, ymin = arr[:, 0].max(), arr[:, 1].max()
            xmax, ymax = arr[:, 2].min(), arr[:, 3].min()
        elif extent == "union":
            xmin, ymin = arr[:, 0].min(), arr[:, 1].min()
            xmax, ymax = arr[:, 2].max(), arr[:, 3].max()
        else:
            raise ValueError(f"Unknown extent mode '{extent}'")
        if xmax <= xmin or ymax <= ymin:
            raise ValueError("Input layers do not overlap; check footprints or use extent: union.")

    # Snap outward to the pixel lattice so grids from separate runs align exactly.
    xmin = np.floor(xmin / res) * res
    ymin = np.floor(ymin / res) * res
    xmax = np.ceil(xmax / res) * res
    ymax = np.ceil(ymax / res) * res
    grid = TargetGrid(float(xmin), float(ymax), int(round((xmax - xmin) / res)), int(round((ymax - ymin) / res)), res)
    _check_utm_zone(grid)
    LOG.info("Target grid %d x %d px @ %.1f m, bounds %s", grid.width, grid.height, res, tuple(round(b) for b in grid.bounds))
    return grid


def _check_utm_zone(grid: TargetGrid) -> None:
    """Warn when the AOI centroid sits outside the active UTM zone."""
    to_geo = Transformer.from_crs(TARGET_CRS, "EPSG:4326", always_xy=True)
    cx = 0.5 * (grid.bounds[0] + grid.bounds[2])
    cy = 0.5 * (grid.bounds[1] + grid.bounds[3])
    lon, lat = to_geo.transform(cx, cy)
    lo, hi = NIGERIA_UTM_ZONES[TARGET_CRS.to_epsg()]
    if not (lo <= lon <= hi):
        better = next((e for e, (a, b) in NIGERIA_UTM_ZONES.items() if a <= lon <= b), None)
        LOG.warning("AOI centroid lon %.2f is outside %s (%gE-%gE); consider EPSG:%s.",
                    lon, TARGET_CRS.to_string(), lo, hi, better)


# =============================================================================
# Warping and gap handling
# =============================================================================
def warp_to_grid(path: Path, grid: TargetGrid, kind: str, band: int = 1) -> Tuple[np.ndarray, float]:
    """Reproject one band onto the target grid. Returns (array with NaN nodata, native res m)."""
    dst = np.full(grid.shape, np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        native = _native_res_m(src)
        src_arr = src.read(band, masked=True).astype(np.float32).filled(np.nan)
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=grid.transform,
            dst_crs=TARGET_CRS,
            dst_nodata=np.nan,
            resampling=RESAMPLING_BY_KIND[kind],
            num_threads=2,
        )
    finite = np.isfinite(dst).mean()
    LOG.info("  warped %-12s native %.1f m -> %.1f m, coverage %.1f%%", path.name, native, grid.res, 100 * finite)
    return dst, native


def fill_nan_nearest(arr: np.ndarray) -> np.ndarray:
    """Fill NaNs with the nearest finite value (Euclidean distance transform)."""
    mask = ~np.isfinite(arr)
    if not mask.any():
        return arr
    if mask.all():
        raise ValueError("Cannot fill an all-NaN array")
    idx = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    return arr[tuple(idx)]


def deblock_geophysics(arr: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian smoothing that respects the coverage footprint (normalised convolution)."""
    if sigma_px <= 0:
        return arr
    valid = np.isfinite(arr)
    num = ndimage.gaussian_filter(np.where(valid, arr, 0.0), sigma_px, mode="nearest")
    den = ndimage.gaussian_filter(valid.astype(np.float32), sigma_px, mode="nearest")
    out = np.where(den > 1e-3, num / np.maximum(den, 1e-6), np.nan).astype(np.float32)
    out[~valid] = np.nan
    return out


# =============================================================================
# Physics-derived layers
# =============================================================================
def first_vertical_derivative(tmi: np.ndarray, dx: float, native_res_m: float) -> np.ndarray:
    """
    First vertical derivative of a potential field via the wavenumber domain:
        F[dT/dz] = |k| F[T]
    A 4th-order Butterworth low-pass at the native survey Nyquist suppresses
    the noise that upsampling and differentiation would otherwise amplify.
    """
    valid = np.isfinite(tmi)
    filled = fill_nan_nearest(tmi.astype(np.float64))
    filled -= filled.mean()
    ny, nx = filled.shape
    pad_y, pad_x = ny // 2, nx // 2
    padded = np.pad(filled, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    py, px = padded.shape
    ky = 2 * np.pi * sfft.fftfreq(py, d=dx)
    kx = 2 * np.pi * sfft.fftfreq(px, d=dx)
    k = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    k_cut = np.pi / max(native_res_m, dx)  # Nyquist of the original survey grid
    lowpass = 1.0 / (1.0 + (k / k_cut) ** 8)
    spec = sfft.fft2(padded) * k * lowpass
    out = np.real(sfft.ifft2(spec))[pad_y:pad_y + ny, pad_x:pad_x + nx].astype(np.float32)
    out[~valid] = np.nan
    return out


def slope_degrees(dem: np.ndarray, dx: float) -> np.ndarray:
    valid = np.isfinite(dem)
    filled = fill_nan_nearest(dem.astype(np.float32))
    gy, gx = np.gradient(filled, dx)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
    slope[~valid] = np.nan
    return slope


def safe_ratio(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    out = np.where(np.abs(b) > eps, a / np.where(np.abs(b) > eps, b, 1.0), np.nan)
    return out.astype(np.float32)


def to_db(arr: np.ndarray, name: str) -> np.ndarray:
    finite = arr[np.isfinite(arr)]
    if finite.size and np.nanmin(finite) < 0:
        LOG.info("  %s already contains negative values; assuming dB, skipping conversion", name)
        return arr
    return (10.0 * np.log10(np.maximum(arr, 1e-6))).astype(np.float32)


# =============================================================================
# Scaling
# =============================================================================
def _sample_finite(arr: np.ndarray, n: int = 2_000_000, seed: int = 0) -> np.ndarray:
    v = arr[np.isfinite(arr)]
    if v.size > n:
        v = np.random.default_rng(seed).choice(v, n, replace=False)
    return v


def fit_scaler(arr: np.ndarray, method: str, cfg: Config) -> dict:
    v = _sample_finite(arr)
    if v.size == 0:
        raise ValueError("Channel has no finite pixels inside the target grid")
    if method == MINMAX:
        lo, hi = np.percentile(v, cfg["scaling"]["clip_percentiles"])
        if hi - lo < 1e-9:
            hi = lo + 1e-6
        return {"method": MINMAX, "lo": float(lo), "hi": float(hi)}
    lo, hi = np.percentile(v, cfg["scaling"]["zscore_clip_percentiles"])
    vc = np.clip(v, lo, hi)
    std = float(vc.std()) or 1e-6
    return {"method": ZSCORE, "lo": float(lo), "hi": float(hi), "mean": float(vc.mean()), "std": std,
            "limit": float(cfg["scaling"]["zscore_limit"])}


def apply_scaler(arr: np.ndarray, stats: dict) -> np.ndarray:
    if stats["method"] == MINMAX:
        out = (np.clip(arr, stats["lo"], stats["hi"]) - stats["lo"]) / (stats["hi"] - stats["lo"])
    else:
        out = (np.clip(arr, stats["lo"], stats["hi"]) - stats["mean"]) / stats["std"]
        out = np.clip(out, -stats["limit"], stats["limit"])
    return out.astype(np.float32)


# =============================================================================
# Harmonizer
# =============================================================================
class MultimodalHarmonizer:
    """Builds the 11-channel analysis-ready stack."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.res = float(cfg["project"]["resolution_m"])
        set_target_crs(cfg["project"]["crs"])
        unknown = [c for c in cfg["channels"] if c not in CHANNEL_LAYERS]
        if unknown:
            raise KeyError(f"No recipe for channels {unknown}")
        needed = sorted({lay for c in cfg["channels"] for lay in CHANNEL_LAYERS[c]})
        absent = [k for k in needed if k not in cfg["layers"]]
        if absent:
            raise KeyError(f"Channels need layers {absent}, which are not defined under 'layers' in the config")
        self.layers = {k: cfg["layers"][k] for k in needed}
        self.layer_paths = {k: cfg.resolve(v["path"]) for k, v in self.layers.items()}
        missing = [str(p) for p in self.layer_paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing input rasters:\n  " + "\n  ".join(missing))
        extent = cfg["project"]["extent"]
        if cfg["project"].get("extent_lonlat"):
            lon0, lat0, lon1, lat1 = map(float, cfg["project"]["extent_lonlat"])
            extent = list(transform_bounds("EPSG:4326", TARGET_CRS, lon0, lat0, lon1, lat1, densify_pts=21))
        self.grid = build_target_grid(self.layer_paths, self.res, extent)
        self._cache: Dict[str, np.ndarray] = {}

    # ---- layer access -----------------------------------------------------
    def _layer(self, key: str) -> np.ndarray:
        """Warp and pre-process a raw layer once, then cache it."""
        if key in self._cache:
            return self._cache[key]
        spec = self.layers[key]
        kind = spec["kind"]
        arr, native = warp_to_grid(self.layer_paths[key], self.grid, kind, int(spec.get("band", 1)))
        if "scale" in spec or "offset" in spec:  # e.g. Sentinel-2 L2A: scale 1e-4, offset -0.1
            arr = arr * float(spec.get("scale", 1.0)) + float(spec.get("offset", 0.0))
        if kind == "sar" and spec.get("to_db", False):
            arr = to_db(arr, key)
        if kind == "geophys":
            sigma = spec.get("smooth_sigma_px")
            sigma = float(sigma) if sigma is not None else 0.5 * native / self.res
            if spec.get("compute_1vd", False):
                # Differentiate on a grid at the survey's native cell size, then upsample.
                # Cheaper (no FFT over the 10 m grid) and avoids differentiating interpolation.
                arr = self._vd_native(key, native)
            arr = deblock_geophysics(arr, sigma)
        self._cache[key] = arr
        return arr

    def _vd_native(self, key: str, native_m: float) -> np.ndarray:
        g = self.grid
        cres = max(g.res, round(native_m / g.res) * g.res)
        coarse = TargetGrid(g.xmin, g.ymax, int(np.ceil(g.width * g.res / cres)),
                            int(np.ceil(g.height * g.res / cres)), cres)
        carr, _ = warp_to_grid(self.layer_paths[key], coarse, "geophys", int(self.layers[key].get("band", 1)))
        carr = first_vertical_derivative(carr, cres, native_m)
        out = np.full(g.shape, np.nan, dtype=np.float32)
        reproject(carr, out, src_transform=coarse.transform, src_crs=TARGET_CRS, src_nodata=np.nan,
                  dst_transform=g.transform, dst_crs=TARGET_CRS, dst_nodata=np.nan, resampling=Resampling.cubic)
        LOG.info("  1VD computed on a %.0f m grid (%d x %d) and upsampled", cres, coarse.width, coarse.height)
        return out

    def _release(self, *keys: str) -> None:
        for k in keys:
            self._cache.pop(k, None)

    # ---- channel recipes --------------------------------------------------
    def _recipes(self) -> Dict[str, Tuple[Callable[[], np.ndarray], str, Tuple[str, ...]]]:
        """channel -> (builder, scaling method, raw layers it can release afterwards)."""
        L = self._layer
        return {
            "B2_blue": (lambda: L("s2_b02"), MINMAX, ("s2_b02",)),
            "B3_green": (lambda: L("s2_b03"), MINMAX, ("s2_b03",)),
            "B4_red": (lambda: L("s2_b04"), MINMAX, ()),
            "B8_nir": (lambda: L("s2_b08"), MINMAX, ()),
            "S1_VV_dB": (lambda: L("s1_vv"), MINMAX, ("s1_vv",)),
            "S1_VH_dB": (lambda: L("s1_vh"), MINMAX, ("s1_vh",)),
            "OH_alteration": (lambda: safe_ratio(L("s2_b11"), L("s2_b12")), MINMAX, ("s2_b11", "s2_b12")),
            "NDVI": (lambda: np.clip(safe_ratio(L("s2_b08") - L("s2_b04"), L("s2_b08") + L("s2_b04")), -1, 1),
                     MINMAX, ("s2_b04", "s2_b08")),
            "MAG_1VD": (lambda: L("mag"), ZSCORE, ("mag",)),
            "RAD_eTh": (lambda: L("th"), ZSCORE, ("th",)),
            "DEM_slope": (lambda: slope_degrees(L("dem"), self.res), MINMAX, ("dem",)),
        }

    # ---- main entry -------------------------------------------------------
    def run(self, out_path: Path, mask_path: Path, stats_path: Path, reuse_stats: bool = False) -> Path:
        channels = list(self.cfg["channels"])
        recipes = self._recipes()

        saved = json.loads(stats_path.read_text()) if (reuse_stats and stats_path.exists()) else {}
        stats_out: Dict[str, dict] = {}
        valid = np.ones(self.grid.shape, dtype=bool)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        profile = dict(
            driver="GTiff", width=self.grid.width, height=self.grid.height, count=len(channels),
            dtype="float32", crs=TARGET_CRS, transform=self.grid.transform, nodata=np.nan,
            tiled=True, blockxsize=256, blockysize=256, compress="deflate", predictor=3,
            BIGTIFF="IF_SAFER", interleave="band",
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            for i, ch in enumerate(channels, start=1):
                build, method, release = recipes[ch]
                LOG.info("Channel %2d/%d  %s", i, len(channels), ch)
                raw = build()
                stats = saved.get(ch) or fit_scaler(raw, method, self.cfg)
                scaled = apply_scaler(raw, stats)
                scaled[~np.isfinite(raw)] = np.nan
                valid &= np.isfinite(scaled)
                dst.write(scaled, i)
                dst.set_band_description(i, ch)
                stats_out[ch] = stats
                self._release(*release)
            dst.update_tags(PROJECT="GeoSentry-SOTA", CHANNELS=",".join(channels), RES_M=str(self.res),
                            CRS=TARGET_CRS.to_string())

        mask_profile = {**profile, "count": 1, "dtype": "uint8", "nodata": 0, "predictor": 1}
        with rasterio.open(mask_path, "w", **mask_profile) as m:
            m.write(valid.astype(np.uint8), 1)
        stats_path.write_text(json.dumps({"crs": TARGET_CRS.to_string(), "res_m": self.res, **stats_out}, indent=2))
        LOG.info("Stack written: %s (valid %.1f%%)", out_path, 100 * valid.mean())
        return out_path


def harmonize(cfg: Config, reuse_stats: bool = False) -> Path:
    h = MultimodalHarmonizer(cfg)
    return h.run(cfg.path("stack"), cfg.path("valid_mask"), cfg.path("scaling_stats"), reuse_stats)


def align_label_raster(src_path: Path, reference_path: Path, out_path: Path, nodata: int = 255) -> Path:
    """
    Warp a label raster (nearest neighbour) onto the harmonized stack grid.
    Source value 255 (or the file's nodata) is treated as "ignore" and kept as 255.
    """
    with rasterio.open(reference_path) as ref:
        prof, shape, transform, crs = ref.profile, ref.shape, ref.transform, ref.crs
    dst = np.full(shape, nodata, dtype=np.uint8)
    with rasterio.open(src_path) as src:
        reproject(rasterio.band(src, 1), dst, src_transform=src.transform, src_crs=src.crs,
                  src_nodata=src.nodata if src.nodata is not None else nodata, dst_transform=transform, dst_crs=crs, dst_nodata=nodata,
                  resampling=Resampling.nearest)
    prof.update(count=1, dtype="uint8", nodata=nodata, predictor=1)
    with rasterio.open(out_path, "w", **prof) as d:
        d.write(dst, 1)
    return out_path


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description="Harmonize raw rasters into the 11-channel GeoSentry stack")
    ap.add_argument("--config", default=None)
    ap.add_argument("--reuse-stats", action="store_true", help="apply saved scaling statistics")
    args = ap.parse_args(argv)
    harmonize(load_config(args.config), args.reuse_stats)


if __name__ == "__main__":
    main()
