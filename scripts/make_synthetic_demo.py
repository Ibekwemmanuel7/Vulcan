"""
Synthetic demo AOI generator
============================

Creates an entirely SYNTHETIC test area so the full pipeline and dashboard can
be exercised before real data arrives. The generator writes the targets
directly into the thorium, SWIR, SAR and terrain inputs, so success on it only
proves the plumbing works. It validates no geological assumption and must never
be shown to investors as results.

What it simulates (6.4 x 6.4 km near 8.6E, 8.9N, inside UTM 32N):
  * NE-SW pegmatite dykes (2-4 px wide) and parallel shear corridors
  * eTh highs over dykes, Al-OH alteration halos, magnetic lows along shears
  * seasonal vegetation and farmland patches that mask optical signal
  * inputs delivered on deliberately mismatched grids:
      S2 10 m bands   EPSG:32632, 10 m, larger extent
      S2 B11/B12      EPSG:32632, 20 m
      S1 VV/VH        EPSG:4326,  ~11 m, linear power
      Magnetics TMI   EPSG:32632, 50 m
      Radiometric eTh EPSG:32631, 100 m (neighbouring UTM zone)
      DEM             EPSG:4326,  ~30 m
  * a label raster and an occurrence CSV that covers only SOME dykes, so the
    remaining dykes play the role of undiscovered deposits for PU learning

    python scripts/make_synthetic_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Configure GDAL/PROJ before importing the geospatial libraries.
import src  # noqa: F401, E402

import numpy as np
import pandas as pd
import rasterio
import yaml
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject
from scipy import ndimage

DEMO_RAW = ROOT / "data" / "demo" / "raw"
DEMO_LABELS = ROOT / "data" / "demo" / "labels"
SRC_CRS = "EPSG:32632"
RES = 10.0
N = 768                      # simulated canvas (px); harmonized AOI is the overlap of all inputs
LON0, LAT0 = 8.60, 8.90


def smooth_noise(rng, shape, sigma):
    f = ndimage.gaussian_filter(rng.standard_normal(shape), sigma)
    return (f - f.mean()) / (f.std() + 1e-9)


def draw_dykes(rng, shape, n, width_range=(2, 4), angle_deg=(35, 55)):
    H, W = shape
    mask = np.zeros(shape, dtype=bool)
    yy, xx = np.mgrid[0:H, 0:W]
    lines = []
    for _ in range(n):
        ang = np.deg2rad(rng.uniform(*angle_deg))
        cy, cx = rng.uniform(0.15 * H, 0.85 * H), rng.uniform(0.15 * W, 0.85 * W)
        length = rng.uniform(80, 260)
        width = rng.uniform(*width_range)
        dx, dy = np.cos(ang), -np.sin(ang)
        t = (xx - cx) * dx + (yy - cy) * dy
        d = np.abs(-(xx - cx) * dy + (yy - cy) * dx)
        seg = (np.abs(t) <= length / 2) & (d <= width / 2)
        mask |= seg
        lines.append(seg)
    return mask, lines


def write(path: Path, arr: np.ndarray, transform, crs, nodata=None, dtype="float32"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1,
                       dtype=dtype, crs=crs, transform=transform, nodata=nodata, compress="deflate") as d:
        d.write(arr.astype(dtype), 1)


def resample_to(arr, src_transform, dst_crs, dst_res, resampling=Resampling.average, nodata=np.nan):
    H, W = arr.shape
    left, top = src_transform.c, src_transform.f
    bounds = (left, top - H * RES, left + W * RES, top)
    tr, w, h = calculate_default_transform(SRC_CRS, dst_crs, W, H, *bounds, resolution=dst_res)
    out = np.full((h, w), nodata, dtype=np.float32)
    reproject(arr.astype(np.float32), out, src_transform=src_transform, src_crs=SRC_CRS, dst_transform=tr,
              dst_crs=dst_crs, resampling=resampling, src_nodata=np.nan, dst_nodata=nodata)
    return out, tr


def main(seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    x0, y0 = Transformer.from_crs("EPSG:4326", SRC_CRS, always_xy=True).transform(LON0, LAT0)
    transform = from_origin(np.floor(x0 / RES) * RES, np.floor(y0 / RES) * RES, RES, RES)
    shape = (N, N)

    # --- Geology ------------------------------------------------------------
    dykes, lines = draw_dykes(rng, shape, n=14)
    shears, _ = draw_dykes(rng, shape, n=5, width_range=(8, 14), angle_deg=(30, 60))
    granite = smooth_noise(rng, shape, 60) > 0.3
    halo = ndimage.binary_dilation(dykes, iterations=4) & ~dykes

    # --- Land cover ---------------------------------------------------------
    veg = np.clip(0.5 + 0.35 * smooth_noise(rng, shape, 25), 0, 1)
    farms = smooth_noise(rng, shape, 12) > 1.0
    veg = np.where(farms, 0.85, veg)

    # --- Surface reflectance (0-1) -----------------------------------------
    soil = 0.18 + 0.03 * smooth_noise(rng, shape, 8) + 0.05 * dykes + 0.03 * granite
    b2 = soil * 0.45 * (1 - 0.6 * veg) + 0.02 * veg + 0.005 * rng.standard_normal(shape)
    b3 = soil * 0.60 * (1 - 0.5 * veg) + 0.05 * veg + 0.005 * rng.standard_normal(shape)
    b4 = soil * 0.80 * (1 - 0.7 * veg) + 0.03 * veg + 0.005 * rng.standard_normal(shape)
    b8 = soil * 1.10 * (1 - 0.2 * veg) + 0.35 * veg + 0.008 * rng.standard_normal(shape)
    b11 = soil * 1.40 * (1 - 0.4 * veg) + 0.12 * veg + 0.006 * rng.standard_normal(shape)
    # Al-OH absorption lowers B12 over altered halos and dykes (weaker under canopy).
    oh_strength = 0.18 * (halo | dykes) * (1 - 0.7 * veg)
    b12 = soil * 1.20 * (1 - 0.45 * veg) * (1 - oh_strength) + 0.06 * veg + 0.006 * rng.standard_normal(shape)

    # --- SAR (linear power) -------------------------------------------------
    rough = 0.6 * dykes + 0.3 * shears + 0.4 * veg + 0.15 * smooth_noise(rng, shape, 5)
    vv = 10 ** ((-12 + 4 * rough + 1.5 * rng.standard_normal(shape)) / 10)
    vh = 10 ** ((-19 + 5 * veg + 2 * rough + 1.5 * rng.standard_normal(shape)) / 10)

    # --- Geophysics -----------------------------------------------------------
    tmi = 33000 + 40 * smooth_noise(rng, shape, 120) + 25 * smooth_noise(rng, shape, 30)
    tmi -= 35 * ndimage.gaussian_filter(shears.astype(float), 6)        # demagnetised shear corridors
    tmi += 0.002 * (np.arange(N)[None, :] * 10)                         # regional gradient
    eth = 12 + 6 * granite + 4 * smooth_noise(rng, shape, 40)
    eth += 45 * ndimage.gaussian_filter(dykes.astype(float), 3)        # Th in monazite-bearing pegmatites
    eth = np.maximum(eth, 0.5)

    # --- Terrain --------------------------------------------------------------
    dem = 450 + 60 * smooth_noise(rng, shape, 90) + 15 * smooth_noise(rng, shape, 25) + 4 * ndimage.gaussian_filter(dykes.astype(float), 2)

    # --- Write raw inputs on mismatched grids ----------------------------------
    for name, arr in {"S2_B02": b2, "S2_B03": b3, "S2_B04": b4, "S2_B08": b8}.items():
        write(DEMO_RAW / f"{name}.tif", arr, transform, SRC_CRS)
    for name, arr in {"S2_B11": b11, "S2_B12": b12}.items():
        a, tr = resample_to(arr, transform, SRC_CRS, 20.0)
        write(DEMO_RAW / f"{name}.tif", a, tr, SRC_CRS, nodata=np.nan)
    for name, arr in {"S1_VV": vv, "S1_VH": vh}.items():
        a, tr = resample_to(arr, transform, "EPSG:4326", 1.0e-4)
        write(DEMO_RAW / f"{name}.tif", a, tr, "EPSG:4326", nodata=np.nan)
    a, tr = resample_to(tmi, transform, SRC_CRS, 50.0)
    write(DEMO_RAW / "MAG_TMI.tif", a, tr, SRC_CRS, nodata=np.nan)
    a, tr = resample_to(eth, transform, "EPSG:32631", 100.0)
    write(DEMO_RAW / "RAD_eTh.tif", a, tr, "EPSG:32631", nodata=np.nan)
    a, tr = resample_to(dem, transform, "EPSG:4326", 2.7e-4, Resampling.bilinear)
    write(DEMO_RAW / "DEM.tif", a, tr, "EPSG:4326", nodata=np.nan)

    # --- Labels -----------------------------------------------------------------
    # Mapped outlines exist for the 9 "known" dykes only; the rest are undiscovered.
    known = np.zeros(shape, bool)
    order = rng.permutation(len(lines))
    for i in order[:9]:
        known |= lines[i]
    label = np.where(known, 1, 0).astype(np.uint8)
    # Undiscovered dykes are unknown ground, not background: mark as ignore.
    label[dykes & ~known] = 255
    write(DEMO_LABELS / "target_mask_raw.tif", label, transform, SRC_CRS, nodata=None, dtype="uint8")

    # Occurrence points (artisanal pits) on the known dykes, with ~15 m GPS jitter.
    rr, cc = np.nonzero(known)
    pick = rng.choice(len(rr), size=45, replace=False)
    xs, ys = rasterio.transform.xy(transform, rr[pick], cc[pick], offset="center")
    xs = np.asarray(xs) + rng.normal(0, 15, 45)
    ys = np.asarray(ys) + rng.normal(0, 15, 45)
    lon, lat = Transformer.from_crs(SRC_CRS, "EPSG:4326", always_xy=True).transform(xs, ys)
    pd.DataFrame({"name": [f"SYN-{i:03d}" for i in range(45)], "commodity": "Li-Ta-Sn pegmatite (synthetic)",
                  "lon": lon, "lat": lat}).to_csv(DEMO_LABELS / "known_occurrences.csv", index=False)

    # --- Demo config ------------------------------------------------------------
    cfg = yaml.safe_load((ROOT / "configs" / "geosentry.yaml").read_text())
    cfg["project"]["name"] = "GeoSentry-SOTA (SYNTHETIC DEMO)"
    cfg["project"]["demo"] = True
    for k, v in list(cfg["paths"].items()):
        cfg["paths"][k] = v.replace("data/", "data/demo/", 1).replace("outputs/", "outputs/demo/", 1)
    for k, v in cfg["layers"].items():
        v["path"] = v["path"].replace("data/raw", "data/demo/raw")
    cfg["stage1"].update({"patch_size": 128, "stride": 64, "epochs": 6, "batch_size": 8, "val_block_px": 256,
                          "early_stop_patience": 8, "base_channels": 16})
    cfg["stage1"]["ssl"]["epochs"] = 3
    cfg["stage1"]["inference"].update({"overlap": 32, "tta": False})
    cfg["stage2"].update({"unlabeled_samples": 8000, "bags": 12, "shap_samples": 800})
    cfg["evaluation"].update({"outer_folds": 3, "block_m": 1600, "dead_zone_m": 200, "cluster_radius_m": 150,
                              "occurrence_buffer_m": 30, "bootstrap": 500})
    cfg["stage2"]["xgb"]["n_estimators"] = 150
    (ROOT / "configs" / "demo.yaml").write_text(
        "# AUTO-GENERATED by scripts/make_synthetic_demo.py. SYNTHETIC DATA ONLY.\n" + yaml.safe_dump(cfg, sort_keys=False))
    print(f"Synthetic demo written to {DEMO_RAW.parent} and configs/demo.yaml")


if __name__ == "__main__":
    main()
