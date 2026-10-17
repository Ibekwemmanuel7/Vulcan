"""
Open-data downloader for a GeoSentry project AOI
================================================

Builds the satellite and terrain inputs for a project from free archives on the
Microsoft Planetary Computer STAC API, aligned to the project's 10 m UTM grid:

  s2   Sentinel-2 L2A dry-season median composite, bands B02 B03 B04 B08 B11 B12,
       as surface reflectance (0-1). The +1000 DN offset of processing baseline
       04.00 and later is removed per scene. Clouds, shadows, cirrus and
       saturated pixels are masked with the SCL layer before the median.
       B11/B12 are 20 m bands resampled to the 10 m grid (bilinear).
  s1   Sentinel-1 RTC (radiometrically terrain corrected gamma0, linear power)
       median of VV and VH over the same season, optionally one orbit direction.
  dem  Copernicus DEM GLO-30 mosaic on a 30 m grid in the project CRS.

Downloads run tile by tile (default 1024 x 1024 px), so memory stays bounded and
an interrupted run resumes where it stopped. A provenance JSON records every
STAC item used, the query and the processing rules.

Requirements (installed by requirements.txt): pystac-client, planetary-computer,
odc-stac. Network access to planetarycomputer.microsoft.com is needed.

    python scripts\\fetch_open_data.py --config configs\\pilot_nasarawa_keffi_li.yaml
    python scripts\\fetch_open_data.py --config configs\\pilot_nasarawa_keffi_li.yaml --only dem
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src  # Configure bundled PROJ/GDAL before rasterio imports.

import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import transform_bounds
from rasterio.windows import Window

from src.config import Config, get_logger, load_config  # noqa: E402

LOG = get_logger("fetch")
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
S2_LAYER_KEYS = {"B02": "s2_b02", "B03": "s2_b03", "B04": "s2_b04", "B08": "s2_b08", "B11": "s2_b11", "B12": "s2_b12"}
# SCL classes kept: 4 vegetation, 5 not vegetated, 6 water, 7 unclassified.
# Dropped: 0 no data, 1 saturated/defective, 2 dark/topographic shadow, 3 cloud shadow,
#          8-9 cloud medium/high probability, 10 thin cirrus, 11 snow.
SCL_KEEP = (4, 5, 6, 7)
# Concurrent asset reads per tile (set by --workers). Downloads are latency-bound:
# one tile reads every scene x band, so parallel reads give most of the speed-up.
# Outputs are identical for any value, so resume is unaffected.
WORKERS = 1


# =============================================================================
# Grid and tiles
# =============================================================================
def project_grid(cfg: Config, res: float) -> Tuple[Affine, int, int, str]:
    """Snapped grid over project.extent_lonlat in project.crs (same rule as the harmonizer)."""
    crs = cfg["project"]["crs"]
    lon0, lat0, lon1, lat1 = map(float, cfg["project"]["extent_lonlat"])
    x0, y0, x1, y1 = transform_bounds("EPSG:4326", crs, lon0, lat0, lon1, lat1, densify_pts=21)
    x0, y0 = np.floor(x0 / res) * res, np.floor(y0 / res) * res
    x1, y1 = np.ceil(x1 / res) * res, np.ceil(y1 / res) * res
    w, h = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    return Affine(res, 0, float(x0), 0, -res, float(y1)), w, h, crs


def tiles(width: int, height: int, size: int) -> List[Window]:
    return [Window(c, r, min(size, width - c), min(size, height - r))
            for r in range(0, height, size) for c in range(0, width, size)]


def tile_lonlat_bounds(transform: Affine, win: Window, crs: str) -> Tuple[float, float, float, float]:
    x0, y1 = transform * (win.col_off, win.row_off)
    x1, y0 = transform * (win.col_off + win.width, win.row_off + win.height)
    return transform_bounds(crs, "EPSG:4326", x0, y0, x1, y1, densify_pts=5)


def create_raster(path: Path, transform: Affine, width: int, height: int, crs: str, desc: str) -> None:
    if path.exists():
        with rasterio.open(path) as d:
            if (d.width, d.height, d.count) != (width, height, 1) or str(d.crs) != crs or not d.transform.almost_equals(transform):
                raise ValueError(f"Existing raster has a different grid: {path}. Use a new output directory.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=1, dtype="float32",
                       crs=crs, transform=transform, nodata=np.nan, tiled=True, blockxsize=512, blockysize=512,
                       compress="deflate", predictor=3, BIGTIFF="IF_SAFER") as d:
        d.set_band_description(1, desc)


# =============================================================================
# Pure compositing functions (unit tested)
# =============================================================================
def s2_offsets_for_times(items: Sequence, times: Sequence[np.datetime64]) -> np.ndarray:
    """
    DN offset per composite time step. Baseline >= 04.00 scenes carry +1000.
    Matches odc-stac 'solar_day' grouping: date of acquisition time shifted by
    longitude / 15 hours.
    """
    by_day: Dict[str, float] = {}
    for it in items:
        lon = 0.5 * (it.bbox[0] + it.bbox[2])
        dt = it.datetime.replace(tzinfo=None) + timedelta(hours=lon / 15.0)
        base = str(it.properties.get("s2:processing_baseline", ""))
        try:
            off = 1000.0 if float(base) >= 4.0 else 0.0
        except ValueError as exc:
            raise ValueError("Missing or invalid Sentinel-2 processing baseline; cannot safely scale reflectance") from exc
        day = dt.date().isoformat()
        if day in by_day and by_day[day] != off:
            raise ValueError("Mixed Sentinel-2 offsets within one solar-day mosaic; use separate acquisitions")
        by_day[day] = off
    out = []
    for t in times:
        day = np.datetime_as_string(np.datetime64(t, "D"))
        if day not in by_day:
            raise ValueError(f"No Sentinel-2 offset metadata for loaded date {day}")
        out.append(by_day[day])
    return np.asarray(out, dtype=np.float32)


def s2_median(dn: Dict[str, np.ndarray], scl: np.ndarray, offsets: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    dn[band]: (t, h, w) uint16 digital numbers (0 = no data); scl: (t, h, w).
    Returns median reflectance per band (h, w) and the count of clear observations.
    """
    clear = np.isin(scl, SCL_KEEP)
    out = {}
    for b, a in dn.items():
        refl = (a.astype(np.float32) - offsets[:, None, None]) / 10000.0
        refl[(a == 0) | ~clear] = np.nan
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                out[b] = np.nanmedian(refl, axis=0).astype(np.float32)
    return out, clear.sum(0).astype(np.float32)


def s1_median(a: np.ndarray) -> np.ndarray:
    """Median of linear gamma0 over time; non-positive and nodata values ignored."""
    a = a.astype(np.float32)
    a[~np.isfinite(a) | (a <= 0)] = np.nan
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(a, axis=0).astype(np.float32)


# =============================================================================
# STAC access
# =============================================================================
def search(collection: str, bbox: Sequence[float], date_ranges: Sequence[str], query: dict = None) -> list:
    import pystac_client

    client = pystac_client.Client.open(STAC_URL)
    items = []
    for dr in date_ranges:
        res = client.search(collections=[collection], bbox=list(bbox), datetime=dr, query=query or {})
        items += list(res.items())
    items = list({it.id: it for it in items}.values())
    LOG.info("%s: %d items over %s", collection, len(items), list(date_ranges))
    return items


def load_tile(items: list, bands: List[str], transform: Affine, win: Window, crs: str, resampling: dict,
              retries: int = 4):
    import odc.stac
    import planetary_computer
    from odc.geo.geobox import GeoBox

    gb = GeoBox((int(win.height), int(win.width)), transform * Affine.translation(win.col_off, win.row_off), crs)
    for attempt in range(1, retries + 1):
        try:
            return odc.stac.load(items, bands=bands, geobox=gb, groupby="solar_day", resampling=resampling,
                                 patch_url=planetary_computer.sign, chunks=None,
                                 pool=WORKERS if WORKERS > 1 else None)
        except Exception as exc:  # network hiccups are common on long runs
            if attempt == retries:
                raise
            wait = 10 * attempt
            LOG.warning("Tile load failed (%s); retry %d/%d in %ds", type(exc).__name__, attempt, retries - 1, wait)
            time.sleep(wait)


def items_for_tile(items: list, lonlat: Tuple[float, float, float, float]) -> list:
    a0, b0, a1, b1 = lonlat
    return [it for it in items if not (it.bbox[2] < a0 or it.bbox[0] > a1 or it.bbox[3] < b0 or it.bbox[1] > b1)]


class Progress:
    def __init__(self, path: Path, signature: str = None):
        self.path = path
        self.signature = signature
        saved = json.loads(path.read_text()) if path.exists() else {"signature": signature, "done": []}
        if not isinstance(saved, dict) or saved.get("signature") != signature:
            raise ValueError(f"Resume settings changed or legacy progress found: {path}. Use a new output directory.")
        self.done = set(saved["done"])

    def mark(self, key: str) -> None:
        self.done.add(key)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps({"signature": self.signature, "done": sorted(self.done)}))
        temp.replace(self.path)


def resume_signature(cfg: Config, tile_px: int, items: list, product: str) -> str:
    payload = {"project": cfg["project"], "acquisition": cfg.get("acquisition", {}),
               "tile": tile_px, "product": product, "items": sorted(it.id for it in items),
               "processing_version": 2}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# =============================================================================
# Products
# =============================================================================
def fetch_s2(cfg: Config, tile_px: int) -> None:
    acq = cfg["acquisition"]
    transform, w, h, crs = project_grid(cfg, float(cfg["project"]["resolution_m"]))
    paths = {b: cfg.resolve(cfg["layers"][S2_LAYER_KEYS[b]]["path"]) for b in S2_BANDS}
    count_path = paths["B02"].with_name("S2_clear_count.tif")
    for b, p in paths.items():
        create_raster(p, transform, w, h, crs, f"Sentinel-2 L2A {b} median surface reflectance")
    create_raster(count_path, transform, w, h, crs, "Sentinel-2 clear observations used")
    items = search("sentinel-2-l2a", cfg["project"]["extent_lonlat"], acq["s2_date_ranges"],
                   {"eo:cloud_cover": {"lt": float(acq["s2_max_cloud_pct"])}})
    if not items:
        raise RuntimeError("No Sentinel-2 scenes found; widen s2_date_ranges or s2_max_cloud_pct")
    prog = Progress(paths["B02"].with_name(".s2_progress.json"), resume_signature(cfg, tile_px, items, "s2"))
    tl = tiles(w, h, tile_px)
    for i, win in enumerate(tl, 1):
        key = f"{win.row_off}_{win.col_off}"
        if key in prog.done:
            continue
        its = items_for_tile(items, tile_lonlat_bounds(transform, win, crs))
        if not its:
            raise RuntimeError(f"No Sentinel-2 coverage for tile {key}")
        ds = load_tile(its, S2_BANDS + ["SCL"], transform, win, crs, {"SCL": "nearest", "*": "bilinear"})
        offsets = s2_offsets_for_times(its, ds.time.values)
        med, cnt = s2_median({b: ds[b].values for b in S2_BANDS}, ds["SCL"].values, offsets)
        for b in S2_BANDS:
            with rasterio.open(paths[b], "r+") as d:
                d.write(med[b], 1, window=win)
        with rasterio.open(count_path, "r+") as d:
            d.write(cnt, 1, window=win)
        prog.mark(key)
        LOG.info("S2 tile %d/%d: %d dates, median clear obs %.0f", i, len(tl), ds.sizes["time"], np.median(cnt))
    write_provenance(paths["B02"].with_name("provenance_s2.json"), "sentinel-2-l2a", items, acq,
                     {"bands": S2_BANDS, "scl_keep": SCL_KEEP, "reducer": "median",
                      "offset_rule": "subtract 1000 DN for s2:processing_baseline >= 04.00, then / 10000",
                      "resampling": "bilinear (B11/B12 from 20 m), SCL nearest"})


def fetch_s1(cfg: Config, tile_px: int) -> None:
    if not os.environ.get("PC_SDK_SUBSCRIPTION_KEY"):
        LOG.warning("Sentinel-1 RTC collection metadata declares account-restricted access, while general "
                    "SAS documentation allows anonymous access. Trying the service; if authorization fails, "
                    "use PC_SDK_SUBSCRIPTION_KEY or download only dem and s2.")
    acq = cfg["acquisition"]
    transform, w, h, crs = project_grid(cfg, float(cfg["project"]["resolution_m"]))
    paths = {"vv": cfg.resolve(cfg["layers"]["s1_vv"]["path"]), "vh": cfg.resolve(cfg["layers"]["s1_vh"]["path"])}
    for b, p in paths.items():
        create_raster(p, transform, w, h, crs, f"Sentinel-1 RTC gamma0 {b.upper()} median (linear)")
    q = {}
    if acq.get("s1_orbit_state"):
        q["sat:orbit_state"] = {"eq": acq["s1_orbit_state"]}
    items = search("sentinel-1-rtc", cfg["project"]["extent_lonlat"], acq["s1_date_ranges"], q)
    items = [it for it in items if all(b in it.assets for b in ("vv", "vh"))]
    if not items:
        raise RuntimeError("No Sentinel-1 RTC scenes found for the requested dates / orbit")
    prog = Progress(paths["vv"].with_name(".s1_progress.json"), resume_signature(cfg, tile_px, items, "s1"))
    tl = tiles(w, h, tile_px)
    for i, win in enumerate(tl, 1):
        key = f"{win.row_off}_{win.col_off}"
        if key in prog.done:
            continue
        its = items_for_tile(items, tile_lonlat_bounds(transform, win, crs))
        if not its:
            raise RuntimeError(f"No Sentinel-1 coverage for tile {key}")
        ds = load_tile(its, ["vv", "vh"], transform, win, crs, {"*": "bilinear"})
        for b in ("vv", "vh"):
            with rasterio.open(paths[b], "r+") as d:
                d.write(s1_median(ds[b].values), 1, window=win)
        prog.mark(key)
        LOG.info("S1 tile %d/%d: %d dates", i, len(tl), ds.sizes["time"])
    write_provenance(paths["vv"].with_name("provenance_s1.json"), "sentinel-1-rtc", items, acq,
                     {"bands": ["vv", "vh"], "units": "gamma0 linear power", "reducer": "median",
                      "orbit_state": acq.get("s1_orbit_state")})


def fetch_dem(cfg: Config, tile_px: int) -> None:
    transform, w, h, crs = project_grid(cfg, 30.0)
    path = cfg.resolve(cfg["layers"]["dem"]["path"])
    create_raster(path, transform, w, h, crs, "Copernicus DEM GLO-30 elevation (m)")
    items = search("cop-dem-glo-30", cfg["project"]["extent_lonlat"], ["2000-01-01/2030-12-31"])
    if not items:
        raise RuntimeError("No DEM coverage for requested area")
    prog = Progress(path.with_name(".dem_progress.json"), resume_signature(cfg, tile_px, items, "dem"))
    for win in tiles(w, h, tile_px):
        key = f"{win.row_off}_{win.col_off}"
        if key in prog.done:
            continue
        its = items_for_tile(items, tile_lonlat_bounds(transform, win, crs))
        if not its:
            raise RuntimeError(f"No DEM coverage for tile {key}")
        ds = load_tile(its, ["data"], transform, win, crs, {"*": "bilinear"})
        a = ds["data"].values.astype(np.float32)
        a[a < -1000] = np.nan
        with rasterio.open(path, "r+") as d:
            d.write(np.nanmax(a, axis=0) if a.shape[0] > 1 else a[0], 1, window=win)
        prog.mark(key)
    write_provenance(path.with_name("provenance_dem.json"), "cop-dem-glo-30", items, {}, {"resolution_m": 30})


def write_provenance(path: Path, collection: str, items: list, acq: dict, rules: dict) -> None:
    path.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "stac_api": STAC_URL,
        "collection": collection, "n_items": len(items), "query": acq, "rules": rules,
        "items": sorted({it.id for it in items}),
    }, indent=2, default=str))
    LOG.info("Provenance written: %s", path)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Download Sentinel-2, Sentinel-1 RTC and Copernicus DEM for a project AOI")
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", nargs="+", choices=["s2", "s1", "dem"], default=["dem", "s1", "s2"])
    ap.add_argument("--tile", type=int, default=256, help="tile size in pixels; peak memory also grows with acquisition count")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel asset reads per tile (8 is a good start); does not change the output")
    ap.add_argument("--catalog-only", action="store_true", help="verify available scenes without downloading rasters")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if not cfg["project"].get("extent_lonlat"):
        raise SystemExit("project.extent_lonlat must be set in the config")
    global WORKERS
    WORKERS = max(1, int(args.workers))
    if args.tile <= 0:
        raise SystemExit("--tile must be positive")
    if args.catalog_only:
        acq = cfg["acquisition"]
        for prod in args.only:
            collection = {"s2": "sentinel-2-l2a", "s1": "sentinel-1-rtc", "dem": "cop-dem-glo-30"}[prod]
            dates = acq.get(f"{prod}_date_ranges", ["2000-01-01/2030-12-31"])
            query = {"eo:cloud_cover": {"lt": float(acq["s2_max_cloud_pct"])}} if prod == "s2" else {}
            if prod == "s1" and acq.get("s1_orbit_state"):
                query["sat:orbit_state"] = {"eq": acq["s1_orbit_state"]}
            items = search(collection, cfg["project"]["extent_lonlat"], dates, query)
            if not items:
                raise SystemExit(f"No {prod} items found for the configured query")
        return
    for prod in args.only:
        LOG.info("========== %s ==========", prod.upper())
        {"s2": fetch_s2, "s1": fetch_s1, "dem": fetch_dem}[prod](cfg, args.tile)


if __name__ == "__main__":
    main()
