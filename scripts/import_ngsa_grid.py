"""
Import NGSA airborne geophysics sheets into one GeoTIFF for GeoSentry
====================================================================

NGSA distributes 1:50,000 total magnetic intensity (TMI) sheets as Geosoft grids
and ASCII files (ngsa.gov.ng, "Overview of Airborne Geophysical Survey"). This
script accepts:

  * Geosoft binary grids (.grd), uncompressed or zlib-compressed, read natively
    because GDAL has no Geosoft driver. The CRS is taken from the sidecar
    .grd.xml (wellknown_epsg) when present. Geosoft stores the first row at the
    south and the origin at the CENTRE of the lower-left cell; both are handled.
  * any GDAL-readable grid (.tif, .ers, ...), or
  * ASCII XYZ text: columns X Y VALUE (header lines and comma, space or tab
    separators are handled). Point data are averaged into square cells.

and mosaics all sheets into one GeoTIFF in the project CRS.

The source CRS of NGSA ASCII files is NOT assumed. Read it from the sheet's
header or interpretation report and pass it with --src-crs. Common Nigerian
choices are EPSG:26332 (Minna / UTM 32N), EPSG:26392 (Minna / Nigeria Mid Belt)
and EPSG:32632 (WGS 84 / UTM 32N). A wrong CRS shifts every anomaly by up to a
few hundred metres, so check the result against a known feature before use.

Adjacent sheets can have small levelling differences; inspect the seams.

    python scripts\\import_ngsa_grid.py --config configs\\pilot_nasarawa_keffi_li.yaml ^
        --layer mag --src-crs EPSG:26332 data\\pilot_nk\\ngsa\\sht208_*\\*.xyz
"""
from __future__ import annotations

import argparse
import glob
import struct
import zlib
import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src  # Configure bundled PROJ/GDAL before rasterio imports.

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject

from src.config import get_logger, load_config  # noqa: E402

LOG = get_logger("ngsa")
NUM = re.compile(r"^[\s,;]*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")


GEOSOFT_FLOAT_TYPES = {(4, 2): "<f4", (8, 3): "<f8"}  # (element size, storage format) -> dtype


def is_geosoft_grd(path: Path) -> bool:
    if path.suffix.lower() != ".grd" or path.stat().st_size < 512:
        return False
    with open(path, "rb") as fh:
        es, sf, ne, nv, kx = struct.unpack("<5i", fh.read(20))
    return (es % 1024) in (1, 2, 4, 8) and 0 <= sf <= 3 and ne > 0 and nv > 0 and kx in (1, -1)


def geosoft_epsg(path: Path) -> Optional[str]:
    """EPSG code from the Geosoft .grd.xml sidecar, if it states one."""
    xml = Path(str(path) + ".xml")
    if not xml.exists():
        return None
    codes = set(re.findall(r'wellknown_epsg=(?:&quot;|")(\d+)', xml.read_text(errors="ignore")))
    if len(codes) > 1:
        raise ValueError(f"{xml} lists several EPSG codes {sorted(codes)}; pass --src-crs explicitly")
    return f"EPSG:{codes.pop()}" if codes else None


def read_geosoft_grd(path: Path) -> Tuple[np.ndarray, object]:
    """
    Read a Geosoft binary grid. Returns (north-up float32 array with NaN for
    dummies, affine transform of cell corners).
    """
    b = open(path, "rb").read()
    es, sf, ne, nv, kx = struct.unpack("<5i", b[0:20])
    de, dv, x0, y0, rot, zbase, zmult = struct.unpack("<7d", b[20:76])
    compressed, esize = es > 1024, es % 1024
    if (esize, sf) not in GEOSOFT_FLOAT_TYPES:
        raise ValueError(f"{path}: unsupported Geosoft element type (size {esize}, format {sf})")
    if rot != 0 or kx != 1:
        raise ValueError(f"{path}: rotated or column-ordered Geosoft grids are not supported")
    dtype = GEOSOFT_FLOAT_TYPES[(esize, sf)]
    n = ne * nv
    if not compressed:
        a = np.frombuffer(b[512:512 + n * esize], dtype)
    else:
        _magic, _ctype, nb, _vpb = struct.unpack("<4i", b[512:528])
        offs = struct.unpack(f"<{nb}q", b[528:528 + 8 * nb])
        sizes = struct.unpack(f"<{nb}i", b[528 + 8 * nb:528 + 12 * nb])
        # Each compressed block carries a 16-byte block header before the zlib stream.
        a = np.concatenate([np.frombuffer(zlib.decompress(b[o + 16:o + sz]), dtype) for o, sz in zip(offs, sizes)])
    if a.size < n:
        raise ValueError(f"{path}: grid data truncated ({a.size} of {n} values)")
    a = a[:n].reshape(nv, ne).astype(np.float64)
    a[~np.isfinite(a) | (a < -1e30)] = np.nan   # Geosoft dummy -1e32
    a = a / (zmult if zmult else 1.0) + zbase
    north_up = a[::-1].astype(np.float32)        # first stored row is the southernmost
    top = y0 + (nv - 1) * dv + dv / 2.0           # x0, y0 are the centre of the lower-left cell
    return north_up, from_origin(x0 - de / 2.0, top, de, dv)


def read_xyz(path: Path, value_col: int = 2, dummy: Optional[float] = None) -> np.ndarray:
    rows = []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if not NUM.match(line):
                continue  # header / comment lines
            parts = re.split(r"[\s,;]+", line.strip())
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            if len(vals) > value_col:
                rows.append((vals[0], vals[1], vals[value_col]))
    a = np.asarray(rows, dtype=np.float64)
    if a.size == 0:
        raise ValueError(f"No numeric X Y VALUE rows found in {path}")
    ok = np.isfinite(a[:, 2])
    if dummy is not None:
        ok &= a[:, 2] != dummy
    ok &= np.abs(a[:, 2]) < 1e30  # Geosoft dummy -1e32 and similar
    return a[ok]


def infer_cell(xy: np.ndarray) -> float:
    """Median spacing between sorted unique X values (gridded ASCII exports)."""
    ux = np.unique(np.round(xy[:, 0], 3))
    d = np.diff(ux)
    d = d[d > 0]
    if d.size == 0:
        raise ValueError("Cannot infer cell size; pass --cell")
    return float(np.median(d))


def grid_points(a: np.ndarray, cell: float) -> Tuple[np.ndarray, object]:
    """Cell-average X Y VALUE points onto a north-up grid whose cell centres match regular inputs."""
    x0, x1 = a[:, 0].min() - cell / 2, a[:, 0].max() + cell / 2
    y0, y1 = a[:, 1].min() - cell / 2, a[:, 1].max() + cell / 2
    w, h = int(np.ceil((x1 - x0) / cell)), int(np.ceil((y1 - y0) / cell))
    c = np.clip(((a[:, 0] - x0) / cell).astype(int), 0, w - 1)
    r = np.clip(((y1 - a[:, 1]) / cell).astype(int), 0, h - 1)
    s = np.zeros((h, w)); n = np.zeros((h, w))
    np.add.at(s, (r, c), a[:, 2]); np.add.at(n, (r, c), 1)
    out = np.where(n > 0, s / np.maximum(n, 1), np.nan).astype(np.float32)
    return out, from_origin(x0, y1, cell, cell)


def to_project_crs(arr: np.ndarray, transform, src_crs: str, dst_crs: str, cell: float, out: Path) -> Path:
    h, w = arr.shape
    if rasterio.crs.CRS.from_user_input(src_crs) == rasterio.crs.CRS.from_user_input(dst_crs):
        # Same CRS: keep the native grid exactly (no resampling).
        with rasterio.open(out, "w", driver="GTiff", width=w, height=h, count=1, dtype="float32", crs=dst_crs,
                           transform=transform, nodata=np.nan, compress="deflate") as d:
            d.write(arr.astype(np.float32), 1)
        return out
    left, top = transform.c, transform.f
    bounds = (left, top - h * transform.a, left + w * transform.a, top)
    tr, dw, dh = calculate_default_transform(src_crs, dst_crs, w, h, *bounds, resolution=cell)
    dst = np.full((dh, dw), np.nan, dtype=np.float32)
    reproject(arr, dst, src_transform=transform, src_crs=src_crs, src_nodata=np.nan, dst_transform=tr,
              dst_crs=dst_crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
    with rasterio.open(out, "w", driver="GTiff", width=dw, height=dh, count=1, dtype="float32", crs=dst_crs,
                       transform=tr, nodata=np.nan, compress="deflate") as d:
        d.write(dst, 1)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Mosaic NGSA airborne grids into a GeoSentry input layer")
    ap.add_argument("inputs", nargs="+", help="grid or XYZ files (wildcards allowed)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--layer", default="mag", help="config layer key to write (mag or th)")
    ap.add_argument("--src-crs", default=None, help="CRS of XYZ inputs (required for XYZ)")
    ap.add_argument("--cell", type=float, default=None, help="cell size in source units (default: inferred)")
    ap.add_argument("--value-col", type=int, default=2)
    ap.add_argument("--dummy", type=float, default=None, help="no-data value in the source")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    dst_crs = cfg["project"]["crs"]
    out_path = cfg.resolve(cfg["layers"][args.layer]["path"])
    tmp = out_path.parent / f"_{args.layer}_parts"
    tmp.mkdir(parents=True, exist_ok=True)
    files = sorted({f for pat in args.inputs for f in glob.glob(pat)})
    if not files:
        raise SystemExit("No input files matched")
    parts: List[Path] = []
    cells = []
    for f in map(Path, files):
        part = tmp / (f.stem + ".tif")
        if is_geosoft_grd(f):
            meta_crs = geosoft_epsg(f)
            if meta_crs and args.src_crs and rasterio.crs.CRS.from_user_input(meta_crs) != rasterio.crs.CRS.from_user_input(args.src_crs):
                raise SystemExit(f"{f}: metadata says {meta_crs} but --src-crs is {args.src_crs}")
            src_crs = meta_crs or args.src_crs
            if not src_crs:
                raise SystemExit(f"{f}: no CRS in the .grd.xml metadata; pass --src-crs")
            arr, tr = read_geosoft_grd(f)
            cell = abs(tr.a)
            to_project_crs(arr, tr, src_crs, dst_crs, cell, part)
            LOG.info("  %s -> %s (Geosoft, %s, cell %.1f, %.0f%% valid)", f.name, part.name, src_crs, cell,
                     100 * np.isfinite(arr).mean())
            parts.append(part)
            cells.append(cell)
            continue
        try:
            with rasterio.open(f) as src:  # GDAL-readable grid
                if src.crs is None and not args.src_crs:
                    raise SystemExit(f"{f} has no CRS; pass --src-crs")
                arr = src.read(1, masked=True).astype(np.float32).filled(np.nan)
                arr[np.abs(arr) > 1e30] = np.nan
                cell = abs(src.transform.a)
                to_project_crs(arr, src.transform, str(src.crs or args.src_crs), dst_crs, cell, part)
        except rasterio.errors.RasterioIOError:
            if not args.src_crs:
                raise SystemExit(f"{f} is ASCII XYZ; --src-crs is required")
            a = read_xyz(f, args.value_col, args.dummy)
            cell = args.cell or infer_cell(a)
            arr, tr = grid_points(a, cell)
            to_project_crs(arr, tr, args.src_crs, dst_crs, cell, part)
        LOG.info("  %s -> %s (cell %.1f)", f.name, part.name, cell)
        parts.append(part)
        cells.append(cell)

    srcs = [rasterio.open(p) for p in parts]
    mosaic, tr = merge(srcs, nodata=np.nan, method="first")
    prof = srcs[0].profile.copy()
    for s in srcs:
        s.close()
    prof.update(width=mosaic.shape[2], height=mosaic.shape[1], transform=tr, compress="deflate",
                tiled=True, blockxsize=256, blockysize=256)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **prof) as d:
        d.write(mosaic[0], 1)
    out_path.with_suffix(".provenance.json").write_text(json.dumps(
        {"inputs": files, "src_crs_argument": args.src_crs, "dst_crs": dst_crs, "cells": cells,
         "method": "Geosoft/GDAL/XYZ read; reprojection (bilinear) only when the source CRS differs; "
                   "mosaic first-valid; XYZ inputs cell-averaged",
         "caution": "Verify source CRS and inspect sheet seams for levelling offsets."}, indent=2))
    LOG.info("Wrote %s (%d x %d)", out_path, mosaic.shape[2], mosaic.shape[1])


if __name__ == "__main__":
    main()
