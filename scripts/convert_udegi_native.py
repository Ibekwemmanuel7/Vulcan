"""Convert Udegi grids without resampling; keep a numerical audit report."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.import_ngsa_grid import read_geosoft_grd, geosoft_epsg, to_project_crs
import numpy as np
import rasterio
from rasterio.warp import transform_bounds

folder = ROOT / 'data/pilot_nk/ngsa/Sheet_229_Udegi'
out = ROOT / 'data/udegi/converted'
out.mkdir(parents=True, exist_ok=True)
report = []
for path in sorted(folder.rglob('*.grd')):
    target = out / (path.stem + '.tif')
    if target.exists():
        raise SystemExit(f'Output exists; preserve it: {target}')
    crs = geosoft_epsg(path)
    if crs != 'EPSG:32632':
        raise ValueError(f'Unexpected source CRS: {crs}')
    a, tr = read_geosoft_grd(path)
    if not np.isfinite(a).any():
        raise ValueError(f'No valid values: {path}')
    to_project_crs(a, tr, crs, crs, abs(tr.a), target)
    with rasterio.open(target) as ds:
        np.testing.assert_equal(ds.read(1), a)
        bounds = transform_bounds(ds.crs, 'EPSG:4326', *ds.bounds)
        item = dict(source=str(path), output=str(target), crs=crs,
                    shape=list(ds.shape), resolution=list(ds.res),
                    geographic_bounds=list(bounds),
                    valid_fraction=float(np.isfinite(a).mean()),
                    percentiles=np.nanpercentile(a, [0, 2, 50, 98, 100]).tolist())
    report.append(item)
    print(json.dumps(item, indent=2))
(out / 'conversion_audit.json').write_text(json.dumps(dict(
    method='Existing project Geosoft decoder; native grid, no resampling',
    verification='GeoTIFF values equal decoded array. Independent Viewer comparison pending.',
    grids=report), indent=2))
