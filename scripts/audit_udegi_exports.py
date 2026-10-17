"""Compare independent Viewer exports without changing source coordinates."""
import json
import sys
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src
import numpy as np
import rasterio

folder = ROOT / 'data/udegi/converted'
results = []
for layer, suffix in [('TMI', 'viewer'), ('eTHORIUM', 'verified'),
                      ('eURANIUM', 'verified'), ('POTASSIUM', 'verified')]:
    native = folder / f'229_{layer}.tif'
    viewer = folder / f'229_{layer}_{suffix}.tif'
    with rasterio.open(native) as a, rasterio.open(viewer) as b:
        if a.shape != b.shape or a.crs != b.crs or a.res != b.res:
            raise ValueError(f'Incompatible grids: {layer}')
        x = a.read(1, masked=True).filled(np.nan)
        y = b.read(1, masked=True).filled(np.nan)
        mask_equal = np.array_equal(np.isfinite(x), np.isfinite(y))
        valid = np.isfinite(x) & np.isfinite(y)
        if not valid.any():
            raise ValueError(f'No shared valid pixels: {layer}')
        error = float(np.max(np.abs(x[valid] - y[valid])))
        source = next((ROOT / 'data/pilot_nk/ngsa/Sheet_229_Udegi').rglob(f'229_{layer}.grd'))
        header = struct.unpack('<5i7d', source.read_bytes()[:76])
        nx, ny = header[2:4]
        dx, dy, x0, y0, rotation = header[5:10]
        if rotation != 0 or header[4] != 1:
            raise ValueError('Unsupported source orientation')
        first = a.transform * (0.5, ny - 0.5)
        last = a.transform * (nx - 0.5, 0.5)
        np.testing.assert_allclose(first, (x0, y0), rtol=0, atol=1e-8)
        np.testing.assert_allclose(last, (x0+(nx-1)*dx, y0+(ny-1)*dy), rtol=0, atol=1e-8)
        if not mask_equal or error != 0:
            raise ValueError(f'Values differ: {layer}')
        item = dict(layer=layer, python_file=str(native), viewer_file=str(viewer),
                    crs=str(b.crs), resolution=list(b.res),
                    masks_equal=mask_equal, max_value_difference=error,
                    viewer_offset_east_m=b.transform.c-a.transform.c,
                    viewer_offset_north_m=b.transform.f-a.transform.f,
                    viewer_transform=list(b.transform),
                    python_transform=list(a.transform))
        item['python_pixel_centres_match_source_header'] = True
        results.append(item)
        print(f'{layer}: masks equal={mask_equal}, max difference={error}, '
              f'offset E/N={item["viewer_offset_east_m"]}/{item["viewer_offset_north_m"]} m')
report = folder / 'viewer_comparison_audit.json'
report.write_text(json.dumps(dict(
    status='Values and masks verified; Python pixel centres match source grid points',
    policy='Use original Python TIFFs as working inputs; preserve Viewer exports as independent value checks.',
    origin_reference='https://github.com/GeosoftInc/gxpy/blob/master/geosoft/gxpy/grid.py',
    rationale='Official xy_from_index maps index zero to x0,y0; extent_2d extends half a cell beyond grid points. Python TIFF centres reproduce those points. Viewer exports read by Rasterio have a half-cell E/S offset.',
    results=results), indent=2))
print(report)
