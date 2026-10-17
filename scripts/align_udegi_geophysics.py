"""Align audited native geophysics to the existing candidate embedding grid."""
import json
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--area', choices=['candidate', 'test'], default='candidate')
args = parser.parse_args()
area = '2025_test' if args.area == 'test' else '2025_lithium_locality_candidate'
reference = ROOT / f'data/udegi/tessera/{area}/tessera_2025_utm32.tif'
stem = 'geophysics_test' if args.area == 'test' else 'geophysics_lithium_candidate'
output = ROOT / f'data/udegi/processed/{stem}_4band.tif'
report = ROOT / ('outputs/udegi/reconnaissance/geophysics_test_alignment.json' if args.area == 'test' else 'outputs/udegi/reconnaissance/geophysics_alignment.json')
if output.exists():
    raise SystemExit(f'Already exists: {output}; preserve existing output.')
layers = ['TMI', 'eTHORIUM', 'eURANIUM', 'POTASSIUM']
audit = json.loads((ROOT / 'data/udegi/converted/viewer_comparison_audit.json').read_text())
if len(audit['results']) != 4 or not all(
    r['masks_equal'] and r['max_value_difference'] == 0 and
    r.get('python_pixel_centres_match_source_header') for r in audit['results']
):
    raise SystemExit('Run the updated audit successfully before alignment.')
output.parent.mkdir(parents=True, exist_ok=True)
report.parent.mkdir(parents=True, exist_ok=True)
stats = []
with rasterio.open(reference) as ref:
    profile = dict(driver='GTiff', width=ref.width, height=ref.height,
                   count=4, dtype='float32', crs=ref.crs, transform=ref.transform,
                   nodata=np.nan, compress='deflate', tiled=True,
                   blockxsize=256, blockysize=256)
    common = np.ones(ref.shape, dtype=bool)
    with rasterio.open(output, 'w', **profile) as dst:
        for index, layer in enumerate(layers, 1):
            path = ROOT / f'data/udegi/converted/229_{layer}.tif'
            with rasterio.open(path) as s:
                a = np.full(ref.shape, np.nan, dtype='float32')
                reproject(source=rasterio.band(s, 1), destination=a,
                          src_transform=s.transform, src_crs=s.crs,
                          src_nodata=s.nodata, dst_transform=ref.transform,
                          dst_crs=ref.crs, dst_nodata=np.nan,
                          resampling=Resampling.bilinear)
                # Conservative mask: retain only pixels whose interpolated
                # source validity is entirely valid, avoiding nodata edges.
                coverage = np.zeros(ref.shape, dtype='float32')
                reproject(source=s.read_masks(1).astype('float32') / 255,
                          destination=coverage, src_transform=s.transform,
                          src_crs=s.crs, dst_transform=ref.transform,
                          dst_crs=ref.crs, dst_nodata=0,
                          resampling=Resampling.bilinear)
                a[coverage < 0.999999] = np.nan
                valid = np.isfinite(a)
                common &= valid
                dst.write(a, index)
                dst.set_band_description(index, layer)
                item = dict(layer=layer, source=str(path),
                            native_resolution=list(s.res),
                            valid_fraction=float(valid.mean()))
                stats.append(item)
                print(f'{layer}: {valid.mean():.1%} candidate coverage')
        dst.update_tags(method='bilinear with conservative source validity mask',
                        caution='10 m output spacing; native information remains 30/35 m. No mineral scores.')
    metadata = dict(reference=str(reference), output=str(output),
                    crs=str(ref.crs), shape=list(ref.shape),
                    transform=list(ref.transform), layers=stats,
                    common_geophysics_valid_fraction=float(common.mean()),
                    caution='Resampling does not improve native spatial resolution. No extrapolation across survey gaps.')
    if any(item['valid_fraction'] == 0 for item in stats):
        metadata['coverage_status'] = 'Candidate lacks at least one geophysical layer; choose another area before combined analysis.'
    else:
        metadata['coverage_status'] = 'Inspect reported coverage before combined analysis.'
report.write_text(json.dumps(metadata, indent=2))
print(f'Common coverage: {common.mean():.1%}')
print(output)
print(report)
with rasterio.open(output) as result, rasterio.open(reference) as ref:
    assert result.crs == ref.crs and result.transform == ref.transform and result.shape == ref.shape
