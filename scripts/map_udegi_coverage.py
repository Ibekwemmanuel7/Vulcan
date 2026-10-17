"""Map observed survey coverage and check existing embedding windows."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

out = ROOT / 'outputs/udegi/reconnaissance'
out.mkdir(parents=True, exist_ok=True)
layers = ['TMI', 'eTHORIUM', 'eURANIUM', 'POTASSIUM']
refs = sorted((ROOT / 'data/udegi/tessera').rglob('tessera_2025_utm32.tif'))
results = []
fig, ax = plt.subplots(figsize=(9, 8))
with rasterio.open(ROOT / 'data/udegi/converted/229_TMI.tif') as s:
    valid = s.read_masks(1) > 0
    ax.imshow(valid, extent=[s.bounds.left,s.bounds.right,s.bounds.bottom,s.bounds.top],
              origin='upper', cmap='Greens', vmin=0, vmax=1, interpolation='nearest')
for i, path in enumerate(refs):
    with rasterio.open(path) as ref:
        common = np.ones(ref.shape, dtype=bool)
        fractions = {}
        for layer in layers:
            with rasterio.open(ROOT / f'data/udegi/converted/229_{layer}.tif') as s:
                coverage = np.zeros(ref.shape, dtype='float32')
                reproject(s.read_masks(1).astype('float32')/255, coverage,
                          src_transform=s.transform, src_crs=s.crs,
                          dst_transform=ref.transform, dst_crs=ref.crs,
                          dst_nodata=0, resampling=Resampling.bilinear)
                ok = coverage >= .999999
                fractions[layer] = float(ok.mean())
                common &= ok
        item = dict(area=path.parent.name, reference=str(path),
                    geophysics_coverage=fractions, common_coverage=float(common.mean()))
        results.append(item)
        print(f'{item["area"]}: common coverage {common.mean():.1%}')
        if ref.crs.to_epsg() == 32632:
            b = ref.bounds
            colour = ['tab:blue','tab:red','tab:orange'][i % 3]
            ax.add_patch(Rectangle((b.left,b.bottom),b.right-b.left,b.top-b.bottom,
                                  fill=False,edgecolor=colour,linewidth=2,label=path.parent.name))
ax.set(title='Udegi magnetic survey footprint and downloaded TESSERA areas',
       xlabel='Easting (m), EPSG:32632', ylabel='Northing (m)')
ax.legend(loc='upper left', fontsize=8)
fig.tight_layout()
fig.savefig(out / 'survey_coverage.png', dpi=160)
plt.close(fig)
(out / 'existing_area_coverage.json').write_text(json.dumps(results, indent=2))
print(out / 'survey_coverage.png')
