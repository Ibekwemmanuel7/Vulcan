"""Descriptive comparison of selected review cells, not target ranking."""
import sys
import json
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import src
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=ROOT/'outputs/udegi/reconnaissance'
cells=json.loads((out/'spatial_review_cells.geojson').read_text())['features']
selected=['R20','R21','R22','R23']
rows=[]
with rasterio.open(ROOT/'data/udegi/processed/away_vegetation_diagnostics.tif') as s:
    a=s.read(masked=True).filled(np.nan)
    for label in selected:
        feature=next(f for f in cells if f['properties']['review_id']==label)
        geom=transform_geom('EPSG:4326',s.crs,feature['geometry'])
        cell=geometry_mask([geom],out_shape=s.shape,transform=s.transform,invert=True)
        ok=cell&np.isfinite(a).all(0)
        if not ok.any():
            raise ValueError(f'No retained pixels: {label}')
        q=np.percentile(a[:,ok],[25,50,75],axis=1)
        rows.append(dict(cell=label,retained_pixels=int(ok.sum()),retained_fraction=float(ok.sum()/cell.sum()),
                         q25=q[0].tolist(),median=q[1].tolist(),q75=q[2].tolist()))
names=['NDVI','B11/B12','Magnetic source','Equivalent thorium','Equivalent uranium','Potassium']
fig,axes=plt.subplots(2,3,figsize=(13,8))
for i,(ax,name) in enumerate(zip(axes.flat,names)):
    med=np.array([r['median'][i] for r in rows])
    lower=np.array([r['q25'][i] for r in rows])
    upper=np.array([r['q75'][i] for r in rows])
    ax.errorbar(selected,med,yerr=[med-lower,upper-med],fmt='o',capsize=5)
    ax.set_title(name)
    ax.set_ylabel('Median and interquartile range')
fig.suptitle('Retained pixels in selected cells — spatial comparison, not independent replicates')
fig.tight_layout()
image=out/f'cell_comparison_{datetime.now():%Y%m%d_%H%M%S_%f}.png'
fig.savefig(image,dpi=160)
plt.close(fig)
report=dict(preview=str(image),band_order=names,cells=rows,
            limitations='Cells selected after inspecting potassium. Unequal retained areas and spatial dependence; neighbours are comparison cells, not verified negative controls. No mineral inference.')
(out/'cell_comparison.json').write_text(json.dumps(report,indent=2))
for r in rows:
    print(r['cell'],f'{r["retained_fraction"]:.1%} retained',dict(zip(names,np.round(r['median'],3))))
print(image)
