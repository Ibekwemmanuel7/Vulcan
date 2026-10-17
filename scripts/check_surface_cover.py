"""Descriptive surface-cover sensitivity check, not mineral inference."""
import sys
import json
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import src
import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

path=ROOT/'data/udegi/processed/combined_test_diagnostics.tif'
out=ROOT/'outputs/udegi/reconnaissance'
out.mkdir(parents=True, exist_ok=True)
with rasterio.open(path) as s:
    a=s.read(masked=True).filled(np.nan)
    resolution=s.res
ndvi=a[0]
valid=np.isfinite(a).all(0)
# Vegetated pixels are only a proxy. This includes vegetation elsewhere in
# the image and does not identify water, alluvium, or the river uniquely.
seed=valid & (ndvi >= .5)
if not seed.any():
    raise SystemExit('No pixels reach NDVI 0.5; no corridor proxy defined.')
distance=distance_transform_edt(~seed,sampling=(resolution[1],resolution[0]))
near=valid & (distance <= 100)
away=valid & (distance >= 300)
if not near.any() or not away.any():
    raise SystemExit('Insufficient pixels for near/away comparison.')
names=['B11/B12','Magnetic source','Equivalent thorium','Equivalent uranium','Potassium']
report=dict(method='NDVI >= 0.5 vegetation proxy; near <=100 m, away >=300 m; middle zone excluded',
            near_pixels=int(near.sum()),away_pixels=int(away.sum()),
            limitations='Vegetation proxy is not a mapped river/alluvium boundary. Spatially correlated pixels are not independent samples. Descriptive differences do not establish cause or mineralization.',
            layers=[])
fig,axes=plt.subplots(2,3,figsize=(13,8))
classes=np.full(valid.shape,np.nan)
classes[valid]=1
classes[near]=0
classes[away]=2
axes.flat[0].imshow(classes,cmap='viridis',vmin=0,vmax=2)
axes.flat[0].set_title('Groups: purple near / yellow away\nGreen excluded buffer')
axes.flat[0].axis('off')
for i,(name,ax) in enumerate(zip(names,list(axes.flat)[1:]),1):
    data=[a[i][near],a[i][away]]
    quantiles=[np.percentile(x,[25,50,75]).tolist() for x in data]
    report['layers'].append(dict(layer=name,near_q25_median_q75=quantiles[0],
                                away_q25_median_q75=quantiles[1]))
    ax.bxp([dict(label=label,q1=q[0],med=q[1],q3=q[2],whislo=q[0],whishi=q[2],fliers=[])
            for label,q in zip(['Near vegetation','Away'],quantiles)],showfliers=False)
    ax.set_title(name)
    ax.set_ylabel('Native source values / ratio')
    print(f'{name}: near median={quantiles[0][1]:.3f}, away median={quantiles[1][1]:.3f}')
fig.suptitle('Surface-cover sensitivity: descriptive medians and interquartile ranges')
fig.tight_layout()
# Always use a new path so an open image viewer cannot block replacement.
image_path=out/f'surface_cover_comparison_{datetime.now():%Y%m%d_%H%M%S_%f}.png'
fig.savefig(image_path,dpi=160)
plt.close(fig)
report['preview']=str(image_path)
(out/'surface_cover_comparison.json').write_text(json.dumps(report,indent=2))
print(f'Near pixels: {near.sum()}, away pixels: {away.sum()}')
print(image_path)
