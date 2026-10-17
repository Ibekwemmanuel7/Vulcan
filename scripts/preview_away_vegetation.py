"""Visualize a vegetation-distance subset without ranking mineral targets."""
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

with rasterio.open(ROOT/'data/udegi/processed/combined_test_diagnostics.tif') as s:
    a=s.read(masked=True).filled(np.nan)
    profile=s.profile.copy()
    res=s.res
valid=np.isfinite(a).all(0)
seed=valid & (a[0]>=.5)
if not seed.any():
    raise SystemExit('No vegetation seeds; cannot define distance subset.')
d=distance_transform_edt(~seed,sampling=(res[1],res[0]))
keep=valid & (d>=300)
if not keep.any():
    raise SystemExit('No pixels retained.')
masked=a.copy()
masked[:,~keep]=np.nan
out=ROOT/'outputs/udegi/reconnaissance'
out.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(2,3,figsize=(14,8))
names=['NDVI','B11/B12 spectral proxy','Magnetic source','Equivalent thorium','Equivalent uranium','Potassium']
for i,(ax,name) in enumerate(zip(axes.flat,names)):
    # Full-window scaling permits comparison with unmasked patterns.
    lo,hi=np.nanpercentile(a[i],[2,98])
    im=ax.imshow(masked[i],cmap='viridis',vmin=lo,vmax=hi)
    ax.set_title(name)
    ax.axis('off')
    fig.colorbar(im,ax=ax,shrink=.7)
fig.suptitle('Pixels >=300 m from NDVI >=0.5 vegetation — exploratory subset')
fig.tight_layout()
image=out/f'away_vegetation_{datetime.now():%Y%m%d_%H%M%S_%f}.png'
fig.savefig(image,dpi=160)
plt.close(fig)
target=ROOT/'data/udegi/processed/away_vegetation_diagnostics.tif'
with rasterio.open(target,'w',**profile) as dst:
    dst.write(masked)
    for i,name in enumerate(names,1):
        dst.set_band_description(i,name)
report=dict(preview=str(image),output=str(target),retained_pixels=int(keep.sum()),
            retained_fraction=float(keep.mean()),ndvi_threshold=.5,distance_m=300,
            limitation='A vegetation-distance subset, not a bedrock exposure mask, river mask, mineral score or field target selection.')
(out/'away_vegetation_metadata.json').write_text(json.dumps(report,indent=2))
print(f'Retained: {keep.sum()} pixels ({keep.mean():.1%} of window)')
print(image)
