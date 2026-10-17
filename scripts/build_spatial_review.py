"""Create neutral 1 km review cells and descriptive summaries, without targets."""
import sys
import json
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import src
import numpy as np
import rasterio
from rasterio.warp import transform as project
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

out=ROOT/'outputs/udegi/reconnaissance'
with rasterio.open(ROOT/'data/udegi/processed/combined_test_diagnostics.tif') as s:
    a=s.read(masked=True).filled(np.nan)
    tr,crs,bounds=s.transform,s.crs,s.bounds
    shape=s.shape
with rasterio.open(ROOT/'data/udegi/processed/away_vegetation_diagnostics.tif') as s:
    assert s.shape==shape and s.crs==crs and s.transform==tr
    keep=np.isfinite(s.read(1,masked=True).filled(np.nan))
with rasterio.open(ROOT/'data/udegi/processed/sentinel_tessera_test_6band.tif') as s:
    assert s.shape==shape and s.crs==crs and s.transform==tr
    rgb=s.read([3,2,1],masked=True).filled(np.nan).transpose(1,2,0)
lo,hi=np.nanpercentile(rgb,[2,98],axis=(0,1))
rgb=np.clip((rgb-lo)/np.maximum(hi-lo,1e-6),0,1)
names=['NDVI','B11_B12','TMI','eTHORIUM','eURANIUM','POTASSIUM']
fig,axes=plt.subplots(1,2,figsize=(14,7))
extent=[bounds.left,bounds.right,bounds.bottom,bounds.top]
axes[0].imshow(rgb,extent=extent)
potassium=np.where(keep,a[5],np.nan)
low,high=np.nanpercentile(a[5],[2,98])
im=axes[1].imshow(potassium,extent=extent,cmap='viridis',vmin=low,vmax=high)
fig.colorbar(im,ax=axes[1],shrink=.7,label='Potassium source values')
features=[]
# Use global UTM kilometre boundaries; edge cells are clipped to the window.
for north in range(int(np.floor(bounds.bottom/1000)*1000),int(bounds.top),1000):
    for east in range(int(np.floor(bounds.left/1000)*1000),int(bounds.right),1000):
        left,right=max(east,bounds.left),min(east+1000,bounds.right)
        bottom,top=max(north,bounds.bottom),min(north+1000,bounds.top)
        cols=tr.c+(np.arange(shape[1])+.5)*tr.a
        rows=tr.f+(np.arange(shape[0])+.5)*tr.e
        cell=(rows[:,None]>=bottom)&(rows[:,None]<top)&(cols[None,:]>=left)&(cols[None,:]<right)
        retained=cell&keep
        if not retained.any():
            continue
        label=f'R{len(features)+1:02d}'
        xs=[left,right,right,left,left]; ys=[bottom,bottom,top,top,bottom]
        lon,lat=project(crs,'EPSG:4326',xs,ys)
        medians={name:float(np.nanmedian(a[i][retained])) for i,name in enumerate(names)}
        properties=dict(review_id=label,window_pixels=int(cell.sum()),retained_pixels=int(retained.sum()),
            retained_fraction=float(retained.sum()/cell.sum()),medians=medians,
            review_status='unreviewed',purpose='Neutral review cell; not a mineral occurrence or ranked target')
        features.append(dict(type='Feature',properties=properties,
            geometry=dict(type='Polygon',coordinates=[list(map(list,zip(lon,lat)))])))
        for ax in axes:
            ax.add_patch(Rectangle((left,bottom),right-left,top-bottom,fill=False,edgecolor='black',linewidth=.6))
            ax.text((left+right)/2,(top+bottom)/2,label,ha='center',va='center',fontsize=8,
                    bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=1))
for ax,title in zip(axes,['Sentinel imagery with review cells','Potassium away from vegetation; same cells']):
    ax.set(title=title,xlabel='Easting (m), EPSG:32632',ylabel='Northing (m)')
    ax.ticklabel_format(style='plain',useOffset=False)
fig.suptitle('Udegi spatial review — cell IDs are geographical, not prospectivity ranks')
fig.tight_layout()
image=out/f'spatial_review_{datetime.now():%Y%m%d_%H%M%S_%f}.png'
fig.savefig(image,dpi=160)
plt.close(fig)
geojson=out/'spatial_review_cells.geojson'
geojson.write_text(json.dumps(dict(type='FeatureCollection',features=features),indent=2))
(out/'spatial_review_metadata.json').write_text(json.dumps(dict(preview=str(image),cells=str(geojson),
    count=len(features),cell_size_m=1000,limitations='Clipped edge cells; statistics use retained pixels only. No verified labels, causal interpretation, accuracy claims, or mineral ranking.'),indent=2))
print(f'Created {len(features)} neutral review cells')
print(image)
print(geojson)
