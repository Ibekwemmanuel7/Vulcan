"""Terrain context for selected cells; slope computed at native DEM spacing."""
import sys
import json
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import src
import numpy as np
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.features import geometry_mask
from rasterio.warp import reproject, transform_geom, Resampling, transform_bounds
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

out=ROOT/'outputs/udegi/reconnaissance'
features=json.loads((out/'spatial_review_cells.geojson').read_text())['features']
labels=['R20','R21','R22','R23']
features=[next(f for f in features if f['properties']['review_id']==label) for label in labels]
with rasterio.open(ROOT/'data/udegi/processed/sentinel_tessera_test_6band.tif') as ref:
    rgb=ref.read([3,2,1],masked=True).filled(np.nan).transpose(1,2,0)
    shape,tr,crs,b=ref.shape,ref.transform,ref.crs,ref.bounds
with rasterio.open(ROOT/'data/udegi/raw/DEM.tif') as dem:
    if not dem.crs.is_projected:
        raise ValueError('Projected DEM in metres required for native slope calculation')
    if dem.crs.linear_units != 'metre':
        raise ValueError('DEM horizontal units must be metres')
    db=transform_bounds(crs,dem.crs,*b)
    w=from_bounds(*db,transform=dem.transform)
    col,row=int(np.floor(w.col_off))-3,int(np.floor(w.row_off))-3
    right,bottom=int(np.ceil(w.col_off+w.width))+3,int(np.ceil(w.row_off+w.height))+3
    w=Window(col,row,right-col,bottom-row).intersection(Window(0,0,dem.width,dem.height))
    z=dem.read(1,window=w,masked=True).astype('float32').filled(np.nan)
    dt=dem.window_transform(w)
    gy,gx=np.gradient(z,abs(dt.e),abs(dt.a))
    slope=np.degrees(np.arctan(np.hypot(gx,gy))).astype('float32')
    aligned=[]
    for source in [z,slope]:
        dest=np.full(shape,np.nan,dtype='float32')
        reproject(source,dest,src_transform=dt,src_crs=dem.crs,src_nodata=np.nan,
                  dst_transform=tr,dst_crs=crs,dst_nodata=np.nan,resampling=Resampling.bilinear)
        aligned.append(dest)
    native_res=list(dem.res)
    records=[]
    with rasterio.open(ROOT/'data/udegi/processed/away_vegetation_diagnostics.tif') as masksrc:
        assert masksrc.crs==crs and masksrc.transform==tr and masksrc.shape==shape
        keep=np.isfinite(masksrc.read(1,masked=True).filled(np.nan))
    for f in features:
        geom=transform_geom('EPSG:4326',dem.crs,f['geometry'])
        cell=geometry_mask([geom],out_shape=z.shape,transform=dt,invert=True)
        item=dict(cell=f['properties']['review_id'])
        for data,name in [(z,'elevation_m'),(slope,'slope_degrees')]:
            ok=cell&np.isfinite(data)
            item[name+'_q25_median_q75']=np.percentile(data[ok],[25,50,75]).tolist()
        records.append(item)
        print(item['cell'],f'elevation median={item["elevation_m_q25_median_q75"][1]:.1f} m, slope median={item["slope_degrees_q25_median_q75"][1]:.2f} degrees')
lo,hi=np.nanpercentile(rgb,[2,98],axis=(0,1))
rgb=np.clip((rgb-lo)/np.maximum(hi-lo,1e-6),0,1)
fig,axes=plt.subplots(1,3,figsize=(17,6))
extent=[b.left,b.right,b.bottom,b.top]
axes[0].imshow(rgb,extent=extent)
for ax,data,title in zip(axes[1:],aligned,['Elevation (m)','Slope (degrees; computed at native DEM spacing)']):
    im=ax.imshow(data,extent=extent,cmap='terrain' if ax==axes[1] else 'magma')
    fig.colorbar(im,ax=ax,shrink=.7)
    ax.set_title(title)
axes[0].set_title('Sentinel surface context')
for ax in axes:
    for f in features:
        geom=transform_geom('EPSG:4326',crs,f['geometry'])
        coords=np.array(geom['coordinates'][0])
        left,bottom=coords.min(0); right,top=coords.max(0)
        ax.add_patch(Rectangle((left,bottom),right-left,top-bottom,fill=False,edgecolor='red',linewidth=1))
        ax.text((left+right)/2,(top+bottom)/2,f['properties']['review_id'],ha='center',bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
    ax.set(xlabel='Easting (m)',ylabel='Northing (m)')
    ax.ticklabel_format(style='plain',useOffset=False)
fig.suptitle('Terrain context — whole-cell summaries; surface classes remain unverified')
fig.tight_layout()
image=out/f'cell_terrain_{datetime.now():%Y%m%d_%H%M%S_%f}.png'
fig.savefig(image,dpi=160)
plt.close(fig)
(out/'cell_terrain.json').write_text(json.dumps(dict(preview=str(image),native_dem_spacing=native_res,
    statistics_support='Native DEM pixel centres within entire review cells, not vegetation-filtered subsets',cells=records,
    limitations='DEM vertical units assumed metres; terrain alone cannot establish bedrock exposure, sediment type or mineralization. Display interpolation does not increase DEM information.'),indent=2))
print(image)
