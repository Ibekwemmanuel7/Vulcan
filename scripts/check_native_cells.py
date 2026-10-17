"""Compare native geophysical cells using the same vegetation-distance subset."""
import sys
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import src
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import reproject, transform_geom, Resampling
from rasterio.windows import from_bounds, Window

out=ROOT/'outputs/udegi/reconnaissance'
features=json.loads((out/'spatial_review_cells.geojson').read_text())['features']
selected=['R20','R21','R22','R23']
results=[]
with rasterio.open(ROOT/'data/udegi/processed/away_vegetation_diagnostics.tif') as masksrc:
    keep=np.isfinite(masksrc.read(1,masked=True).filled(np.nan)).astype('uint8')
    for layer in ['TMI','eTHORIUM','eURANIUM','POTASSIUM']:
        with rasterio.open(ROOT/f'data/udegi/converted/229_{layer}.tif') as s:
            assert s.crs==masksrc.crs
            w=from_bounds(*masksrc.bounds,transform=s.transform)
            x0,y0=int(np.floor(w.col_off)),int(np.floor(w.row_off))
            x1,y1=int(np.ceil(w.col_off+w.width)),int(np.ceil(w.row_off+w.height))
            w=Window(x0,y0,x1-x0,y1-y0).intersection(Window(0,0,s.width,s.height))
            a=s.read(1,window=w,masked=True).filled(np.nan)
            tr=s.window_transform(w)
            fraction=np.zeros(a.shape,dtype='float32')
            reproject(keep,fraction,src_transform=masksrc.transform,src_crs=masksrc.crs,
                      dst_transform=tr,dst_crs=s.crs,resampling=Resampling.average)
            # Require almost the entire native cell to belong to retained area.
            retained=(fraction>=.99)&np.isfinite(a)
            for label in selected:
                f=next(f for f in features if f['properties']['review_id']==label)
                geom=transform_geom('EPSG:4326',s.crs,f['geometry'])
                cell=geometry_mask([geom],out_shape=a.shape,transform=tr,invert=True)
                values=a[cell&retained]
                if not values.size:
                    raise ValueError(f'No native values: {layer} {label}')
                q=np.percentile(values,[25,50,75])
                results.append(dict(layer=layer,cell=label,native_pixels=int(values.size),
                                    resolution=list(s.res),q25=float(q[0]),median=float(q[1]),q75=float(q[2])))
                print(f'{layer:10s} {label}: median={q[1]:.3f}, native pixels={values.size}')
summary={}
for layer in ['eTHORIUM','eURANIUM','POTASSIUM']:
    v={r['cell']:r['median'] for r in results if r['layer']==layer}
    summary[layer]=dict(R21_and_R22_above_both_neighbours=min(v['R21'],v['R22'])>max(v['R20'],v['R23']))
report=dict(results=results,summary=summary,
    method='Original native values; native pixel centre inside cell, >=99% vegetation-subset overlap. No value resampling.',
    limitations='Native grid pixels are spatially correlated and do not represent independent observations or survey footprint resolution. Selected cells are exploratory, not verified occurrences.')
(out/'native_cell_comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps(summary,indent=2))
