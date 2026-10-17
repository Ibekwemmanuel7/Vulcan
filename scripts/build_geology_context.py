"""Regional map context only: PDF datum metadata is unsuitable for precise GIS."""
import sys
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import src
import numpy as np
import rasterio
from rasterio.warp import transform
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=ROOT/'outputs/udegi/reconnaissance'
meta=json.loads((out/'regional_geology_coordinates.json').read_text())
points=np.array(meta['geographic_points']).reshape(-1,2)
south,north=points[:,0].min(),points[:,0].max()
west,east=points[:,1].min(),points[:,1].max()
image=Image.open(out/'regional_geology_map.png')
with rasterio.open(ROOT/'data/udegi/tessera/2025_test/tessera_2025_utm32.tif') as ref:
    b=ref.bounds
    xs,ys=transform(ref.crs,'EPSG:4326',[b.left,b.right,b.right,b.left,b.left],
                    [b.top,b.top,b.bottom,b.bottom,b.top])
cx,cy=float(np.mean(xs[:4])),float(np.mean(ys[:4]))
# Crop an approximately 45 km contextual region. Coordinates use the printed
# graticule directly, not an unsupported NAD27-to-WGS84 datum transformation.
left,right=max(west,cx-.2),min(east,cx+.2)
bottom,top=max(south,cy-.2),min(north,cy+.2)
box=(round((left-west)/(east-west)*image.width),round((north-top)/(north-south)*image.height),
     round((right-west)/(east-west)*image.width),round((north-bottom)/(north-south)*image.height))
crop=image.crop(box)
fig,ax=plt.subplots(figsize=(10,8))
ax.imshow(crop,extent=[left,right,bottom,top],origin='upper')
ax.plot(xs,ys,color='red',linewidth=2,label='Test area — approximate placement')
ax.set(xlabel='Longitude (printed map graticule)',ylabel='Latitude (printed map graticule)',
       title='Udegi regional geology context — NGSA 2025, 1:1,000,000')
ax.legend(loc='upper left')
fig.text(.5,.02,'Context only: embedded PDF datum says assumed NAD27; not validated for precise geological boundaries.',
         ha='center',fontsize=8)
fig.tight_layout(rect=[0,.04,1,1])
fig.savefig(out/'geology_test_context.png',dpi=180)
plt.close(fig)
meta.update(test_boundary_lonlat=list(zip(xs,ys)),
    status='Approximate regional context, not a georeferenced modelling layer',
    limitation='Embedded CRS is Assumed Geographic / NAD27, inappropriate for an unverified Nigeria datum. Printed graticule used as context only; no precise datum transform or lithology labels generated.')
(out/'geology_context_metadata.json').write_text(json.dumps(meta,indent=2))
print(out/'geology_test_context.png')
