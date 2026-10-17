"""Descriptive sensitivity sweep; no independence or mineral claims."""
import sys
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import src
import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

with rasterio.open(ROOT/'data/udegi/processed/combined_test_diagnostics.tif') as s:
    a=s.read(masked=True).filled(np.nan)
    res=s.res
valid=np.isfinite(a).all(0)
names=['B11/B12','Magnetic source','Equivalent thorium','Equivalent uranium','Potassium']
results=[]
for threshold in [.4,.5,.6]:
    seed=valid & (a[0]>=threshold)
    if not seed.any():
        continue
    d=distance_transform_edt(~seed,sampling=(res[1],res[0]))
    for near_distance,away_distance in [(50,200),(100,300),(150,500)]:
        near=valid & (d<=near_distance)
        away=valid & (d>=away_distance)
        if not near.any() or not away.any():
            continue
        differences={name:float(np.median(a[i][near])-np.median(a[i][away]))
                     for i,name in enumerate(names,1)}
        results.append(dict(ndvi_threshold=threshold,near_m=near_distance,away_m=away_distance,
                            near_pixels=int(near.sum()),away_pixels=int(away.sum()),
                            median_near_minus_away=differences))
summary={}
for name in names:
    v=np.array([r['median_near_minus_away'][name] for r in results])
    summary[name]=dict(settings=len(v),positive=int((v>0).sum()),negative=int((v<0).sum()),
                       minimum=float(v.min()),maximum=float(v.max()))
    print(f'{name}: positive {int((v>0).sum())}/{len(v)}, difference range {v.min():.3f} to {v.max():.3f}')
out=ROOT/'outputs/udegi/reconnaissance/surface_cover_sensitivity.json'
out.write_text(json.dumps(dict(summary=summary,results=results,
    limitations='Overlapping settings on the same area are not independent replicates. Vegetation is not a river/alluvium classification; persistent association does not establish causation or mineralization.'),indent=2))
print(out)
