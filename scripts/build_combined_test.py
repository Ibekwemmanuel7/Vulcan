"""Exploratory maps of the covered Udegi test window; no mineral predictions."""
import sys
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src
import numpy as np
import rasterio
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

paths = [ROOT / 'data/udegi/processed/sentinel_tessera_test_6band.tif',
         ROOT / 'data/udegi/tessera/2025_test/tessera_2025_utm32.tif',
         ROOT / 'data/udegi/processed/geophysics_test_4band.tif']
arrays = []
with rasterio.open(paths[0]) as ref:
    profile = ref.profile.copy()
    shape, crs, transform = ref.shape, ref.crs, ref.transform
for path in paths:
    with rasterio.open(path) as s:
        if s.shape != shape or s.crs != crs or s.transform != transform:
            raise ValueError(f'Grid mismatch: {path}')
        arrays.append(s.read(masked=True).astype('float32').filled(np.nan))
optical, embedding, geo = arrays
assert optical.shape[0] == 6 and embedding.shape[0] == 128 and geo.shape[0] == 4
valid = np.isfinite(optical).all(0) & np.isfinite(embedding).all(0) & np.isfinite(geo).all(0)
if not valid.any():
    raise ValueError('No common valid pixels')
blue, green, red, nir, swir1, swir2 = optical
ndvi = np.divide(nir-red,nir+red,out=np.full(shape,np.nan,dtype='float32'),where=np.abs(nir+red)>1e-6)
ratio = np.divide(swir1,swir2,out=np.full(shape,np.nan,dtype='float32'),where=swir2>1e-6)
def stretch(a):
    low, high = np.nanpercentile(a,[2,98],axis=(0,1))
    return np.clip((a-low)/np.maximum(high-low,1e-6),0,1)
rgb = stretch(np.stack([red,green,blue],axis=-1))
x = embedding[:,valid].T
rng = np.random.default_rng(42)
sample = rng.choice(len(x),min(50000,len(x)),replace=False)
pca = PCA(n_components=3,svd_solver='randomized',random_state=42).fit(x[sample])
pcs = np.full((*shape,3),np.nan,dtype='float32')
pcs[valid] = pca.transform(x)
fig, axes = plt.subplots(2,4,figsize=(17,9))
axes = axes.ravel()
axes[0].imshow(rgb); axes[0].set_title('Sentinel true colour')
axes[1].imshow(stretch(pcs)); axes[1].set_title('TESSERA PCA colours (exploratory)')
panels = [(ndvi,'NDVI','RdYlGn'),(ratio,'B11/B12 spectral proxy','viridis')]
panels += [(geo[i],name,'viridis') for i,name in enumerate(['Magnetic source grid','Equivalent thorium','Equivalent uranium','Potassium'])]
for ax,(a,title,cmap) in zip(axes[2:],panels):
    lo,hi = np.nanpercentile(a,[2,98])
    im=ax.imshow(a,cmap=cmap,vmin=lo,vmax=hi)
    ax.set_title(title)
    fig.colorbar(im,ax=ax,shrink=.7)
for ax in axes:
    ax.axis('off')
fig.suptitle('Udegi covered test area — reconnaissance only; no verified mineral labels',fontsize=14)
fig.tight_layout()
out=ROOT/'outputs/udegi/reconnaissance'
out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'combined_test_preview.png',dpi=160)
plt.close(fig)
diagnostics=np.stack([ndvi,ratio,*geo])
diagnostics[:,~valid]=np.nan
profile.update(driver='GTiff',count=6,dtype='float32',nodata=np.nan,compress='deflate')
target=ROOT/'data/udegi/processed/combined_test_diagnostics.tif'
with rasterio.open(target,'w',**profile) as dst:
    dst.write(diagnostics)
    for i,name in enumerate(['NDVI','B11_B12','TMI','eTHORIUM','eURANIUM','POTASSIUM'],1):
        dst.set_band_description(i,name)
(out/'combined_test_metadata.json').write_text(json.dumps(dict(
    inputs=[str(p) for p in paths],common_valid_fraction=float(valid.mean()),
    crs=str(crs),shape=list(shape),pca_explained_variance=pca.explained_variance_ratio_.tolist(),
    limitations='No occurrence labels or mineral scores. Geophysics native 30/35 m resampled to 10 m. Units/processing require source metadata review. Annual embeddings and seasonal optical composites differ in temporal support.'),indent=2))
print(f'Common valid coverage: {valid.mean():.1%}')
print(out/'combined_test_preview.png')
print(target)
