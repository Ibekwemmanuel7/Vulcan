"""Build exploratory candidate-window layers; no mineral prediction or labels."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

ROOT = Path(__file__).resolve().parents[1]


def stretch(a):
    lo, hi = np.nanpercentile(a, [2, 98], axis=(0, 1))
    return np.clip((a - lo) / np.maximum(hi - lo, 1e-6), 0, 1)


def main():
    optical = ROOT / "data/udegi/processed/sentinel_lithium_candidate_6band.tif"
    embeddings = ROOT / "data/udegi/tessera/2025_lithium_locality_candidate/tessera_2025_utm32.tif"
    dem = ROOT / "data/udegi/raw/DEM.tif"
    out = ROOT / "outputs/udegi/reconnaissance"
    out.mkdir(parents=True, exist_ok=True)
    with rasterio.open(optical) as s:
        bands = s.read(masked=True).filled(np.nan)
        profile = s.profile.copy()
        transform, crs, shape = s.transform, s.crs, s.shape
    with rasterio.open(embeddings) as t:
        if t.crs != crs or t.shape != shape or not t.transform.almost_equals(transform):
            raise ValueError("TESSERA and Sentinel grids differ")
        e = t.read(masked=True).filled(np.nan)
    if bands.shape[0] != 6 or e.shape[0] != 128:
        raise ValueError("Expected six Sentinel bands and 128 embedding dimensions")
    blue, green, red, nir, swir1, swir2 = bands
    valid = np.isfinite(bands).all(0)
    ndvi = np.divide(nir-red, nir+red, out=np.full(shape, np.nan, np.float32),
                     where=valid & (np.abs(nir+red) > 1e-6))
    ratio = np.divide(swir1, swir2, out=np.full(shape, np.nan, np.float32),
                      where=valid & (swir2 > 1e-6))
    elevation = np.full(shape, np.nan, np.float32)
    with rasterio.open(dem) as d:
        reproject(source=rasterio.band(d, 1), destination=elevation,
                  src_transform=d.transform, src_crs=d.crs, src_nodata=d.nodata,
                  dst_transform=transform, dst_crs=crs, dst_nodata=np.nan,
                  resampling=Resampling.bilinear)
    # Slope remains derived from a coarser DEM resampled onto this alignment grid.
    dy, dx = np.gradient(elevation, abs(transform.e), abs(transform.a))
    slope = np.degrees(np.arctan(np.hypot(dx, dy))).astype(np.float32)
    X = e.reshape(128, -1).T
    good = np.isfinite(X).all(1)
    ids = np.flatnonzero(good)
    if len(ids) < 3:
        raise ValueError("Insufficient valid TESSERA pixels")
    sample = np.random.default_rng(42).choice(ids, min(50000, len(ids)), replace=False)
    pca = PCA(n_components=3, random_state=42).fit(X[sample])
    pc = np.full((len(X), 3), np.nan, np.float32)
    pc[good] = pca.transform(X[good])
    pc = pc.reshape(*shape, 3)
    rgb = stretch(np.stack([red, green, blue], axis=-1))
    layers = [ndvi, ratio, elevation, slope] + [pc[..., i] for i in range(3)]
    names = ["NDVI", "B11_B12_ratio", "DEM_elevation_m", "DEM_slope_deg",
             "TESSERA_PC1", "TESSERA_PC2", "TESSERA_PC3"]
    profile.update(count=len(layers), dtype="float32", nodata=np.nan, compress="deflate")
    with rasterio.open(out / "candidate_diagnostics.tif", "w", **profile) as dst:
        for i, (a, name) in enumerate(zip(layers, names), 1):
            dst.write(a.astype(np.float32), i)
            dst.set_band_description(i, name)
        dst.update_tags(PURPOSE="Exploratory reconnaissance; no lithium prediction")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    panels = [(rgb, "Sentinel true colour", None), (ndvi, "NDVI", "RdYlGn"),
              (ratio, "B11/B12 spectral proxy", "viridis"),
              (elevation, "Elevation (m)", "terrain"),
              (slope, "Slope (degrees; derived from GLO-30)", "magma"),
              (stretch(pc), "TESSERA PCA (exploratory)", None)]
    for ax, (a, title, cmap) in zip(axes.ravel(), panels):
        if cmap:
            lo, hi = (-1, 1) if title == "NDVI" else np.nanpercentile(a, [2, 98])
            im = ax.imshow(a, cmap=cmap, vmin=lo, vmax=hi)
            fig.colorbar(im, ax=ax, shrink=0.7)
        else:
            ax.imshow(np.nan_to_num(a))
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Udegi candidate window: observations and embeddings, not mineral predictions")
    fig.tight_layout()
    fig.savefig(out / "candidate_reconnaissance.png", dpi=160)
    plt.close(fig)
    (out / "reconnaissance_metadata.json").write_text(json.dumps({
        "inputs": [str(optical), str(embeddings), str(dem)],
        "crs": str(crs), "shape": shape, "optical_valid_fraction": float(valid.mean()),
        "tessera_valid_fraction": float(good.mean()),
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "limits": "Approximate locality. No training labels, mineral scores or accuracy claims. "
                  "Annual 2025 embeddings and multi-season Sentinel composites differ in temporal support."
    }, indent=2), encoding="utf-8")
    print(out / "candidate_reconnaissance.png")
    print(out / "candidate_diagnostics.tif")


if __name__ == "__main__":
    main()
