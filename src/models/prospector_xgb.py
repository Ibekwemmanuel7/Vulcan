"""
Stage 2: Positive-Unlabeled prospectivity engine
================================================

Problem framing
---------------
Known occurrences (NGSA records, artisanal workings) are positives. Every other
cell is UNLABELED, not negative: it may be barren or unexplored. The model is a
bagged PU classifier (Mordelet and Vert, 2014):

    for b in 1..B:
        U_b = bootstrap sample of the unlabeled pool, |U_b| = r * |P|
        f_b = classifier trained on P (label 1) vs U_b (label 0)
    score(x) = sigmoid( mean_b margin_b(x) )          0.00 to 1.00

Bags are averaged in log-odds space. This keeps the score in [0, 1] and makes
TreeSHAP over ALL bags an exact additive explanation of the deployed score's
log-odds (review finding 7). Averaging does not guarantee that hidden positives
inside U are cancelled; it only dilutes them in each bag.

What the score is
-----------------
A relative ranking of ground. It is not the probability that an economic
deposit exists. Band 2 of prospectivity.tif, `top_area_fraction`, is the share
of ALL valid AOI cells scoring >= the cell (ties included; review finding 3):
0.03 means "inside the top 3 % of the AOI".

Performance numbers
-------------------
This module fits and applies the deployment model only. All performance metrics
come from `src.evaluation.nested_cv`, which retrains Stage 1 and Stage 2 inside
shared geographic folds (review finding 1).

    python -m src.models.prospector_xgb --config configs/geosentry.yaml
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import Window
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import Config, get_logger, load_config, seed_everything
from src.evaluation.folds import (
    buffer_cells,
    check_same_grid,
    cluster_occurrences,
    drop_invalid_occurrences,
    load_occurrences,
)
from src.evaluation.metrics import ScoreCDF

LOG = get_logger("prospector")

FEATURE_GROUPS: Dict[str, str] = {
    "B2_blue": "Optical reflectance",
    "B3_green": "Optical reflectance",
    "B4_red": "Optical reflectance",
    "B8_nir": "Optical reflectance",
    "S1_VV_dB": "SAR backscatter",
    "S1_VH_dB": "SAR backscatter",
    "OH_alteration": "SWIR ratio B11/B12 (broad Al-OH proxy)",
    "NDVI": "Vegetation cover (NDVI)",
    "MAG_1VD": "Magnetic 1VD",
    "RAD_eTh": "Radiometric eTh",
    "DEM_slope": "Terrain slope",
    "stage1_prob": "Stage 1 CNN probability",
}


def feature_group(name: str) -> str:
    if name in FEATURE_GROUPS:
        return FEATURE_GROUPS[name]
    if name.startswith("latent_"):
        return "Stage 1 latent embedding"
    if name.startswith("attn_gate"):
        return "Stage 1 attention gates (internal)"
    return name


def sigmoid(z: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))).astype(np.float32)


# =============================================================================
# Raster feature access
# =============================================================================
class FeatureRasters:
    """Pixel-aligned access to the harmonized stack plus (optionally) a Stage 1 feature raster."""

    def __init__(self, stack_path: Path, stage1_path: Optional[Path], valid_mask_path: Path,
                 physical_only: bool = False):
        use_s1 = (not physical_only) and stage1_path is not None and Path(stage1_path).exists()
        self.paths = [Path(stack_path)] + ([Path(stage1_path)] if use_s1 else [])
        self.valid_mask_path = Path(valid_mask_path)
        check_same_grid(self.paths + [self.valid_mask_path])  # CRS, shape and transform
        names: List[str] = []
        for p in self.paths:
            with rasterio.open(p) as src:
                names += [d or f"{p.stem}_b{i + 1}" for i, d in enumerate(src.descriptions)]
        with rasterio.open(self.paths[0]) as src:
            self.shape, self.transform, self.crs, self.profile = src.shape, src.transform, src.crs, src.profile
        self.feature_names = names
        if not physical_only and not use_s1:
            LOG.warning("Stage 1 feature raster not found; using the physical channels only")

    def read_block(self, window: Window) -> np.ndarray:
        arrs = []
        for p in self.paths:
            with rasterio.open(p) as src:
                arrs.append(src.read(window=window).astype(np.float32))
        return np.concatenate(arrs, axis=0)

    def read_valid(self) -> np.ndarray:
        with rasterio.open(self.valid_mask_path) as src:
            return src.read(1).astype(bool)

    def extract(self, rows: np.ndarray, cols: np.ndarray, block: int = 1024) -> np.ndarray:
        rows, cols = np.asarray(rows, int), np.asarray(cols, int)
        out = np.full((rows.size, len(self.feature_names)), np.nan, dtype=np.float32)
        if rows.size == 0:
            return out
        br, bc = rows // block, cols // block
        keys = br * 1_000_000 + bc
        for key in np.unique(keys):
            sel = np.flatnonzero(keys == key)
            r0, c0 = int(br[sel[0]]) * block, int(bc[sel[0]]) * block
            h, w = min(block, self.shape[0] - r0), min(block, self.shape[1] - c0)
            data = self.read_block(Window(c0, r0, w, h))
            out[sel] = data[:, rows[sel] - r0, cols[sel] - c0].T
        return out

    def rowcol_to_xy(self, rows: np.ndarray, cols: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = rasterio.transform.xy(self.transform, rows, cols, offset="center")
        return np.asarray(xs, float), np.asarray(ys, float)

    def windows(self, block: int):
        H, W = self.shape
        for r0 in range(0, H, block):
            for c0 in range(0, W, block):
                yield Window(c0, r0, min(block, W - c0), min(block, H - r0))


# =============================================================================
# PU bagging model
# =============================================================================
@dataclass
class PUBaggingProspector:
    params: dict
    estimator: str = "xgb"                    # "xgb" or "logreg"
    bags: int = 50
    unlabeled_ratio: float = 1.0
    seed: int = 42
    feature_names: List[str] = field(default_factory=list)
    models: list = field(default_factory=list)
    area_cdf: Optional[dict] = None          # ScoreCDF.to_dict() over all valid AOI cells

    def _new(self, b: int):
        if self.estimator == "xgb":
            return XGBClassifier(**self.params, random_state=int(self.seed + b), n_jobs=-1,
                                 eval_metric="logloss", missing=np.nan)
        if self.estimator == "logreg":
            return make_pipeline(StandardScaler(), LogisticRegression(C=float(self.params.get("C", 1.0)), max_iter=2000))
        raise ValueError(self.estimator)

    def fit(self, X_pos: np.ndarray, X_unl: np.ndarray) -> "PUBaggingProspector":
        rng = np.random.default_rng(self.seed)
        n_p, n_u = len(X_pos), len(X_unl)
        if n_p == 0 or n_u == 0:
            raise ValueError("PU fit needs positive and unlabeled samples")
        k = int(min(n_u, max(1, round(self.unlabeled_ratio * n_p))))
        self.models = []
        for b in range(self.bags):
            idx_u = rng.choice(n_u, size=k, replace=True)
            X = np.vstack([X_pos, X_unl[idx_u]])
            y = np.r_[np.ones(n_p), np.zeros(k)]
            m = self._new(b)
            m.fit(X, y)
            self.models.append(m)
        return self

    def _margin(self, m, X: np.ndarray) -> np.ndarray:
        if self.estimator == "xgb":
            return m.predict(X, output_margin=True).astype(np.float64)
        return m.decision_function(X).astype(np.float64)

    def margin(self, X: np.ndarray, subset: Optional[Sequence[int]] = None) -> np.ndarray:
        if not self.models:
            raise RuntimeError("Model is not fitted")
        idx = range(len(self.models)) if subset is None else subset
        return np.mean([self._margin(self.models[i], X) for i in idx], axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.margin(X))

    def top_area_fraction(self, scores: np.ndarray) -> np.ndarray:
        if self.area_cdf is None:
            raise RuntimeError("Area reference not set; run predict_grid first")
        return ScoreCDF.from_dict(self.area_cdf).top_area_fraction(scores)

    def shap_values(self, X: np.ndarray, subset: Optional[Sequence[int]] = None) -> Tuple[np.ndarray, float]:
        """
        TreeSHAP (log-odds) averaged over the bags in `subset` (default ALL).
        Because the deployed score is sigmoid(mean margin), base + sum(SHAP) equals
        the deployed log-odds exactly when every bag is included.
        Returns (values (n, f), base value).
        """
        if self.estimator != "xgb":
            raise ValueError("TreeSHAP requires the xgb estimator")
        import shap

        idx = list(range(len(self.models))) if subset is None else list(subset)
        acc = np.zeros(X.shape, dtype=np.float64)
        base = 0.0
        for i in idx:
            ex = shap.TreeExplainer(self.models[i])
            sv = ex.shap_values(X)
            acc += sv if not isinstance(sv, list) else sv[1]
            ev = ex.expected_value
            base += float(np.ravel(ev)[-1])
        return acc / len(idx), base / len(idx)


# =============================================================================
# Training-set construction
# =============================================================================
def sample_unlabeled(eligible: Callable[[np.ndarray, np.ndarray], np.ndarray], shape: Tuple[int, int],
                     pos_xy: np.ndarray, fr: FeatureRasters, n: int, exclusion_m: float,
                     rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform random eligible cells farther than `exclusion_m` from every training occurrence."""
    H, W = shape
    tree = cKDTree(pos_xy) if len(pos_xy) else None
    rows, cols = [], []
    need, tries = n, 0
    while need > 0 and tries < 60:
        tries += 1
        r = rng.integers(0, H, size=need * 3)
        c = rng.integers(0, W, size=need * 3)
        keep = eligible(r, c)
        r, c = r[keep], c[keep]
        if tree is not None and r.size:
            xs, ys = fr.rowcol_to_xy(r, c)
            d, _ = tree.query(np.c_[xs, ys], k=1)
            r, c = r[d > exclusion_m], c[d > exclusion_m]
        rows.append(r[:need]); cols.append(c[:need])
        need -= min(need, r.size)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    lin = np.unique(rows.astype(np.int64) * W + cols)
    return (lin // W).astype(int), (lin % W).astype(int)


def build_training_set(fr: FeatureRasters, occ: pd.DataFrame, valid: np.ndarray, s2: dict,
                       eligible: Callable[[np.ndarray, np.ndarray], np.ndarray], rng: np.random.Generator,
                       exclude_xy: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Positive cells (disc buffers of training occurrences) and sampled unlabeled cells.
    Unlabeled cells keep `exclusion_radius_m` from every point in `exclude_xy`
    (default: the training occurrences; pass all known non-test occurrences so
    dropped clusters are never sampled as unlabeled).
    """
    pr, pc, _ = buffer_cells(occ["row"].to_numpy(), occ["col"].to_numpy(), occ["cluster_id"].to_numpy(),
                             int(s2["positive_buffer_px"]), fr.shape, valid)
    ok = eligible(pr, pc)
    pr, pc = pr[ok], pc[ok]
    excl = occ[["x", "y"]].to_numpy() if exclude_xy is None else exclude_xy
    ur, uc = sample_unlabeled(lambda r, c: valid[r, c] & eligible(r, c), fr.shape, excl,
                              fr, int(s2["unlabeled_samples"]), float(s2["exclusion_radius_m"]), rng)
    X_pos, X_unl = fr.extract(pr, pc), fr.extract(ur, uc)
    X_pos = X_pos[np.isfinite(X_pos).all(1)]
    X_unl = X_unl[np.isfinite(X_unl).all(1)]
    info = {"n_positive_cells": int(len(X_pos)), "n_unlabeled_cells": int(len(X_unl)),
            "n_occurrences": int(len(occ)), "n_clusters": int(occ["cluster_id"].nunique())}
    return X_pos, X_unl, info


# =============================================================================
# Grid prediction
# =============================================================================
def predict_grid(model: PUBaggingProspector, fr: FeatureRasters, out_path: Path, block: int) -> Path:
    """
    Two passes. Pass 1 scores every valid cell, writes band 1 and accumulates the
    full-AOI score distribution. Pass 2 converts each score to top_area_fraction
    against that distribution (ties included).
    """
    prof = fr.profile.copy()
    prof.update(count=2, dtype="float32", nodata=np.nan, BIGTIFF="IF_SAFER", predictor=3, compress="deflate")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cdf = ScoreCDF()
    with rasterio.open(fr.valid_mask_path) as vm, rasterio.open(out_path, "w", **prof) as dst:
        dst.set_band_description(1, "prospectivity_score")
        dst.set_band_description(2, "top_area_fraction")
        for win in fr.windows(block):
            data = fr.read_block(win)
            valid = vm.read(1, window=win).astype(bool) & np.isfinite(data).all(axis=0)
            score = np.full(valid.shape, np.nan, dtype=np.float32)
            if valid.any():
                score[valid] = model.predict_proba(data[:, valid].T)
                cdf.update(score[valid])
            dst.write(score, 1, window=win)
    model.area_cdf = cdf.to_dict()
    with rasterio.open(out_path, "r+") as dst:
        for win in fr.windows(block):
            s = dst.read(1, window=win)
            dst.write(cdf.top_area_fraction(s), 2, window=win)
        dst.update_tags(PROJECT="GeoSentry-SOTA", STAGE="2", METHOD=f"bagged PU {model.estimator}, logit-mean",
                        BAGS=str(model.bags), AREA_REFERENCE="all valid AOI cells, ties included",
                        VALID_CELLS=str(cdf.total))
    LOG.info("Prospectivity grid written: %s (%d valid cells)", out_path, cdf.total)
    return out_path


# =============================================================================
# Targets and explanations
# =============================================================================
def find_targets(prospectivity_path: Path, top_area_fraction: float = 0.02, min_pixels: int = 20) -> pd.DataFrame:
    """Connected clusters inside the top X % of AOI area, ranked by peak score.
    Note: reads both bands in full; tile very large AOIs first."""
    with rasterio.open(prospectivity_path) as src:
        score, taf = src.read(1), src.read(2)
        transform, crs = src.transform, src.crs
    mask = np.nan_to_num(taf, nan=1.0) <= top_area_fraction
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return pd.DataFrame()
    idx = np.arange(1, n + 1)
    sizes = ndimage.sum(mask, lab, idx)
    s0 = np.nan_to_num(score, nan=0)
    peak = ndimage.maximum(s0, lab, idx)
    mean = ndimage.mean(s0, lab, idx)
    peak_pos = ndimage.maximum_position(s0, lab, idx)
    to_geo = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    px_area_km2 = abs(transform.a * transform.e) / 1e6
    rows = []
    for i, lbl in enumerate(idx):
        if sizes[i] < min_pixels:
            continue
        r, c = peak_pos[i]
        x, y = rasterio.transform.xy(transform, r, c, offset="center")
        lon, lat = to_geo.transform(x, y)
        rows.append({"cluster_id": int(lbl), "peak_score": float(peak[i]), "mean_score": float(mean[i]),
                     "area_km2": float(sizes[i] * px_area_km2), "peak_row": int(r), "peak_col": int(c),
                     "easting": float(x), "northing": float(y), "lon": float(lon), "lat": float(lat)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("peak_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def _group_shares(mean_abs: np.ndarray, names: Sequence[str]) -> pd.Series:
    s = pd.Series(mean_abs, index=[feature_group(n) for n in names]).groupby(level=0).sum()
    return 100 * s / s.sum()


def explain_global(model: PUBaggingProspector, X: np.ndarray, names: Sequence[str]
                   ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Global TreeSHAP over ALL bags in log-odds space, plus two diagnostics:
      completeness  max |base + sum(SHAP) - deployed log-odds| (should be ~0)
      stability     agreement between two disjoint halves of the bags
    Correlated channels can trade attribution; read group totals, not single channels.
    """
    sv, base = model.shap_values(X)
    deployed = model.margin(X)
    completeness = float(np.max(np.abs(base + sv.sum(1) - deployed)))
    mean_abs = np.abs(sv).mean(axis=0)
    df = pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs, "group": [feature_group(n) for n in names]})
    df["share_pct"] = 100 * df["mean_abs_shap"] / df["mean_abs_shap"].sum()
    direction = []
    for j in range(X.shape[1]):
        xj, sj = X[:, j], sv[:, j]
        direction.append(float(np.corrcoef(xj, sj)[0, 1]) if np.std(xj) > 0 and np.std(sj) > 0 else 0.0)
    df["direction"] = direction
    groups = (df.groupby("group", as_index=False)[["mean_abs_shap", "share_pct"]].sum()
              .sort_values("share_pct", ascending=False))

    half = len(model.models) // 2
    stability = {}
    if half >= 2:
        a, _ = model.shap_values(X, range(half))
        b, _ = model.shap_values(X, range(half, len(model.models)))
        ma, mb = np.abs(a).mean(0), np.abs(b).mean(0)
        ga, gb = _group_shares(ma, names), _group_shares(mb, names)
        stability = {"feature_rank_spearman": float(spearmanr(ma, mb).statistic),
                     "group_share_max_abs_diff_pct": float((ga - gb).abs().max())}
    diag = {"output_space": "log-odds of the deployed score (sigmoid of mean bag margin)",
            "bags_explained": len(model.models), "n_samples": int(len(X)),
            "completeness_max_abs_error": completeness, "stability_between_bag_halves": stability,
            "caution": "Attributions describe this model, not mineralisation mechanisms. Correlated channels "
                       "can exchange attribution; compare with modality ablation before drawing conclusions."}
    return df.sort_values("share_pct", ascending=False), groups, diag


def explain_targets(model: PUBaggingProspector, fr: FeatureRasters, targets: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    if targets.empty:
        return targets
    X = fr.extract(targets["peak_row"].to_numpy(), targets["peak_col"].to_numpy())
    sv, _ = model.shap_values(X)
    names = list(fr.feature_names)
    drivers = []
    for i in range(len(targets)):
        grp = pd.Series(sv[i], index=[feature_group(n) for n in names]).groupby(level=0).sum()
        top = grp.sort_values(ascending=False).head(top_k)
        drivers.append("; ".join(f"{g} ({v:+.2f})" for g, v in top.items()))
    out = targets.copy()
    out["top_drivers_logodds"] = drivers
    return out


def prepare_occurrences(cfg: Config, fr: FeatureRasters, valid: np.ndarray) -> Tuple[pd.DataFrame, dict]:
    s2, ev = cfg["stage2"], cfg["evaluation"]
    occ, counts = load_occurrences(cfg.path("positives_csv"), fr.crs, fr.transform, fr.shape,
                                   s2.get("commodity_filter"))
    occ, n_drop = drop_invalid_occurrences(occ, valid, int(s2["positive_buffer_px"]))
    counts["dropped_no_valid_data"] = n_drop
    occ = cluster_occurrences(occ, float(ev["cluster_radius_m"]))
    counts["retained"] = int(len(occ))
    counts["clusters"] = int(occ["cluster_id"].nunique())
    return occ, counts


# =============================================================================
# Deployment run
# =============================================================================
def run(cfg: Config) -> Path:
    s2 = cfg["stage2"]
    seed = int(cfg["project"]["seed"])
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    reports = cfg.path("reports_dir"); reports.mkdir(parents=True, exist_ok=True)
    models_dir = cfg.path("models_dir"); models_dir.mkdir(parents=True, exist_ok=True)

    fr = FeatureRasters(cfg.path("stack"), cfg.path("stage1_features"), cfg.path("valid_mask"))
    valid = fr.read_valid()
    occ, counts = prepare_occurrences(cfg, fr, valid)
    LOG.info("Occurrences: %s", counts)
    if counts["clusters"] < 5:
        raise RuntimeError(f"Only {counts['clusters']} occurrence clusters; PU learning needs at least 5 (ideally 30+).")

    X_pos, X_unl, info = build_training_set(fr, occ, valid, s2, lambda r, c: np.ones(np.shape(r), bool), rng)
    LOG.info("Training set: %s", info)
    model = PUBaggingProspector(dict(s2["xgb"]), "xgb", int(s2["bags"]), float(s2["unlabeled_per_bag_ratio"]),
                                seed, fr.feature_names)
    model.fit(X_pos, X_unl)
    out = predict_grid(model, fr, cfg.path("prospectivity"), int(s2["predict_block"]))
    joblib.dump(model, models_dir / "pu_prospector.joblib")

    LOG.info("Computing TreeSHAP over all %d bags", len(model.models))
    n_shap = int(s2["shap_samples"])
    idx_u = rng.choice(len(X_unl), size=min(n_shap, len(X_unl)), replace=False)
    idx_p = rng.choice(len(X_pos), size=min(max(1, n_shap // 4), len(X_pos)), replace=False)
    X_shap = np.vstack([X_pos[idx_p], X_unl[idx_u]])
    feat_df, group_df, diag = explain_global(model, X_shap, fr.feature_names)
    feat_df.to_csv(reports / "shap_global.csv", index=False)
    group_df.to_csv(reports / "shap_groups.csv", index=False)
    diag["sample"] = f"{len(idx_p)} occurrence-buffer cells + {len(idx_u)} sampled unlabeled cells"
    (reports / "shap_diagnostics.json").write_text(json.dumps(diag, indent=2))

    targets = explain_targets(model, fr, find_targets(out, float(s2.get("target_top_area", 0.02))))
    targets.to_csv(reports / "targets.csv", index=False)
    (reports / "deployment_model.json").write_text(json.dumps(
        {"occurrence_counts": counts, "training_set": info, "features": fr.feature_names,
         "commodity_filter": s2.get("commodity_filter"),
         "note": "Deployment model fitted on all occurrences. It has no held-out score; see evaluation_report.json."},
        indent=2))
    LOG.info("SHAP completeness error %.2e | stability %s", diag["completeness_max_abs_error"], diag["stability_between_bag_halves"])
    LOG.info("%d target clusters written to targets.csv", len(targets))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Stage 2 PU prospectivity (deployment model)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
