"""Fast unit tests for the GeoSentry-SOTA core.   Run:  python -m pytest -q"""
import numpy as np
import pytest
import torch

from src.models.attention_unet import (
    AttentionUNet,
    FocalDiceLoss,
    PatchWindow,
    apply_d4,
    invert_d4,
    spatial_split,
)
from src.models.prospector_xgb import PUBaggingProspector
from src.utils.harmonizer import apply_scaler, first_vertical_derivative, slope_degrees


@pytest.mark.parametrize("code", range(8))
def test_d4_roundtrip_is_exact(code):
    a = np.random.rand(11, 32, 32).astype(np.float32)
    assert np.array_equal(invert_d4(apply_d4(a, code), code), a)


def test_attention_unet_shapes():
    m = AttentionUNet(in_channels=11, base=8, depth=3, latent_channels=4).eval()
    x = torch.randn(2, 11, 64, 64)
    logits, z, alphas = m(x, return_features=True)
    assert logits.shape == (2, 1, 64, 64)
    assert z.shape == (2, 4, 64, 64)
    assert [a.shape[-1] for a in alphas] == [64, 32, 16]
    assert all(((a >= 0) & (a <= 1)).all() for a in alphas)


def test_focal_dice_ignores_ignore_index():
    loss = FocalDiceLoss()
    logits = torch.zeros(1, 1, 4, 4)
    t = torch.zeros(1, 4, 4, dtype=torch.long)
    t[0, 0, 0] = 1
    l1, _ = loss(logits, t)
    t2 = t.clone()
    t2[0, 3, 3] = 255
    l2, _ = loss(logits, t2)
    assert torch.isfinite(l1) and torch.isfinite(l2)
    assert 0 <= float(l1) < 3


def test_spatial_split_has_no_shared_pixels():
    wins = [PatchWindow(r, c, (r + c) % 512 == 0) for r in range(0, 1024, 64) for c in range(0, 1024, 64)]
    tr, va, _ = spatial_split(wins, 128, 256, 0.25, 0)
    occ = np.zeros((1200, 1200), np.uint8)
    for w in tr:
        occ[w.row:w.row + 128, w.col:w.col + 128] |= 1
    for w in va:
        occ[w.row:w.row + 128, w.col:w.col + 128] |= 2
    assert tr and va and not (occ == 3).any()


def test_first_vertical_derivative_of_plane_is_zero():
    yy, xx = np.mgrid[0:64, 0:64]
    plane = (0.0 * xx + 5.0).astype(np.float32)
    out = first_vertical_derivative(plane, 10.0, 50.0)
    assert np.allclose(out, 0, atol=1e-4)


def test_slope_of_45_degree_ramp():
    yy, xx = np.mgrid[0:32, 0:32]
    dem = (xx * 10.0).astype(np.float32)  # rises 10 m per 10 m pixel
    assert np.allclose(slope_degrees(dem, 10.0)[5:-5, 5:-5], 45.0, atol=1e-3)


def test_scalers():
    a = np.linspace(0, 10, 101, dtype=np.float32)
    mm = apply_scaler(a, {"method": "minmax", "lo": 2.0, "hi": 8.0})
    assert mm.min() == 0 and mm.max() == 1
    z = apply_scaler(a, {"method": "zscore", "lo": 0, "hi": 10, "mean": 5, "std": 1, "limit": 3})
    assert z.min() == -3 and z.max() == 3


def test_pu_bagging_recovers_hidden_positives():
    rng = np.random.default_rng(0)
    pos = rng.normal(2, 0.5, size=(60, 3))
    hidden = rng.normal(2, 0.5, size=(60, 3))
    bg = rng.normal(0, 0.7, size=(1500, 3))
    unl = np.vstack([hidden, bg])
    m = PUBaggingProspector({"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1}, bags=8, seed=0).fit(pos, unl)
    assert m.predict_proba(hidden).mean() > m.predict_proba(bg).mean() + 0.4


# ---------------------------------------------------------------------------
# Review-driven tests: metrics, folds, explanations
# ---------------------------------------------------------------------------
import pandas as pd
from affine import Affine
from sklearn.metrics import average_precision_score

from src.evaluation.folds import FoldLayout, cluster_occurrences
from src.evaluation.metrics import BinnedPR, ScoreCDF, occurrence_metrics
from src.models.attention_unet import lattice_origins, owned_slices


def test_score_cdf_counts_ties_inclusively():
    cdf = ScoreCDF()
    cdf.update(np.array([0.1, 0.5, 0.5, 0.5, 0.9]))
    taf = cdf.top_area_fraction(np.array([0.9, 0.5, 0.1]))
    assert np.allclose(taf, [0.2, 0.8, 1.0])          # ties at 0.5 all count as flagged
    assert cdf.realized_area(0.5) == 0.8


def test_binned_ap_matches_sklearn_without_ties():
    rng = np.random.default_rng(1)
    y = rng.random(20000) < 0.01
    p = np.clip(rng.random(20000) * 0.5 + 0.5 * y * rng.random(20000), 0, 0.9999)
    p = np.round(p, 4)  # align with 1e-4 bins
    pr = BinnedPR(10_000)
    pr.update(p, y)
    assert abs(pr.average_precision() - average_precision_score(y, p)) < 0.02


@pytest.mark.parametrize("n", [256, 300, 513, 1000])
def test_owned_slices_cover_every_cell_once(n):
    patch = 128
    cover = np.zeros(n, int)
    for o in lattice_origins(n, patch, patch):
        sl, _ = owned_slices(o, 0, patch, (n, n))
        cover[o + sl.start: o + sl.stop] += 1
    assert (cover == 1).all()


def test_cluster_occurrences_single_linkage():
    df = pd.DataFrame({"x": [0, 300, 600, 5000], "y": [0, 0, 0, 0]})
    out = cluster_occurrences(df, 400)
    assert out["cluster_id"].nunique() == 2
    assert out["cluster_id"].iloc[0] == out["cluster_id"].iloc[2]


def _layout():
    tr = Affine(10, 0, 0, 0, -10, 10_000)  # 1000 x 1000 cells, 10 km
    rng = np.random.default_rng(0)
    xs, ys = rng.uniform(500, 9500, 40), rng.uniform(500, 9500, 40)
    rows, cols = ((10_000 - ys) // 10).astype(int), (xs // 10).astype(int)
    occ = cluster_occurrences(pd.DataFrame({"x": xs, "y": ys, "row": rows, "col": cols}), 300)
    return FoldLayout.build(tr, (1000, 1000), "EPSG:32632", occ, 2500, 4, 500, 0), occ


def test_fold_layout_dead_zone_and_indivisible_clusters():
    lay, occ = _layout()
    rr, cc = np.mgrid[0:1000:7, 0:1000:7]
    rr, cc = rr.ravel(), cc.ravel()
    fold = lay.fold_at(rr, cc)
    for k in range(lay.n_folds):
        elig = lay.train_eligible(rr, cc, k)
        assert not (elig & (fold == k)).any()
        # every eligible cell is > 500 m (minus one coarse cell) from every fold-k cell
        test_xy = np.c_[cc[fold == k] * 10.0, rr[fold == k] * 10.0]
        from scipy.spatial import cKDTree
        d, _ = cKDTree(test_xy).query(np.c_[cc[elig] * 10.0, rr[elig] * 10.0])
        assert d.min() >= 500 - lay.coarse_res * 1.5
        train, test, dropped = lay.split_clusters(occ, k)
        assert not set(train) & set(test)
        assert set(train) | set(test) | set(dropped) == set(occ["cluster_id"])


def test_occurrence_metrics_area_for_80():  # noqa: D103
    a = np.array([0.01, 0.02, 0.03, 0.04, 0.5])
    m = occurrence_metrics(a)
    assert m["area_for_80pct_recall"] == 0.04 and m["recall_at_top_5pct_area"] == 0.8


def test_shap_is_exact_for_logit_mean_ensemble():
    rng = np.random.default_rng(3)
    pos, unl = rng.normal(1, 1, (80, 4)), rng.normal(0, 1, (800, 4))
    m = PUBaggingProspector({"n_estimators": 30, "max_depth": 3}, bags=4, seed=1).fit(pos, unl)
    X = rng.normal(0, 1, (50, 4))
    sv, base = m.shap_values(X)
    assert np.allclose(base + sv.sum(1), m.margin(X), atol=1e-4)


# ---------------------------------------------------------------------------
# Regression tests for the second review (P1 and P2 findings)
# ---------------------------------------------------------------------------
import rasterio
from rasterio.transform import from_origin

from src.evaluation.folds import write_fold_training_rasters
from src.evaluation.nested_cv import _ssl_provenance, tied_auc
from src.config import Config


def _crossing_case(dead_zone=0.0):
    """Review reproduction: 200 x 400 cells at 10 m, 1 km blocks, 2 folds, seed 2."""
    tr = from_origin(0, 2000, 10, 10)
    xs = np.array([800.0, 1400.0, 2500.0, 3500.0])
    ys = np.array([1500.0, 1500.0, 500.0, 500.0])
    rows, cols = rasterio.transform.rowcol(tr, xs, ys)
    occ = pd.DataFrame({"x": xs, "y": ys, "row": np.asarray(rows), "col": np.asarray(cols)})
    occ = cluster_occurrences(occ, 700)
    lay = FoldLayout.build(tr, (200, 400), "EPSG:32632", occ, 1000, 2, dead_zone, 2, footprint_m=30)
    return tr, occ, lay


@pytest.mark.parametrize("dz", [0.0, 200.0])
def test_p1_cluster_never_straddles_folds(dz):
    _, occ, lay = _crossing_case(dz)
    assert occ.loc[0, "cluster_id"] == occ.loc[1, "cluster_id"]
    member_folds = lay.fold_at(occ["row"].to_numpy(), occ["col"].to_numpy())
    for cid, f in lay.cluster_fold.items():
        assert set(member_folds[occ["cluster_id"].to_numpy() == cid]) == {f}
    for k in range(lay.n_folds):
        _, test, _ = lay.split_clusters(occ, k)
        m = occ["cluster_id"].isin(test).to_numpy()
        assert not lay.train_eligible(occ["row"].to_numpy()[m], occ["col"].to_numpy()[m], k).any()


def test_p1_mapped_outline_linked_to_test_cluster_is_removed(tmp_path):
    tr, occ, lay = _crossing_case(0.0)
    k = lay.cluster_fold[int(occ.loc[0, "cluster_id"])]
    # A mapped outline running from the test cluster east across the whole AOI.
    tgt = np.zeros((200, 400), np.uint8)
    r0 = int(occ.loc[0, "row"])
    tgt[r0:r0 + 2, int(occ.loc[0, "col"]):] = 1
    prof = dict(driver="GTiff", width=400, height=200, count=1, dtype="uint8", crs="EPSG:32632", transform=tr)
    with rasterio.open(tmp_path / "t.tif", "w", nodata=255, **prof) as d:
        d.write(tgt, 1)
    with rasterio.open(tmp_path / "v.tif", "w", nodata=0, **prof) as d:
        d.write(np.ones((200, 400), np.uint8), 1)
    test_m = occ["cluster_id"].map(lay.cluster_fold).to_numpy() == k
    audit = write_fold_training_rasters(lay, k, tmp_path / "v.tif", tmp_path / "t.tif", tmp_path / "ov.tif",
                                        tmp_path / "ot.tif", occ["row"].to_numpy()[test_m],
                                        occ["col"].to_numpy()[test_m], 3)
    with rasterio.open(tmp_path / "ot.tif") as d:
        out = d.read(1)
    with rasterio.open(tmp_path / "ov.tif") as d:
        ov = d.read(1)
    assert not (out == 1).any()                 # no label of the linked outline survives anywhere
    assert ov[r0, -1] == 0                      # its imagery is zeroed even in training territory
    assert audit["linked_cells_outside_test_area"] > 0


def test_p2_saturated_ties_report_empty_selection():
    cdf = ScoreCDF()
    cdf.update(np.ones(3))
    assert cdf.realized_area_for_budget(0.02) == 0.0
    assert cdf.threshold_for_area(0.02) is None
    assert (cdf.top_area_fraction(np.ones(3)) <= 0.02).sum() == 0
    cdf2 = ScoreCDF()
    cdf2.update(np.r_[np.ones(10), np.linspace(0, 0.5, 90)])   # 10 % tie block at the top
    assert cdf2.realized_area_for_budget(0.05) == 0.0
    assert abs(cdf2.realized_area_for_budget(0.10) - 0.10) < 1e-12
    assert abs(cdf2.realized_area_for_budget(0.15) - 0.15) < 1e-12


@pytest.mark.parametrize("v", [0.0, 1.0])
def test_p2_auc_ties_at_endpoints(v):
    cdf = ScoreCDF()
    cdf.update(np.full(50, v))
    assert tied_auc(cdf, np.full(5, v)) == 0.5


def test_p2_auc_matches_sklearn_with_ties():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(4)
    neg = np.round(rng.random(2000), 2)
    pos = np.round(np.clip(rng.random(200) + 0.2, 0, 1), 2)
    cdf = ScoreCDF()
    cdf.update(neg)
    ref = roc_auc_score(np.r_[np.ones(200), np.zeros(2000)], np.r_[pos, neg])
    assert abs(tied_auc(cdf, pos) - ref) < 1e-9


def test_p2_missing_clusters_stay_in_denominator():
    m = occurrence_metrics(np.array([0.01, np.inf]))
    assert m["recall_at_top_2pct_area"] == 0.5 and m["n_missed_unevaluable"] == 1
    assert m["area_for_80pct_recall"] is None
    with pytest.raises(ValueError):
        occurrence_metrics(np.array([0.01, np.nan]))
    rep = __import__("src.evaluation.metrics", fromlist=["x"]).occurrence_report(np.array([0.01, np.inf]), n_boot=50)
    assert rep["point"]["recall_at_top_2pct_area"] == 0.5


def test_p2_transductive_ssl_requires_checkpoint(tmp_path):
    cfg = Config({"evaluation": {"ssl_mode": "transductive"}, "channels": ["a"],
                  "paths": {"ssl_weights": str(tmp_path / "missing.pt")}})
    with pytest.raises(FileNotFoundError):
        _ssl_provenance(cfg)
    torch.save({"encoder": {}, "channels": ["a"], "epoch": 3}, tmp_path / "ok.pt")
    cfg["paths"]["ssl_weights"] = str(tmp_path / "ok.pt")
    info = _ssl_provenance(cfg)
    assert info["mode"] == "transductive" and len(info["sha256"]) == 64
    cfg["channels"] = ["b"]
    with pytest.raises(ValueError):
        _ssl_provenance(cfg)
