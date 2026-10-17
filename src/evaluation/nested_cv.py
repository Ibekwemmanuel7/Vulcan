"""
Nested geographic evaluation of the full two-stage pipeline and baselines
=========================================================================

This is the ONLY source of performance numbers in GeoSentry.

Protocol
--------
1. Occurrences are filtered (optional commodity), checked for valid data and
   clustered (`evaluation.cluster_radius_m`). A cluster is the independent unit.
2. One FoldLayout is built and written to disk (fold_map.tif, fold_assignment.csv).
   Every block touched by a cluster footprint (members plus positional and label
   buffers) is tied to that cluster, and tied blocks go to one fold together, so a
   cluster never straddles folds. Every method uses the same folds.
3. For each outer fold k:
     * TEST AREA = fold-k blocks. TRAIN-ELIGIBLE = outside fold k and farther than
       `evaluation.dead_zone_m`. Training clusters touching the test area or dead
       zone are dropped whole.
     * Stage 1 labels are additionally removed (label 255, imagery zeroed) wherever
       they are linked to a test cluster: within `stage2.exclusion_radius_m` of a
       member, or on a mapped outline touching that radius. This holds even if a
       mapped outline extends out of the test area.
     * Stage 1 is retrained from scratch with model selection inside the training
       area, then applied to the whole grid. Stage 2 is fitted on training clusters
       and unlabeled cells from the train-eligible area, with every known
       occurrence kept `exclusion_radius_m` away from unlabeled samples.
     * EVALUABLE POPULATION (declared before any model is scored, identical for all
       methods): test clusters with at least one valid cell of their positional
       buffer inside the test area. Hit buffers are restricted to test-area cells.
       A method that cannot score an evaluable cluster stops the run.
     * area_needed = share of the fold's test area scoring at least the cluster's
       best buffer score (ties included).
4. Each method is reported on two populations:
     evaluable          headline; conditional on clusters the data can test
     all_test_clusters  conservative; unevaluable clusters counted as misses

SSL provenance: with evaluation.ssl_mode = transductive the SSL checkpoint must
exist and match the config; its SHA-256 and the per-fold initialisation actually
used are recorded. The encoder has seen unlabeled imagery of the test areas
(never labels). Use ssl_mode = none to remove it.

    python -m src.evaluation.nested_cv --config configs/geosentry.yaml
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import rasterio
import torch

from src.config import Config, get_logger, load_config, seed_everything
from src.evaluation.folds import FoldLayout, buffer_cells, write_fold_training_rasters
from src.evaluation.metrics import AREA_LEVELS, ScoreCDF, occurrence_metrics, occurrence_report, success_curve
from src.models.prospector_xgb import (
    FeatureRasters,
    PUBaggingProspector,
    build_training_set,
    prepare_occurrences,
)

LOG = get_logger("evaluate")
ALL_METHODS = ["two_stage", "stage1_only", "pu_xgb_physical", "pu_logreg_physical"]
NEEDS_STAGE1 = {"two_stage", "stage1_only"}


def tied_auc(cdf: ScoreCDF, pos_scores: np.ndarray) -> float:
    """
    Mann-Whitney AUC of positive cells against the CDF population, ties counting
    one half: mean over positives of P(U < s) + 0.5 * P(U = s).
    """
    if len(pos_scores) == 0:
        return float("nan")
    ge = cdf.top_area_fraction(pos_scores).astype(np.float64)
    gt = cdf.strictly_above_fraction(pos_scores)
    return float(np.mean(1.0 - ge + 0.5 * (ge - gt)))


def _score_test_area(score_fn, fr: FeatureRasters, layout: FoldLayout, k: int, valid_path: Path,
                     block: int) -> ScoreCDF:
    cdf = ScoreCDF()
    with rasterio.open(valid_path) as vm:
        for win in fr.windows(block):
            test, _ = layout.window_masks(win, k)
            if not test.any():
                continue
            data = fr.read_block(win)
            m = test & vm.read(1, window=win).astype(bool) & np.isfinite(data).all(0)
            if m.any():
                cdf.update(score_fn(data[:, m].T))
    return cdf


def _evaluate_fold(cfg: Config, k: int, layout: FoldLayout, occ: pd.DataFrame, valid: np.ndarray,
                   methods: List[str], fold_dir: Path, device: Optional[str], seed: int,
                   ssl_info: dict) -> List[dict]:
    s2, ev = cfg["stage2"], cfg["evaluation"]
    rng = np.random.default_rng(seed + 1000 * k)
    res = abs(layout.transform.a)
    train_c, test_c, dropped_c = layout.split_clusters(occ, k)
    test_occ = occ[occ["cluster_id"].isin(test_c)]
    train_occ = occ[occ["cluster_id"].isin(train_c)]

    # ---- predeclared evaluable population (method independent) ---------------------
    buf_px = max(0, int(round(float(ev["occurrence_buffer_m"]) / res)))
    br, bc, bg = buffer_cells(test_occ["row"].to_numpy(), test_occ["col"].to_numpy(),
                              test_occ["cluster_id"].to_numpy(), buf_px, layout.shape, valid)
    in_test = layout.fold_at(br, bc) == k
    br, bc, bg = br[in_test], bc[in_test], bg[in_test]
    evaluable = np.array(sorted(set(bg.tolist())), dtype=int)
    unevaluable = np.array(sorted(set(test_c.tolist()) - set(evaluable.tolist())), dtype=int)
    audit = {"fold": k, "seed": seed, "train_clusters": int(len(train_c)), "test_clusters": int(len(test_c)),
             "evaluable_test_clusters": int(len(evaluable)), "unevaluable_test_clusters": int(len(unevaluable)),
             "dropped_dead_zone_clusters": int(len(dropped_c))}
    LOG.info("Fold %d: %s", k, audit)
    if len(evaluable) == 0 or len(train_c) < 3:
        LOG.warning("Fold %d skipped (needs >=1 evaluable test cluster and >=3 training clusters)", k)
        return [{**audit, "method": m, "skipped": True} for m in methods]

    stack, valid_path = cfg.path("stack"), cfg.path("valid_mask")
    fold_dir.mkdir(parents=True, exist_ok=True)
    s1_feat, s1_init, excl_audit = None, None, {}
    if NEEDS_STAGE1 & set(methods):
        from src.pipeline.stage1 import infer, train

        fv, ft = fold_dir / "train_valid.tif", fold_dir / "train_target.tif"
        excl_px = int(round(float(s2["exclusion_radius_m"]) / res))
        excl_audit = write_fold_training_rasters(layout, k, valid_path, cfg.path("target_mask"), fv, ft,
                                                 test_occ["row"].to_numpy(), test_occ["col"].to_numpy(), excl_px)
        transductive = ev.get("ssl_mode", "transductive") == "transductive"
        w = train(cfg, device, use_ssl=transductive, require_ssl=transductive, target_path=ft, valid_path=fv,
                  weights_out=fold_dir / "stage1.pt", report_tag=f"evaluation/fold{k}_seed{seed}_stage1")
        s1_init = torch.load(w, map_location="cpu", weights_only=False).get("init")
        if transductive and (s1_init or {}).get("sha256") != ssl_info.get("sha256"):
            raise RuntimeError(f"Fold {k} Stage 1 was not initialised from the declared SSL checkpoint")
        s1_feat = infer(cfg, device, weights=w, out_path=fold_dir / "stage1_features.tif")

    exclude_xy = occ[["x", "y"]].to_numpy()  # unlabeled samples keep away from every known occurrence
    eligible = lambda r, c: layout.train_eligible(r, c, k)  # noqa: E731

    rows = []
    for method in methods:
        t0 = time.time()
        physical = method.endswith("_physical")
        fr = FeatureRasters(stack, None if physical else s1_feat, valid_path, physical_only=physical)
        info = {}
        if method == "stage1_only":
            j = fr.feature_names.index("stage1_prob")
            score_fn = lambda X, j=j: X[:, j]  # noqa: E731
        else:
            X_pos, X_unl, info = build_training_set(fr, train_occ, valid, s2, eligible, rng, exclude_xy=exclude_xy)
            est = "logreg" if "logreg" in method else "xgb"
            params = dict(s2["xgb"]) if est == "xgb" else {"C": float(ev.get("logreg_C", 1.0))}
            model = PUBaggingProspector(params, est, int(s2["bags"]), float(s2["unlabeled_per_bag_ratio"]),
                                        seed + k, fr.feature_names).fit(X_pos, X_unl)
            score_fn = model.predict_proba
        cdf = _score_test_area(score_fn, fr, layout, k, valid_path, int(s2["predict_block"]))
        Xb = fr.extract(br, bc)
        ok = np.isfinite(Xb).all(1)
        cell_scores = pd.Series(np.asarray(score_fn(Xb[ok]), dtype=np.float64), index=bg[ok])
        best = cell_scores.groupby(level=0).max().reindex(evaluable)
        if best.isna().any():
            raise RuntimeError(f"Method {method} could not score evaluable clusters "
                               f"{best.index[best.isna()].tolist()} in fold {k}")
        area_eval = cdf.top_area_fraction(best.to_numpy()).astype(np.float64)
        area_all = np.r_[area_eval, np.full(len(unevaluable), np.inf)]
        realized = {f"realized_area_at_{int(round(100 * a))}pct_budget": cdf.realized_area_for_budget(a)
                    for a in AREA_LEVELS}
        m_eval = occurrence_metrics(area_eval)
        row = {**audit, **{f"excl_{k_}": v for k_, v in excl_audit.items()}, "method": method, "skipped": False,
               "stage1_init": (s1_init or {}).get("type") if method in NEEDS_STAGE1 else None,
               "test_area_cells": cdf.total, "obs_positive_vs_test_area_auc": tied_auc(cdf, cell_scores.to_numpy()),
               **m_eval, **realized, **info, "seconds": round(time.time() - t0, 1),
               "_area_eval": area_eval.tolist(), "_area_all": area_all.tolist(),
               "_eval_ids": evaluable.tolist(), "_all_ids": evaluable.tolist() + unevaluable.tolist()}
        LOG.info("  %-20s recall@5%% %.2f  recall@10%% %.2f  area80 %s  (%d evaluable clusters)", method,
                 m_eval["recall_at_top_5pct_area"], m_eval["recall_at_top_10pct_area"],
                 "n/a" if m_eval["area_for_80pct_recall"] is None else f"{m_eval['area_for_80pct_recall']:.3f}",
                 len(evaluable))
        rows.append(row)
    return rows


def _summarise(rows: List[dict], seeds: List[int], key: str, ids: str, n_boot: int) -> dict:
    per_seed = {}
    for seed in seeds:
        rr = [r for r in rows if r["seed"] == seed]
        a = np.concatenate([r[key] for r in rr])
        g = np.concatenate([[r["fold"]] * len(r[key]) for r in rr])
        per_seed[str(seed)] = occurrence_report(a, n_boot=n_boot, seed=seed, groups=g)
    keys = [k for k in per_seed[str(seeds[0])]["point"] if k.startswith(("recall_", "area_"))]
    across = {}
    for k in keys:
        vals = [per_seed[str(s)]["point"][k] for s in seeds]
        vals = [np.nan if v is None else v for v in vals]
        across[k] = {"mean": None if np.all(np.isnan(vals)) else float(np.nanmean(vals)),
                     "sd": None if np.all(np.isnan(vals)) else float(np.nanstd(vals))}
    return {"per_seed": per_seed, "across_seeds": across}


def _ssl_provenance(cfg: Config) -> dict:
    ev = cfg["evaluation"]
    mode = ev.get("ssl_mode", "transductive")
    if mode == "none":
        return {"mode": "none", "description": "Stage 1 trained from random initialisation in every fold"}
    if mode != "transductive":
        raise ValueError(f"evaluation.ssl_mode must be 'transductive' or 'none', got {mode!r}")
    from src.pipeline.stage1 import file_sha256

    p = cfg.path("ssl_weights")
    if not p.exists():
        raise FileNotFoundError(f"ssl_mode is transductive but {p} does not exist; run the ssl step or set ssl_mode: none")
    st = torch.load(p, map_location="cpu", weights_only=False)
    if list(st.get("channels", [])) != list(cfg["channels"]):
        raise ValueError("SSL checkpoint channels do not match the config")
    return {"mode": "transductive", "path": str(cfg["paths"]["ssl_weights"]), "sha256": file_sha256(p),
            "ssl_epoch": st.get("epoch"), "ssl_masked_mse": st.get("masked_mse"),
            "origin": "local masked pretraining on this AOI's unlabeled stack (src.models.ssl_pretrain)",
            "description": "Encoder saw unlabeled imagery of the whole AOI, including test areas, but no labels."}


def run(cfg: Config, device: Optional[str] = None) -> Path:
    ev, s2 = cfg["evaluation"], cfg["stage2"]
    methods = [m for m in ev.get("methods", ALL_METHODS) if m in ALL_METHODS]
    seeds = [int(s) for s in ev.get("seeds", [cfg["project"]["seed"]])]
    n_boot = int(ev.get("bootstrap", 2000))
    out_dir = cfg.path("reports_dir") / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    work = cfg.path("models_dir") / "evaluation"
    skipped_methods = {}
    if NEEDS_STAGE1 & set(methods) and not cfg.path("target_mask").exists():
        # Phase 1 of a real pilot: occurrences exist but no mapped target outlines yet.
        for m in sorted(NEEDS_STAGE1 & set(methods)):
            skipped_methods[m] = f"no target mask at {cfg['paths']['target_mask']}; Stage 1 cannot be trained"
            LOG.warning("Skipping %s: %s", m, skipped_methods[m])
        methods = [m for m in methods if m not in NEEDS_STAGE1]
    if not methods:
        raise RuntimeError("No evaluation method can run with the available inputs")
    ssl_info = _ssl_provenance(cfg) if NEEDS_STAGE1 & set(methods) else {"mode": "not used"}

    fr0 = FeatureRasters(cfg.path("stack"), None, cfg.path("valid_mask"), physical_only=True)
    res = abs(fr0.transform.a)
    valid = fr0.read_valid()
    occ, counts = prepare_occurrences(cfg, fr0, valid)
    LOG.info("Occurrence audit: %s", counts)
    footprint_m = max(float(ev["occurrence_buffer_m"]), int(s2["positive_buffer_px"]) * res)
    layout = FoldLayout.build(fr0.transform, fr0.shape, fr0.crs, occ, float(ev["block_m"]), int(ev["outer_folds"]),
                              float(ev["dead_zone_m"]), int(cfg["project"]["seed"]), footprint_m=footprint_m)
    layout.write(out_dir, occ)

    all_rows: List[dict] = []
    for seed in seeds:
        c = Config(copy.deepcopy(dict(cfg)))
        c["project"]["seed"] = seed
        seed_everything(seed)
        for k in range(layout.n_folds):
            all_rows += _evaluate_fold(c, k, layout, occ, valid, methods, work / f"seed{seed}" / f"fold{k}",
                                       device, seed, ssl_info)

    per_fold = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in all_rows])
    per_fold.to_csv(out_dir / "per_fold.csv", index=False)

    summary: Dict[str, dict] = {}
    curves, cluster_rows = [], []
    for method in methods:
        rs = [r for r in all_rows if r["method"] == method and not r.get("skipped")]
        if not rs:
            continue
        summary[method] = {
            "evaluable": _summarise(rs, seeds, "_area_eval", "_eval_ids", n_boot),
            "all_test_clusters": _summarise(rs, seeds, "_area_all", "_all_ids", n_boot),
            "obs_positive_vs_test_area_auc_mean": float(np.nanmean([r["obs_positive_vs_test_area_auc"] for r in rs])),
        }
        for seed in seeds:
            rr = [r for r in rs if r["seed"] == seed]
            for r in rr:
                cluster_rows += [{"method": method, "seed": seed, "fold": r["fold"], "cluster_id": cid,
                                  "area_needed": (None if not np.isfinite(v) else v),
                                  "status": "evaluable" if np.isfinite(v) else "unevaluable (counted as miss)"}
                                 for cid, v in zip(r["_all_ids"], r["_area_all"])]
            cv = success_curve(np.concatenate([r["_area_eval"] for r in rr]))
            curves += [{"method": method, "seed": seed, "area_flagged": x, "clusters_captured": y}
                       for x, y in zip(cv["area_flagged"], cv["clusters_captured"])]

    pd.DataFrame(curves).to_csv(out_dir / "success_curves.csv", index=False)
    pd.DataFrame(cluster_rows).to_csv(out_dir / "cluster_scores.csv", index=False)
    report = {
        "protocol": {
            "outer_folds": layout.n_folds, "block_m": layout.block_m, "dead_zone_m": layout.dead_zone_m,
            "cluster_radius_m": float(ev["cluster_radius_m"]), "occurrence_buffer_m": float(ev["occurrence_buffer_m"]),
            "cluster_footprint_m": footprint_m, "label_exclusion_m": float(s2["exclusion_radius_m"]),
            "seeds": seeds, "methods": methods, "methods_skipped": skipped_methods,
            "channels": list(cfg["channels"]), "commodity_filter": s2.get("commodity_filter"),
            "unit": "occurrence cluster; all blocks its footprint touches belong to one fold",
            "evaluable_population": "test clusters with >=1 valid buffer cell inside the test area; declared before "
                                    "scoring and identical for all methods; hit buffers restricted to the test area",
            "headline_population": "evaluable; all_test_clusters counts unevaluable clusters as misses",
            "area_reference": "all valid cells of the fold's test area; ties (same 2^-20 bin) included",
            "stage1": "retrained per fold; labels outside the train-eligible area and labels linked to test "
                      "clusters excluded, with their imagery zeroed",
            "auc_definition": "observed positive buffer cells vs all test-area cells, ties counted one half; "
                              "NOT a bound on true ROC-AUC",
            "ci": "95 % percentile bootstrap over clusters, stratified by fold, on the same population as the point estimate",
            "ssl": ssl_info,
        },
        "occurrence_audit": counts,
        "fold_audit": per_fold[[c for c in per_fold.columns if c in {
            "seed", "fold", "train_clusters", "test_clusters", "evaluable_test_clusters", "unevaluable_test_clusters",
            "dropped_dead_zone_clusters", "excl_linked_cells", "excl_linked_cells_outside_test_area", "stage1_init"}]]
        .drop_duplicates(subset=["seed", "fold"]).to_dict("records"),
        "results": summary,
    }
    path = out_dir / "evaluation_report.json"
    path.write_text(json.dumps(report, indent=2, default=lambda o: None))
    LOG.info("Evaluation report written: %s", path)
    for m, s_ in summary.items():
        a = s_["evaluable"]["across_seeds"]
        LOG.info("%-20s recall@5%% %.2f | recall@10%% %.2f | area80 %s", m, a["recall_at_top_5pct_area"]["mean"],
                 a["recall_at_top_10pct_area"]["mean"], a["area_for_80pct_recall"]["mean"])
    return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Nested geographic evaluation")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    run(load_config(args.config), args.device)


if __name__ == "__main__":
    main()
