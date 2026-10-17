"""
GeoSentry-SOTA end-to-end orchestrator.

    python run_pipeline.py --config configs/geosentry.yaml                 # all steps
    python run_pipeline.py --config configs/geosentry.yaml --steps harmonize ssl
    python run_pipeline.py --config configs/demo.yaml                      # synthetic demo

Steps (in order): harmonize, labels, ssl, train, infer, prospect, evaluate

    train/infer/prospect  fit and apply the DEPLOYMENT model on all labels (no test score)
    evaluate              nested geographic evaluation: retrains Stage 1 and Stage 2 per
                          fold and compares baselines; the only source of performance numbers

Every run writes reports/run_manifest_<time>.json with input checksums, the
config, package versions and the steps executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import yaml

from src.config import Config, get_logger, load_config

LOG = get_logger("pipeline")
ALL_STEPS = ["harmonize", "labels", "ssl", "train", "infer", "prospect", "evaluate"]
PACKAGES = ["numpy", "scipy", "pandas", "rasterio", "pyproj", "torch", "scikit-learn", "xgboost", "shap", "streamlit"]


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write_manifest(cfg: Config, cfg_path: str, steps: list, seconds: dict) -> Path:
    inputs = {}
    for name, spec in cfg["layers"].items():
        p = cfg.resolve(spec["path"])
        inputs[name] = {"path": str(spec["path"]), "sha256": sha256(p) if p.exists() else None}
    for key in ("target_mask", "positives_csv"):
        p = cfg.path(key)
        inputs[key] = {"path": str(cfg["paths"][key]), "sha256": sha256(p) if p.exists() else None}
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    manifest = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_file": cfg_path,
        "config_sha256": hashlib.sha256(yaml.safe_dump(dict(cfg), sort_keys=True).encode()).hexdigest(),
        "config": dict(cfg),
        "steps": steps,
        "step_seconds": seconds,
        "inputs": inputs,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }
    out = cfg.path("reports_dir")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"run_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Run the GeoSentry-SOTA pipeline")
    ap.add_argument("--config", default=None)
    ap.add_argument("--steps", nargs="+", choices=ALL_STEPS, default=ALL_STEPS)
    ap.add_argument("--device", default=None, help="cpu or cuda (auto by default)")
    ap.add_argument("--reuse-stats", action="store_true", help="reuse saved scaling statistics")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    t_all = time.time()
    run_steps = [s for s in ALL_STEPS if s in args.steps]
    seconds = {}

    for step in run_steps:
        t0 = time.time()
        LOG.info("========== %s ==========", step.upper())
        if step == "harmonize":
            from src.utils.harmonizer import harmonize
            harmonize(cfg, args.reuse_stats)
        elif step == "labels":
            # A label raster on any grid/CRS named <target_mask>_raw.tif is snapped onto the stack grid.
            from src.utils.harmonizer import align_label_raster
            tgt = cfg.path("target_mask")
            raw = tgt.with_name(tgt.stem + "_raw" + tgt.suffix)
            if raw.exists():
                align_label_raster(raw, cfg.path("stack"), tgt)
                LOG.info("Aligned %s -> %s", raw.name, tgt.name)
            elif not tgt.exists():
                LOG.warning("No target mask found (%s); Stage 1 training will fail", tgt)
        elif step == "ssl":
            from src.models.ssl_pretrain import pretrain
            pretrain(cfg, args.device)
        elif step == "train":
            from src.pipeline.stage1 import train
            train(cfg, args.device)
        elif step == "infer":
            from src.pipeline.stage1 import infer
            infer(cfg, args.device)
        elif step == "prospect":
            from src.models.prospector_xgb import run
            run(cfg)
        elif step == "evaluate":
            from src.evaluation.nested_cv import run as evaluate
            evaluate(cfg, args.device)
        seconds[step] = round(time.time() - t0, 1)
        LOG.info("%s finished in %.1fs", step, seconds[step])
    manifest = write_manifest(cfg, str(args.config), run_steps, seconds)
    LOG.info("Pipeline complete in %.1f min; manifest %s", (time.time() - t_all) / 60, manifest.name)


if __name__ == "__main__":
    main()
