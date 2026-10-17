# GeoSentry-SOTA

Experimental multimodal Positive-Unlabeled (PU) prospectivity framework for rare earth elements (monazite, xenotime) and lithium pegmatites in deeply weathered, vegetated terrain of the Nigerian Basement Complex.

Pipeline: multimodal harmonization (10 m UTM grid) -> local self-supervised encoder pretraining -> spatial-channel Attention U-Net (Focal-Dice) -> bagged PU XGBoost -> TreeSHAP, Grad-CAM and integrated-gradient diagnostics -> nested geographic evaluation against baselines -> Streamlit portal.

Status: the software runs end to end on synthetic data. It has not been validated on real geology. Until it beats simpler baselines on independent geological regions, describe it as an experimental framework, not a validated discovery system.

## 1. Install (Windows PowerShell)

```powershell
cd C:\Users\taylo\ML_mineral
powershell -ExecutionPolicy Bypass -File .\setup_workspace.ps1          # CPU
powershell -ExecutionPolicy Bypass -File .\setup_workspace.ps1 -Cuda    # NVIDIA GPU
.\.venv\Scripts\Activate.ps1
python -m pytest -q
```

Use Python 3.11 (3.10 to 3.12 are supported). If `pytest` is missing, the install step did not finish; rerun the setup script.

## 2. Run the synthetic demo first

```powershell
python scripts\make_synthetic_demo.py
python run_pipeline.py --config configs\demo.yaml      # about 8 minutes on CPU
streamlit run main.py
```

The demo writes its targets directly into the thorium, SWIR, SAR and terrain inputs. Success on it only shows that the software runs. The portal shows a banner when the demo configuration is loaded.

## 3. Run on real data

Put these files in `data\raw\` (any CRS, any pixel size, GeoTIFF):

| File | Content | Notes |
|---|---|---|
| `S2_B02/B03/B04/B08.tif` | Sentinel-2 L2A 10 m bands | Dry-season, cloud-masked median composite |
| `S2_B11.tif`, `S2_B12.tif` | Sentinel-2 L2A 20 m SWIR | Used for the B11/B12 ratio |
| `S1_VV.tif`, `S1_VH.tif` | Sentinel-1 GRD, terrain corrected | Linear power (`to_db: true`) or dB |
| `MAG_TMI.tif` | Airborne total magnetic intensity (NGSA) | Set `compute_1vd: false` if the grid is already 1VD |
| `RAD_eTh.tif` | Airborne radiometric equivalent thorium (ppm) | |
| `DEM.tif` | Copernicus GLO-30 or SRTM | Converted to slope |

Labels in `data\labels\`:

* `target_mask_raw.tif`: 1 = mapped target, 0 = confirmed background, 255 = unknown. Any grid; it is snapped onto the stack. Mark unexplored ground as 255, not 0.
* `known_occurrences.csv`: columns `lon`, `lat` (WGS 84) or `easting`, `northing` (project CRS), plus `commodity` and `name`. Record the source of each occurrence. Use 30 or more independent clusters.

Run lithium pegmatites and REE as separate mineral systems with `stage2.commodity_filter` (for example `"Li"` or `"REE"`), each with its own config file and output folders.

```powershell
python run_pipeline.py --config configs\geosentry.yaml                       # all steps
python run_pipeline.py --config configs\geosentry.yaml --steps evaluate      # evaluation only
python -m src.xai.gradcam --config configs\geosentry.yaml --row 1200 --col 3400
streamlit run main.py
```

For Sentinel-2 L2A data with processing baseline 04.00 or later, add `scale: 0.0001` and `offset: -0.1` to each S2 layer so band ratios use true reflectance.

## 3a. Real-data pilot: Nasarawa-Keffi lithium pegmatites

The first real pilot is set up in `configs/pilot_nasarawa_keffi_li.yaml`. The plan, data inventory and labelling rules are in `docs/Pilot_Plan_Nasarawa_Keffi_Li.docx`.

* `scripts/fetch_open_data.py` downloads the Sentinel-2 dry-season median, Sentinel-1 RTC and Copernicus DEM from the Microsoft Planetary Computer, tile by tile, with resume and provenance files.
* `scripts/import_ngsa_grid.py` mosaics NGSA magnetic sheets (Geosoft-readable grids or ASCII XYZ) into the project CRS. Pass the source CRS explicitly.
* `scripts/build_occurrence_csv.py` exports verified rows of `data/pilot_nk/labels/occurrence_register.csv` to the pipeline.
* Channels are configurable. The pilot runs without radiometrics (10 channels) until NGSA radiometric grids are obtained.
* Without a target mask, `evaluate` runs the physical PU baselines and skips the Stage 1 methods. The report records this.

## 4. Steps

| Step | What it does |
|---|---|
| `harmonize` | Warps all inputs to the project grid, derives NDVI, B11/B12, magnetic 1VD and slope, scales, and saves scaling statistics |
| `labels` | Snaps `target_mask_raw.tif` onto the stack grid |
| `ssl` | Masked multimodal pretraining on unlabeled imagery of this AOI |
| `train`, `infer` | Deployment Stage 1 model on all labels, applied to the whole grid |
| `prospect` | Deployment Stage 2 model on all occurrences, prospectivity grid, SHAP, ranked targets. Produces no performance score |
| `evaluate` | Nested geographic evaluation. The only source of performance numbers |

Every run writes `outputs\reports\run_manifest_<time>.json` with input SHA-256 checksums, the full config, package versions and the steps run.

## 5. Evaluation protocol

`evaluate` (`src/evaluation/nested_cv.py`) applies one protocol to every method:

1. Occurrences within `evaluation.cluster_radius_m` are merged into clusters. A cluster is the independent unit.
2. The AOI is cut into `block_m` blocks. A cluster's footprint is the bounding box of its members, expanded by the largest positional or label buffer. Every block that footprint touches is tied to the cluster. Tied blocks form one group, and each group goes to a single fold, so a cluster never straddles folds. `fold_map.tif` and `fold_assignment.csv` are written for inspection.
3. For each fold, the test area is the fold's blocks. Training uses only cells farther than `dead_zone_m` from the test area. A training cluster that touches the test area or dead zone is dropped whole.
4. Stage 1 labels are also removed wherever they are linked to a test cluster: within `stage2.exclusion_radius_m` of a member, or on a mapped outline that touches that radius, even where the outline runs out of the test area. Their imagery is zeroed too. The report audits how many linked cells lay outside the test area.
5. Stage 1 is retrained per fold. Stage 2 is fitted on that fold's Stage 1 features. Unlabeled samples keep `exclusion_radius_m` away from every known occurrence.
6. Before any model is scored, the evaluable population is declared once per fold and is the same for every method. It is the test clusters with at least one valid positional-buffer cell inside the test area. Hit buffers use test-area cells only. If a method cannot score an evaluable cluster, the run stops.
7. A cluster counts as captured at area fraction *a* if its best buffer score lies in the top *a* of the fold's test area (ties included). Area budgets use the same rank rule, so a tie block larger than the budget gives an empty selection, not a full one.
8. Methods compared on the same folds: `two_stage`, `stage1_only`, `pu_xgb_physical`, `pu_logreg_physical`.

Each method is reported on two populations:

* `evaluable` is the headline figure, conditional on the clusters the data can test.
* `all_test_clusters` is the conservative figure, with unevaluable clusters counted as misses.

Both populations get recall at 2/5/10/20 % of test area and the area needed for 80 % recall (`null` if 80 % is never reached), with 95 % cluster-bootstrap intervals on the same population. The report also includes per-fold values, the count audit, realized area per budget, and the observed-positive vs test-area AUC (ties count one half). That AUC is not a bound on the true ROC-AUC.

SSL provenance: with `ssl_mode: transductive`, the SSL checkpoint must exist and match the config channels, or the run stops. The report records its SHA-256 and the initialization each fold actually used. The encoder sees unlabeled imagery of the test areas, but never their labels. Set `ssl_mode: none` to measure transfer without it.

Add seeds to `evaluation.seeds` to measure seed variance. Stage 1 validation scores in `stage1_metrics.json` are tuning scores; do not quote them as test performance.

## 6. Outputs

* `prospectivity.tif` band 1: score from 0 to 1. Bags are averaged in log-odds. The score ranks ground; it is not the probability that an economic deposit exists.
* Band 2 `top_area_fraction`: the share of all valid AOI cells that score at least as high (ties included). 0.03 means "inside the top 3 % of the AOI". The portal filters on this band and reports the realized area flagged.
* `reports\shap_groups.csv`, `shap_diagnostics.json`: TreeSHAP over all bags in the log-odds of the deployed score, with a completeness check and stability between bag halves. These describe the model, not mineralisation mechanisms.
* `stage1_attribution_*.json`: integrated gradients against blur, mean and zero baselines, with completeness errors, plus modality ablation. Attention-gate maps are internal network weights.
* `reports\evaluation\`: `evaluation_report.json`, `per_fold.csv`, `cluster_scores.csv`, `success_curves.csv`, fold map.

## 7. Known limits

* The 10 m grid is an alignment choice. SWIR is 20 m and airborne surveys are much coarser, so interpolation does not add geological detail.
* B11/B12 is a broad Al-OH proxy, not mineral identification. eTh is not REE grade. SAR backscatter does not detect pegmatites directly.
* Known workings partly record where people looked. Check accessibility and exploration history before reading scores as geology.
* Supported CRSs are EPSG:32631, 32632 and 32633 (`project.crs`). The harmonizer warns when the AOI centroid is outside the chosen zone.
* Memory is not fully bounded. Harmonization holds one full-AOI band, patch planning reads full masks, the magnetic FFT pads the AOI, and `find_targets` reads both prospectivity bands. Profile realistic extents first. Tiling needs halos and seam checks.
* Before adding a larger architecture or a pretrained Earth-observation encoder (Prithvi-EO-2.0, TerraMind, OlmoEarth), first show that the current model beats the physical-feature baselines on real folds.
