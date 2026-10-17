# Vulcan

Experimental mineral-prospectivity research framework, developed from GeoSentry / ML_mineral. Combines satellite, terrain and geophysical inputs with spatial evaluation and interpretable models. It is not a validated mineral-discovery product.

## Relationship to Lillie US

Vulcan is the reusable mineral research repository. [Lillie US](https://github.com/Ibekwemmanuel7/Lillie_US) consumes it as a Git submodule pinned to an exact commit at `modules/ml_mineral`. Southern US pilot requirements and customer workflows belong in Lillie US; reusable mineral processing and evaluation belong here.

## Setup

Use a dedicated environment. Python 3.11 is recommended as a starting point. From the Vulcan checkout:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch
python -m pip install -r requirements.txt
python -m pytest -q
```

Select the appropriate CPU or CUDA PyTorch distribution for your machine. The Windows `setup_workspace.ps1` is an alternative setup workflow; inspect it before use. Dependencies are not fully locked.

## Synthetic demo

```powershell
python scripts/make_synthetic_demo.py
python run_pipeline.py --config configs/demo.yaml
streamlit run main.py
```

Synthetic performance demonstrates execution only. Training, downloads and a full pipeline run are not performed by repository migration.

## Scientific scope

Existing configurations target Nigerian lithium pegmatites and REE prospectivity. Southern US use needs suitable data, commodity-specific geology, independent reference records, coordinate-system adaptation and geographic holdout evaluation. Prospectivity scores rank investigation targets; they do not establish grades, economic reserves, mineral ownership or extraction permission.

See [the original detailed framework documentation](docs/GEOSENTRY_ORIGINAL_README.md) for methods and limitations. Its original local paths and source-document references are historical context; datasets and those source documents are not included here.

## Provenance

The mineral-component history was extracted from Lillie US without changing its Python code, tests or scientific configurations. `provenance/ML_mineral_import_2026-10-17.json` preserves the original local import manifest. `SOURCE_MANIFEST.json` identifies the current selected source snapshot. Git commits are the integration pin.

Datasets, survey grids, installers, trained models, credentials, environments and generated outputs are excluded. The original local ML_mineral folder remains separate and unchanged.
