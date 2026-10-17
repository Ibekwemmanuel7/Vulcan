# Mineral component in Lillie US

Imported on 17 October 2026 as a source snapshot from the local `ML_mineral` project. There was no source Git commit or configured remote. `SOURCE_MANIFEST.json` records the exact imported files, hashes and combined tree hash; the root `source-pins.json` references it. Integration notes and test-discovery configuration are maintained by Lillie US separately from that original snapshot.

## Included and excluded

Included: Python source, scripts and tests; YAML configurations; the Streamlit entry point, pipeline entry point, dependency list, setup script, original README and display configuration.

Excluded: source Git metadata, virtual environments, Geosoft installer, survey grids and other datasets, generated outputs, trained models, notebooks, source documentation and caches. The original source folder is unchanged. Source README references to local datasets or omitted documents are not a promise that these materials are included here.

## Setup and run

Use a separate environment for this component to isolate geospatial and machine-learning dependencies. Python 3.11 is the recommended starting point in its original documentation. From the Lillie US root:

```powershell
cd modules/ml_mineral
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch
python -m pip install -r requirements.txt
python -m pytest -q
```

Choose an appropriate CPU or CUDA PyTorch distribution for the target machine. The copied setup script is an alternative Windows setup workflow; inspect it before running. Dependencies are version-constrained where specified upstream, not fully locked.

For the synthetic demo, from the same component directory:

```powershell
python scripts/make_synthetic_demo.py
python run_pipeline.py --config configs/demo.yaml
streamlit run main.py
```

The demo creates synthetic input data and outputs locally. It demonstrates execution, not geological predictive skill. The integration does not automatically run downloads, training or the web server.

## Southern US adaptation

The existing examples concern Nigerian lithium pegmatites and REE prospectivity. Do not apply their AOIs, geological assumptions or reported evidence to Southern US customers. A US pilot needs commodity-specific geology, suitable survey and satellite inputs, independent occurrence/reference records, an appropriate coordinate reference system and geographic holdout evaluation. Existing CRS restrictions must be adapted and tested before using US coordinates.

Scores rank ground for investigation; they do not establish mineral ownership, economic reserves, grades, regulatory permission or deep-brine lithium concentrations.

## Updating the snapshot

Review intentional changes against the manifest. If refreshing from another source snapshot, regenerate file checksums and the combined tree hash and update `source-pins.json` together. Preserve exclusions and the original source provenance. If the upstream project is later published and packaged, it can instead become a commit-pinned external dependency.
