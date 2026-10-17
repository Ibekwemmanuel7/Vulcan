<#
.SYNOPSIS
    GeoSentry-SOTA workspace deployment for Windows PowerShell 5.1 / PowerShell 7.

.DESCRIPTION
    1. Creates the production folder layout (idempotent; never deletes anything).
    2. Writes requirements.txt and a data-leak-safe .gitignore (only if missing, or with -Force).
    3. Scaffolds Python package files (__init__.py, .gitkeep).
    4. Creates a virtual environment in .venv and installs PyTorch (CPU or CUDA) plus requirements.
    5. Initialises git and verifies that every core library imports.

.EXAMPLE
    # From the project folder (C:\Users\<you>\ML_mineral):
    powershell -ExecutionPolicy Bypass -File .\setup_workspace.ps1
    powershell -ExecutionPolicy Bypass -File .\setup_workspace.ps1 -Cuda          # NVIDIA GPU (CUDA 12.4 wheels)
    powershell -ExecutionPolicy Bypass -File .\setup_workspace.ps1 -SkipInstall   # folders and files only
#>
[CmdletBinding()]
param(
    [switch]$Cuda,
    [switch]$SkipInstall,
    [switch]$Force,
    [string]$PythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"
$Root = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
Set-Location $Root
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Write-Step([string]$msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok([string]$msg)   { Write-Host "    [ok] $msg" -ForegroundColor Green }
function Write-Skip([string]$msg) { Write-Host "    [--] $msg" -ForegroundColor DarkGray }

function Write-TextFile([string]$RelPath, [string]$Content) {
    $full = Join-Path $Root $RelPath
    if ((Test-Path $full) -and -not $Force) { Write-Skip "$RelPath exists (use -Force to overwrite)"; return }
    $dir = Split-Path $full -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    [System.IO.File]::WriteAllText($full, $Content.Replace("`r`n", "`n"), $Utf8NoBom)
    Write-Ok "wrote $RelPath"
}

# -----------------------------------------------------------------------------
Write-Step "Creating folder layout in $Root"
$folders = @(
    "configs",
    "data\raw", "data\processed", "data\labels", "data\demo",
    "outputs\models", "outputs\rasters", "outputs\reports",
    "src\utils", "src\models", "src\pipeline", "src\xai", "src\evaluation",
    "scripts", "tests", "notebooks", "docs", ".streamlit"
)
foreach ($f in $folders) {
    $p = Join-Path $Root $f
    if (-not (Test-Path $p)) { New-Item -ItemType Directory -Path $p | Out-Null; Write-Ok $f } else { Write-Skip $f }
}

# -----------------------------------------------------------------------------
Write-Step "Writing requirements.txt"
Write-TextFile "requirements.txt" @'
# GeoSentry-SOTA runtime dependencies (Python 3.10 - 3.12).
# PyTorch is installed separately by setup_workspace.ps1 so the CPU or CUDA build can be chosen.
numpy>=1.26,<3
scipy>=1.11
pandas>=2.1
pyyaml>=6.0
joblib>=1.3
rasterio>=1.3.10
pyproj>=3.6
scikit-learn>=1.4
xgboost>=2.0
shap>=0.45
matplotlib>=3.8
altair>=5.2
folium>=0.17
branca>=0.7
streamlit>=1.50
streamlit-folium>=0.22
# Open-data download (scripts/fetch_open_data.py)
pystac-client>=0.8
planetary-computer>=1.0
odc-stac>=0.3.10
pytest>=8.0
'@

# -----------------------------------------------------------------------------
Write-Step "Writing .gitignore (blocks rasters, labels, models and secrets from GitHub)"
Write-TextFile ".gitignore" @'
# ---- Python / environments ----
.venv/
venv/
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ipynb_checkpoints/

# ---- ALL data and outputs stay local (licensed NGSA / survey data, client labels) ----
data/**
!data/**/
!data/**/.gitkeep
outputs/**
!outputs/**/
!outputs/**/.gitkeep

# ---- Geospatial and model binaries anywhere in the tree ----
*.tif
*.tiff
*.vrt
*.img
*.jp2
*.nc
*.h5
*.hdf
*.grd
*.ers
*.shp
*.shx
*.dbf
*.prj
*.cpg
*.gpkg
*.geojson
*.kml
*.kmz
*.zip
*.pt
*.pth
*.ckpt
*.onnx
*.joblib
*.pkl
*.npy
*.npz

# ---- Secrets and credentials ----
.env
.env.*
*.key
*.pem
credentials*.json
service-account*.json
.streamlit/secrets.toml
configs/*secret*

# ---- OS / editor ----
Thumbs.db
desktop.ini
.DS_Store
.vscode/
.idea/
'@

Write-TextFile ".streamlit\config.toml" @'
[theme]
base = "light"
primaryColor = "#eb6834"
font = "serif"

[server]
maxUploadSize = 2048
'@

# -----------------------------------------------------------------------------
Write-Step "Scaffolding package files"
$initFiles = @("src\__init__.py", "src\utils\__init__.py", "src\models\__init__.py",
               "src\pipeline\__init__.py", "src\xai\__init__.py", "src\evaluation\__init__.py",
               "tests\__init__.py")
foreach ($f in $initFiles) {
    $p = Join-Path $Root $f
    if (-not (Test-Path $p)) { [System.IO.File]::WriteAllText($p, "", $Utf8NoBom); Write-Ok $f } else { Write-Skip $f }
}
$keep = @("data\raw", "data\processed", "data\labels", "outputs\models", "outputs\rasters", "outputs\reports", "notebooks")
foreach ($k in $keep) {
    $p = Join-Path $Root "$k\.gitkeep"
    if (-not (Test-Path $p)) { [System.IO.File]::WriteAllText($p, "", $Utf8NoBom) }
}
$core = @("main.py", "run_pipeline.py", "configs\geosentry.yaml", "src\utils\harmonizer.py",
          "src\models\attention_unet.py", "src\models\ssl_pretrain.py", "src\models\prospector_xgb.py",
          "src\pipeline\stage1.py", "src\xai\gradcam.py", "src\evaluation\folds.py",
          "src\evaluation\metrics.py", "src\evaluation\nested_cv.py", "scripts\make_synthetic_demo.py")
$missing = $core | Where-Object { -not (Test-Path (Join-Path $Root $_)) }
if ($missing) { Write-Warning ("Source files not found (copy them into the project): " + ($missing -join ", ")) }
else { Write-Ok "all core source files present" }

if ($SkipInstall) { Write-Step "Skipping environment install (-SkipInstall)"; return }

# -----------------------------------------------------------------------------
Write-Step "Creating virtual environment (.venv)"
$venvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    $created = $false
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py "-$PythonVersion" -m venv .venv
        $created = ($LASTEXITCODE -eq 0)
    }
    if (-not $created) {
        if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
            throw "Python $PythonVersion not found. Install it from python.org (tick 'Add to PATH') and rerun."
        }
        Write-Warning "py launcher for $PythonVersion unavailable; using 'python' on PATH"
        & python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
    }
    Write-Ok ".venv created"
} else { Write-Skip ".venv exists" }

& $venvPy -m pip install --upgrade pip wheel setuptools --quiet
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

Write-Step ("Installing PyTorch (" + $(if ($Cuda) { "CUDA 12.4" } else { "CPU" }) + ")")
$torchIndex = if ($Cuda) { "https://download.pytorch.org/whl/cu124" } else { "https://download.pytorch.org/whl/cpu" }
& $venvPy -m pip install torch --index-url $torchIndex
if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed" }

Write-Step "Installing requirements.txt"
& $venvPy -m pip install -r (Join-Path $Root "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "requirements install failed" }

# -----------------------------------------------------------------------------
Write-Step "Initialising git"
if (Get-Command git -ErrorAction SilentlyContinue) {
    if (-not (Test-Path (Join-Path $Root ".git"))) { & git init -q; Write-Ok "git repository initialised" }
    else { Write-Skip "git repository exists" }
} else { Write-Warning "git not found; skipping" }

Write-Step "Verifying imports"
& $venvPy -c "import torch, rasterio, pyproj, scipy, xgboost, shap, sklearn, streamlit, streamlit_folium; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available()); print('rasterio', rasterio.__version__, '| GDAL', rasterio.__gdal_version__)"
if ($LASTEXITCODE -ne 0) { throw "Import check failed" }

Write-Host "`nGeoSentry-SOTA workspace ready." -ForegroundColor Green
Write-Host "Next:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python scripts\make_synthetic_demo.py"
Write-Host "  python run_pipeline.py --config configs\demo.yaml"
Write-Host "  streamlit run main.py"
