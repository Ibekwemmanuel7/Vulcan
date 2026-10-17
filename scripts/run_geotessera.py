"""Run GeoTessera with this environment's bundled GDAL/PROJ data.

Use .venv_tessera/Scripts/python.exe scripts/run_geotessera.py download ...
All arguments are passed unchanged to the installed GeoTessera CLI.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src  # Select bundled geospatial data before GeoTessera imports rasterio.


if __name__ == "__main__":
    from geotessera.cli import main

    main()
