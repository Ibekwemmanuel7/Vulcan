"""Verify startup under inherited PostGIS environment variables."""
import os
import subprocess
import sys
from pathlib import Path


def test_project_startup_overrides_external_proj_database():
    env = os.environ.copy()
    env.update(PROJ_LIB="C:/external-postgis/proj", PROJ_DATA="C:/external-postgis/proj",
               GDAL_DATA="C:/external-postgis/gdal")
    subprocess.run(
        [sys.executable, "-c",
         "import src; from rasterio.crs import CRS; from pyproj import Transformer; "
         "assert CRS.from_epsg(32632).to_epsg() == 32632; "
         "x,y=Transformer.from_crs(4326,32632,always_xy=True).transform(8.6,8.9); "
         "assert 100000 < x < 900000 and 0 < y < 2000000"],
        cwd=Path(__file__).resolve().parents[1], env=env, check=True,
        capture_output=True, text=True,
    )
