"""Select bundled GDAL/PROJ data without changing Windows settings."""
from importlib.util import find_spec
import os
from pathlib import Path


def configure_geospatial_runtime() -> None:
    spec = find_spec("rasterio")
    if spec is None or spec.origin is None:
        return
    package = Path(spec.origin).parent
    proj = package / "proj_data"
    gdal = package / "gdal_data"
    if (proj / "proj.db").is_file():
        # An inherited PostGIS path may contain an incompatible database.
        os.environ["PROJ_DATA"] = str(proj)
        os.environ["PROJ_LIB"] = str(proj)
    if gdal.is_dir():
        os.environ["GDAL_DATA"] = str(gdal)
