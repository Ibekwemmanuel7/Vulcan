"""Tests for the real-data pilot tooling (no network needed)."""
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from scripts.fetch_open_data import s1_median, s2_median, s2_offsets_for_times, tiles
from scripts.fetch_open_data import Progress
from scripts.import_ngsa_grid import grid_points, infer_cell, read_xyz
from src.models.attention_unet import MODALITY_GROUPS, modality_groups


def _item(dt, baseline, lon=7.7):
    return SimpleNamespace(datetime=dt, bbox=[lon - 0.5, 8.0, lon + 0.5, 9.0],
                           properties={"s2:processing_baseline": baseline})


def test_s2_offsets_follow_processing_baseline():
    items = [_item(datetime(2021, 12, 5, 9, 50, tzinfo=timezone.utc), "03.01"),
             _item(datetime(2025, 1, 10, 9, 50, tzinfo=timezone.utc), "05.11")]
    times = np.array(["2021-12-05T09:50", "2025-01-10T09:50"], dtype="datetime64[ns]")
    assert s2_offsets_for_times(items, times).tolist() == [0.0, 1000.0]


def test_missing_baseline_is_rejected():
    item = _item(datetime(2025, 1, 10, 9, 50, tzinfo=timezone.utc), "unknown")
    with pytest.raises(ValueError, match="baseline"):
        s2_offsets_for_times([item], np.array(["2025-01-10"], dtype="datetime64[D]"))


def test_resume_rejects_changed_query(tmp_path):
    p = tmp_path / "progress.json"
    Progress(p, "query-one").mark("0_0")
    assert Progress(p, "query-one").done == {"0_0"}
    with pytest.raises(ValueError, match="Resume settings changed"):
        Progress(p, "query-two")


def test_s2_median_masks_clouds_and_removes_offset():
    t, h, w = 3, 2, 2
    dn = {"B04": np.array([[[1500] * w] * h, [[3000] * w] * h, [[1600] * w] * h], dtype=np.uint16)}
    scl = np.array([[[4] * w] * h, [[9] * w] * h, [[5] * w] * h])   # middle scene is cloud
    med, cnt = s2_median(dn, scl, np.array([1000, 1000, 1000], np.float32))
    assert np.allclose(med["B04"], 0.055)    # median of 0.05 and 0.06, cloud excluded
    assert (cnt == 2).all()


def test_s2_median_all_cloudy_is_nan():
    med, cnt = s2_median({"B02": np.full((2, 1, 1), 1200, np.uint16)}, np.full((2, 1, 1), 8), np.zeros(2, np.float32))
    assert np.isnan(med["B02"]).all() and (cnt == 0).all()


def test_s1_median_ignores_nodata():
    a = np.array([[[0.02]], [[-32768.0]], [[0.04]]], np.float32)
    assert np.allclose(s1_median(a), 0.03)


def test_tiles_cover_grid_exactly():
    cover = np.zeros((2500, 3100), int)
    for wnd in tiles(3100, 2500, 1024):
        cover[wnd.row_off:wnd.row_off + wnd.height, wnd.col_off:wnd.col_off + wnd.width] += 1
    assert (cover == 1).all()


def test_xyz_import_roundtrip(tmp_path):
    xs, ys = np.meshgrid(np.arange(300000, 302000, 100.0), np.arange(950000, 951500, 100.0))
    v = 33000 + xs / 1000.0
    p = tmp_path / "sheet.xyz"
    with open(p, "w") as fh:
        fh.write("/ Geosoft XYZ export\n/ X Y TMI\n")
        for x, y, z in zip(xs.ravel(), ys.ravel(), v.ravel()):
            fh.write(f"{x:.1f} {y:.1f} {z:.3f}\n")
        fh.write("301000.0 950000.0 -1e32\n")  # dummy value must be ignored
    a = read_xyz(p)
    assert infer_cell(a) == 100.0
    arr, tr = grid_points(a, 100.0)
    assert arr.shape == (15, 20) and np.isfinite(arr).all()
    # value at a known node
    r, c = rasterio.transform.rowcol(tr, 300500.0, 951000.0)
    assert abs(arr[r, c] - (33000 + 300.5)) < 1e-3


def test_modality_groups_without_radiometrics():
    names = ["B2_blue", "B3_green", "B4_red", "B8_nir", "S1_VV_dB", "S1_VH_dB", "OH_alteration", "NDVI",
             "MAG_1VD", "DEM_slope"]
    g = modality_groups(names)
    assert g["geophysics"] == [8] and g["terrain"] == [9]
    assert MODALITY_GROUPS["geophysics"] == [8, 9]


def _write_geosoft(path, a_south_first, de, x0, y0, compressed=True, vpb=2):
    """Minimal Geosoft binary grid writer (test fixture). a_south_first[0] is the southern row."""
    import struct
    import zlib
    nv, ne = a_south_first.shape
    hdr = struct.pack("<5i", 1028 if compressed else 4, 2, ne, nv, 1) + struct.pack("<7d", de, de, x0, y0, 0, 0, 1)
    hdr = hdr.ljust(512, b"\0")
    data = a_south_first.astype("<f4")
    if not compressed:
        return path.write_bytes(hdr + data.tobytes())
    blocks = [zlib.compress(data[i:i + vpb].tobytes()) for i in range(0, nv, vpb)]
    nb = len(blocks)
    table_start = 512 + 16
    first = table_start + 12 * nb
    offs, pos = [], first
    for blk in blocks:
        offs.append(pos)
        pos += 16 + len(blk)
    body = struct.pack("<4i", -0x07182739, 2, nb, vpb) + struct.pack(f"<{nb}q", *offs) + \
        struct.pack(f"<{nb}i", *[16 + len(blk) for blk in blocks])
    out = hdr + body + b"".join(b"\0" * 16 + blk for blk in blocks)
    path.write_bytes(out)


@pytest.mark.parametrize("compressed", [True, False])
def test_geosoft_reader_orientation_and_origin(tmp_path, compressed):
    from scripts.import_ngsa_grid import geosoft_epsg, is_geosoft_grd, read_geosoft_grd
    south_first = np.arange(5 * 4, dtype=float).reshape(5, 4)     # row 0 = south
    south_first[0, 0] = -1e32                                     # Geosoft dummy
    p = tmp_path / "t.grd"
    _write_geosoft(p, south_first, 30.0, 1000.0, 2000.0, compressed)
    (tmp_path / "t.grd.xml").write_text('<projection wellknown_epsg="32632" name="WGS 84 / UTM zone 32N"/>')
    assert is_geosoft_grd(p) and geosoft_epsg(p) == "EPSG:32632"
    a, tr = read_geosoft_grd(p)
    assert a.shape == (5, 4)
    assert a[0, 3] == 19.0 and np.isnan(a[-1, 0])                 # north-up: last stored row on top
    assert (tr.c, tr.f) == (985.0, 2000.0 + 4 * 30 + 15)          # origin = centre of lower-left cell
    # centre of the south-west cell maps back to (x0, y0)
    x, y = rasterio.transform.xy(tr, 4, 0, offset="center")
    assert (x, y) == (1000.0, 2000.0)
