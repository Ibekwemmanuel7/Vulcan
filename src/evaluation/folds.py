"""
Occurrence handling and geographic fold layout shared by every stage.

Rules enforced here (review findings 1, 2 and 4):
  * Occurrences closer than `cluster_radius_m` are merged into one cluster.
    A cluster is the independent evaluation unit and is never split across folds.
  * The AOI is cut into square blocks; whole blocks are assigned to folds,
    balancing the number of clusters per fold.
  * For fold k, TEST = cells in fold-k blocks. TRAIN-ELIGIBLE = cells outside
    fold k AND farther than `dead_zone_m` from it. Cells in between are used
    by nothing. A training cluster with any member inside the test area or the
    dead zone is dropped whole.
  * The layout is written to disk (fold map GeoTIFF and cluster table) so it can
    be inspected before any score is reported.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from rasterio.transform import from_origin
from rasterio.windows import Window
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from src.config import get_logger

LOG = get_logger("folds")


# =============================================================================
# Occurrences
# =============================================================================
def load_occurrences(csv_path: Path, crs, transform, shape: Tuple[int, int],
                     commodity_filter: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Read occurrences (lon/lat WGS 84, or easting/northing in the grid CRS), apply an
    optional case-insensitive substring filter on the `commodity` column, and keep
    those inside the grid. Returns the table and a count audit.
    """
    df = pd.read_csv(csv_path)
    counts = {"in_file": int(len(df))}
    low = {c.lower(): c for c in df.columns}
    if commodity_filter:
        if "commodity" not in low:
            raise ValueError("commodity_filter is set but the occurrence CSV has no 'commodity' column")
        df = df[df[low["commodity"]].astype(str).str.contains(commodity_filter, case=False, regex=False)]
        counts["after_commodity_filter"] = int(len(df))
    if "lon" in low and "lat" in low:
        tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x, y = tr.transform(df[low["lon"]].to_numpy(float), df[low["lat"]].to_numpy(float))
    elif "easting" in low and "northing" in low:
        x, y = df[low["easting"]].to_numpy(float), df[low["northing"]].to_numpy(float)
    else:
        raise ValueError("Occurrence CSV needs lon/lat (WGS 84) or easting/northing columns")
    x, y = np.asarray(x, float), np.asarray(y, float)
    r, c = rasterio.transform.rowcol(transform, x, y)
    r, c = np.asarray(r, int), np.asarray(c, int)
    inside = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
    df = df.loc[inside].reset_index(drop=True).copy()
    df["x"], df["y"], df["row"], df["col"] = x[inside], y[inside], r[inside], c[inside]
    df["occ_id"] = np.arange(len(df))
    counts["inside_aoi"] = int(len(df))
    return df, counts


def cluster_occurrences(df: pd.DataFrame, radius_m: float) -> pd.DataFrame:
    """Single-linkage clustering: occurrences within `radius_m` of each other share a cluster."""
    df = df.copy()
    n = len(df)
    if n == 0:
        df["cluster_id"] = []
        return df
    pairs = np.array(sorted(cKDTree(df[["x", "y"]].to_numpy()).query_pairs(radius_m)))
    if pairs.size:
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        _, labels = connected_components(g, directed=False)
    else:
        labels = np.arange(n)
    df["cluster_id"] = labels
    return df


def drop_invalid_occurrences(df: pd.DataFrame, valid: np.ndarray, buffer_px: int) -> Tuple[pd.DataFrame, int]:
    """Drop occurrences with no valid data cell inside their positional buffer."""
    keep = []
    H, W = valid.shape
    for r, c in zip(df["row"], df["col"]):
        r0, r1 = max(0, r - buffer_px), min(H, r + buffer_px + 1)
        c0, c1 = max(0, c - buffer_px), min(W, c + buffer_px + 1)
        keep.append(bool(valid[r0:r1, c0:c1].any()))
    keep = np.array(keep, bool)
    return df.loc[keep].reset_index(drop=True), int((~keep).sum())


def buffer_cells(rows: np.ndarray, cols: np.ndarray, ids: np.ndarray, radius_px: int, shape: Tuple[int, int],
                 valid: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand points into disc-shaped cell sets. Each cell is returned once per id."""
    offs = np.array([(dr, dc) for dr in range(-radius_px, radius_px + 1) for dc in range(-radius_px, radius_px + 1)
                     if dr * dr + dc * dc <= radius_px * radius_px], dtype=int)
    rr = (np.asarray(rows)[:, None] + offs[None, :, 0]).ravel()
    cc = (np.asarray(cols)[:, None] + offs[None, :, 1]).ravel()
    gg = np.repeat(np.asarray(ids), len(offs))
    H, W = shape
    ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    rr, cc, gg = rr[ok], cc[ok], gg[ok]
    if valid is not None:
        ok = valid[rr, cc]
        rr, cc, gg = rr[ok], cc[ok], gg[ok]
    key = np.unique(np.c_[gg, rr, cc], axis=0)
    return key[:, 1], key[:, 2], key[:, 0]


# =============================================================================
# Fold layout
# =============================================================================
@dataclass
class FoldLayout:
    """Block-based geographic folds with dead zones, evaluated on a coarse lattice."""

    transform: object
    shape: Tuple[int, int]
    crs: object
    block_m: float
    n_folds: int
    dead_zone_m: float
    coarse_res: float
    fold_coarse: np.ndarray            # (h, w) int8 fold id per coarse cell
    dist_coarse: np.ndarray            # (K, h, w) float32 distance (m) to fold k
    cluster_fold: Dict[int, int]

    # ---- construction --------------------------------------------------------
    @classmethod
    def build(cls, transform, shape, crs, occ: pd.DataFrame, block_m: float, n_folds: int,
              dead_zone_m: float, seed: int, footprint_m: float = 0.0) -> "FoldLayout":
        """
        Every block touched by a cluster's footprint (bounding box of its members
        expanded by `footprint_m`, the largest positional / label buffer) is tied to
        that cluster. Blocks tied through shared clusters are merged into one group,
        and each group goes to a single fold. A cluster, its buffers and the
        segments between its members therefore lie entirely inside one fold.
        """
        res = abs(transform.a)
        coarse = max(res, min(block_m / 20.0, dead_zone_m / 4.0 if dead_zone_m > 0 else block_m / 20.0))
        f = max(1, int(round(coarse / res)))
        coarse = f * res
        h, w = int(np.ceil(shape[0] / f)), int(np.ceil(shape[1] / f))
        x0, y0 = transform.c, transform.f
        cx = x0 + (np.arange(w) + 0.5) * coarse
        cy = y0 - (np.arange(h) + 0.5) * coarse
        bx = np.floor((cx - x0) / block_m).astype(int)
        by = np.floor((y0 - cy) / block_m).astype(int)
        nbx, nby = int(bx.max() + 1), int(by.max() + 1)
        block_id = by[:, None] * nbx + bx[None, :]

        # ---- blocks touched by each cluster footprint --------------------------
        parent: Dict[int, int] = {}

        def find(a: int) -> int:
            parent.setdefault(a, a)
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        cluster_blocks: Dict[int, List[int]] = {}
        for cid, g in occ.groupby("cluster_id"):
            xmin, xmax = g["x"].min() - footprint_m, g["x"].max() + footprint_m
            ymin, ymax = g["y"].min() - footprint_m, g["y"].max() + footprint_m
            c0, c1 = [int(np.clip(np.floor((v - x0) / block_m), 0, nbx - 1)) for v in (xmin, xmax)]
            r0, r1 = [int(np.clip(np.floor((y0 - v) / block_m), 0, nby - 1)) for v in (ymax, ymin)]
            blocks = [r * nbx + c for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
            cluster_blocks[int(cid)] = blocks
            for b in blocks:
                union(blocks[0], b)
        group_of_cluster = {cid: find(bl[0]) for cid, bl in cluster_blocks.items()}
        load: Dict[int, int] = {}
        for gid in group_of_cluster.values():
            load[gid] = load.get(gid, 0) + 1
        n_groups = len(load)
        k = int(min(n_folds, n_groups))
        if k < 2:
            raise RuntimeError(f"All occurrence clusters form {n_groups} connected block group(s) at {block_m:g} m; "
                               "geographic evaluation needs at least 2. Reduce evaluation.block_m.")
        if k < n_folds:
            LOG.warning("Only %d independent block groups contain clusters; using %d folds instead of %d",
                        n_groups, k, n_folds)
        if max(load.values()) > 0.5 * len(group_of_cluster):
            LOG.warning("One block group holds %d of %d clusters; folds will be unbalanced",
                        max(load.values()), len(group_of_cluster))

        # ---- assign groups to folds (balance clusters), then empty blocks (balance area)
        rng = np.random.default_rng(seed)
        fold_of_group: Dict[int, int] = {}
        fold_clusters = np.zeros(k, int)
        for gid in sorted(load, key=lambda g_: (-load[g_], rng.random())):
            j = int(np.argmin(fold_clusters + rng.random(k) * 1e-3))
            fold_of_group[gid] = j
            fold_clusters[j] += load[gid]
        fold_of_block: Dict[int, int] = {b: fold_of_group[find(b)] for b in parent}
        counts = np.bincount(block_id.ravel(), minlength=nbx * nby)
        fold_area = np.zeros(k)
        for b, j in fold_of_block.items():
            fold_area[j] += counts[b]
        empty = [int(b) for b in np.unique(block_id) if int(b) not in fold_of_block]
        rng.shuffle(empty)
        for b in empty:
            j = int(np.argmin(fold_area))
            fold_of_block[b] = j
            fold_area[j] += counts[b]

        lut = np.zeros(nbx * nby, dtype=np.int8)
        for b, j in fold_of_block.items():
            lut[b] = j
        fold_coarse = lut[block_id]
        dist = np.stack([ndimage.distance_transform_edt(fold_coarse != j).astype(np.float32) * coarse
                         for j in range(k)])
        cluster_fold = {cid: int(fold_of_group[gid]) for cid, gid in group_of_cluster.items()}
        lay = cls(transform, shape, crs, block_m, k, dead_zone_m, coarse, fold_coarse, dist, cluster_fold)
        LOG.info("Fold layout: %d folds, %g m blocks, %d block groups, %g m dead zone, clusters per fold %s",
                 k, block_m, n_groups, dead_zone_m, fold_clusters.tolist())
        return lay

    # ---- queries ---------------------------------------------------------------
    def _coarse_idx(self, rows: np.ndarray, cols: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f = int(round(self.coarse_res / abs(self.transform.a)))
        return np.asarray(rows) // f, np.asarray(cols) // f

    def fold_at(self, rows, cols) -> np.ndarray:
        i, j = self._coarse_idx(rows, cols)
        return self.fold_coarse[i, j]

    def train_eligible(self, rows, cols, k: int) -> np.ndarray:
        i, j = self._coarse_idx(rows, cols)
        return (self.fold_coarse[i, j] != k) & (self.dist_coarse[k, i, j] > self.dead_zone_m)

    def window_masks(self, win: Window, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """(test, train_eligible) boolean masks for a raster window."""
        rows = np.arange(win.row_off, win.row_off + win.height)
        cols = np.arange(win.col_off, win.col_off + win.width)
        i, j = self._coarse_idx(rows, cols)
        fold = self.fold_coarse[np.ix_(i, j)]
        dist = self.dist_coarse[k][np.ix_(i, j)]
        return fold == k, (fold != k) & (dist > self.dead_zone_m)

    def split_clusters(self, occ: pd.DataFrame, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(train cluster ids, test cluster ids, dropped cluster ids) for fold k."""
        elig = self.train_eligible(occ["row"].to_numpy(), occ["col"].to_numpy(), k)
        ok = pd.Series(elig).groupby(occ["cluster_id"].to_numpy()).all()
        test = np.array([c for c, f in self.cluster_fold.items() if f == k])
        train = np.array([c for c in ok.index[ok.to_numpy()] if self.cluster_fold[int(c)] != k])
        dropped = np.array([c for c in ok.index if c not in set(train) and c not in set(test)])
        return train, test, dropped

    # ---- outputs --------------------------------------------------------------
    def write(self, out_dir: Path, occ: pd.DataFrame) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        tr = from_origin(self.transform.c, self.transform.f, self.coarse_res, self.coarse_res)
        with rasterio.open(out_dir / "fold_map.tif", "w", driver="GTiff", width=self.fold_coarse.shape[1],
                           height=self.fold_coarse.shape[0], count=1, dtype="int16", crs=self.crs,
                           transform=tr, nodata=-1, compress="deflate") as d:
            d.write(self.fold_coarse.astype(np.int16), 1)
        t = occ.copy()
        t["fold"] = t["cluster_id"].map(self.cluster_fold)
        t.to_csv(out_dir / "fold_assignment.csv", index=False)


def write_fold_training_rasters(layout: FoldLayout, k: int, valid_path: Path, target_path: Path,
                                out_valid: Path, out_target: Path, test_rows: Optional[np.ndarray] = None,
                                test_cols: Optional[np.ndarray] = None, exclusion_px: int = 0,
                                strip: int = 1024) -> Dict[str, int]:
    """
    Fold-specific Stage 1 inputs. A cell is usable for training only if it is
    train-eligible for fold k AND not linked to a test cluster. Linked means within
    `exclusion_px` of a test-cluster member, or part of a mapped target outline
    (8-connected component of label 1) that touches such a disc. Unusable cells get
    label 255 and valid 0, so their imagery is also zeroed in training patches.
    Returns an audit of excluded cells.
    """
    with rasterio.open(target_path) as ts:
        tgt_full = ts.read(1)
    H, W = tgt_full.shape
    linked = np.zeros((H, W), dtype=bool)
    if test_rows is not None and len(test_rows):
        disc = np.zeros((H, W), dtype=bool)
        r_, c_, _ = buffer_cells(np.asarray(test_rows), np.asarray(test_cols), np.zeros(len(test_rows), int),
                                 int(exclusion_px), (H, W))
        disc[r_, c_] = True
        comp, n = ndimage.label(tgt_full == 1, structure=np.ones((3, 3)))
        if n:
            touched = np.unique(comp[disc & (comp > 0)])
            linked = np.isin(comp, touched[touched > 0])
        linked |= disc
        del comp
    audit = {"linked_cells": int(linked.sum()), "linked_cells_outside_test_area": 0}
    with rasterio.open(valid_path) as vs:
        pv = vs.profile.copy()
        with rasterio.open(target_path) as ts:
            pt = ts.profile.copy()
        out_valid.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_valid, "w", **pv) as dv, rasterio.open(out_target, "w", **pt) as dt:
            for r0 in range(0, H, strip):
                win = Window(0, r0, W, min(strip, H - r0))
                test, elig = layout.window_masks(win, k)
                lk = linked[r0:r0 + win.height]
                audit["linked_cells_outside_test_area"] += int((lk & ~test).sum())
                use = elig & ~lk
                v = vs.read(1, window=win)
                t = tgt_full[r0:r0 + win.height].copy()
                dv.write((v.astype(bool) & use).astype(v.dtype), 1, window=win)
                t[~use] = 255
                dt.write(t, 1, window=win)
    return audit


def check_same_grid(paths: List[Path]) -> None:
    """Raise if rasters differ in CRS, shape or transform."""
    ref = None
    for p in paths:
        with rasterio.open(p) as s:
            meta = (CRS.from_user_input(s.crs) if s.crs else None, s.shape, s.transform)
        if ref is None:
            ref, ref_p = meta, p
            continue
        if meta[0] != ref[0]:
            raise ValueError(f"CRS mismatch: {p} ({meta[0]}) vs {ref_p} ({ref[0]})")
        if meta[1] != ref[1] or not meta[2].almost_equals(ref[2]):
            raise ValueError(f"Grid mismatch: {p} vs {ref_p}")
