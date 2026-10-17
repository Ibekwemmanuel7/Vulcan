"""
Evaluation metrics with explicit population and tie conventions.

ScoreCDF        Exact-count histogram of scores over a defined population of cells
                (the whole valid AOI, or one fold's test area). Resolution 2**20 bins
                on [0, 1]. Counts are exact per bin; scores that fall in the same bin
                (width ~1e-6) are treated as ties. Tie convention: a cell's
                `top_area_fraction` is the share of the population in its bin or above
                (ties INCLUDED), so selecting it flags at least that much area. This
                is the conservative convention for "area needed". Area budgets are
                evaluated with the same rule (`realized_area_for_budget`), so reported
                area always equals what a rank selection would flag, including an
                empty selection when a tie block is larger than the budget.
BinnedPR        Unique-cell precision-recall from binned counts. Every eligible cell
                contributes once, without subsampling, in constant memory.
occurrence_report  One success criterion per independent occurrence cluster, with
                cluster-bootstrap confidence intervals.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

AREA_LEVELS = (0.02, 0.05, 0.10, 0.20)


class ScoreCDF:
    """Population score distribution on [0, 1] from exact bin counts."""

    def __init__(self, nbins: int = 2 ** 20):
        self.nbins = int(nbins)
        self.counts = np.zeros(self.nbins, dtype=np.int64)

    def _bin(self, s: np.ndarray) -> np.ndarray:
        s = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)
        return np.minimum((s * self.nbins).astype(np.int64), self.nbins - 1)

    def update(self, scores: np.ndarray) -> None:
        s = np.asarray(scores).ravel()
        s = s[np.isfinite(s)]
        if s.size:
            self.counts += np.bincount(self._bin(s), minlength=self.nbins)

    @property
    def total(self) -> int:
        return int(self.counts.sum())

    def _ge(self) -> np.ndarray:
        """ge[b] = cells in bins >= b; ge[nbins] = 0."""
        return np.r_[np.cumsum(self.counts[::-1])[::-1], 0]

    def _lookup(self, scores: np.ndarray, strict: bool) -> np.ndarray:
        if self.total == 0:
            raise RuntimeError("ScoreCDF is empty")
        ge = self._ge()
        s = np.asarray(scores, dtype=np.float64)
        out = np.full(s.shape, np.nan, dtype=np.float64)
        ok = np.isfinite(s)
        b = self._bin(s[ok]) + (1 if strict else 0)
        out[ok] = ge[b] / self.total
        return out

    def top_area_fraction(self, scores: np.ndarray) -> np.ndarray:
        """Share of the population in the score's bin or above (ties included). NaN stays NaN."""
        return self._lookup(scores, strict=False).astype(np.float32)

    def strictly_above_fraction(self, scores: np.ndarray) -> np.ndarray:
        """Share of the population in bins strictly above the score's bin (0 above the last bin)."""
        return self._lookup(scores, strict=True)

    def threshold_for_area(self, fraction: float) -> Optional[float]:
        """Lowest bin edge whose flagged area (ties included) does not exceed `fraction`; None if no
        non-empty selection fits the budget."""
        ge = self._ge()[:-1] / max(self.total, 1)
        ok = np.flatnonzero((ge <= fraction) & (ge > 0))
        return float(ok[0] / self.nbins) if ok.size else None

    def realized_area_for_budget(self, fraction: float) -> float:
        """Area actually flagged by the rank rule top_area_fraction <= fraction (may be 0)."""
        ge = self._ge()[:-1] / max(self.total, 1)
        ok = ge[ge <= fraction]
        return float(ok.max()) if ok.size else 0.0

    def realized_area(self, threshold: float) -> float:
        """Area with score in the threshold's bin or above."""
        b = int(self._bin(np.array([threshold]))[0])
        return float(self.counts[b:].sum() / max(self.total, 1))

    def to_dict(self) -> dict:
        nz = np.flatnonzero(self.counts)
        return {"nbins": self.nbins, "idx": nz.astype(np.int64), "cnt": self.counts[nz]}

    @classmethod
    def from_dict(cls, d: dict) -> "ScoreCDF":
        c = cls(d["nbins"])
        c.counts[np.asarray(d["idx"])] = np.asarray(d["cnt"])
        return c


class BinnedPR:
    """Average precision and threshold metrics over unique cells, from binned counts."""

    def __init__(self, nbins: int = 10_000):
        self.nbins = nbins
        self.pos = np.zeros(nbins, dtype=np.int64)
        self.neg = np.zeros(nbins, dtype=np.int64)

    def update(self, probs: np.ndarray, labels: np.ndarray) -> None:
        p = np.clip(np.asarray(probs, dtype=np.float64).ravel(), 0, 1)
        y = np.asarray(labels).ravel().astype(bool)
        b = np.minimum((p * self.nbins).astype(np.int64), self.nbins - 1)
        self.pos += np.bincount(b[y], minlength=self.nbins)
        self.neg += np.bincount(b[~y], minlength=self.nbins)

    def average_precision(self) -> float:
        P = self.pos.sum()
        if P == 0:
            return float("nan")
        tp = np.cumsum(self.pos[::-1])
        fp = np.cumsum(self.neg[::-1])
        keep = (self.pos[::-1] + self.neg[::-1]) > 0
        tp, fp = tp[keep], fp[keep]
        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / P
        return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))

    def at_threshold(self, t: float = 0.5) -> Dict[str, float]:
        b = min(int(t * self.nbins), self.nbins - 1)
        tp, fp = self.pos[b:].sum(), self.neg[b:].sum()
        fn = self.pos[:b].sum()
        return {"iou": float(tp / max(tp + fp + fn, 1)), "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
                "n_pos_cells": int(self.pos.sum()), "n_neg_cells": int(self.neg.sum())}


def _clean(area_needed: np.ndarray) -> np.ndarray:
    """inf = explicit miss (unevaluable cluster). NaN is a bug upstream and is rejected."""
    a = np.asarray(area_needed, dtype=np.float64)
    if np.isnan(a).any():
        raise ValueError("area_needed contains NaN; mark unevaluable clusters as np.inf explicitly")
    return a


def _area_for_recall(a: np.ndarray, recall: float) -> Optional[float]:
    if a.size == 0:
        return None
    k = int(np.ceil(recall * a.size)) - 1
    v = float(np.sort(a)[min(max(k, 0), a.size - 1)])
    return v if np.isfinite(v) else None   # None: the recall is never reached


def occurrence_metrics(area_needed: np.ndarray) -> Dict[str, object]:
    """
    area_needed[i] = share of test area that must be flagged to capture cluster i;
    np.inf = cluster counted as missed. Every entry stays in the denominator.
    """
    a = _clean(area_needed)
    out: Dict[str, object] = {f"recall_at_top_{int(round(100 * lv))}pct_area": float((a <= lv).mean()) if a.size
                              else None for lv in AREA_LEVELS}
    out["area_for_80pct_recall"] = _area_for_recall(a, 0.8)
    out["n_clusters"] = int(a.size)
    out["n_missed_unevaluable"] = int(np.isinf(a).sum())
    return out


def occurrence_report(area_needed: np.ndarray, n_boot: int = 2000, seed: int = 0,
                      groups: Optional[Sequence[int]] = None) -> Dict[str, dict]:
    """
    Point estimates plus 95 % percentile bootstrap intervals, resampling clusters
    (the independent unit), stratified by `groups` (e.g. fold ids) when given.
    The same population, including explicit misses, is used for the point
    estimate and the interval.
    """
    a = _clean(area_needed)
    point = occurrence_metrics(a)
    rng = np.random.default_rng(seed)
    idx_all = np.arange(a.size)
    strata = [idx_all] if groups is None else [idx_all[np.asarray(groups) == g] for g in np.unique(groups)]
    keys = [k for k in point if k.startswith(("recall_", "area_"))]
    boots = {k: [] for k in keys}
    for _ in range(n_boot):
        samp = np.concatenate([rng.choice(s_, size=s_.size, replace=True) for s_ in strata if s_.size])
        m = occurrence_metrics(a[samp])
        for k in keys:
            boots[k].append(np.inf if m[k] is None else m[k])
    ci = {}
    for k, v in boots.items():
        with np.errstate(invalid="ignore"):  # inf entries (never reached) are expected
            lo, hi = np.percentile(np.asarray(v, dtype=np.float64), [2.5, 97.5], method="lower")
        ci[k] = [float(lo) if np.isfinite(lo) else None, float(hi) if np.isfinite(hi) else None]
    return {"point": point, "ci95": ci}


def success_curve(area_needed: np.ndarray, steps: int = 201) -> Dict[str, list]:
    a = _clean(area_needed)
    grid = np.linspace(0, 1, steps)
    return {"area_flagged": grid.tolist(),
            "clusters_captured": [float((a <= g).mean()) if a.size else None for g in grid]}
