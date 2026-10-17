"""
GeoSentry-SOTA Investor WebGIS Portal
=====================================

    streamlit run main.py

Reads the outputs of the pipeline (prospectivity.tif, reports/*.csv/json,
models/pu_prospector.joblib) and presents:
  * the continuous prospectivity grid over satellite basemaps, re-projected to
    Web Mercator so it registers exactly with the tiles
  * interactive filters on score or on "top X % of area"
  * KPIs from spatial block cross-validation (not training-set scores)
  * global TreeSHAP attribution grouped by geological factor, computed from the
    trained model, and a live local SHAP explanation for any clicked pixel
  * a ranked, downloadable target list

Every number on this page comes from the trained model and the validation run.
When the loaded configuration is the synthetic demo, a banner says so.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import altair as alt
import folium
import joblib
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from branca.colormap import LinearColormap
from matplotlib.colors import LinearSegmentedColormap
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from scipy import ndimage
from streamlit_folium import st_folium

from src.config import PROJECT_ROOT, Config, load_config
from src.models.prospector_xgb import FeatureRasters, feature_group

# -----------------------------------------------------------------------------
# Visual tokens
# -----------------------------------------------------------------------------
SEQ_COLORS = ["#fde4d4", "#f7a77f", "#eb6834", "#b8441a", "#7a2a0c"]  # single-hue orange ramp, light -> dark
BAR_COLOR = "#2a78d6"
NEG_COLOR = "#8a8984"
INK, INK_2 = "#0b0b0b", "#52514e"
SEQ_CMAP = LinearSegmentedColormap.from_list("geosentry_orange", SEQ_COLORS)
BASEMAPS = {
    "Esri World Imagery": ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                           "Tiles &copy; Esri, Maxar, Earthstar Geographics"),
    "OpenTopoMap": ("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
                    "&copy; OpenStreetMap contributors, SRTM | &copy; OpenTopoMap (CC-BY-SA)"),
    "CartoDB Positron": ("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
                         "&copy; OpenStreetMap contributors &copy; CARTO"),
}

st.set_page_config(page_title="GeoSentry-SOTA | Prospectivity Portal", page_icon="🛰️", layout="wide")
st.markdown(
    """
    <style>
    html, body, [class*="css"], .stMarkdown, .stMetric, button, input, label, h1, h2, h3, h4 {
        font-family: "Times New Roman", Times, serif !important; }
    div[data-testid="stMetricValue"] { font-size: 1.9rem; }
    .gs-banner { padding: 0.6rem 1rem; border-radius: 6px; background: #fff4e5; border: 1px solid #eb6834;
                 color: #7a2a0c; font-weight: 600; margin-bottom: 0.8rem; }
    .gs-sub { color: #52514e; margin-top: -0.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Data access (cached on file modification time)
# -----------------------------------------------------------------------------
def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner=False)
def load_cfg(path: str, _m: float) -> dict:
    return dict(load_config(path))


@st.cache_data(show_spinner="Re-projecting prospectivity grid to Web Mercator ...")
def load_grid_3857(path: str, _m: float, max_px: int = 1600) -> Tuple[np.ndarray, np.ndarray, list, float]:
    """Returns (score_3857, top_area_fraction_3857, [[s, w], [n, e]], native pixel area km2)."""
    with rasterio.open(path) as src:
        scale = max(1.0, max(src.width, src.height) / max_px)
        dst_w, dst_h = int(src.width / scale), int(src.height / scale)
        tr, w, h = calculate_default_transform(src.crs, "EPSG:3857", src.width, src.height, *src.bounds,
                                               dst_width=dst_w, dst_height=dst_h)
        out = np.full((2, h, w), np.nan, dtype=np.float32)
        for b in (1, 2):
            reproject(rasterio.band(src, b), out[b - 1], src_transform=src.transform, src_crs=src.crs,
                      dst_transform=tr, dst_crs="EPSG:3857", resampling=Resampling.bilinear,
                      src_nodata=np.nan, dst_nodata=np.nan)
        left, top = tr.c, tr.f
        right, bottom = left + w * tr.a, top + h * tr.e
        west, south, east, north = transform_bounds("EPSG:3857", "EPSG:4326", left, bottom, right, top)
        px_km2 = abs(src.transform.a * src.transform.e) / 1e6
    return out[0], out[1], [[south, west], [north, east]], px_km2


@st.cache_data(show_spinner=False)
def load_native(path: str, _m: float) -> Tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        return src.read(1), src.read(2)


@st.cache_resource(show_spinner="Loading PU ensemble ...")
def load_model(path: str, _m: float):
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_feature_rasters(stack: str, stage1: str, valid: str):
    return FeatureRasters(Path(stack), Path(stage1), Path(valid))


def read_json(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def colorize(score: np.ndarray, show: np.ndarray, opacity: float) -> np.ndarray:
    """RGBA uint8 image: sequential orange ramp, transparent outside the filter."""
    s = np.nan_to_num(score, nan=0.0)
    rgba = (SEQ_CMAP(np.clip(s, 0, 1)) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(show, int(255 * opacity), 0).astype(np.uint8)
    return rgba


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
configs = sorted((PROJECT_ROOT / "configs").glob("*.yaml"))
if not configs:
    st.error("No configuration files found in configs/.")
    st.stop()
with st.sidebar:
    st.header("GeoSentry-SOTA")
    # Default to the first configuration that already has pipeline outputs.
    ready = [i for i, c in enumerate(configs) if load_config(c).path("prospectivity").exists()]
    cfg_file = st.selectbox("Project configuration", configs, index=ready[0] if ready else 0,
                            format_func=lambda p: p.name)
cfg = Config(load_cfg(str(cfg_file), _mtime(cfg_file)))
is_demo = bool(cfg["project"].get("demo", False))
prosp_path = cfg.path("prospectivity")
reports = cfg.path("reports_dir")

if not prosp_path.exists():
    st.title("GeoSentry-SOTA Prospectivity Portal")
    st.warning(f"No prospectivity grid at `{prosp_path.relative_to(PROJECT_ROOT)}`. Run the pipeline first:")
    st.code(f"python run_pipeline.py --config {cfg_file.relative_to(PROJECT_ROOT).as_posix()}", language="powershell")
    st.stop()

with st.sidebar:
    st.subheader("Target filter")
    mode = st.radio("Filter by", ["Top % of area", "Prospectivity score"], horizontal=False)
    if mode == "Top % of area":
        top_pct = st.slider("Show the top ... % of the AOI", 0.5, 30.0, 5.0, 0.5,
                            help="Uses band 2 (share of all valid AOI cells scoring at least as high; ties included).")
        lo, hi = None, None
    else:
        lo, hi = st.slider("Prospectivity score range", 0.0, 1.0, (0.70, 1.0), 0.01)
        top_pct = None
    st.subheader("Map")
    basemap = st.selectbox("Basemap", list(BASEMAPS))
    opacity = st.slider("Overlay opacity", 0.1, 1.0, 0.75, 0.05)
    show_occ = st.checkbox("Known occurrences", True)
    show_targets = st.checkbox("Ranked target clusters", True)
    n_targets = st.slider("Targets to label", 5, 50, 15)

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("GeoSentry-SOTA Prospectivity Portal")
st.markdown(f"<div class='gs-sub'>{cfg['project']['name']} &middot; REE (monazite / xenotime) and Li pegmatite targeting "
            f"&middot; {cfg['project']['crs']} &middot; {cfg['project']['resolution_m']:.0f} m grid</div>", unsafe_allow_html=True)
if is_demo:
    st.markdown("<div class='gs-banner'>SYNTHETIC DEMO DATA. Every layer, label and score on this page is simulated "
                "to test the software. Do not present these results as geological findings.</div>", unsafe_allow_html=True)

score3857, taf3857, bounds, px_km2 = load_grid_3857(str(prosp_path), _mtime(prosp_path))
score_nat, taf_nat = load_native(str(prosp_path), _mtime(prosp_path))

if top_pct is not None:
    sel_nat = np.nan_to_num(taf_nat, nan=2.0) <= top_pct / 100
    sel_map = np.nan_to_num(taf3857, nan=2.0) <= top_pct / 100
    filt_label = f"top {top_pct:g}% of area"
else:
    sel_nat = (np.nan_to_num(score_nat, nan=-1) >= lo) & (np.nan_to_num(score_nat, nan=-1) <= hi)
    sel_map = (np.nan_to_num(score3857, nan=-1) >= lo) & (np.nan_to_num(score3857, nan=-1) <= hi)
    filt_label = f"score {lo:.2f} to {hi:.2f}"

evr = read_json(reports / "evaluation" / "evaluation_report.json")
# Reports written before the evaluable / all_test_clusters schema are not shown: their
# protocol lacked the cluster-footprint fold grouping and label exclusion fixes.
stale_report = bool(evr) and any("evaluable" not in r for r in evr.get("results", {}).values())
if stale_report:
    st.warning("The evaluation report on disk was produced by an older GeoSentry version and is not shown. "
               "Rerun it with:  python run_pipeline.py --config "
               f"{cfg_file.relative_to(PROJECT_ROOT).as_posix()} --steps evaluate")
    evr = {}
shap_diag = read_json(reports / "shap_diagnostics.json")
# Headline KPIs describe the deployed model: two-stage when Stage 1 exists, otherwise
# the physical-channel PU XGBoost (phase 1 of a real pilot, before target outlines exist).
_results = evr.get("results", {})
kpi_method = "two_stage" if "two_stage" in _results else ("pu_xgb_physical" if "pu_xgb_physical" in _results else None)
two = _results.get(kpi_method, {}) if kpi_method else {}
two_seed = next(iter(two["evaluable"]["per_seed"].values()), {}) if two else {}
two_all = next(iter(two["all_test_clusters"]["per_seed"].values()), {}) if two else {}


def pct(v, digits: int = 0) -> str:
    """Percent string; None (not reached / not defined) shown as n/a."""
    return "n/a" if v is None else f"{100 * v:.{digits}f}%"


def ci_str(ci_pair, digits: int = 0) -> str:
    lo, hi = ci_pair
    return f"95% CI {pct(lo, digits)} to {pct(hi, digits)}"
MIN_CLUSTER_PX = 20  # ignore speckle: clusters smaller than 20 px (0.002 km2 at 10 m) are not counted
labels, n_raw = ndimage.label(sel_nat, structure=np.ones((3, 3)))
n_clusters = int((np.bincount(labels.ravel())[1:] >= MIN_CLUSTER_PX).sum()) if n_raw else 0
aoi_km2 = float(np.isfinite(score_nat).sum() * px_km2)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Area flagged (realised)", f"{sel_nat.sum() * px_km2:,.2f} km²",
          f"{100 * sel_nat.sum() * px_km2 / max(aoi_km2, 1e-9):.1f}% of valid AOI", delta_color="off")
k2.metric("Target clusters (>= 20 px)", f"{n_clusters:,}", filt_label, delta_color="off")
if two_seed:
    pt, ci = two_seed["point"], two_seed["ci95"]
    proto = evr["protocol"]
    k3.metric("Clusters in top 10% area", pct(pt["recall_at_top_10pct_area"]),
              ci_str(ci["recall_at_top_10pct_area"]), delta_color="off")
    k4.metric("Area for 80% recall", pct(pt["area_for_80pct_recall"], 1),
              ci_str(ci["area_for_80pct_recall"], 1), delta_color="off")
    n_unev = two_all["point"]["n_missed_unevaluable"] if two_all else 0
    k5.metric("Clusters tested", f"{pt['n_clusters']}",
              f"{proto['outer_folds']} folds; {n_unev} unevaluable", delta_color="off")
    if kpi_method != "two_stage":
        st.caption(f"Held-out figures are for `{kpi_method}` (no Stage 1 model in this evaluation: "
                   f"{'; '.join(proto.get('methods_skipped', {}).values()) or 'not requested'}).")
else:
    k3.metric("Held-out cluster recall", "not evaluated")
    k4.metric("Area for 80% recall", "not evaluated")
    k5.metric("Clusters tested", "0")
    st.info("No nested geographic evaluation found. Run `python run_pipeline.py --steps evaluate` before quoting "
            "any performance figure.")

# -----------------------------------------------------------------------------
# Map
# -----------------------------------------------------------------------------
map_col, side_col = st.columns([3, 2], gap="large")
with map_col:
    st.subheader("Prospectivity map")
    center = [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2]
    m = folium.Map(location=center, zoom_start=13, tiles=None, control_scale=True)
    url, attr = BASEMAPS[basemap]
    folium.TileLayer(url, attr=attr, name=basemap, max_zoom=19).add_to(m)
    folium.raster_layers.ImageOverlay(colorize(score3857, sel_map, opacity), bounds=bounds, name="Prospectivity",
                                      mercator_project=False, interactive=False, zindex=10).add_to(m)
    LinearColormap(SEQ_COLORS, vmin=0, vmax=1, caption="Prospectivity score (0 to 1)").add_to(m)

    occ_path = cfg.path("positives_csv")
    if show_occ and occ_path.exists():
        occ = pd.read_csv(occ_path)
        cl = {c.lower(): c for c in occ.columns}
        if "lon" in cl and "lat" in cl:
            fg = folium.FeatureGroup("Known occurrences")
            for _, r in occ.iterrows():
                tip = str(r[cl["name"]]) if "name" in cl else "Occurrence"
                folium.CircleMarker([r[cl["lat"]], r[cl["lon"]]], radius=4, color="#ffffff", weight=1.5,
                                    fill=True, fill_color=BAR_COLOR, fill_opacity=1, tooltip=tip).add_to(fg)
            fg.add_to(m)

    targets = read_csv(reports / "targets.csv")
    if show_targets and not targets.empty:
        fg = folium.FeatureGroup("Ranked targets")
        for _, r in targets.head(n_targets).iterrows():
            html = (f"<b>Target #{int(r['rank'])}</b><br>Peak score {r['peak_score']:.2f}<br>"
                    f"Area {r['area_km2']:.3f} km²<br><small>{r.get('top_drivers_logodds', '')}</small>")
            folium.Marker([r["lat"], r["lon"]], tooltip=f"Target #{int(r['rank'])}",
                          popup=folium.Popup(html, max_width=320),
                          icon=folium.DivIcon(html=f"<div style='font:700 12px Times New Roman;color:#fff;"
                                                   f"background:#7a2a0c;border:2px solid #fff;border-radius:50%;"
                                                   f"width:24px;height:24px;line-height:20px;text-align:center'>"
                                                   f"{int(r['rank'])}</div>")).add_to(fg)
        fg.add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    map_state = st_folium(m, height=620, use_container_width=True, returned_objects=["last_clicked"])
    st.caption("Click anywhere on the AOI for a pixel-level SHAP explanation. Overlay is warped to Web Mercator "
               "before display so it registers with the basemap tiles.")

# -----------------------------------------------------------------------------
# Explainability
# -----------------------------------------------------------------------------
def bar_chart(df: pd.DataFrame, value: str, label: str, title: str, fmt: str, signed: bool = False) -> alt.Chart:
    base = alt.Chart(df).encode(
        y=alt.Y(f"{label}:N", sort="-x" if not signed else alt.EncodingSortField(value, order="descending"),
                title=None, axis=alt.Axis(labelLimit=260, labelColor=INK, labelFont="Times New Roman")),
        x=alt.X(f"{value}:Q", title=title, axis=alt.Axis(grid=True, gridOpacity=0.25, labelColor=INK_2,
                                                          titleColor=INK_2, titleFont="Times New Roman")),
        tooltip=[alt.Tooltip(f"{label}:N", title="Factor"), alt.Tooltip(f"{value}:Q", title=title, format=fmt)],
    )
    color = (alt.condition(f"datum['{value}'] >= 0", alt.value(BAR_COLOR), alt.value(NEG_COLOR)) if signed
             else alt.value(BAR_COLOR))
    bars = base.mark_bar(cornerRadiusEnd=4, height=14).encode(color=color)
    text = base.mark_text(align="left", dx=4, color=INK, font="Times New Roman").encode(
        text=alt.Text(f"{value}:Q", format=fmt))
    return (bars + text).properties(height=alt.Step(24)).configure_view(stroke=None)


with side_col:
    st.subheader("Why the model ranks ground highly")
    groups = read_csv(reports / "shap_groups.csv")
    if groups.empty:
        st.info("No SHAP report found. Run the `prospect` step.")
    else:
        g = groups.copy()
        g["label"] = g["group"] + ": " + g["share_pct"].map(lambda v: f"{v:.0f}%")
        st.altair_chart(bar_chart(g, "share_pct", "label", "Share of |SHAP| (%)", ".1f"), width="stretch")
        stab = shap_diag.get("stability_between_bag_halves", {})
        if shap_diag:
            st.caption(f"Completeness error {shap_diag.get('completeness_max_abs_error', float('nan')):.1e} (log-odds). "
                       f"Stability between bag halves: rank correlation {stab.get('feature_rank_spearman', float('nan')):.2f}, "
                       f"max group-share difference {stab.get('group_share_max_abs_diff_pct', float('nan')):.1f} points.")
        st.caption("TreeSHAP over all PU bags, in the log-odds of the deployed score, on a sample of occurrence and "
                   "background cells, grouped by data source. It describes what this model uses, not why "
                   "mineralisation occurs; correlated channels can trade attribution, so read the group totals.")

    st.subheader("Pixel explanation")
    clicked = (map_state or {}).get("last_clicked")
    model_path = cfg.path("models_dir") / "pu_prospector.joblib"
    if not clicked:
        st.write("Click the map to explain a location.")
    elif not model_path.exists():
        st.info("Model file not found; run the `prospect` step.")
    else:
        fr = load_feature_rasters(str(cfg.path("stack")), str(cfg.path("stage1_features")), str(cfg.path("valid_mask")))
        x, y = Transformer.from_crs("EPSG:4326", fr.crs, always_xy=True).transform(clicked["lng"], clicked["lat"])
        row, col = rasterio.transform.rowcol(fr.transform, x, y)
        if not (0 <= row < fr.shape[0] and 0 <= col < fr.shape[1]):
            st.warning("That point is outside the AOI.")
        else:
            model = load_model(str(model_path), _mtime(model_path))
            X = fr.extract(np.array([row]), np.array([col]))
            if not np.isfinite(X).all():
                st.warning("No valid data at that pixel (nodata in one or more layers).")
            else:
                score = float(model.predict_proba(X)[0])
                taf = float(model.top_area_fraction(np.array([score]))[0])
                sv = model.shap_values(X)[0][0]
                loc = (pd.DataFrame({"feature": fr.feature_names, "shap": sv})
                       .assign(group=lambda d: d["feature"].map(feature_group))
                       .groupby("group", as_index=False)["shap"].sum())
                loc = loc.reindex(loc["shap"].abs().sort_values(ascending=False).index).head(8)
                c1, c2 = st.columns(2)
                c1.metric("Score", f"{score:.2f}")
                c2.metric("Rank in AOI", f"top {100 * taf:.1f}%")
                st.altair_chart(bar_chart(loc, "shap", "group", "Contribution (log-odds)", "+.2f", signed=True),
                                width="stretch")
                st.caption(f"{clicked['lat']:.5f}°N, {clicked['lng']:.5f}°E · {cfg['project']['crs']} {x:,.0f} E, {y:,.0f} N. "
                           "Blue raises prospectivity, grey lowers it.")

# -----------------------------------------------------------------------------
# Validation and targets
# -----------------------------------------------------------------------------
v_col, t_col = st.columns([1, 1], gap="large")
METHOD_LABELS = {"two_stage": "Two-stage (product)", "stage1_only": "Stage 1 U-Net only",
                 "pu_xgb_physical": "PU XGBoost, physical channels", "pu_logreg_physical": "PU logistic, physical channels"}
METHOD_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order
with v_col:
    st.subheader("Held-out success curves")
    curves = read_csv(reports / "evaluation" / "success_curves.csv")
    if curves.empty:
        st.info("Run the evaluate step to compare methods on shared geographic folds.")
    else:
        seed0 = curves["seed"].iloc[0]
        c = curves[curves["seed"] == seed0].assign(
            area_pct=lambda d: 100 * d["area_flagged"], captured_pct=lambda d: 100 * d["clusters_captured"],
            Method=lambda d: d["method"].map(METHOD_LABELS))
        order = [METHOD_LABELS[m] for m in METHOD_LABELS if m in set(c["method"])]
        xmax = st.slider("Zoom: max area flagged (%)", 5, 100, 25, 5)
        c = c[c["area_pct"] <= xmax]
        lines = alt.Chart(c).mark_line(strokeWidth=2).encode(
            x=alt.X("area_pct:Q", title="Test area flagged (%)", scale=alt.Scale(domain=[0, xmax])),
            y=alt.Y("captured_pct:Q", title="Held-out clusters captured (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Method:N", sort=order, scale=alt.Scale(domain=order, range=METHOD_COLORS[:len(order)]),
                            legend=alt.Legend(orient="bottom", columns=1, title=None, labelFont="Times New Roman")),
            tooltip=[alt.Tooltip("Method:N"), alt.Tooltip("area_pct:Q", title="Area flagged %", format=".1f"),
                     alt.Tooltip("captured_pct:Q", title="Clusters captured %", format=".0f")])
        rand = alt.Chart(pd.DataFrame({"a": [0, xmax]})).mark_line(strokeWidth=1.5, strokeDash=[4, 4],
                                                                   color=NEG_COLOR).encode(x="a:Q", y="a:Q")
        st.altair_chart((rand + lines).properties(height=320).configure_view(stroke=None), width="stretch")
        rows = []
        for m, r in evr.get("results", {}).items():
            p = next(iter(r["evaluable"]["per_seed"].values()))
            pa = next(iter(r["all_test_clusters"]["per_seed"].values()))
            rows.append({"Method": METHOD_LABELS.get(m, m),
                         "Recall @5% area": f"{pct(p['point']['recall_at_top_5pct_area'])} "
                                            f"({pct(p['ci95']['recall_at_top_5pct_area'][0])} to "
                                            f"{pct(p['ci95']['recall_at_top_5pct_area'][1])})",
                         "Area for 80% recall": pct(p["point"]["area_for_80pct_recall"], 1),
                         "Recall @5%, all test clusters": pct(pa["point"]["recall_at_top_5pct_area"])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        proto = evr.get("protocol", {})
        st.caption(f"Dashed grey: random selection. {proto.get('outer_folds')} geographic folds, "
                   f"{proto.get('dead_zone_m', 0):.0f} m dead zone, occurrences merged within "
                   f"{proto.get('cluster_radius_m', 0):.0f} m into indivisible clusters; Stage 1 retrained per fold. "
                   f"Area is relative to each fold's test area, ties included. 95% CI by cluster bootstrap. "
                   + proto.get("ssl", {}).get("description", ""))

with t_col:
    st.subheader("Ranked targets")
    if targets.empty:
        st.info("No targets file found.")
    else:
        show = targets[["rank", "peak_score", "mean_score", "area_km2", "lat", "lon", "top_drivers_logodds"]].rename(
            columns={"rank": "Rank", "peak_score": "Peak score", "mean_score": "Mean score", "area_km2": "Area (km²)",
                     "lat": "Lat", "lon": "Lon", "top_drivers_logodds": "Top drivers (log-odds)"})
        st.dataframe(show, hide_index=True, width="stretch", height=300,
                     column_config={"Peak score": st.column_config.NumberColumn(format="%.3f"),
                                    "Mean score": st.column_config.NumberColumn(format="%.3f"),
                                    "Area (km²)": st.column_config.NumberColumn(format="%.3f"),
                                    "Lat": st.column_config.NumberColumn(format="%.5f"),
                                    "Lon": st.column_config.NumberColumn(format="%.5f")})
        st.download_button("Download target list (CSV)", targets.to_csv(index=False).encode(),
                           file_name="geosentry_targets.csv", mime="text/csv")

with st.expander("Methodology and limitations"):
    st.markdown(
        """
**Stage 0, harmonization.** Sentinel-2, Sentinel-1, airborne magnetics, airborne radiometrics and DEM are warped
to one 10 m grid in WGS 84 / UTM. The 10 m grid is an alignment choice: SWIR is 20 m and airborne surveys are much
coarser, and interpolation does not add geological detail. Derived layers: NDVI, the B11/B12 SWIR ratio (a broad
Al-OH proxy, not mineral identification), FFT magnetic first vertical derivative, slope.

**Stage 1.** A spatial-channel Attention U-Net, initialised from a masked self-supervised encoder trained on this
AOI only (not a general foundation model), segments mapped target outlines. Attention gates are internal network
weights, not geological evidence.

**Stage 2.** Known occurrences are positives; all other ground is unlabeled. A bagged PU XGBoost ensemble, averaged
in log-odds, produces the 0 to 1 score.

**Validation.** All performance figures come from nested geographic folds shared by every method: occurrences are
clustered and never split, a dead zone separates training and test ground, and Stage 1 is retrained inside each
fold. Baselines are scored on the same folds. Results on the synthetic demo only show that the software runs.

**Reading the score.** It ranks ground; it is not the probability of an economic deposit. Equivalent thorium is not
REE grade, and known workings partly reflect where people have looked. Field verification against matched control
areas remains the real test.
        """
    )
