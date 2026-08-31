"""
Orchard Diagnostic Intelligence v2
=================================
Research-oriented UAV hyperspectral + 3-D orchard health screening framework.

Major methodological changes relative to the original app
----------------------------------------------------------
1. LAI is REMOVED from all diagnostic rules and from the map overlay.
2. Radius_m is REMOVED from all diagnostic rules. It may remain in the source
   table only as descriptive metadata; this app never uses it to classify health.
3. Structure is represented by point-cloud/DSM-derived CHM metrics. CHM_P95 is
   preferred; CHM_max is used only as a fallback if P95 is unavailable.
4. The healthy reference is multi-domain (vigor + red edge/chlorophyll + water
   + low senescence), rather than NDVI + LAI.
5. New hyperspectral features can be computed from the averaged tree spectra
   when wavelength metadata are available: NDRE, MTCI, CIred-edge, SIPI,
   VOG1, MTVI2, CCCI, maxLARE, and selected literature narrow-band ratios.
6. Diagnostic outputs are anomaly / management-screening candidates, NOT
   ground-truth-validated disease diagnoses.
7. Internal validation includes:
      - spectral consistency vs internally defined reference trees,
      - multi-index concordance scores,
      - threshold sensitivity (P20/P80, P25/P75, P30/P70),
      - Jaccard agreement among threshold schemes,
      - statistical comparison with the reference group,
      - structural corroboration when CHM is available,
      - PCA feature-space corroboration,
      - exploratory kNN Moran's I spatial coherence.
8. The LLM layer remains downstream of deterministic rules and is instructed
   not to convert anomaly classes into confirmed diagnoses.

Expected project structure
--------------------------
project/
  orchard_diagnostic_intelligence_v2.py
  data/
    data.zip                                  # shapefile bundle; one polygon/tree
    master_hyperspectral_signatures_averaged.csv
    band_wavelengths.csv                      # recommended (optional if .hdr exists)
    *.hdr                                     # ENVI/Headwall header, optional
    CHM.tif / CHM_1.tif / canopy_height_model.tif  # optional map overlay

Recommended band_wavelengths.csv format
---------------------------------------
band,wavelength_nm
Band_1,400.52
Band_2,402.31
...

If no wavelength mapping is available, the app still runs with the existing
precomputed indices in the shapefile, but it cannot calculate new narrow-band
indices from the spectral CSV.
"""

from __future__ import annotations

import base64
import html
import io
import math
import os
import re
import tempfile
import warnings
import zipfile
from itertools import combinations
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import rasterio
import streamlit as st
from rasterio.warp import transform_bounds
from shapely.geometry import Point
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from streamlit_folium import st_folium

try:
    from scipy.stats import mannwhitneyu
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

warnings.filterwarnings("ignore")


# =============================================================================
# 0. CONFIGURATION
# =============================================================================

APP_TITLE = "🍋 Orchard Diagnostic Intelligence Platform — Research Workflow v2"
DATA_DIR = Path(__file__).resolve().parent / "data"
ZIP_PATH = DATA_DIR / "data.zip"
SPECTRAL_CSV_CANDIDATES = [
    Path(__file__).resolve().parent / "master_hyperspectral_signatures_averaged.csv",
    DATA_DIR / "master_hyperspectral_signatures_averaged.csv",
]
WAVELENGTH_CSV_CANDIDATES = [
    DATA_DIR / "band_wavelengths.csv",
    Path(__file__).resolve().parent / "band_wavelengths.csv",
]
CHM_RASTER_CANDIDATES = [
    DATA_DIR / "CHM.tif",
    DATA_DIR / "CHM_1.tif",
    DATA_DIR / "canopy_height_model.tif",
]

# Operational percentile rule (sensitivity analysis perturbs this later).
LOWER_Q = 0.25
UPPER_Q = 0.75

# IMPORTANT: verify against the exact PRI convention used in your ENVI output.
# Standard PRI often uses (R531-R570)/(R531+R570), for which lower values are
# commonly interpreted as reduced photochemical efficiency in this workflow.
PRI_STRESS_DIRECTION_DEFAULT = "low"  # allowed: "low" or "high"

# Maximum wavelength mismatch permitted for new narrow-band HSI features.
NARROW_BAND_TOLERANCE_NM = 6.0

# Gap-analysis defaults preserved from original app; expose in sidebar.
DEFAULT_TREE_SPACING_M = 5.5
DEFAULT_ROW_DISTANCE_THRESHOLD_M = 2.5
DEFAULT_GRID_ANGLE_DEG = 75.0
DEFAULT_MAX_EMPTY_SPACE_M = 20.0


# =============================================================================
# 1. GENERAL HELPERS
# =============================================================================

def safe_divide(a, b):
    """Vectorized safe division returning NaN where denominator is ~0."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full(np.broadcast(a, b).shape, np.nan, dtype=float)
    np.divide(a, b, out=out, where=np.abs(b) > 1e-12)
    return out


def numeric_band_suffix(col: str) -> float:
    """Return numeric suffix for Band_### sorting; inf if unavailable."""
    m = re.search(r"(-?\d+(?:\.\d+)?)$", str(col))
    return float(m.group(1)) if m else float("inf")


def first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None


def qvalue(df: pd.DataFrame, col: str, q: float):
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s):
            return float(s.quantile(q))
    return np.nan


def meanvalue(df: pd.DataFrame, col: str):
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s):
            return float(s.mean())
    return np.nan


def bool_false(df: pd.DataFrame):
    return pd.Series(False, index=df.index)


def cond_lt(df, col, value):
    if col not in df.columns or not np.isfinite(value):
        return bool_false(df)
    return pd.to_numeric(df[col], errors="coerce") < value


def cond_le(df, col, value):
    if col not in df.columns or not np.isfinite(value):
        return bool_false(df)
    return pd.to_numeric(df[col], errors="coerce") <= value


def cond_gt(df, col, value):
    if col not in df.columns or not np.isfinite(value):
        return bool_false(df)
    return pd.to_numeric(df[col], errors="coerce") > value


def cond_ge(df, col, value):
    if col not in df.columns or not np.isfinite(value):
        return bool_false(df)
    return pd.to_numeric(df[col], errors="coerce") >= value


def cond_between_or(df, col, low, high):
    if col not in df.columns or not (np.isfinite(low) and np.isfinite(high)):
        return bool_false(df)
    s = pd.to_numeric(df[col], errors="coerce")
    return (s < low) | (s > high)


def confidence_from_score(score: pd.Series, max_score: int = 3) -> pd.Series:
    """Generic evidence-tier conversion; does not imply ground-truth accuracy."""
    score = pd.to_numeric(score, errors="coerce").fillna(0)
    if max_score <= 2:
        return pd.cut(score, [-1, 0, 1, np.inf], labels=["No evidence", "Screening", "Supported"])
    return pd.cut(
        score,
        [-1, 0, 1, 2, np.inf],
        labels=["No evidence", "Screening", "Supported", "High priority"],
    )


def rank_biserial_from_u(u, n1, n2):
    if n1 == 0 or n2 == 0:
        return np.nan
    return (2.0 * u) / (n1 * n2) - 1.0


# =============================================================================
# 1B. FIELD-NAVIGATION / KML HELPERS
# =============================================================================

FIELD_FLAG_COLUMNS = [
    "Flag_A", "Flag_B", "Flag_C", "Flag_D",
    "Flag_E", "Flag_F", "Flag_G", "Flag_H",
]


def _kml_color_for_scenario(scenario_key: str) -> str:
    """KML colors are AABBGGRR, not normal RGB."""
    return {
        "Flag_A": "ffff0000",       # blue
        "Flag_B": "ffff00ff",       # magenta/purple
        "Flag_C": "ff0000ff",       # red
        "Flag_D": "ff000080",       # dark red
        "Flag_E": "ffffff00",       # cyan
        "Flag_F": "ff00a5ff",       # orange
        "Flag_G": "ff0080ff",       # orange-red
        "Flag_H": "ff00aa00",       # green
        "HIGH_PRIORITY": "ff00ffff", # yellow
    }.get(scenario_key, "ffffffff")


def _short_flag_name(flag_col: str) -> str:
    if flag_col.startswith("Flag_"):
        return flag_col.replace("Flag_", "")
    if flag_col == "HIGH_PRIORITY":
        return "PRIORITY"
    return str(flag_col)


def make_navigation_points(source_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create canopy-centroid navigation points and return them in EPSG:4326."""
    if source_gdf is None or source_gdf.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

    work = source_gdf.copy()
    if work.crs is None:
        raise ValueError("Canopy layer has no CRS; KML navigation coordinates cannot be generated safely.")

    try:
        if work.crs.is_geographic:
            projected_crs = work.estimate_utm_crs()
            if projected_crs is None:
                raise ValueError("Could not estimate projected CRS.")
            projected = work.to_crs(projected_crs)
        else:
            projected = work.copy()

        points_projected = projected.copy()
        points_projected.geometry = projected.geometry.centroid
        points = points_projected.to_crs(epsg=4326)
    except Exception:
        # Fallback stays inside the crown if projected centroid creation fails.
        points = work.to_crs(epsg=4326).copy()
        points.geometry = points.geometry.representative_point()

    points["Longitude"] = points.geometry.x
    points["Latitude"] = points.geometry.y
    return points


def active_flags_for_row(row: pd.Series) -> str:
    active = []
    for flag in FIELD_FLAG_COLUMNS:
        if flag in row.index:
            try:
                if bool(row[flag]):
                    active.append(_short_flag_name(flag))
            except Exception:
                pass
    return ",".join(active) if active else "None"


def build_tree_navigation_kml(
    source_gdf: gpd.GeoDataFrame,
    layer_title: str,
    scenario_key: str = "",
) -> bytes:
    """Create a mobile-ready KML file in memory, one placemark per tree."""
    points = make_navigation_points(source_gdf)
    if points.empty:
        return b""

    kml_color = _kml_color_for_scenario(scenario_key)
    safe_title = html.escape(str(layer_title))

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        '<Document>',
        f'<name>{safe_title}</name>',
        '<Style id="treeTarget">',
        '<IconStyle>',
        f'<color>{kml_color}</color>',
        '<scale>1.15</scale>',
        '<Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>',
        '</IconStyle>',
        '<LabelStyle><scale>0.85</scale></LabelStyle>',
        '</Style>',
    ]

    optional_fields = [
        "NDVI_mn", "NDRE_mn", "WBI_mn", "PRI_mn", "PSRI_mn",
        "CHM_P95", "CHM_max", "Domain_Count", "Overall_Evidence",
        "Row_ID", "row_id", "Tree_in_row", "tree_in_row",
    ]

    for _, row in points.iterrows():
        tree_id = row.get("generated_id", row.get("tree_id", "NA"))
        try:
            tree_label = f"T{int(tree_id)}"
        except Exception:
            tree_label = f"T{tree_id}"

        scenario_text = active_flags_for_row(row)
        point_name = (
            f"{tree_label} [{_short_flag_name(scenario_key)}]"
            if scenario_key else f"{tree_label} [{scenario_text}]"
        )

        description_lines = [
            f"<b>Tree ID:</b> {html.escape(str(tree_id))}",
            f"<b>Scenario(s):</b> {html.escape(scenario_text)}",
            f"<b>Latitude:</b> {float(row['Latitude']):.7f}",
            f"<b>Longitude:</b> {float(row['Longitude']):.7f}",
        ]

        if "tree_id" in row.index and pd.notna(row.get("tree_id")):
            description_lines.insert(
                1,
                f"<b>Source tree_id:</b> {html.escape(str(row.get('tree_id')))}",
            )

        for field in optional_fields:
            if field in row.index and pd.notna(row.get(field)):
                value = row.get(field)
                if isinstance(value, (float, np.floating)):
                    value = f"{float(value):.4f}"
                description_lines.append(
                    f"<b>{html.escape(field)}:</b> {html.escape(str(value))}"
                )

        description = "<br>".join(description_lines)
        lon = float(row["Longitude"])
        lat = float(row["Latitude"])

        parts.extend([
            '<Placemark>',
            f'<name>{html.escape(point_name)}</name>',
            '<styleUrl>#treeTarget</styleUrl>',
            f'<description><![CDATA[{description}]]></description>',
            '<Point>',
            f'<coordinates>{lon:.8f},{lat:.8f},0</coordinates>',
            '</Point>',
            '</Placemark>',
        ])

    parts.extend(['</Document>', '</kml>'])
    return "\n".join(parts).encode("utf-8")

# =============================================================================
# 2. DATA INGESTION
# =============================================================================

@st.cache_data(show_spinner=False)
def load_tree_data() -> gpd.GeoDataFrame:
    """Load the zipped shapefile and preserve the original generated_id logic."""
    if not ZIP_PATH.exists():
        st.error(f"File not found on server: {ZIP_PATH}")
        st.stop()

    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    shp_file = None
    for root, _, files in os.walk(temp_dir):
        for file in files:
            if file.lower().endswith(".shp"):
                shp_file = os.path.join(root, file)
                break
        if shp_file:
            break

    if not shp_file:
        st.error("No .shp file was found inside data.zip.")
        st.stop()

    gdf = gpd.read_file(shp_file)
    if "tree_id" in gdf.columns:
        gdf = gdf.drop_duplicates(subset=["tree_id"]).copy()
    else:
        gdf = gdf.drop_duplicates().copy()

    # Preserve original cross-file ID alignment assumption.
    gdf["generated_id"] = range(1, len(gdf) + 1)

    # Keep radius only as descriptive metadata. NEVER use it in rules.
    # LAI columns can remain in source data for provenance but are ignored.

    # Descriptive crown polygon area only; not used in health rules by default.
    try:
        utm_crs = gdf.estimate_utm_crs()
        if utm_crs is not None:
            area_gdf = gdf.to_crs(utm_crs)
            gdf["CrownArea_m2_desc"] = area_gdf.geometry.area.values
    except Exception:
        pass

    return gdf.to_crs(epsg=4326)


@st.cache_data(show_spinner=False)
def load_spectral_data() -> pd.DataFrame:
    csv_path = first_existing(SPECTRAL_CSV_CANDIDATES)
    if csv_path is None:
        return pd.DataFrame()
    return pd.read_csv(csv_path)


# =============================================================================
# 3. WAVELENGTH METADATA + NEW HYPERSPECTRAL FEATURES
# =============================================================================

def parse_envi_wavelengths(hdr_path: Path):
    try:
        txt = hdr_path.read_text(errors="ignore")
        match = re.search(r"wavelength\s*=\s*\{(.*?)\}", txt, flags=re.I | re.S)
        if not match:
            return []
        vals = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", match.group(1))
        return [float(v) for v in vals]
    except Exception:
        return []


def discover_wavelength_map(band_cols):
    """
    Return {band_column: wavelength_nm}, source_description.

    Priority:
      1) explicit band_wavelengths.csv,
      2) ENVI .hdr wavelength list,
      3) Band_<nm> naming convention if suffixes themselves look like nm.
    """
    band_cols = sorted(band_cols, key=numeric_band_suffix)

    # 1. Explicit CSV mapping.
    wl_csv = first_existing(WAVELENGTH_CSV_CANDIDATES)
    if wl_csv is not None:
        try:
            mdf = pd.read_csv(wl_csv)
            lower_cols = {c.lower(): c for c in mdf.columns}
            wave_col = next((lower_cols[k] for k in lower_cols if "wavelength" in k), None)
            band_col = next((lower_cols[k] for k in lower_cols if k in {"band", "band_name", "band_col", "column"}), None)
            if wave_col is not None:
                if band_col is not None:
                    mapping = {}
                    for _, r in mdf.iterrows():
                        key = str(r[band_col])
                        if key in band_cols:
                            mapping[key] = float(r[wave_col])
                        else:
                            # Permit numeric band identifier such as 1 for Band_1.
                            candidate = f"Band_{key}"
                            if candidate in band_cols:
                                mapping[candidate] = float(r[wave_col])
                    if len(mapping) >= max(3, int(0.8 * len(band_cols))):
                        return mapping, f"CSV mapping: {wl_csv.name}"
                elif len(mdf) == len(band_cols):
                    return dict(zip(band_cols, pd.to_numeric(mdf[wave_col], errors="coerce"))), f"CSV ordered wavelengths: {wl_csv.name}"
        except Exception:
            pass

    # 2. ENVI header.
    hdr_candidates = list(DATA_DIR.rglob("*.hdr")) if DATA_DIR.exists() else []
    for hdr in hdr_candidates:
        waves = parse_envi_wavelengths(hdr)
        if len(waves) == len(band_cols):
            return dict(zip(band_cols, waves)), f"ENVI/Headwall header: {hdr.name}"

    # 3. Column suffixes may already be wavelengths.
    suffixes = [numeric_band_suffix(c) for c in band_cols]
    finite = [x for x in suffixes if np.isfinite(x)]
    if finite and min(finite) >= 350 and max(finite) <= 2500:
        return dict(zip(band_cols, suffixes)), "Band column suffix interpreted as wavelength (nm)"

    return {}, "No wavelength mapping found"


def nearest_band(wavelength_map, target_nm, tolerance_nm=NARROW_BAND_TOLERANCE_NM):
    if not wavelength_map:
        return None, np.nan
    items = [(col, float(wl)) for col, wl in wavelength_map.items() if pd.notna(wl)]
    if not items:
        return None, np.nan
    col, wl = min(items, key=lambda x: abs(x[1] - target_nm))
    if abs(wl - target_nm) > tolerance_nm:
        return None, wl
    return col, wl


def compute_new_hsi_features(spectral_df: pd.DataFrame, band_cols, wavelength_map):
    """Compute literature-supported narrow-band features from tree-average spectra."""
    if spectral_df.empty or not band_cols or not wavelength_map:
        return pd.DataFrame(), {}

    out = pd.DataFrame(index=spectral_df.index)
    if "generated_id" in spectral_df.columns:
        out["generated_id"] = spectral_df["generated_id"].values
    elif "tree_id" in spectral_df.columns:
        out["tree_id"] = spectral_df["tree_id"].values
    else:
        return pd.DataFrame(), {}

    used = {}

    def R(target):
        col, actual = nearest_band(wavelength_map, target)
        if col is None:
            return None, None, actual
        return pd.to_numeric(spectral_df[col], errors="coerce").to_numpy(float), col, actual

    def record(name, pairs):
        used[name] = {f"target_{t}": {"column": c, "actual_nm": a} for t, c, a in pairs}

    # NDRE: representative narrow-band red-edge implementation from reviewed UAV-HSI literature.
    r770, c770, a770 = R(770)
    r750, c750, a750 = R(750)
    if r770 is not None and r750 is not None:
        out["NDRE_mn"] = safe_divide(r770 - r750, r770 + r750)
        record("NDRE", [(770, c770, a770), (750, c750, a750)])

    # MTCI.
    r710, c710, a710 = R(710)
    r680, c680, a680 = R(680)
    if r750 is not None and r710 is not None and r680 is not None:
        out["MTCI_mn"] = safe_divide(r750 - r710, r710 - r680)
        record("MTCI", [(750, c750, a750), (710, c710, a710), (680, c680, a680)])

    # Chlorophyll Index red-edge.
    r780, c780, a780 = R(780)
    if r780 is not None and r710 is not None:
        out["CIRED_mn"] = safe_divide(r780, r710) - 1.0
        record("CIred-edge", [(780, c780, a780), (710, c710, a710)])

    # SIPI.
    r800, c800, a800 = R(800)
    r445, c445, a445 = R(445)
    if r800 is not None and r445 is not None and r680 is not None:
        out["SIPI_mn"] = safe_divide(r800 - r445, r800 - r680)
        record("SIPI", [(800, c800, a800), (445, c445, a445), (680, c680, a680)])

    # VOG1 (research / secondary red-edge feature).
    r745, c745, a745 = R(745)
    r720, c720, a720 = R(720)
    if r745 is not None and r720 is not None:
        out["VOG1_mn"] = safe_divide(r745, r720)
        record("VOG1", [(745, c745, a745), (720, c720, a720)])

    # MTVI2 (exploratory spectral structural proxy; CHM remains the primary structural domain).
    r550, c550, a550 = R(550)
    r670, c670, a670 = R(670)
    if r800 is not None and r550 is not None and r670 is not None:
        numerator = 1.5 * (1.2 * (r800 - r550) - 2.5 * (r670 - r550))
        inside = (2 * r800 + 1) ** 2 - (6 * r800 - 5 * np.sqrt(np.clip(r670, 0, None))) - 0.5
        denominator = np.sqrt(np.where(inside > 0, inside, np.nan))
        out["MTVI2_mn"] = safe_divide(numerator, denominator)
        record("MTVI2", [(800, c800, a800), (550, c550, a550), (670, c670, a670)])

    # First derivative red-edge metric: maximum derivative in 690-710 nm.
    ordered = sorted(
        [(c, float(wavelength_map[c])) for c in band_cols if c in wavelength_map and pd.notna(wavelength_map[c])],
        key=lambda x: x[1],
    )
    if len(ordered) >= 3:
        cols_sorted = [x[0] for x in ordered]
        waves_sorted = np.array([x[1] for x in ordered], dtype=float)
        mask = (waves_sorted >= 690) & (waves_sorted <= 710)
        if mask.sum() >= 2:
            matrix = spectral_df[cols_sorted].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            deriv = np.gradient(matrix, waves_sorted, axis=1)
            out["maxLARE"] = np.nanmax(deriv[:, mask], axis=1)
            used["maxLARE"] = {"range_nm": "690-710", "bands_used": int(mask.sum())}

    # Literature narrow-band ratios kept as research features, not hard diagnostic rules.
    ratio_pairs = [(710, 714), (714, 718), (750, 754), (754, 758), (894, 898), (963, 967)]
    for a, b in ratio_pairs:
        ra, ca, aa = R(a)
        rb, cb, ab = R(b)
        if ra is not None and rb is not None and ca != cb:
            name = f"NB_{a}_{b}"
            out[name] = safe_divide(ra, rb)
            record(name, [(a, ca, aa), (b, cb, ab)])

    return out, used


def merge_new_features(gdf: gpd.GeoDataFrame, feature_df: pd.DataFrame):
    if feature_df.empty:
        return gdf

    key = "generated_id" if "generated_id" in feature_df.columns and "generated_id" in gdf.columns else None
    if key is None and "tree_id" in feature_df.columns and "tree_id" in gdf.columns:
        key = "tree_id"
    if key is None:
        return gdf

    # Preserve existing precomputed values if the shapefile already contains a feature.
    new_cols = [c for c in feature_df.columns if c == key or c not in gdf.columns]
    return gdf.merge(feature_df[new_cols], on=key, how="left")


def add_derived_indices(gdf: gpd.GeoDataFrame):
    # CCCI uses the already available NDVI and newly derived / precomputed NDRE.
    if "NDRE_mn" in gdf.columns and "NDVI_mn" in gdf.columns and "CCCI_mn" not in gdf.columns:
        gdf["CCCI_mn"] = safe_divide(
            pd.to_numeric(gdf["NDRE_mn"], errors="coerce"),
            pd.to_numeric(gdf["NDVI_mn"], errors="coerce"),
        )
    return gdf


# =============================================================================
# 4. STRUCTURAL DOMAIN (NO LAI, NO RADIUS IN RULES)
# =============================================================================

def choose_structural_metric(gdf: pd.DataFrame):
    """Prefer robust point-cloud-derived P95 height; use CHM_max only as fallback."""
    candidates = [
        "CHM_P95",
        "CHM_p95",
        "CHM95",
        "CHM_95",
        "Height_P95",
        "height_p95",
        "CHM_max",  # fallback only
    ]
    for c in candidates:
        if c in gdf.columns and pd.to_numeric(gdf[c], errors="coerce").notna().any():
            return c
    return None


# =============================================================================
# 5. RULE ENGINE
# =============================================================================

def apply_rules(
    source_gdf: gpd.GeoDataFrame,
    lower_q=LOWER_Q,
    upper_q=UPPER_Q,
    pri_stress_direction=PRI_STRESS_DIRECTION_DEFAULT,
):
    """
    Apply independent Boolean anomaly rules.

    NOTE: Outputs are screening/anomaly classes, not externally validated diagnoses.
    """
    gdf = source_gdf.copy()
    structural_col = choose_structural_metric(gdf)

    # Core distribution thresholds.
    cols = [
        "NDVI_mn", "WBI_mn", "MCARI_mn", "NDRE_mn", "MTCI_mn", "CIRED_mn",
        "PSRI_mn", "PRI_mn", "SIPI_mn", "CRI1_mn", "CRI2_mn",
        "NDVI_mi", "NDVI_sd", "CRI1_sd", "SIPI_sd", "PSRI_sd",
    ]
    if structural_col:
        cols.append(structural_col)

    thresholds = {}
    for c in cols:
        thresholds[c] = {
            "low": qvalue(gdf, c, lower_q),
            "median": qvalue(gdf, c, 0.50),
            "high": qvalue(gdf, c, upper_q),
            "mean": meanvalue(gdf, c),
        }

    # PRI stress condition is configurable because sign/order convention must be verified.
    pri_low = thresholds.get("PRI_mn", {}).get("low", np.nan)
    pri_high = thresholds.get("PRI_mn", {}).get("high", np.nan)
    if pri_stress_direction == "high":
        pri_stress = cond_gt(gdf, "PRI_mn", pri_high)
    else:
        pri_stress = cond_lt(gdf, "PRI_mn", pri_low)

    # -------------------------------------------------------------------------
    # Healthy-reference population: multi-domain reference, no LAI.
    # -------------------------------------------------------------------------
    healthy = cond_ge(gdf, "NDVI_mn", thresholds["NDVI_mn"]["high"])

    if "NDRE_mn" in gdf.columns:
        healthy &= cond_ge(gdf, "NDRE_mn", thresholds["NDRE_mn"]["median"])
    elif "MCARI_mn" in gdf.columns:
        healthy &= cond_ge(gdf, "MCARI_mn", thresholds["MCARI_mn"]["median"])

    if "WBI_mn" in gdf.columns:
        healthy &= cond_ge(gdf, "WBI_mn", thresholds["WBI_mn"]["median"])
    if "PSRI_mn" in gdf.columns:
        healthy &= cond_le(gdf, "PSRI_mn", thresholds["PSRI_mn"]["median"])

    gdf["IS_HEALTHY_REF"] = healthy.fillna(False)

    # -------------------------------------------------------------------------
    # A. Water-stress candidate.
    # Existing logic retained conceptually; PRI adds independent support.
    # -------------------------------------------------------------------------
    wbi_low = cond_lt(gdf, "WBI_mn", thresholds["WBI_mn"]["low"])
    ndvi_not_collapsed = cond_gt(gdf, "NDVI_mn", thresholds["NDVI_mn"]["low"])
    ndvi_below_mean = cond_le(gdf, "NDVI_mn", thresholds["NDVI_mn"]["mean"])
    gdf["Flag_A"] = (wbi_low & ndvi_not_collapsed & ndvi_below_mean).fillna(False)
    gdf["Water_Score"] = (
        wbi_low.astype(int)
        + pri_stress.astype(int)
        + cond_le(gdf, "NDVI_mn", thresholds["NDVI_mn"]["median"]).astype(int)
    )
    gdf["Water_Confidence"] = confidence_from_score(gdf["Water_Score"], 3).astype(str)

    # -------------------------------------------------------------------------
    # B. Chlorophyll / nutrient-related anomaly.
    # Replaces old LAI + MCARI rule with multi-index biochemical concordance.
    # -------------------------------------------------------------------------
    mcari_low = cond_lt(gdf, "MCARI_mn", thresholds.get("MCARI_mn", {}).get("low", np.nan))
    ndre_low = cond_lt(gdf, "NDRE_mn", thresholds.get("NDRE_mn", {}).get("low", np.nan))
    mtci_low = cond_lt(gdf, "MTCI_mn", thresholds.get("MTCI_mn", {}).get("low", np.nan))
    cired_low = cond_lt(gdf, "CIRED_mn", thresholds.get("CIRED_mn", {}).get("low", np.nan))
    ndvi_reasonable = cond_ge(gdf, "NDVI_mn", thresholds["NDVI_mn"]["median"])

    rededge_support = (mtci_low | cired_low)
    gdf["Nutrient_Score"] = mcari_low.astype(int) + ndre_low.astype(int) + rededge_support.astype(int)
    gdf["Flag_B"] = (ndvi_reasonable & mcari_low & ndre_low).fillna(False)
    gdf["Flag_B_plus"] = (gdf["Flag_B"] & rededge_support).fillna(False)
    gdf["Nutrient_Confidence"] = confidence_from_score(gdf["Nutrient_Score"], 3).astype(str)

    # -------------------------------------------------------------------------
    # C. Chronic canopy decline candidate.
    # Radius and LAI are completely removed.
    # -------------------------------------------------------------------------
    ndvi_low = cond_lt(gdf, "NDVI_mn", thresholds["NDVI_mn"]["low"])
    psri_high = cond_gt(gdf, "PSRI_mn", thresholds.get("PSRI_mn", {}).get("high", np.nan))

    if structural_col:
        structural_low = cond_lt(gdf, structural_col, thresholds[structural_col]["low"])
    else:
        structural_low = bool_false(gdf)

    gdf["Flag_C"] = (ndvi_low & psri_high).fillna(False)
    gdf["Flag_C_plus"] = (gdf["Flag_C"] & structural_low).fillna(False)
    gdf["Decline_Score"] = ndvi_low.astype(int) + psri_high.astype(int) + structural_low.astype(int)
    gdf["Decline_Confidence"] = confidence_from_score(gdf["Decline_Score"], 3).astype(str)

    # -------------------------------------------------------------------------
    # D. Localized canopy anomaly.
    # Retains original spatial heterogeneity idea but avoids causal pest diagnosis.
    # -------------------------------------------------------------------------
    ndvi_mean_ok = cond_gt(gdf, "NDVI_mn", thresholds["NDVI_mn"]["low"])
    ndvi_sd_high = cond_gt(gdf, "NDVI_sd", thresholds.get("NDVI_sd", {}).get("high", np.nan))
    cri1_sd_high = cond_gt(gdf, "CRI1_sd", thresholds.get("CRI1_sd", {}).get("high", np.nan))
    ndvi_min_low = cond_lt(gdf, "NDVI_mi", thresholds.get("NDVI_mi", {}).get("low", np.nan))
    gdf["Flag_D"] = (ndvi_mean_ok & ndvi_sd_high & cri1_sd_high & ndvi_min_low).fillna(False)

    local_support = bool_false(gdf)
    if "SIPI_sd" in gdf.columns:
        local_support |= cond_gt(gdf, "SIPI_sd", thresholds["SIPI_sd"]["high"])
    if "PSRI_sd" in gdf.columns:
        local_support |= cond_gt(gdf, "PSRI_sd", thresholds["PSRI_sd"]["high"])
    gdf["Flag_D_plus"] = (gdf["Flag_D"] & local_support).fillna(False)
    gdf["Localized_Score"] = (
        ndvi_sd_high.astype(int)
        + cri1_sd_high.astype(int)
        + ndvi_min_low.astype(int)
        + local_support.astype(int)
    )

    # -------------------------------------------------------------------------
    # E. Acute physiological stress candidate.
    # LAI removed. WBI splits water-associated from non-water physiological stress.
    # -------------------------------------------------------------------------
    psri_above_median = cond_gt(gdf, "PSRI_mn", thresholds.get("PSRI_mn", {}).get("median", np.nan))
    gdf["Flag_E"] = (pri_stress & ndvi_not_collapsed & psri_above_median).fillna(False)
    gdf["Flag_E_water"] = (gdf["Flag_E"] & wbi_low).fillna(False)
    gdf["Flag_E_nonwater"] = (gdf["Flag_E"] & ~wbi_low).fillna(False)
    gdf["Physiology_Score"] = pri_stress.astype(int) + psri_above_median.astype(int) + ndvi_not_collapsed.astype(int)
    gdf["Physiology_Confidence"] = confidence_from_score(gdf["Physiology_Score"], 3).astype(str)

    # -------------------------------------------------------------------------
    # F. Early biochemical anomaly: red-edge changes while broadband vigor remains.
    # -------------------------------------------------------------------------
    psri_not_advanced = cond_le(gdf, "PSRI_mn", thresholds.get("PSRI_mn", {}).get("high", np.nan))
    gdf["Flag_F"] = (ndvi_reasonable & ndre_low & mtci_low & psri_not_advanced).fillna(False)

    # -------------------------------------------------------------------------
    # G. Pigment / potential biotic-stress candidate.
    # Uses IQR-type abnormality rather than assuming SIPI/CRI direction universally.
    # -------------------------------------------------------------------------
    sipi_abnormal = cond_between_or(
        gdf,
        "SIPI_mn",
        thresholds.get("SIPI_mn", {}).get("low", np.nan),
        thresholds.get("SIPI_mn", {}).get("high", np.nan),
    )
    cri1_abnormal = cond_between_or(
        gdf,
        "CRI1_mn",
        thresholds.get("CRI1_mn", {}).get("low", np.nan),
        thresholds.get("CRI1_mn", {}).get("high", np.nan),
    )
    cri2_abnormal = cond_between_or(
        gdf,
        "CRI2_mn",
        thresholds.get("CRI2_mn", {}).get("low", np.nan),
        thresholds.get("CRI2_mn", {}).get("high", np.nan),
    )
    pigment_support = cri1_abnormal | cri2_abnormal | cri1_sd_high
    gdf["Flag_G"] = (psri_high & sipi_abnormal & pigment_support).fillna(False)
    gdf["Pigment_Score"] = psri_high.astype(int) + sipi_abnormal.astype(int) + pigment_support.astype(int)
    gdf["Pigment_Confidence"] = confidence_from_score(gdf["Pigment_Score"], 3).astype(str)

    # -------------------------------------------------------------------------
    # H. Structural anomaly: point-cloud/CHM evidence only.
    # -------------------------------------------------------------------------
    gdf["Flag_H"] = structural_low.fillna(False)
    gdf["CHM_PROFILE"] = gdf[structural_col].notna() if structural_col else False

    # -------------------------------------------------------------------------
    # Multi-domain evidence count. A tree can legitimately carry multiple flags.
    # -------------------------------------------------------------------------
    domain_cols = ["Flag_A", "Flag_B", "Flag_C", "Flag_D", "Flag_E", "Flag_F", "Flag_G", "Flag_H"]
    gdf["Domain_Count"] = gdf[domain_cols].astype(int).sum(axis=1)
    gdf["HIGH_PRIORITY"] = gdf["Domain_Count"] >= 3
    gdf["Overall_Evidence"] = pd.cut(
        gdf["Domain_Count"],
        bins=[-1, 0, 1, 2, np.inf],
        labels=["No flagged domain", "Screening", "Supported", "High priority"],
    ).astype(str)

    return gdf, thresholds, structural_col


# =============================================================================
# 6. GAP ANALYSIS (PRESERVED, PARAMETERS EXPOSED)
# =============================================================================

def calculate_gaps(gdf, expected_tree_spacing, row_distance_threshold, grid_angle_degrees, max_empty_space_m):
    gap_calc_gdf = gdf.copy()
    utm_crs = gap_calc_gdf.estimate_utm_crs()
    gap_calc_gdf = gap_calc_gdf.to_crs(utm_crs)

    gap_calc_gdf["centroid"] = gap_calc_gdf.geometry.centroid
    raw_x = gap_calc_gdf["centroid"].apply(lambda p: p.x)
    raw_y = gap_calc_gdf["centroid"].apply(lambda p: p.y)
    mean_x, mean_y = raw_x.mean(), raw_y.mean()

    def rotate_coords(x, y, angle_deg):
        rx = (x - mean_x) * math.cos(math.radians(angle_deg)) + (y - mean_y) * math.sin(math.radians(angle_deg))
        ry = -(x - mean_x) * math.sin(math.radians(angle_deg)) + (y - mean_y) * math.cos(math.radians(angle_deg))
        return rx, ry

    def unrotate_coords(rx, ry, angle_deg):
        x = rx * math.cos(math.radians(angle_deg)) - ry * math.sin(math.radians(angle_deg))
        y = rx * math.sin(math.radians(angle_deg)) + ry * math.cos(math.radians(angle_deg))
        return x + mean_x, y + mean_y

    rotated = [rotate_coords(x, y, grid_angle_degrees) for x, y in zip(raw_x, raw_y)]
    gap_calc_gdf["x"] = [c[0] for c in rotated]
    gap_calc_gdf["y"] = [c[1] for c in rotated]

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=row_distance_threshold,
        linkage="average",
    )
    gap_calc_gdf["Row_ID"] = clustering.fit_predict(gap_calc_gdf[["y"]].to_numpy())
    row_centers = gap_calc_gdf.groupby("Row_ID")["y"].mean().to_dict()
    gap_calc_gdf["Row_Center_Y"] = gap_calc_gdf["Row_ID"].map(row_centers)

    gaps = []
    for _, group in gap_calc_gdf.groupby("Row_ID"):
        group = group.sort_values("x").reset_index(drop=True)
        for i in range(len(group) - 1):
            tree_a, tree_b = group.iloc[i], group.iloc[i + 1]
            dist = tree_b["x"] - tree_a["x"]
            if (expected_tree_spacing * 1.5) < dist <= max_empty_space_m:
                missing_count = int(np.round(dist / expected_tree_spacing)) - 1
                for j in range(1, missing_count + 1):
                    gap_x_rotated = tree_a["x"] + j * (dist / (missing_count + 1))
                    real_x, real_y = unrotate_coords(gap_x_rotated, tree_a["Row_Center_Y"], grid_angle_degrees)
                    gaps.append(Point(real_x, real_y))

    gaps_gdf = gpd.GeoDataFrame({"geometry": gaps}, crs=gap_calc_gdf.crs)
    if len(gaps_gdf):
        tree_buffers = gap_calc_gdf.geometry.buffer(2.0).unary_union
        gaps_gdf = gaps_gdf[~gaps_gdf.intersects(tree_buffers)]
    gaps_wgs84 = gaps_gdf.to_crs(epsg=4326)

    total_trees = len(gap_calc_gdf)
    total_gaps = len(gaps_wgs84)
    ideal_capacity = total_trees + total_gaps
    yield_loss_percentage = (total_gaps / ideal_capacity) * 100 if ideal_capacity > 0 else 0.0
    return gaps_wgs84, total_trees, total_gaps, yield_loss_percentage


# =============================================================================
# 7. INTERNAL VALIDATION / ROBUSTNESS FUNCTIONS (NO GROUND TRUTH REQUIRED)
# =============================================================================

def jaccard(a: pd.Series, b: pd.Series):
    a = a.astype(bool).to_numpy()
    b = b.astype(bool).to_numpy()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return np.nan
    return np.logical_and(a, b).sum() / union


def threshold_sensitivity(source_gdf, flag_col, pri_direction):
    schemes = {
        "P20/P80": (0.20, 0.80),
        "P25/P75": (0.25, 0.75),
        "P30/P70": (0.30, 0.70),
    }
    runs = {}
    for name, (lo, hi) in schemes.items():
        rgdf, _, _ = apply_rules(source_gdf, lo, hi, pri_direction)
        if flag_col in rgdf.columns:
            runs[name] = rgdf[flag_col].astype(bool).reset_index(drop=True)

    if not runs:
        return pd.DataFrame(), pd.DataFrame(), None

    stability = pd.DataFrame(runs)
    stability["stability"] = stability.mean(axis=1)

    names = list(runs)
    jac = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            jac.loc[a, b] = jaccard(runs[a], runs[b])
    return stability, jac, runs


def spectral_angle_deg(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return np.nan
    a = a[mask]
    b = b[mask]
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den <= 0:
        return np.nan
    cosang = np.clip(np.dot(a, b) / den, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def spectral_rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2))) if mask.sum() else np.nan


def comparison_statistics(gdf, flag_col, metrics):
    if flag_col not in gdf.columns or "IS_HEALTHY_REF" not in gdf.columns:
        return pd.DataFrame()

    flagged = gdf[gdf[flag_col].astype(bool)]
    reference = gdf[gdf["IS_HEALTHY_REF"].astype(bool) & ~gdf[flag_col].astype(bool)]
    rows = []
    for c in metrics:
        if c not in gdf.columns:
            continue
        x = pd.to_numeric(flagged[c], errors="coerce").dropna().to_numpy(float)
        y = pd.to_numeric(reference[c], errors="coerce").dropna().to_numpy(float)
        if len(x) == 0 or len(y) == 0:
            continue
        row = {
            "Metric": c,
            "Flagged_n": len(x),
            "Reference_n": len(y),
            "Flagged_median": float(np.median(x)),
            "Reference_median": float(np.median(y)),
            "Median_difference": float(np.median(x) - np.median(y)),
        }
        if SCIPY_AVAILABLE:
            try:
                u, p = mannwhitneyu(x, y, alternative="two-sided")
                row["Mann_Whitney_p"] = float(p)
                row["Rank_biserial"] = rank_biserial_from_u(u, len(x), len(y))
            except Exception:
                row["Mann_Whitney_p"] = np.nan
                row["Rank_biserial"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def pca_feature_space(gdf, feature_cols, flag_col):
    available = [c for c in feature_cols if c in gdf.columns and pd.to_numeric(gdf[c], errors="coerce").notna().sum() >= 3]
    if len(available) < 3:
        return pd.DataFrame(), None, []

    X = gdf[available].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)

    plot_df = pd.DataFrame({
        "PC1": pcs[:, 0],
        "PC2": pcs[:, 1],
        "Group": np.where(gdf[flag_col].astype(bool), "Flagged", "Other") if flag_col in gdf.columns else "Other",
        "Reference": np.where(gdf["IS_HEALTHY_REF"].astype(bool), "Healthy reference", "Non-reference"),
        "generated_id": gdf["generated_id"].values,
    })
    explained = pca.explained_variance_ratio_ * 100
    return plot_df, explained, available


def morans_i_knn(gdf, flag_col, k=4, permutations=199, random_seed=42):
    """Exploratory binary Moran's I using row-standardized kNN weights."""
    if flag_col not in gdf.columns or len(gdf) < max(5, k + 1):
        return np.nan, np.nan, np.nan, np.nan

    y = gdf[flag_col].astype(int).to_numpy(float)
    prevalence = float(y.mean())
    if prevalence in (0.0, 1.0):
        return np.nan, np.nan, prevalence, np.nan

    try:
        projected = gdf.to_crs(gdf.estimate_utm_crs())
        centroids = projected.geometry.centroid
        coords = np.c_[centroids.x, centroids.y]
    except Exception:
        return np.nan, np.nan, prevalence, np.nan

    n = len(y)
    k_eff = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(coords)
    neigh = nn.kneighbors(return_distance=False)[:, 1:]

    # Row-standardized weights: 1/k per neighbor.
    z = y - y.mean()
    denom = np.sum(z ** 2)
    if denom == 0:
        return np.nan, np.nan, prevalence, np.nan

    def calc_i(values):
        zz = values - values.mean()
        den = np.sum(zz ** 2)
        if den == 0:
            return np.nan
        num = 0.0
        for i in range(n):
            num += np.mean(zz[i] * zz[neigh[i]])
        # S0 = n because row-standardized; n/S0 = 1.
        return num / den

    observed_i = calc_i(y)

    flagged_idx = np.where(y == 1)[0]
    observed_neighbor_rate = float(np.mean([y[neigh[i]].mean() for i in flagged_idx])) if len(flagged_idx) else np.nan
    lift = observed_neighbor_rate / prevalence if prevalence > 0 else np.nan

    rng = np.random.default_rng(random_seed)
    perm_vals = []
    for _ in range(permutations):
        perm_vals.append(calc_i(rng.permutation(y)))
    perm_vals = np.asarray(perm_vals, dtype=float)
    pseudo_p = (np.sum(np.abs(perm_vals) >= abs(observed_i)) + 1) / (len(perm_vals) + 1)
    return observed_i, pseudo_p, prevalence, lift


# =============================================================================
# 8. LOAD + PREPARE FEATURES
# =============================================================================

st.set_page_config(page_title="Orchard Diagnostic Intelligence", layout="wide")
st.title(APP_TITLE)
st.caption(
    "UAV hyperspectral + point-cloud/CHM decision support. Outputs are internally assessed anomaly candidates; "
    "no external ground-truth accuracy is claimed."
)

base_gdf = load_tree_data()
spectral_df = load_spectral_data()

band_cols = []
wavelength_map = {}
wavelength_source = "No spectral data"
computed_index_details = {}

if not spectral_df.empty:
    band_cols = sorted([c for c in spectral_df.columns if str(c).startswith("Band_")], key=numeric_band_suffix)
    wavelength_map, wavelength_source = discover_wavelength_map(band_cols)
    new_features, computed_index_details = compute_new_hsi_features(spectral_df, band_cols, wavelength_map)
    base_gdf = merge_new_features(base_gdf, new_features)

base_gdf = add_derived_indices(base_gdf)

# PRI convention user control, because this must match the actual precomputed PRI.
st.sidebar.header("Diagnostic Controls")
pri_direction = st.sidebar.selectbox(
    "PRI stress convention",
    ["low", "high"],
    index=0 if PRI_STRESS_DIRECTION_DEFAULT == "low" else 1,
    help="Verify this against the exact PRI wavelength order/formula used by ENVI. This changes Flag E and water-support logic.",
)

# Apply the operational P25/P75 rule set.
gdf, thresholds, structural_col = apply_rules(base_gdf, LOWER_Q, UPPER_Q, pri_direction)

if structural_col == "CHM_max":
    st.sidebar.warning("CHM_P95 not found; CHM_max is being used as a structural fallback. Prefer P95 from the point cloud when available.")
elif structural_col is None:
    st.sidebar.warning("No CHM structural metric found. Structural corroboration and Flag H are disabled.")


# =============================================================================
# 9. SCENARIO DEFINITIONS
# =============================================================================

scenario_dict = {
    "Flag_A": (
        "A: Water-Stress Candidate",
        "blue",
        "Low WBI with reduced-but-not-collapsed vigor.",
        "**Inputs:** WBI, NDVI; PRI contributes independent physiological support.\n\n"
        "**Interpretation:** relative water-status anomaly requiring irrigation/field inspection; not a ground-truth drought diagnosis.",
    ),
    "Flag_B": (
        "B: Chlorophyll / Nutrient Anomaly",
        "purple",
        "Reasonable broadband vigor but concordantly low chlorophyll/red-edge response.",
        "**Inputs:** NDVI, MCARI, NDRE; MTCI/CIred-edge raise confidence.\n\n"
        "**Change:** replaces the old LAI + MCARI 'hidden hunger' rule. No LAI is used.",
    ),
    "Flag_C": (
        "C: Chronic Canopy Decline",
        "red",
        "Low vigor plus high senescence; CHM can provide independent structural corroboration.",
        "**Inputs:** NDVI, PSRI; CHM_P95 preferred for structural support.\n\n"
        "**Change:** Radius_m and LAI were removed. This is a decline candidate, not 'root rot detected'.",
    ),
    "Flag_D": (
        "D: Localized Canopy Anomaly",
        "darkred",
        "High within-crown spectral heterogeneity despite acceptable mean vigor.",
        "**Inputs:** NDVI mean, NDVI SD, NDVI minimum, CRI1 SD; optional SIPI/PSRI SD support.\n\n"
        "**Interpretation:** localized canopy anomaly; causal pest/pathogen assignment requires field confirmation.",
    ),
    "Flag_E": (
        "E: Acute Physiological Stress Candidate",
        "cyan",
        "PRI anomaly with retained vigor and elevated pigment/senescence response.",
        "**Inputs:** PRI, NDVI, PSRI; WBI separates water-associated from non-water physiological stress.\n\n"
        "**Change:** LAI removed and 'heat/frost' is no longer asserted without independent evidence.",
    ),
    "Flag_F": (
        "F: Early Biochemical Anomaly",
        "orange",
        "Broadband vigor remains acceptable while red-edge indicators are already depressed.",
        "**Inputs:** NDVI, NDRE, MTCI, PSRI.\n\n"
        "**Purpose:** exploit hyperspectral red-edge sensitivity before gross canopy decline is evident.",
    ),
    "Flag_G": (
        "G: Pigment / Potential Biotic-Stress Candidate",
        "magenta",
        "Senescence plus abnormal SIPI and carotenoid-related response.",
        "**Inputs:** PSRI, SIPI, CRI1/CRI2 or CRI1 heterogeneity.\n\n"
        "**Interpretation:** pigment/biotic-stress candidate only; disease identity is not inferred without ground truth.",
    ),
    "Flag_H": (
        "H: Structural Anomaly",
        "green",
        "Low point-cloud/CHM structural metric relative to the orchard population.",
        f"**Input:** {structural_col or 'No structural metric available'}.\n\n"
        "**Change:** 3-D structure replaces unvalidated LAI and indicative radius in the diagnostic framework.",
    ),
    "HIGH_PRIORITY": (
        "⚠️ Multi-Domain High Priority",
        "black",
        "Three or more independent anomaly domains are simultaneously flagged.",
        "**Purpose:** prioritize field inspection where several independent physiological/structural domains agree.",
    ),
    "CHM_PROFILE": (
        "🌳 Canopy Height / Structure Profile",
        "green",
        "Displays trees with available CHM structural information.",
        f"**Structural metric selected:** {structural_col or 'none'}. P95 is preferred over absolute maximum height when available.",
    ),
    "GAP_ANALYSIS": (
        "🍋 Geometric Gap & Yield Analysis",
        "red",
        "Estimates missing planting positions from orchard-row geometry.",
        "This module is preserved as an orchard inventory tool and is logically separate from spectral health classification.",
    ),
}

selected_scenario = st.sidebar.selectbox(
    "Select Target Scenario",
    options=list(scenario_dict.keys()),
    format_func=lambda x: scenario_dict[x][0],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Wavelength metadata: {wavelength_source}")
if wavelength_map:
    st.sidebar.success(f"Mapped {len(wavelength_map)} spectral bands to wavelength (nm).")
else:
    st.sidebar.info("New narrow-band indices are used only if already present in the shapefile or if wavelength metadata can be resolved.")


# =============================================================================
# 10. GAP-ANALYSIS CONTROLS + TARGET SELECTION
# =============================================================================

if selected_scenario == "GAP_ANALYSIS":
    with st.sidebar.expander("Gap-analysis parameters", expanded=False):
        expected_tree_spacing = st.number_input("Expected tree spacing (m)", 0.5, 20.0, DEFAULT_TREE_SPACING_M, 0.1)
        row_distance_threshold = st.number_input("Row clustering threshold (m)", 0.5, 10.0, DEFAULT_ROW_DISTANCE_THRESHOLD_M, 0.1)
        grid_angle_degrees = st.number_input("Grid rotation angle (deg)", -180.0, 180.0, DEFAULT_GRID_ANGLE_DEG, 1.0)
        max_empty_space_m = st.number_input("Maximum gap considered (m)", 2.0, 100.0, DEFAULT_MAX_EMPTY_SPACE_M, 1.0)

    gaps_folium_gdf, total_trees, total_gaps, yield_loss_percentage = calculate_gaps(
        gdf,
        expected_tree_spacing,
        row_distance_threshold,
        grid_angle_degrees,
        max_empty_space_m,
    )
    target_gdf = gpd.GeoDataFrame()
else:
    total_trees = len(gdf)
    if selected_scenario in gdf.columns:
        target_gdf = gdf[gdf[selected_scenario].astype(bool)].copy()
    else:
        target_gdf = gpd.GeoDataFrame()


# =============================================================================
# 11. HYPERSPECTRAL X-AXIS
# =============================================================================

if band_cols:
    if wavelength_map and all(c in wavelength_map for c in band_cols):
        spectral_x = np.array([wavelength_map[c] for c in band_cols], dtype=float)
        spectral_x_title = "Wavelength (nm)"
        x_is_wavelength = True
    else:
        spectral_x = np.array([numeric_band_suffix(c) for c in band_cols], dtype=float)
        spectral_x_title = "Band Number"
        x_is_wavelength = False
else:
    spectral_x = np.array([])
    spectral_x_title = "Band"
    x_is_wavelength = False


def add_em_regions(figure):
    """Use true wavelength regions only when wavelength metadata are known."""
    if x_is_wavelength:
        figure.add_vrect(x0=400, x1=680, opacity=0.08, layer="below", line_width=0, annotation_text="VIS")
        figure.add_vrect(x0=680, x1=750, opacity=0.08, layer="below", line_width=0, annotation_text="Red edge")
        figure.add_vrect(x0=750, x1=1000, opacity=0.08, layer="below", line_width=0, annotation_text="NIR")
    return figure


def get_reference_spectrum():
    if spectral_df.empty or not band_cols:
        return None
    healthy_ids = gdf.loc[gdf["IS_HEALTHY_REF"], "generated_id"].tolist()
    if healthy_ids:
        subset = spectral_df[spectral_df["generated_id"].isin(healthy_ids)] if "generated_id" in spectral_df.columns else pd.DataFrame()
        if not subset.empty:
            return subset[band_cols].apply(pd.to_numeric, errors="coerce").mean().to_numpy(float)
    return spectral_df[band_cols].apply(pd.to_numeric, errors="coerce").mean().to_numpy(float)


# =============================================================================
# 12. MAIN TABS
# =============================================================================

tab_map, tab_validation, tab_workflow = st.tabs([
    "🗺️ Diagnostic Map",
    "🧪 Internal Validation & Robustness",
    "🧭 Workflow / What Changed",
])


# -----------------------------------------------------------------------------
# TAB 1 — MAP + SPECTRA
# -----------------------------------------------------------------------------
with tab_map:
    col1, col2 = st.columns([3, 2])

    with col1:
        toggle_col1, toggle_col2 = st.columns(2)
        with toggle_col1:
            show_chm = st.checkbox("Load CHM raster overlay (if available)", value=True)
        with toggle_col2:
            show_canopies = st.checkbox("Show canopy vectors", value=True)

        map_center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
        m = folium.Map(location=map_center, zoom_start=18, max_zoom=22, tiles="CartoDB dark_matter")

        # CHM overlay replaces the previous LAI overlay.
        chm_path = first_existing(CHM_RASTER_CANDIDATES)
        if show_chm and chm_path is not None:
            try:
                with rasterio.open(chm_path) as src:
                    minx, miny, maxx, maxy = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
                    scale = min(1.0, 1500.0 / src.width)
                    arr = src.read(
                        1,
                        out_shape=(int(src.height * scale), int(src.width * scale)),
                        resampling=rasterio.enums.Resampling.bilinear,
                    ).astype(float)
                    nodata = src.nodata
                    if nodata is not None:
                        arr[arr == nodata] = np.nan
                    masked = np.ma.masked_invalid(arr)
                    valid = masked.compressed()
                    if len(valid):
                        vmin, vmax = np.percentile(valid, [5, 95])
                        colored = (plt.cm.viridis(plt.Normalize(vmin=vmin, vmax=vmax)(masked)) * 255).astype(np.uint8)
                        colored[..., 3] = np.where(masked.mask, 0, 210)
                        buf = io.BytesIO()
                        plt.imsave(buf, colored, format="png")
                        buf.seek(0)
                        folium.raster_layers.ImageOverlay(
                            image=f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}",
                            bounds=[[miny, minx], [maxy, maxx]],
                            opacity=0.75,
                            name="Canopy Height Model",
                        ).add_to(m)
            except Exception as exc:
                st.warning(f"CHM overlay could not be rendered: {exc}")

        if show_canopies:
            # All crowns as outline context.
            folium.GeoJson(
                gdf,
                style_function=lambda _: {"fillColor": "none", "color": "#00FFCC", "weight": 1.0, "fillOpacity": 0.0},
                name="All orchard canopies",
            ).add_to(m)

            if selected_scenario == "GAP_ANALYSIS":
                for _, row in gaps_folium_gdf.iterrows():
                    folium.CircleMarker(
                        location=[row.geometry.y, row.geometry.x],
                        radius=5,
                        color="#000000",
                        weight=2,
                        fill=True,
                        fill_color="#FFFFFF",
                        fill_opacity=1.0,
                        tooltip="Calculated crop gap",
                    ).add_to(m)
            elif not target_gdf.empty:
                tooltip_candidates = [
                    "generated_id", "tree_id", "NDVI_mn", "NDRE_mn", "MCARI_mn", "WBI_mn",
                    "PRI_mn", "PSRI_mn", structural_col, "Overall_Evidence", "Domain_Count",
                ]
                fields = [f for f in tooltip_candidates if f and f in target_gdf.columns]
                aliases = [f"{f}:" for f in fields]
                tooltip = folium.GeoJsonTooltip(fields=fields, aliases=aliases, localize=True) if fields else None
                folium.GeoJson(
                    target_gdf,
                    style_function=lambda _: {
                        "fillColor": scenario_dict[selected_scenario][1],
                        "color": "white",
                        "weight": 2.0,
                        "fillOpacity": 0.70,
                    },
                    tooltip=tooltip,
                    name="Selected targets",
                ).add_to(m)

        folium.LayerControl().add_to(m)
        st_data = st_folium(m, height=650, use_container_width=True, returned_objects=["last_active_drawing"])

    with col2:
        st.header("Scenario Details")
        st.subheader(scenario_dict[selected_scenario][0])
        st.write(scenario_dict[selected_scenario][2])
        with st.expander("🔬 Scientific rationale / interpretation"):
            st.markdown(scenario_dict[selected_scenario][3])

        if selected_scenario == "GAP_ANALYSIS":
            st.metric("Total Orchard Trees", total_trees)
            st.metric("Calculated Crop Gaps", total_gaps)
            st.metric("Estimated Planting-Capacity Loss", f"{yield_loss_percentage:.2f}%")
            if total_gaps:
                st.download_button(
                    f"Download {total_gaps} gaps (GeoJSON)",
                    gaps_folium_gdf.to_json(),
                    file_name="calculated_orchard_gaps.geojson",
                    mime="application/geo+json",
                )
        else:
            target_count = len(target_gdf)
            st.metric(
                "Targeted Trees",
                target_count,
                delta=f"{(target_count / total_trees * 100) if total_trees else 0:.1f}% of block",
                delta_color="inverse",
            )

            if not target_gdf.empty and "Overall_Evidence" in target_gdf.columns:
                st.dataframe(
                    target_gdf["Overall_Evidence"].value_counts().rename_axis("Evidence tier").reset_index(name="Trees"),
                    hide_index=True,
                    use_container_width=True,
                )

            st.markdown("---")

            # Hyperspectral signature comparison retained and improved to wavelength axis when possible.
            if not spectral_df.empty and selected_scenario != "CHM_PROFILE" and band_cols:
                selected_canopy_id = None
                if st_data and st_data.get("last_active_drawing"):
                    props = st_data["last_active_drawing"].get("properties", {})
                    selected_canopy_id = props.get("generated_id")

                baseline = get_reference_spectrum()

                if selected_canopy_id is not None and "generated_id" in spectral_df.columns:
                    subset = spectral_df[spectral_df["generated_id"] == selected_canopy_id]
                    if not subset.empty:
                        canopy = subset[band_cols].apply(pd.to_numeric, errors="coerce").iloc[0].to_numpy(float)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=spectral_x, y=baseline, mode="lines", name="Internal healthy-reference mean"))
                        fig.add_trace(go.Scatter(x=spectral_x, y=canopy, mode="lines", name=f"Tree {selected_canopy_id}"))
                        fig = add_em_regions(fig)
                        fig.update_layout(
                            title=f"Tree {selected_canopy_id} vs Internal Reference",
                            xaxis_title=spectral_x_title,
                            yaxis_title="Reflectance",
                            height=400,
                            margin=dict(l=0, r=0, t=40, b=0),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                elif target_count > 0 and "generated_id" in spectral_df.columns:
                    target_ids = target_gdf["generated_id"].tolist()
                    target_spec = spectral_df[spectral_df["generated_id"].isin(target_ids)]
                    if not target_spec.empty:
                        target_mean = target_spec[band_cols].apply(pd.to_numeric, errors="coerce").mean().to_numpy(float)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=spectral_x, y=baseline, mode="lines", name="Internal healthy-reference mean"))
                        fig.add_trace(go.Scatter(x=spectral_x, y=target_mean, mode="lines", name="Flagged-group mean"))
                        fig = add_em_regions(fig)
                        fig.update_layout(
                            title="Flagged Group vs Internal Reference",
                            xaxis_title=spectral_x_title,
                            yaxis_title="Reflectance",
                            height=400,
                            margin=dict(l=0, r=0, t=40, b=0),
                        )
                        st.plotly_chart(fig, use_container_width=True)
            elif spectral_df.empty:
                st.warning("Hyperspectral CSV not found; signature comparison is unavailable.")

            st.markdown("---")
            st.header("📥 Export & Field Navigation")
            st.caption(
                "KML points are generated from projected canopy centroids and exported in WGS84. "
                "Open them in Google Earth on your phone. Phone GPS can still be off by several metres, "
                "so confirm the Tree ID against the UAV canopy layout when adjacent trees are close."
            )

            if not target_gdf.empty:
                export_col1, export_col2 = st.columns(2)

                with export_col1:
                    st.download_button(
                        f"Download {target_count} targets (GeoJSON)",
                        target_gdf.to_json(),
                        file_name=f"field_targets_{selected_scenario}.geojson",
                        mime="application/geo+json",
                    )

                with export_col2:
                    active_kml = build_tree_navigation_kml(
                        target_gdf,
                        layer_title=scenario_dict[selected_scenario][0],
                        scenario_key=selected_scenario,
                    )
                    st.download_button(
                        f"📍 Download {target_count} targets (KML)",
                        data=active_kml,
                        file_name=f"field_navigation_{selected_scenario}.kml",
                        mime="application/vnd.google-earth.kml+xml",
                    )

            # Combined navigation files are always available from the active map page.
            available_field_flags = [f for f in FIELD_FLAG_COLUMNS if f in gdf.columns]
            if available_field_flags:
                all_flagged_mask = gdf[available_field_flags].fillna(False).astype(bool).any(axis=1)
                all_flagged_gdf = gdf[all_flagged_mask].copy()
            else:
                all_flagged_gdf = gdf.iloc[0:0].copy()

            if not all_flagged_gdf.empty:
                nav_col1, nav_col2 = st.columns(2)

                with nav_col1:
                    all_kml = build_tree_navigation_kml(
                        all_flagged_gdf,
                        layer_title="All Flagged Orchard Trees",
                    )
                    st.download_button(
                        f"🗺️ Download ALL flagged trees ({len(all_flagged_gdf)}) KML",
                        data=all_kml,
                        file_name="field_navigation_ALL_FLAGGED_TREES.kml",
                        mime="application/vnd.google-earth.kml+xml",
                    )

                with nav_col2:
                    # 2+ active A-H flags = field-navigation priority only.
                    multi_flag_count = gdf[available_field_flags].fillna(False).astype(bool).sum(axis=1)
                    multi_flag_gdf = gdf[multi_flag_count >= 2].copy()
                    if not multi_flag_gdf.empty:
                        priority_kml = build_tree_navigation_kml(
                            multi_flag_gdf,
                            layer_title="Multi-Flag Field Priority Trees",
                            scenario_key="HIGH_PRIORITY",
                        )
                        st.download_button(
                            f"⭐ Download multi-flag priority ({len(multi_flag_gdf)}) KML",
                            data=priority_kml,
                            file_name="field_navigation_MULTI_FLAG_PRIORITY.kml",
                            mime="application/vnd.google-earth.kml+xml",
                        )
                    else:
                        st.info("No trees currently have two or more A-H flags.")


# -----------------------------------------------------------------------------
# TAB 2 — INTERNAL VALIDATION / ROBUSTNESS
# -----------------------------------------------------------------------------
with tab_validation:
    st.header("Internal Validation and Robustness Assessment")
    st.warning(
        "No independent ground truth is available. The analyses below evaluate internal consistency, robustness, "
        "spectral/structural corroboration, and separability. They do NOT provide diagnostic accuracy."
    )

    if selected_scenario in {"GAP_ANALYSIS", "CHM_PROFILE"}:
        st.info("Select one of Flags A–H or Multi-Domain High Priority to run rule robustness diagnostics.")
    else:
        flag_col = selected_scenario
        target_count = int(gdf[flag_col].sum()) if flag_col in gdf.columns else 0
        ref_count = int(gdf["IS_HEALTHY_REF"].sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Flagged trees", target_count)
        c2.metric("Internal reference trees", ref_count)
        c3.metric("Reference fraction", f"{100 * ref_count / len(gdf):.1f}%" if len(gdf) else "0%")

        # 2.1 Multi-index concordance.
        st.subheader("1. Multi-index concordance")
        score_map = {
            "Flag_A": ("Water_Score", 3),
            "Flag_B": ("Nutrient_Score", 3),
            "Flag_C": ("Decline_Score", 3),
            "Flag_D": ("Localized_Score", 4),
            "Flag_E": ("Physiology_Score", 3),
            "Flag_G": ("Pigment_Score", 3),
        }
        if flag_col in score_map:
            score_col, max_score = score_map[flag_col]
            if score_col in gdf.columns:
                dist = gdf.loc[gdf[flag_col], score_col].value_counts().sort_index().rename_axis("Evidence score").reset_index(name="Trees")
                st.dataframe(dist, hide_index=True, use_container_width=True)
                st.caption("Higher concordance means more independent indicators support the same anomaly; it is not an accuracy probability.")
        else:
            st.caption("This scenario is a composite or direct structural rule; no dedicated concordance score is defined.")

        # 2.2 Spectral consistency.
        st.subheader("2. Spectral consistency against the internal reference")
        baseline = get_reference_spectrum()
        if target_count > 0 and baseline is not None and "generated_id" in spectral_df.columns:
            ids = gdf.loc[gdf[flag_col], "generated_id"].tolist()
            tdf = spectral_df[spectral_df["generated_id"].isin(ids)]
            if not tdf.empty:
                target_mean = tdf[band_cols].apply(pd.to_numeric, errors="coerce").mean().to_numpy(float)
                sam = spectral_angle_deg(target_mean, baseline)
                rmse = spectral_rmse(target_mean, baseline)
                d1, d2 = st.columns(2)
                d1.metric("Spectral angle vs reference", f"{sam:.3f}°" if np.isfinite(sam) else "NA")
                d2.metric("Reflectance RMSE vs reference", f"{rmse:.5f}" if np.isfinite(rmse) else "NA")

                delta = target_mean - baseline
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=spectral_x, y=delta, mode="lines", name="Flagged − reference"))
                fig.add_hline(y=0, line_dash="dash")
                fig = add_em_regions(fig)
                fig.update_layout(
                    title="Mean Spectral Difference",
                    xaxis_title=spectral_x_title,
                    yaxis_title="Δ Reflectance",
                    height=360,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Spectral comparison requires flagged trees, a reference group, and the averaged hyperspectral CSV.")

        # 2.3 Threshold robustness.
        st.subheader("3. Threshold sensitivity and Jaccard robustness")
        stability, jac, runs = threshold_sensitivity(base_gdf, flag_col, pri_direction)
        if not stability.empty:
            base_mask = runs.get("P25/P75")
            if base_mask is not None and base_mask.any():
                stable_fraction = float((stability.loc[base_mask, "stability"] == 1.0).mean())
                mean_stability = float(stability.loc[base_mask, "stability"].mean())
                e1, e2 = st.columns(2)
                e1.metric("Main-rule targets stable in all 3 schemes", f"{100 * stable_fraction:.1f}%")
                e2.metric("Mean target stability", f"{100 * mean_stability:.1f}%")
            st.write("**Jaccard similarity among threshold schemes**")
            st.dataframe(jac.round(3), use_container_width=True)
            st.caption("P20/P80, P25/P75 and P30/P70 perturb the percentile cutoffs. High Jaccard agreement indicates lower sensitivity to one arbitrary threshold choice.")
        else:
            st.info("Threshold sensitivity could not be computed for this scenario.")

        # 2.4 Statistical comparison with internally defined reference group.
        st.subheader("4. Flagged vs reference feature distributions")
        scenario_metrics = {
            "Flag_A": ["WBI_mn", "PRI_mn", "NDVI_mn"],
            "Flag_B": ["MCARI_mn", "NDRE_mn", "MTCI_mn", "CIRED_mn", "CCCI_mn"],
            "Flag_C": ["NDVI_mn", "PSRI_mn", structural_col],
            "Flag_D": ["NDVI_sd", "CRI1_sd", "NDVI_mi", "SIPI_sd", "PSRI_sd"],
            "Flag_E": ["PRI_mn", "WBI_mn", "PSRI_mn", "NDVI_mn"],
            "Flag_F": ["NDVI_mn", "NDRE_mn", "MTCI_mn", "PSRI_mn"],
            "Flag_G": ["PSRI_mn", "SIPI_mn", "CRI1_mn", "CRI2_mn", "CRI1_sd"],
            "Flag_H": [structural_col],
            "HIGH_PRIORITY": ["NDVI_mn", "NDRE_mn", "WBI_mn", "PRI_mn", "PSRI_mn", structural_col],
        }
        metrics = [m for m in scenario_metrics.get(flag_col, []) if m]
        stats_df = comparison_statistics(gdf, flag_col, metrics)
        if not stats_df.empty:
            st.dataframe(stats_df.round(5), hide_index=True, use_container_width=True)
            if not SCIPY_AVAILABLE:
                st.caption("SciPy unavailable: medians are shown, but Mann–Whitney tests were skipped.")
            else:
                st.caption("These tests assess distributional differences from the internal reference; they do not establish causal diagnosis or external accuracy.")
        else:
            st.info("Insufficient flagged/reference observations for statistical comparison.")

        # 2.5 Structural corroboration.
        st.subheader("5. Structural corroboration")
        if structural_col:
            st.write(f"Primary structural metric: **{structural_col}**")
            if structural_col == "CHM_max":
                st.warning("CHM_max is a fallback. Generate per-tree CHM P95 in the upstream point-cloud workflow for a more outlier-robust structural metric.")
            if flag_col in {"Flag_C", "Flag_H", "HIGH_PRIORITY"} and not stats_df.empty:
                st.caption("For decline/structural classes, agreement between spectral anomalies and reduced CHM strengthens internal plausibility but is not ground-truth validation.")
        else:
            st.info("No point-cloud/CHM metric is available in the tree table; structural corroboration is disabled.")

        # 2.6 PCA feature-space corroboration.
        st.subheader("6. Multivariate feature-space corroboration (PCA)")
        pca_features = [
            "NDVI_mn", "OSAVI_mn", "MCARI_mn", "LCI_mn", "NDRE_mn", "MTCI_mn", "CIRED_mn",
            "WBI_mn", "PRI_mn", "PSRI_mn", "CRI1_mn", "CRI2_mn", "SIPI_mn", structural_col,
        ]
        pca_features = [c for c in pca_features if c]
        pca_df, explained, used_features = pca_feature_space(gdf, pca_features, flag_col)
        if not pca_df.empty:
            fig = px.scatter(
                pca_df,
                x="PC1",
                y="PC2",
                color="Group",
                symbol="Reference",
                hover_data=["generated_id"],
                title=f"PCA feature space — PC1 {explained[0]:.1f}%, PC2 {explained[1]:.1f}%",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Exploratory only: separability supports feature-space distinction but does not prove biological class identity.")
            with st.expander("PCA features used"):
                st.write(used_features)
        else:
            st.info("At least three sufficiently populated numeric features are needed for PCA.")

        # 2.7 Spatial coherence.
        st.subheader("7. Spatial coherence (exploratory kNN Moran's I)")
        I, p_perm, prevalence, lift = morans_i_knn(gdf, flag_col, k=4, permutations=199)
        if np.isfinite(I):
            s1, s2, s3 = st.columns(3)
            s1.metric("Moran's I", f"{I:.3f}")
            s2.metric("Permutation pseudo-p", f"{p_perm:.3f}")
            s3.metric("Flagged-neighbor lift", f"{lift:.2f}×" if np.isfinite(lift) else "NA")
            st.caption("Positive spatial coherence can indicate systematic orchard patterns; it does not identify the causal stressor.")
        else:
            st.info("Spatial coherence requires a non-trivial mix of flagged and unflagged trees.")

        # 2.8 Limitations / future external validation.
        st.subheader("8. Interpretation boundary")
        st.markdown(
            "- Report these analyses as **internal consistency and robustness assessment**, not ground-truth accuracy.\n"
            "- Use terms such as *water-stress candidate*, *nutrient/chlorophyll anomaly*, *chronic canopy decline*, and *localized canopy anomaly*.\n"
            "- Future campaigns should add independent leaf chemistry/SPAD, water-status/soil-moisture, disease scoring, and field tree-height/crown measurements for external validation."
        )


# -----------------------------------------------------------------------------
# TAB 3 — WORKFLOW + CHANGE LOG
# -----------------------------------------------------------------------------
with tab_workflow:
    st.header("Integrated Research Workflow")
    st.code(
        """
UAV Headwall hyperspectral acquisition + GNSS/IMU + reference target
                              │
                              ▼
Headwall proprietary preprocessing
(radiometric calibration → reflectance → geometric correction → orthorectification)
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
Analysis-ready hyperspectral             Derived 3-D products
orthomosaic / tree spectra               point cloud → DSM/DTM → CHM
             │                                 │
             └────────────────┬────────────────┘
                              ▼
                    Individual tree crowns
                              │
        ┌─────────────────────┼──────────────────────┐
        ▼                     ▼                      ▼
Spectral/biochemical       Structure             Spatial heterogeneity
NDVI/OSAVI                 CHM P95/mean          within-crown SD/min/CV
MCARI/LCI                  (no LAI rule)          (no radius rule)
NDRE/MTCI/CIred-edge
WBI/PRI
CRI1/CRI2/PSRI/SIPI
REP + derivatives
        │                     │                      │
        └─────────────────────┼──────────────────────┘
                              ▼
                    Tree-level feature database
                              │
                              ▼
          Multi-domain rule engine + evidence tiers
                              │
                              ▼
 Internal consistency / robustness / spectral-structural corroboration
                              │
                              ▼
                 Streamlit mapping + target export
                              │
                              ▼
              Future field confirmation / external validation
        """,
        language="text",
    )

    st.subheader("What changed from the original code")
    changes = pd.DataFrame(
        [
            ["Healthy reference", "NDVI ≥ P75 + LAI ≥ P75", "High NDVI + adequate NDRE/MCARI + WBI + low PSRI", "Removes unvalidated LAI dependence and uses multiple physiological domains."],
            ["Nutrient rule", "High LAI + low MCARI", "Reasonable NDVI + low MCARI + low NDRE; MTCI/CIred-edge support", "Red-edge concordance is more hyperspectral and less dependent on one model-derived variable."],
            ["Decline rule", "Radius + LAI + PSRI", "Low NDVI + high PSRI; CHM supports high-confidence decline", "Radius was indicative and LAI was not trusted; both removed from classification."],
            ["Localized pest rule", "NDVI/CRI variability → pest", "Same heterogeneity logic → localized canopy anomaly", "Avoids unsupported causal diagnosis."],
            ["Heat/frost rule", "PRI + LAI + PSRI", "PRI + NDVI + PSRI; WBI separates water-associated stress", "Removes LAI and avoids asserting heat/frost without meteorological/field evidence."],
            ["Structure", "CHM >= 0.5 m profile", "CHM P95 preferred; orchard-relative P25 structural anomaly", "Relative 3-D structure is more defensible than a universal 0.5 m health cutoff."],
            ["Radius_m", "Used in Flag C", "Never used in health rules", "Indicative radius could propagate geometric error into classification."],
            ["LAI raster", "Displayed as UAV overlay", "Removed; CHM raster overlay used instead", "UI now matches the revised structural methodology."],
            ["Hyperspectral spectra", "Band-number x-axis with hard-coded regions", "Actual wavelength (nm) when metadata are available", "Improves reproducibility and physical interpretability."],
            ["New indices", "Existing ENVI index set", "NDRE, MTCI, CIred-edge, SIPI + research VOG1/MTVI2/maxLARE/narrow ratios", "Covers red-edge, pigment, and genuinely hyperspectral information."],
            ["Validation", "No formal internal robustness layer", "Concordance + spectral comparison + sensitivity + Jaccard + statistics + PCA + spatial coherence", "Supports internal plausibility without falsely claiming ground-truth accuracy."],
            ["LLM", "Can generate prescriptions from active flag", "Still downstream, explicitly told flags are unvalidated anomaly candidates", "Keeps deterministic science separate from natural-language assistance."],
        ],
        columns=["Component", "Old workflow", "New workflow", "Research rationale"],
    )
    st.dataframe(changes, hide_index=True, use_container_width=True)

    st.subheader("Feature availability in the current dataset")
    desired = [
        "NDVI_mn", "OSAVI_mn", "GCI_mn", "LCI_mn", "MCARI_mn", "REP_mn",
        "NDRE_mn", "MTCI_mn", "CIRED_mn", "CCCI_mn", "WBI_mn", "PRI_mn",
        "CRI1_mn", "CRI2_mn", "PSRI_mn", "SIPI_mn", "maxLARE", structural_col,
    ]
    desired = list(dict.fromkeys([x for x in desired if x]))
    availability = pd.DataFrame({
        "Feature": desired,
        "Available": [x in gdf.columns and pd.to_numeric(gdf[x], errors="coerce").notna().any() for x in desired],
    })
    st.dataframe(availability, hide_index=True, use_container_width=True)

    if computed_index_details:
        st.subheader("New HSI features calculated from spectral CSV")
        detail_rows = []
        for name, info in computed_index_details.items():
            detail_rows.append({"Feature": name, "Band selection / metadata": str(info)})
        st.dataframe(pd.DataFrame(detail_rows), hide_index=True, use_container_width=True)

    st.subheader("Point-cloud / CHM integration note")
    st.markdown(
        "The Streamlit app is a **downstream analysis and decision-support layer**. It does not reconstruct the point cloud. "
        "Describe the point cloud in the manuscript as a **derived processing product** of the UAV processing chain only if your Headwall processing report confirms that provenance. "
        "The app consumes per-tree CHM metrics (preferably CHM P95) and/or an optional CHM raster generated upstream."
    )


# =============================================================================
# 13. LLM ASSISTANT — DOWNSTREAM ONLY
# =============================================================================

with st.sidebar.expander("🤖 LLM Diagnostic Assistant", expanded=False):
    st.markdown("Natural-language support only. Deterministic flags are calculated before this assistant is called.")

    if not GEMINI_AVAILABLE:
        st.info("google-generativeai is not installed; the rest of the app remains fully functional.")
    else:
        api_key = st.text_input("Enter Gemini API Key", type="password")
        model_name = st.text_input("Gemini model", value=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        active_name = scenario_dict[selected_scenario][0]
        active_rationale = scenario_dict[selected_scenario][3]

        if selected_scenario == "GAP_ANALYSIS":
            spatial_context = (
                f"Total trees: {total_trees}; calculated gaps: {total_gaps}; "
                f"estimated planting-capacity loss: {yield_loss_percentage:.2f}%."
            )
        else:
            target_count = len(target_gdf)
            spatial_context = (
                f"Total trees: {len(gdf)}; flagged targets: {target_count} "
                f"({100 * target_count / len(gdf) if len(gdf) else 0:.1f}% of block)."
            )

        system_prompt = f"""
You are an expert precision-agriculture decision-support assistant for an arid orchard UAV hyperspectral study.

CURRENT ACTIVE LAYER: {active_name}
SCIENTIFIC INTERPRETATION: {active_rationale}
CURRENT SPATIAL STATISTICS: {spatial_context}

Critical constraints:
- The remote-sensing classes have NO independent ground truth in this campaign.
- Treat all health outputs as anomaly candidates / inspection priorities, not confirmed diagnoses.
- Do not convert 'chronic canopy decline' into root rot, or 'localized anomaly' into a pest/pathogen diagnosis without field evidence.
- LAI and Radius_m are not used in the diagnostic rules.
- CHM/point-cloud metrics provide the independent structural domain where available.
- Keep answers concise, scientific, and focused on agronomy/remote sensing.
"""

        if st.button("Generate Field Inspection Plan"):
            if not api_key:
                st.warning("Please enter a valid API key.")
            else:
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(model_name, system_instruction=system_prompt)
                    auto_prompt = (
                        "Using the active anomaly layer and counts, give a short field-inspection plan. "
                        "Separate what the remote-sensing evidence supports from what must be confirmed in the field."
                    )
                    with st.chat_message("assistant"):
                        placeholder = st.empty()
                        full = ""
                        for chunk in model.generate_content(auto_prompt, stream=True):
                            full += chunk.text
                            placeholder.markdown(full + "▌")
                        placeholder.markdown(full)
                    st.session_state.messages.append({"role": "assistant", "content": full})
                except Exception as exc:
                    st.error(f"API Error: {exc}")

        if user_query := st.chat_input("Ask a follow-up question..."):
            if not api_key:
                st.warning("Please enter a valid API key.")
            else:
                st.session_state.messages.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.markdown(user_query)
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(model_name, system_instruction=system_prompt)
                    history = [
                        {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
                        for m in st.session_state.messages[:-1]
                    ]
                    chat = model.start_chat(history=history)
                    with st.chat_message("assistant"):
                        placeholder = st.empty()
                        full = ""
                        response = chat.send_message(user_query, stream=True)
                        for chunk in response:
                            full += chunk.text
                            placeholder.markdown(full + "▌")
                        placeholder.markdown(full)
                    st.session_state.messages.append({"role": "assistant", "content": full})
                except Exception as exc:
                    st.error(f"API Error: {exc}")
