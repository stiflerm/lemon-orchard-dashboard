"""
Orchard Diagnostic Intelligence v3 — Final Three-Domain Engine
===============================================================

Final deterministic domains
---------------------------
1. WATER STATUS
2. CANOPY BIOCHEMICAL STATUS
3. STRUCTURE

The application consumes the finalized tree-level rule database produced upstream.
It does NOT recalculate the hyperspectral or LAS measurements used for diagnosis.

Preferred input
---------------
project/
  orchard_diagnostic_intelligence_v3_three_domain.py
  data/
    data.zip
    master_tree_multidomain_FINAL_compact.csv
    master_tree_spectra_CONSENSUS.csv           # optional, spectral display only

Fallback input
--------------
If master_tree_multidomain_FINAL_compact.csv is absent, the app can rebuild the
same final three-domain table from:
  data/master_tree_rule_database_VNIR_SWIR.csv
  data/structural_domain_FINAL_compact.csv

Scientific boundary
-------------------
Cross-domain agreement is internal corroboration. It is NOT causal proof and is
NOT ground-truth diagnostic accuracy. "No supported anomaly" means no domain
reached the supported-evidence condition; it does not mean "healthy".
"""

from __future__ import annotations

import html
import math
import os
import tempfile
import warnings
import zipfile
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from shapely.geometry import Point
from sklearn.cluster import AgglomerativeClustering
from streamlit_folium import st_folium

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

warnings.filterwarnings("ignore")


# =============================================================================
# 0. CONFIGURATION
# =============================================================================

APP_TITLE = "🍋 Orchard Diagnostic Intelligence — Three-Domain Engine"
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
ZIP_PATH = DATA_DIR / "data.zip"

FINAL_DB_CANDIDATES = [
    DATA_DIR / "master_tree_multidomain_FINAL_compact.csv",
    DATA_DIR / "master_tree_multidomain_FINAL.csv",
    APP_DIR / "master_tree_multidomain_FINAL_compact.csv",
    APP_DIR / "master_tree_multidomain_FINAL.csv",
]

VNIR_SWIR_CANDIDATES = [
    DATA_DIR / "master_tree_rule_database_VNIR_SWIR.csv",
    APP_DIR / "master_tree_rule_database_VNIR_SWIR.csv",
]

STRUCTURE_CANDIDATES = [
    DATA_DIR / "structural_domain_FINAL_compact.csv",
    APP_DIR / "structural_domain_FINAL_compact.csv",
]

# Optional full-spectrum product used ONLY for per-tree spectral visualization.
# It is NOT used to calculate Water, Biochemical, Structure, or the final priority.
# Prefer ONLY the post-QC consensus spectrum. The pre-QC master_tree_spectra.csv
# is deliberately not used in the final app, even for display.
SPECTRAL_CSV_CANDIDATES = [
    DATA_DIR / "master_tree_spectra_CONSENSUS.csv",
    APP_DIR / "master_tree_spectra_CONSENSUS.csv",
    Path(r"D:\hx_UAV_data_processing\object_based_tree_hsi\spectral_consensus_qc\master_tree_spectra_CONSENSUS.csv"),
]

# Final frozen operational thresholds from the completed study workflow.
BIO_THRESHOLDS = {
    "NDRE_LOW_MAX": 0.2047129778647948,
    "CIRED_LOW_MAX": 0.5487572048699663,
    "REP_LOW_MAX_NM": 725.3890241067692,
    "PSRI_HIGH_MIN": 0.2317542293068353,
    "SIPI_HIGH_MIN": 1.6044129002460532,
    "ARI1_HIGH_MIN": 4.409562434704582,
}

STRUCTURE_THRESHOLDS = {
    "H_P95_P20_M": 1.492920999526977,
    "H_P95_P25_M": 1.6373457133769989,
    "H_P95_P30_M": 1.7331793653964995,
    "H_IQR_P75_M": 1.3969939574599266,
    "RASTER_CHM_P95_P25_M": 1.413568115234375,
}

DEFAULT_TREE_SPACING_M = 5.5
DEFAULT_ROW_DISTANCE_THRESHOLD_M = 2.5
DEFAULT_GRID_ANGLE_DEG = 75.0
DEFAULT_MAX_EMPTY_SPACE_M = 20.0


# =============================================================================
# 1. GENERAL HELPERS
# =============================================================================

def first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None


def fmt(value, digits=3):
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def as_bool(series):
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def valid_index(df, value_col, qc_col):
    values = pd.to_numeric(df[value_col], errors="coerce")
    qc = df[qc_col].astype(str)
    return values.notna() & qc.isin(["PASS", "WARN"])


# =============================================================================
# 2. FINAL THREE-DOMAIN TABLE — FALLBACK RECONSTRUCTION
# =============================================================================

def reconstruct_final_database(master: pd.DataFrame, structure: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the same finalized three-domain table used in the analysis."""

    required = [
        "tree_id", "WATER_EVIDENCE_STATUS",
        "NDRE_VALUE", "NDRE_QC", "CIRED_EDGE_VALUE", "CIRED_EDGE_QC",
        "REP_D1_NM_VALUE", "REP_D1_NM_QC", "PSRI_VALUE", "PSRI_QC",
        "SIPI_VALUE", "SIPI_QC", "ARI1_VALUE", "ARI1_QC",
    ]
    missing = [c for c in required if c not in master.columns]
    if missing:
        raise ValueError("VNIR/SWIR master is missing required columns: " + ", ".join(missing))

    if "tree_id" not in structure.columns:
        raise ValueError("Structural table has no tree_id column.")

    if master["tree_id"].duplicated().any() or structure["tree_id"].duplicated().any():
        raise ValueError("Duplicate tree_id detected in final-engine source tables.")

    if set(master["tree_id"]) != set(structure["tree_id"]):
        raise ValueError("tree_id sets do not match between VNIR/SWIR and structure tables.")

    df = master.copy()

    ndre_valid = valid_index(df, "NDRE_VALUE", "NDRE_QC")
    cired_valid = valid_index(df, "CIRED_EDGE_VALUE", "CIRED_EDGE_QC")
    rep_valid = valid_index(df, "REP_D1_NM_VALUE", "REP_D1_NM_QC")
    psri_valid = valid_index(df, "PSRI_VALUE", "PSRI_QC")
    sipi_valid = valid_index(df, "SIPI_VALUE", "SIPI_QC")
    ari1_valid = valid_index(df, "ARI1_VALUE", "ARI1_QC")

    NDRE = pd.to_numeric(df["NDRE_VALUE"], errors="coerce")
    CIRED = pd.to_numeric(df["CIRED_EDGE_VALUE"], errors="coerce")
    REP = pd.to_numeric(df["REP_D1_NM_VALUE"], errors="coerce")
    PSRI = pd.to_numeric(df["PSRI_VALUE"], errors="coerce")
    SIPI = pd.to_numeric(df["SIPI_VALUE"], errors="coerce")
    ARI1 = pd.to_numeric(df["ARI1_VALUE"], errors="coerce")

    df["BIO_NDRE_LOW"] = ndre_valid & (NDRE <= BIO_THRESHOLDS["NDRE_LOW_MAX"])
    df["BIO_CIRED_LOW"] = cired_valid & (CIRED <= BIO_THRESHOLDS["CIRED_LOW_MAX"])
    df["BIO_REP_LOW"] = rep_valid & (REP <= BIO_THRESHOLDS["REP_LOW_MAX_NM"])
    df["BIO_CHL_AMPLITUDE_ABNORMAL"] = df["BIO_NDRE_LOW"] & df["BIO_CIRED_LOW"]
    df["BIO_CHL_RE_SUPPORTED"] = df["BIO_CHL_AMPLITUDE_ABNORMAL"] & df["BIO_REP_LOW"]

    df["BIO_PSRI_HIGH"] = psri_valid & (PSRI >= BIO_THRESHOLDS["PSRI_HIGH_MIN"])
    df["BIO_SIPI_HIGH"] = sipi_valid & (SIPI >= BIO_THRESHOLDS["SIPI_HIGH_MIN"])
    df["BIO_PIGMENT_SUPPORTED"] = df["BIO_PSRI_HIGH"] & df["BIO_SIPI_HIGH"]
    df["BIO_ARI1_HIGH"] = ari1_valid & (ARI1 >= BIO_THRESHOLDS["ARI1_HIGH_MIN"])

    df["BIOCHEMICAL_FULLY_ASSESSABLE"] = (
        ndre_valid & cired_valid & rep_valid & psri_valid & sipi_valid & ari1_valid
    )

    df["BIOCHEMICAL_STATUS"] = "NOT_FULLY_ASSESSABLE"
    ok = df["BIOCHEMICAL_FULLY_ASSESSABLE"]
    chl = df["BIO_CHL_RE_SUPPORTED"]
    pig = df["BIO_PIGMENT_SUPPORTED"]
    ari = df["BIO_ARI1_HIGH"]

    df.loc[ok & ~chl & ~pig, "BIOCHEMICAL_STATUS"] = "NO_CONCORDANT_BIOCHEMICAL_DECLINE"
    df.loc[ok & chl & ~pig, "BIOCHEMICAL_STATUS"] = "CHLOROPHYLL_RED_EDGE_ONLY"
    df.loc[ok & ~chl & pig, "BIOCHEMICAL_STATUS"] = "SENESCENCE_PIGMENT_ONLY"
    df.loc[ok & chl & pig, "BIOCHEMICAL_STATUS"] = "CONCORDANT_BIOCHEMICAL_DECLINE"
    df.loc[ok & chl & pig & ari, "BIOCHEMICAL_STATUS"] = "STRONG_CONCORDANT_BIOCHEMICAL_DECLINE"

    structure_cols = [c for c in structure.columns if c != "generated_id"]
    df = df.merge(structure[structure_cols], on="tree_id", how="left", validate="one_to_one")

    def water_state(s):
        if s in {"STRONG_INTERNAL_WATER_EVIDENCE", "SUPPORTED_WATER_STATUS_ANOMALY"}:
            return "SUPPORTED_ANOMALY"
        if s in {"ISOLATED_DIRECT_WATER_ANOMALY", "NOT_FULLY_ASSESSABLE_ISOLATED_ANOMALY"}:
            return "SCREENING_ANOMALY"
        if s == "NO_CONCORDANT_WATER_ANOMALY":
            return "NO_CONCORDANT_ANOMALY"
        if s in {"SWIR_DISCORDANT_INSUFFICIENT_EVIDENCE", "NOT_FULLY_ASSESSABLE_SWIR_DISCORDANT"}:
            return "INCONCLUSIVE"
        if s == "WATER_STATUS_NOT_FULLY_ASSESSABLE":
            return "NOT_FULLY_ASSESSABLE"
        return "UNKNOWN"

    def bio_state(s):
        if s in {"CONCORDANT_BIOCHEMICAL_DECLINE", "STRONG_CONCORDANT_BIOCHEMICAL_DECLINE"}:
            return "SUPPORTED_ANOMALY"
        if s in {"CHLOROPHYLL_RED_EDGE_ONLY", "SENESCENCE_PIGMENT_ONLY"}:
            return "SCREENING_ANOMALY"
        if s == "NO_CONCORDANT_BIOCHEMICAL_DECLINE":
            return "NO_CONCORDANT_ANOMALY"
        if s == "NOT_FULLY_ASSESSABLE":
            return "NOT_FULLY_ASSESSABLE"
        return "UNKNOWN"

    def structure_state(s):
        if s == "CORROBORATED_LOW_STRUCTURAL_STATURE":
            return "LOW_STATURE_CORROBORATED"
        if s == "LOW_STATURE_LAS_ONLY":
            return "LOW_STATURE_LAS_ONLY"
        if s == "NO_LOW_STATURE_EVIDENCE":
            return "NO_LOW_STATURE_EVIDENCE"
        if s == "NOT_ASSESSABLE":
            return "NOT_ASSESSABLE"
        return "UNKNOWN"

    df["WATER_DOMAIN_STATE"] = df["WATER_EVIDENCE_STATUS"].apply(water_state)
    df["BIOCHEMICAL_DOMAIN_STATE"] = df["BIOCHEMICAL_STATUS"].apply(bio_state)
    df["STRUCTURE_DOMAIN_STATE"] = df["STRUCTURE_STATUS_FINAL"].apply(structure_state)

    df["WATER_SUPPORTED"] = df["WATER_DOMAIN_STATE"].eq("SUPPORTED_ANOMALY")
    df["BIOCHEMICAL_SUPPORTED"] = df["BIOCHEMICAL_DOMAIN_STATE"].eq("SUPPORTED_ANOMALY")
    df["STRUCTURE_LOW_STATURE"] = df["STRUCTURE_DOMAIN_STATE"].isin(
        ["LOW_STATURE_CORROBORATED", "LOW_STATURE_LAS_ONLY"]
    )
    df["STRUCTURE_CORROBORATED"] = df["STRUCTURE_DOMAIN_STATE"].eq("LOW_STATURE_CORROBORATED")

    df["WATER_SCREENING_ONLY"] = df["WATER_DOMAIN_STATE"].eq("SCREENING_ANOMALY")
    df["BIOCHEMICAL_SCREENING_ONLY"] = df["BIOCHEMICAL_DOMAIN_STATE"].eq("SCREENING_ANOMALY")
    df["ANY_SCREENING_LEVEL_EVIDENCE"] = df["WATER_SCREENING_ONLY"] | df["BIOCHEMICAL_SCREENING_ONLY"]

    df["SUPPORTED_DOMAIN_COUNT"] = (
        df["WATER_SUPPORTED"].astype(int)
        + df["BIOCHEMICAL_SUPPORTED"].astype(int)
        + df["STRUCTURE_LOW_STATURE"].astype(int)
    )

    df["WATER_FULLY_ASSESSABLE"] = ~df["WATER_EVIDENCE_STATUS"].isin([
        "WATER_STATUS_NOT_FULLY_ASSESSABLE",
        "NOT_FULLY_ASSESSABLE_ISOLATED_ANOMALY",
        "NOT_FULLY_ASSESSABLE_SWIR_DISCORDANT",
    ])
    df["BIOCHEMICAL_ASSESSABLE"] = ~df["BIOCHEMICAL_STATUS"].eq("NOT_FULLY_ASSESSABLE")
    df["STRUCTURE_ASSESSABLE"] = ~df["STRUCTURE_STATUS_FINAL"].eq("NOT_ASSESSABLE")
    df["ASSESSABLE_DOMAIN_COUNT"] = (
        df["WATER_FULLY_ASSESSABLE"].astype(int)
        + df["BIOCHEMICAL_ASSESSABLE"].astype(int)
        + df["STRUCTURE_ASSESSABLE"].astype(int)
    )
    df["MULTIDOMAIN_COMPLETENESS"] = np.select(
        [df["ASSESSABLE_DOMAIN_COUNT"].eq(3), df["ASSESSABLE_DOMAIN_COUNT"].eq(2)],
        ["COMPLETE_3_OF_3", "PARTIAL_2_OF_3"],
        default="LIMITED_0_OR_1_OF_3",
    )

    W = df["WATER_SUPPORTED"]
    B = df["BIOCHEMICAL_SUPPORTED"]
    S = df["STRUCTURE_LOW_STATURE"]
    df["CROSS_DOMAIN_PATTERN"] = np.select(
        [W & B & S, W & B & ~S, W & ~B & S, ~W & B & S, W & ~B & ~S, ~W & B & ~S, ~W & ~B & S],
        [
            "WATER_BIOCHEMICAL_STRUCTURE",
            "WATER_BIOCHEMICAL_ONLY",
            "WATER_STRUCTURE_ONLY",
            "BIOCHEMICAL_STRUCTURE_ONLY",
            "WATER_ONLY",
            "BIOCHEMICAL_ONLY",
            "STRUCTURE_ONLY",
        ],
        default="NO_SUPPORTED_MAJOR_DOMAIN_ANOMALY",
    )

    inconclusive = df["WATER_DOMAIN_STATE"].eq("INCONCLUSIVE")
    df["FIELD_INSPECTION_TIER"] = np.select(
        [
            df["SUPPORTED_DOMAIN_COUNT"].eq(3),
            df["SUPPORTED_DOMAIN_COUNT"].eq(2),
            df["SUPPORTED_DOMAIN_COUNT"].eq(1),
            df["SUPPORTED_DOMAIN_COUNT"].eq(0) & df["ANY_SCREENING_LEVEL_EVIDENCE"],
            df["SUPPORTED_DOMAIN_COUNT"].eq(0) & (df["ASSESSABLE_DOMAIN_COUNT"].lt(3) | inconclusive),
        ],
        [
            "HIGH_MULTI_DOMAIN_PRIORITY",
            "MULTI_DOMAIN_PRIORITY",
            "SINGLE_DOMAIN_PRIORITY",
            "SCREENING_PRIORITY",
            "DATA_LIMITED_REVIEW",
        ],
        default="NO_SUPPORTED_ANOMALY",
    )

    def interpretation(row):
        w, b, s = bool(row["WATER_SUPPORTED"]), bool(row["BIOCHEMICAL_SUPPORTED"]), bool(row["STRUCTURE_LOW_STATURE"])
        if w and b and s:
            text = "Supported water-status anomaly and supported biochemical decline coincide with relatively low structural stature."
        elif w and b:
            text = "Supported water-status and biochemical anomalies are present without current low-stature evidence."
        elif w and s:
            text = "Supported water-status anomaly coincides with relatively low structural stature; biochemical decline is not supported at domain level."
        elif b and s:
            text = "Supported biochemical decline coincides with relatively low structural stature; a supported water-status anomaly is not present."
        elif w:
            text = "Supported water-status anomaly is present without another supported major-domain signal."
        elif b:
            text = "Supported biochemical decline is present without another supported major-domain signal."
        elif s:
            text = "Relatively low structural stature is present without another supported major-domain anomaly."
        elif bool(row["ANY_SCREENING_LEVEL_EVIDENCE"]):
            text = "Screening-level spectral evidence is present, but no major domain reaches the supported multi-indicator condition."
        elif row["ASSESSABLE_DOMAIN_COUNT"] < 3 or row["WATER_DOMAIN_STATE"] == "INCONCLUSIVE":
            text = "No supported major-domain anomaly is identified, but the assessment is incomplete or contains inconclusive evidence."
        else:
            text = "No supported concordant anomaly evidence is identified across the assessed domains."
        if row["STRUCTURE_DOMAIN_STATE"] == "LOW_STATURE_LAS_ONLY":
            text += " Structural low-stature evidence is based on LAS H_P95 without raster-CHM corroboration."
        return text

    df["CROSS_DOMAIN_INTERPRETATION"] = df.apply(interpretation, axis=1)
    df["INTERPRETATION_BOUNDARY"] = (
        "Cross-domain agreement represents internal corroboration. It does not prove that water status, "
        "biochemical change, disease, or another factor caused the structural condition."
    )
    return df


# =============================================================================
# 3. DATA INGESTION
# =============================================================================

@st.cache_data(show_spinner=False)
def load_tree_geometry() -> gpd.GeoDataFrame:
    if not ZIP_PATH.exists():
        st.error(f"Canopy bundle not found: {ZIP_PATH}")
        st.stop()

    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(temp_dir)

    shp_file = None
    for root, _, files in os.walk(temp_dir):
        for file in files:
            if file.lower().endswith(".shp"):
                shp_file = os.path.join(root, file)
                break
        if shp_file:
            break

    if shp_file is None:
        st.error("No .shp file found inside data.zip")
        st.stop()

    gdf = gpd.read_file(shp_file)
    if "tree_id" not in gdf.columns:
        st.error("The canopy shapefile must contain the original tree_id field. The app will not regenerate IDs.")
        st.stop()

    gdf = gdf.drop_duplicates(subset=["tree_id"]).copy()
    if gdf["tree_id"].duplicated().any():
        st.error("Duplicate tree_id remains after geometry de-duplication.")
        st.stop()

    return gdf.to_crs(epsg=4326)


@st.cache_data(show_spinner=False)
def load_final_rule_database():
    final_path = first_existing(FINAL_DB_CANDIDATES)
    if final_path is not None:
        df = pd.read_csv(final_path)
        return df, f"Final multidomain database: {final_path.name}"

    vnir_path = first_existing(VNIR_SWIR_CANDIDATES)
    structure_path = first_existing(STRUCTURE_CANDIDATES)
    if vnir_path is None or structure_path is None:
        st.error(
            "Final rule database is missing. Put master_tree_multidomain_FINAL_compact.csv in data/. "
            "Fallback requires both master_tree_rule_database_VNIR_SWIR.csv and structural_domain_FINAL_compact.csv."
        )
        st.stop()

    master = pd.read_csv(vnir_path)
    structure = pd.read_csv(structure_path)
    df = reconstruct_final_database(master, structure)
    return df, f"Reconstructed from {vnir_path.name} + {structure_path.name}"


@st.cache_data(show_spinner=False)
def load_spectral_data():
    path = first_existing(SPECTRAL_CSV_CANDIDATES)
    if path is None:
        return pd.DataFrame()
    return pd.read_csv(path)


def merge_geometry_and_rules(geometry: gpd.GeoDataFrame, rules: pd.DataFrame) -> gpd.GeoDataFrame:
    if "tree_id" not in rules.columns:
        raise ValueError("Final rule database has no tree_id column.")
    if rules["tree_id"].duplicated().any():
        raise ValueError("Final rule database contains duplicate tree_id values.")

    geometry_ids = set(geometry["tree_id"])
    rule_ids = set(rules["tree_id"])
    if geometry_ids != rule_ids:
        only_geo = sorted(geometry_ids - rule_ids)[:20]
        only_rule = sorted(rule_ids - geometry_ids)[:20]
        raise ValueError(
            "tree_id mismatch between canopy geometry and final rule database. "
            f"Only geometry: {only_geo}; only rules: {only_rule}"
        )

    # Never use a regenerated sequential ID. tree_id is the canonical join key.
    dup_cols = [c for c in rules.columns if c in geometry.columns and c != "tree_id"]
    safe_geometry = geometry.drop(columns=dup_cols, errors="ignore")
    merged = safe_geometry.merge(rules, on="tree_id", how="left", validate="one_to_one")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=geometry.crs)


# =============================================================================
# 4. FIELD NAVIGATION / KML
# =============================================================================

SCENARIO_COLORS = {
    "OVERVIEW": "#CCCCCC",
    "HIGH_MULTI_DOMAIN_PRIORITY": "#FF0000",
    "MULTI_DOMAIN_PRIORITY": "#FF8C00",
    "SINGLE_DOMAIN_PRIORITY": "#FFD700",
    "SCREENING_PRIORITY": "#00BFFF",
    "DATA_LIMITED_REVIEW": "#808080",
    "NO_SUPPORTED_ANOMALY": "#2E8B57",
    "WATER_SUPPORTED": "#1E90FF",
    "BIOCHEMICAL_SUPPORTED": "#8A2BE2",
    "STRUCTURE_LOW_STATURE": "#32CD32",
    "WATER_BIOCHEMICAL_STRUCTURE": "#B22222",
    "WATER_BIOCHEMICAL_ONLY": "#8B008B",
    "WATER_STRUCTURE_ONLY": "#008080",
    "BIOCHEMICAL_STRUCTURE_ONLY": "#D2691E",
    "STRUCTURE_PROFILE": "#228B22",
    "GAP_ANALYSIS": "#FFFFFF",
}


def make_navigation_points(source_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if source_gdf is None or source_gdf.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

    work = source_gdf.copy()
    if work.crs is None:
        raise ValueError("Canopy layer has no CRS.")

    try:
        projected = work.to_crs(work.estimate_utm_crs()) if work.crs.is_geographic else work.copy()
        points = projected.copy()
        points.geometry = projected.geometry.centroid
        points = points.to_crs(epsg=4326)
    except Exception:
        points = work.to_crs(epsg=4326).copy()
        points.geometry = points.geometry.representative_point()

    points["Longitude"] = points.geometry.x
    points["Latitude"] = points.geometry.y
    return points


def build_tree_navigation_kml(source_gdf: gpd.GeoDataFrame, layer_title: str, color_hex="#FF0000") -> bytes:
    points = make_navigation_points(source_gdf)
    if points.empty:
        return b""

    # KML uses AABBGGRR.
    h = color_hex.lstrip("#")
    rr, gg, bb = h[0:2], h[2:4], h[4:6]
    kml_color = f"ff{bb}{gg}{rr}"

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        '<Document>',
        f'<name>{html.escape(layer_title)}</name>',
        '<Style id="treeTarget"><IconStyle>',
        f'<color>{kml_color}</color><scale>1.15</scale>',
        '<Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>',
        '</IconStyle></Style>',
    ]

    detail_fields = [
        "FIELD_INSPECTION_TIER", "CROSS_DOMAIN_PATTERN", "SUPPORTED_DOMAIN_COUNT",
        "WATER_EVIDENCE_STATUS", "BIOCHEMICAL_STATUS", "STRUCTURE_STATUS_FINAL",
        "WBI_VALUE", "NDMI2_VALUE", "NDSI_RWC_VALUE", "PRI_VALUE",
        "NDRE_VALUE", "CIRED_EDGE_VALUE", "REP_D1_NM_VALUE", "PSRI_VALUE", "SIPI_VALUE", "ARI1_VALUE",
        "H_P95_m", "RASTER_CHM_P95_m", "H_IQR_m",
    ]

    for _, row in points.iterrows():
        tree_id = row.get("tree_id", "NA")
        desc = [
            f"<b>Tree ID:</b> {html.escape(str(tree_id))}",
            f"<b>Latitude:</b> {float(row['Latitude']):.7f}",
            f"<b>Longitude:</b> {float(row['Longitude']):.7f}",
        ]
        for field in detail_fields:
            if field in row.index and pd.notna(row.get(field)):
                value = row.get(field)
                if isinstance(value, (float, np.floating)):
                    value = f"{float(value):.4f}"
                desc.append(f"<b>{html.escape(field)}:</b> {html.escape(str(value))}")

        parts.extend([
            '<Placemark>',
            f'<name>Tree {html.escape(str(tree_id))}</name>',
            '<styleUrl>#treeTarget</styleUrl>',
            f'<description><![CDATA[{"<br>".join(desc)}]]></description>',
            '<Point>',
            f'<coordinates>{float(row["Longitude"]):.8f},{float(row["Latitude"]):.8f},0</coordinates>',
            '</Point>',
            '</Placemark>',
        ])

    parts.extend(['</Document>', '</kml>'])
    return "\n".join(parts).encode("utf-8")


# =============================================================================
# 5. GAP ANALYSIS — PRESERVED AS INVENTORY TOOL
# =============================================================================

def calculate_gaps(gdf, expected_tree_spacing, row_distance_threshold, grid_angle_degrees, max_empty_space_m):
    work = gdf.copy()
    utm_crs = work.estimate_utm_crs()
    work = work.to_crs(utm_crs)
    work["centroid"] = work.geometry.centroid
    raw_x = work["centroid"].apply(lambda p: p.x)
    raw_y = work["centroid"].apply(lambda p: p.y)
    mean_x, mean_y = raw_x.mean(), raw_y.mean()

    def rotate(x, y, angle_deg):
        rx = (x - mean_x) * math.cos(math.radians(angle_deg)) + (y - mean_y) * math.sin(math.radians(angle_deg))
        ry = -(x - mean_x) * math.sin(math.radians(angle_deg)) + (y - mean_y) * math.cos(math.radians(angle_deg))
        return rx, ry

    def unrotate(rx, ry, angle_deg):
        x = rx * math.cos(math.radians(angle_deg)) - ry * math.sin(math.radians(angle_deg))
        y = rx * math.sin(math.radians(angle_deg)) + ry * math.cos(math.radians(angle_deg))
        return x + mean_x, y + mean_y

    rotated = [rotate(x, y, grid_angle_degrees) for x, y in zip(raw_x, raw_y)]
    work["x"] = [c[0] for c in rotated]
    work["y"] = [c[1] for c in rotated]

    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=row_distance_threshold, linkage="average")
    work["Row_ID"] = clustering.fit_predict(work[["y"]].to_numpy())
    row_centers = work.groupby("Row_ID")["y"].mean().to_dict()
    work["Row_Center_Y"] = work["Row_ID"].map(row_centers)

    gaps = []
    for _, group in work.groupby("Row_ID"):
        group = group.sort_values("x").reset_index(drop=True)
        for i in range(len(group) - 1):
            a, b = group.iloc[i], group.iloc[i + 1]
            dist = b["x"] - a["x"]
            if (expected_tree_spacing * 1.5) < dist <= max_empty_space_m:
                missing_count = int(np.round(dist / expected_tree_spacing)) - 1
                for j in range(1, missing_count + 1):
                    gx = a["x"] + j * (dist / (missing_count + 1))
                    real_x, real_y = unrotate(gx, a["Row_Center_Y"], grid_angle_degrees)
                    gaps.append(Point(real_x, real_y))

    gaps_gdf = gpd.GeoDataFrame({"geometry": gaps}, crs=work.crs)
    if len(gaps_gdf):
        tree_buffers = work.geometry.buffer(2.0).unary_union
        gaps_gdf = gaps_gdf[~gaps_gdf.intersects(tree_buffers)]
    gaps_wgs84 = gaps_gdf.to_crs(epsg=4326)

    total_trees = len(work)
    total_gaps = len(gaps_wgs84)
    ideal_capacity = total_trees + total_gaps
    loss = (total_gaps / ideal_capacity) * 100 if ideal_capacity else 0.0
    return gaps_wgs84, total_trees, total_gaps, loss


# =============================================================================
# 6. SCENARIO FILTERS
# =============================================================================

SCENARIOS = {
    "OVERVIEW": (
        "🧭 Orchard Overview",
        "All 386 trees, colored only when a selected target layer is activated.",
        "Use the Tree Inspector and summary panels to review all three domains without forcing a single diagnosis.",
    ),
    "HIGH_MULTI_DOMAIN_PRIORITY": (
        "🔴 High Multi-Domain Priority",
        "Water + Biochemical + Structure all show supported evidence.",
        "Strongest internal cross-domain corroboration. It does not prove a causal chain or disease identity.",
    ),
    "MULTI_DOMAIN_PRIORITY": (
        "🟠 Two-Domain Priority",
        "Exactly two of the three major domains show supported evidence.",
        "Useful for identifying paired physiological/biochemical/structural expressions without requiring all domains to agree.",
    ),
    "SINGLE_DOMAIN_PRIORITY": (
        "🟡 Single-Domain Priority",
        "Exactly one major domain shows supported evidence.",
        "A single supported domain remains meaningful but lacks independent cross-domain corroboration.",
    ),
    "SCREENING_PRIORITY": (
        "🔵 Screening Priority",
        "Partial or isolated spectral evidence without a supported major-domain condition.",
        "These trees are candidates for inspection, not supported multi-domain anomalies.",
    ),
    "DATA_LIMITED_REVIEW": (
        "⚪ Data-Limited Review",
        "No supported major-domain anomaly, but one or more domains are incomplete or inconclusive.",
        "Missing evidence is never interpreted as normal.",
    ),
    "NO_SUPPORTED_ANOMALY": (
        "🟢 No Supported Anomaly",
        "Adequately assessed trees with no supported major-domain anomaly.",
        "This does not mean confirmed healthy; it only means none of the finalized supported-evidence conditions were met.",
    ),
    "WATER_SUPPORTED": (
        "💧 Supported Water-Status Anomaly",
        "Water domain reaches the finalized supported-evidence condition.",
        "WBI and the SWIR water block provide direct evidence; PRI can strengthen but does not independently create the supported class.",
    ),
    "BIOCHEMICAL_SUPPORTED": (
        "🧪 Supported Biochemical Decline",
        "Chlorophyll/red-edge and pigment/senescence sub-blocks agree.",
        "NDRE + CIred-edge + REP and PSRI + SIPI are treated as related sub-blocks, not separate final domains. ARI1 is corroborative.",
    ),
    "STRUCTURE_LOW_STATURE": (
        "🌳 Low Structural Stature",
        "LAS H_P95 is below the orchard-relative operational P25 threshold.",
        "Raster CHM P95 can corroborate the structural measurement. Structure describes relative stature, not a stress cause.",
    ),
    "WATER_BIOCHEMICAL_STRUCTURE": (
        "🔴 Water + Biochemical + Structure",
        "All three final domains coincide.",
        "Highest cross-domain inspection priority in the current framework.",
    ),
    "WATER_BIOCHEMICAL_ONLY": (
        "🟣 Water + Biochemical",
        "Water and biochemical evidence agree without low-stature evidence.",
        "Potentially consistent with a spectral/physiological response that has not expressed as low canopy stature.",
    ),
    "WATER_STRUCTURE_ONLY": (
        "🟦 Water + Structure",
        "Supported water anomaly coincides with low stature; biochemical decline is not supported.",
        "Do not infer that water caused the structural condition without field evidence.",
    ),
    "BIOCHEMICAL_STRUCTURE_ONLY": (
        "🟤 Biochemical + Structure",
        "Supported biochemical decline coincides with low stature; water evidence is not supported.",
        "Useful for separating structural/biochemical co-expression from a water-associated pattern.",
    ),
    "STRUCTURE_PROFILE": (
        "🌲 Structural Profile",
        "Trees with an assessable structural measurement.",
        "H_P95 is primary; raster CHM P95 is measurement corroboration and H_IQR is context only.",
    ),
    "GAP_ANALYSIS": (
        "🍋 Geometric Gap Analysis",
        "Estimated missing planting positions from orchard-row geometry.",
        "Inventory tool only; logically separate from health/anomaly classification.",
    ),
}


def scenario_mask(gdf, key):
    if key == "OVERVIEW":
        return pd.Series(True, index=gdf.index)
    if key == "HIGH_MULTI_DOMAIN_PRIORITY":
        return gdf["FIELD_INSPECTION_TIER"].eq("HIGH_MULTI_DOMAIN_PRIORITY")
    if key == "MULTI_DOMAIN_PRIORITY":
        return gdf["FIELD_INSPECTION_TIER"].eq("MULTI_DOMAIN_PRIORITY")
    if key == "SINGLE_DOMAIN_PRIORITY":
        return gdf["FIELD_INSPECTION_TIER"].eq("SINGLE_DOMAIN_PRIORITY")
    if key == "SCREENING_PRIORITY":
        return gdf["FIELD_INSPECTION_TIER"].eq("SCREENING_PRIORITY")
    if key == "DATA_LIMITED_REVIEW":
        return gdf["FIELD_INSPECTION_TIER"].eq("DATA_LIMITED_REVIEW")
    if key == "NO_SUPPORTED_ANOMALY":
        return gdf["FIELD_INSPECTION_TIER"].eq("NO_SUPPORTED_ANOMALY")
    if key == "WATER_SUPPORTED":
        return as_bool(gdf["WATER_SUPPORTED"])
    if key == "BIOCHEMICAL_SUPPORTED":
        return as_bool(gdf["BIOCHEMICAL_SUPPORTED"])
    if key == "STRUCTURE_LOW_STATURE":
        return as_bool(gdf["STRUCTURE_LOW_STATURE"])
    if key in {"WATER_BIOCHEMICAL_STRUCTURE", "WATER_BIOCHEMICAL_ONLY", "WATER_STRUCTURE_ONLY", "BIOCHEMICAL_STRUCTURE_ONLY"}:
        return gdf["CROSS_DOMAIN_PATTERN"].eq(key)
    if key == "STRUCTURE_PROFILE":
        return ~gdf["STRUCTURE_STATUS_FINAL"].eq("NOT_ASSESSABLE")
    return pd.Series(False, index=gdf.index)


# =============================================================================
# 7. APP LOAD
# =============================================================================

st.set_page_config(page_title="Orchard Three-Domain Engine", layout="wide")
st.title(APP_TITLE)
st.caption(
    "Final tree-level decision support using Water + Biochemical + Structure. "
    "Outputs are inspection priorities/anomaly evidence, not confirmed disease diagnoses."
)

geometry_gdf = load_tree_geometry()
rule_df, rule_source = load_final_rule_database()
spectral_df = load_spectral_data()

try:
    gdf = merge_geometry_and_rules(geometry_gdf, rule_df)
except Exception as exc:
    st.error(f"Could not align geometry and final rule database: {exc}")
    st.stop()

# Core dataset validation: freeze expected 386-tree implementation when applicable.
if len(gdf) == 386:
    observed = gdf["SUPPORTED_DOMAIN_COUNT"].value_counts().sort_index().to_dict()
    expected = {0: 259, 1: 90, 2: 27, 3: 10}
    if observed != expected:
        st.warning(f"Three-domain count check differs from finalized dataset. Expected {expected}; observed {observed}.")

st.sidebar.header("Three-Domain Controls")
st.sidebar.caption(rule_source)
selected_scenario = st.sidebar.selectbox(
    "Map / inspection layer",
    options=list(SCENARIOS.keys()),
    format_func=lambda x: SCENARIOS[x][0],
)

if selected_scenario == "GAP_ANALYSIS":
    with st.sidebar.expander("Gap-analysis parameters", expanded=False):
        expected_tree_spacing = st.number_input("Expected tree spacing (m)", 0.5, 20.0, DEFAULT_TREE_SPACING_M, 0.1)
        row_distance_threshold = st.number_input("Row clustering threshold (m)", 0.5, 10.0, DEFAULT_ROW_DISTANCE_THRESHOLD_M, 0.1)
        grid_angle_degrees = st.number_input("Grid rotation angle (deg)", -180.0, 180.0, DEFAULT_GRID_ANGLE_DEG, 1.0)
        max_empty_space_m = st.number_input("Maximum gap considered (m)", 2.0, 100.0, DEFAULT_MAX_EMPTY_SPACE_M, 1.0)
    gaps_gdf, total_trees, total_gaps, yield_loss_percentage = calculate_gaps(
        gdf, expected_tree_spacing, row_distance_threshold, grid_angle_degrees, max_empty_space_m
    )
    target_gdf = gdf.iloc[0:0].copy()
else:
    gaps_gdf = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
    total_trees = len(gdf)
    mask = scenario_mask(gdf, selected_scenario)
    target_gdf = gdf[mask].copy()


# =============================================================================
# 8. TREE INSPECTOR
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.subheader("Tree Inspector")
all_tree_ids = sorted(gdf["tree_id"].tolist())
selected_tree_id = st.sidebar.selectbox("Tree ID", all_tree_ids)
selected_tree = gdf.loc[gdf["tree_id"] == selected_tree_id].iloc[0]

with st.sidebar.expander("Selected tree — final evidence", expanded=True):
    st.write(f"**Inspection tier:** {selected_tree.get('FIELD_INSPECTION_TIER', 'NA')}")
    st.write(f"**Cross-domain pattern:** {selected_tree.get('CROSS_DOMAIN_PATTERN', 'NA')}")
    st.write(f"**Supported domains:** {selected_tree.get('SUPPORTED_DOMAIN_COUNT', 'NA')}/3")
    st.write(f"**Assessment completeness:** {selected_tree.get('MULTIDOMAIN_COMPLETENESS', 'NA')}")
    st.markdown("**Water**")
    st.write(selected_tree.get("WATER_EVIDENCE_STATUS", "NA"))
    st.markdown("**Biochemical**")
    st.write(selected_tree.get("BIOCHEMICAL_STATUS", "NA"))
    st.markdown("**Structure**")
    st.write(selected_tree.get("STRUCTURE_STATUS_FINAL", "NA"))
    if pd.notna(selected_tree.get("CROSS_DOMAIN_INTERPRETATION", np.nan)):
        st.caption(str(selected_tree.get("CROSS_DOMAIN_INTERPRETATION")))


# =============================================================================
# 9. MAIN TABS
# =============================================================================

tab_map, tab_tree, tab_summary, tab_methods = st.tabs([
    "🗺️ Three-Domain Map",
    "🌳 Tree Evidence",
    "📊 Orchard Summary",
    "🧪 Rules / Methods",
])


# -----------------------------------------------------------------------------
# TAB 1 — MAP
# -----------------------------------------------------------------------------
with tab_map:
    col_map, col_detail = st.columns([3, 2])

    with col_map:
        show_canopies = st.checkbox("Show all canopy outlines", value=True)

        map_center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
        m = folium.Map(location=map_center, zoom_start=18, max_zoom=22, tiles="CartoDB dark_matter")

        # No CHM GeoTIFF is required. The structural domain has already been
        # quantified upstream and is carried tree-by-tree in the final database
        # (LAS H_P95, raster-derived CHM P95 cross-check, H_IQR, QC/status).
        if show_canopies:
            folium.GeoJson(
                gdf,
                style_function=lambda _: {"fillColor": "none", "color": "#00FFCC", "weight": 0.8, "fillOpacity": 0.0},
                name="All tree crowns",
            ).add_to(m)

        # Selected tree is always highlighted.
        selected_geom = gdf[gdf["tree_id"] == selected_tree_id]
        folium.GeoJson(
            selected_geom,
            style_function=lambda _: {"fillColor": "#FFFFFF", "color": "#FFFFFF", "weight": 3.0, "fillOpacity": 0.25},
            tooltip=folium.GeoJsonTooltip(fields=["tree_id"], aliases=["Selected Tree ID:"]),
            name="Selected tree",
        ).add_to(m)

        if selected_scenario == "GAP_ANALYSIS":
            for _, row in gaps_gdf.iterrows():
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x], radius=5, color="#000000", weight=2,
                    fill=True, fill_color="#FFFFFF", fill_opacity=1.0, tooltip="Calculated planting gap"
                ).add_to(m)
        elif selected_scenario != "OVERVIEW" and not target_gdf.empty:
            tooltip_candidates = [
                "tree_id", "FIELD_INSPECTION_TIER", "CROSS_DOMAIN_PATTERN", "SUPPORTED_DOMAIN_COUNT",
                "WATER_EVIDENCE_STATUS", "BIOCHEMICAL_STATUS", "STRUCTURE_STATUS_FINAL",
                "WBI_VALUE", "NDMI2_VALUE", "NDSI_RWC_VALUE", "PRI_VALUE",
                "NDRE_VALUE", "CIRED_EDGE_VALUE", "REP_D1_NM_VALUE", "PSRI_VALUE", "SIPI_VALUE", "ARI1_VALUE",
                "H_P95_m", "RASTER_CHM_P95_m",
            ]
            fields = [f for f in tooltip_candidates if f in target_gdf.columns]
            tooltip = folium.GeoJsonTooltip(fields=fields, aliases=[f"{f}:" for f in fields], localize=True)
            color = SCENARIO_COLORS.get(selected_scenario, "#FF0000")
            folium.GeoJson(
                target_gdf,
                style_function=lambda _, c=color: {"fillColor": c, "color": "white", "weight": 2.0, "fillOpacity": 0.72},
                tooltip=tooltip,
                name="Selected three-domain targets",
            ).add_to(m)

        folium.LayerControl().add_to(m)
        st_folium(m, height=650, use_container_width=True)

    with col_detail:
        st.header(SCENARIOS[selected_scenario][0])
        st.write(SCENARIOS[selected_scenario][1])
        with st.expander("Scientific interpretation", expanded=True):
            st.markdown(SCENARIOS[selected_scenario][2])

        if selected_scenario == "GAP_ANALYSIS":
            st.metric("Orchard trees", total_trees)
            st.metric("Calculated planting gaps", total_gaps)
            st.metric("Estimated planting-capacity loss", f"{yield_loss_percentage:.2f}%")
        elif selected_scenario == "OVERVIEW":
            st.metric("Orchard trees", len(gdf))
            st.dataframe(
                gdf["FIELD_INSPECTION_TIER"].value_counts().rename_axis("Inspection tier").reset_index(name="Trees"),
                hide_index=True, use_container_width=True,
            )
        else:
            target_count = len(target_gdf)
            st.metric("Target trees", target_count, f"{100*target_count/len(gdf):.1f}% of orchard")
            if target_count:
                st.dataframe(
                    target_gdf["CROSS_DOMAIN_PATTERN"].value_counts().rename_axis("Pattern").reset_index(name="Trees"),
                    hide_index=True, use_container_width=True,
                )

        st.markdown("---")
        st.subheader("Field export")
        if selected_scenario != "GAP_ANALYSIS" and selected_scenario != "OVERVIEW" and not target_gdf.empty:
            st.download_button(
                f"Download {len(target_gdf)} target crowns (GeoJSON)",
                target_gdf.to_json(),
                file_name=f"three_domain_{selected_scenario}.geojson",
                mime="application/geo+json",
            )
            kml = build_tree_navigation_kml(
                target_gdf,
                layer_title=SCENARIOS[selected_scenario][0],
                color_hex=SCENARIO_COLORS.get(selected_scenario, "#FF0000"),
            )
            st.download_button(
                f"📍 Download {len(target_gdf)} target trees (KML)",
                data=kml,
                file_name=f"three_domain_{selected_scenario}.kml",
                mime="application/vnd.google-earth.kml+xml",
            )
        elif selected_scenario == "GAP_ANALYSIS" and len(gaps_gdf):
            st.download_button(
                f"Download {len(gaps_gdf)} planting gaps (GeoJSON)", gaps_gdf.to_json(),
                file_name="calculated_orchard_gaps.geojson", mime="application/geo+json"
            )

        # Always provide all 2/3-domain field priorities.
        priority = gdf[gdf["SUPPORTED_DOMAIN_COUNT"] >= 2].copy()
        if len(priority):
            priority_kml = build_tree_navigation_kml(priority, "All Two/Three-Domain Priority Trees", "#FF4500")
            st.download_button(
                f"⭐ Download all 2/3-domain priority trees ({len(priority)}) KML",
                data=priority_kml,
                file_name="field_navigation_ALL_MULTI_DOMAIN_PRIORITY.kml",
                mime="application/vnd.google-earth.kml+xml",
            )


# -----------------------------------------------------------------------------
# TAB 2 — TREE EVIDENCE
# -----------------------------------------------------------------------------
with tab_tree:
    st.header(f"Tree {selected_tree_id} — Three-Domain Evidence")

    a, b, c, d = st.columns(4)
    a.metric("Supported domains", f"{int(selected_tree.get('SUPPORTED_DOMAIN_COUNT', 0))}/3")
    b.metric("Assessable domains", f"{int(selected_tree.get('ASSESSABLE_DOMAIN_COUNT', 0))}/3")
    c.metric("Inspection tier", str(selected_tree.get("FIELD_INSPECTION_TIER", "NA")))
    d.metric("Pattern", str(selected_tree.get("CROSS_DOMAIN_PATTERN", "NA")))

    st.info(str(selected_tree.get("CROSS_DOMAIN_INTERPRETATION", "No interpretation available.")))

    water_col, bio_col, struct_col = st.columns(3)

    with water_col:
        st.subheader("💧 Water")
        st.write(f"**State:** {selected_tree.get('WATER_DOMAIN_STATE', 'NA')}")
        st.write(f"**Final status:** {selected_tree.get('WATER_EVIDENCE_STATUS', 'NA')}")
        st.write(f"**Measurement quality:** {selected_tree.get('WATER_MEASUREMENT_QUALITY', 'NA')}")
        water_table = pd.DataFrame([
            ["WBI", fmt(selected_tree.get("WBI_VALUE"), 6), "Direct VNIR water block"],
            ["NDMI2", fmt(selected_tree.get("NDMI2_VALUE"), 6), "SWIR water block"],
            ["NDSI_RWC", fmt(selected_tree.get("NDSI_RWC_VALUE"), 6), "SWIR water block"],
            ["PRI", fmt(selected_tree.get("PRI_VALUE"), 6), "Physiological corroboration"],
        ], columns=["Indicator", "Value", "Role"])
        st.dataframe(water_table, hide_index=True, use_container_width=True)
        if "WBI_WATER_BLOCK" in selected_tree.index:
            st.caption(f"WBI block: {selected_tree.get('WBI_WATER_BLOCK')} | SWIR block: {selected_tree.get('SWIR_WATER_BLOCK')} | PRI support: {selected_tree.get('PRI_PHYSIOLOGICAL_SUPPORT')}")

    with bio_col:
        st.subheader("🧪 Biochemical")
        st.write(f"**State:** {selected_tree.get('BIOCHEMICAL_DOMAIN_STATE', 'NA')}")
        st.write(f"**Final status:** {selected_tree.get('BIOCHEMICAL_STATUS', 'NA')}")
        bio_table = pd.DataFrame([
            ["NDRE", fmt(selected_tree.get("NDRE_VALUE"), 6), "Chlorophyll/red-edge amplitude"],
            ["CIred-edge", fmt(selected_tree.get("CIRED_EDGE_VALUE"), 6), "Chlorophyll/red-edge amplitude"],
            ["REP", fmt(selected_tree.get("REP_D1_NM_VALUE"), 3), "Red-edge position"],
            ["PSRI", fmt(selected_tree.get("PSRI_VALUE"), 6), "Senescence/pigment"],
            ["SIPI", fmt(selected_tree.get("SIPI_VALUE"), 6), "Pigment balance"],
            ["ARI1", fmt(selected_tree.get("ARI1_VALUE"), 6), "Corroborative only"],
            ["NDVI", fmt(selected_tree.get("NDVI_VALUE"), 6), "Vigor context only"],
        ], columns=["Indicator", "Value", "Role"])
        st.dataframe(bio_table, hide_index=True, use_container_width=True)
        if "BIO_CHL_RE_SUPPORTED" in selected_tree.index:
            st.caption(
                f"Chlorophyll/red-edge supported: {selected_tree.get('BIO_CHL_RE_SUPPORTED')} | "
                f"Pigment supported: {selected_tree.get('BIO_PIGMENT_SUPPORTED')} | ARI1 corroboration: {selected_tree.get('BIO_ARI1_HIGH')}"
            )

    with struct_col:
        st.subheader("🌳 Structure")
        st.write(f"**State:** {selected_tree.get('STRUCTURE_DOMAIN_STATE', 'NA')}")
        st.write(f"**Final status:** {selected_tree.get('STRUCTURE_STATUS_FINAL', 'NA')}")
        st.write(f"**Evidence strength:** {selected_tree.get('STRUCTURE_EVIDENCE_STRENGTH', 'NA')}")
        struct_table = pd.DataFrame([
            ["LAS H_P95 (m)", fmt(selected_tree.get("H_P95_m"), 3), "Primary structural indicator"],
            ["Raster CHM P95 (m)", fmt(selected_tree.get("RASTER_CHM_P95_m"), 3), "Measurement corroboration"],
            ["H_IQR (m)", fmt(selected_tree.get("H_IQR_m"), 3), "Vertical-complexity context"],
        ], columns=["Metric", "Value", "Role"])
        st.dataframe(struct_table, hide_index=True, use_container_width=True)
        st.caption(f"Structural QC: {selected_tree.get('STRUCTURE_QC_FINAL', 'NA')}")

    st.markdown("---")
    st.subheader("Optional post-QC consensus hyperspectral signature — display only")
    if spectral_df.empty:
        st.info("Post-QC consensus spectral CSV is not present. This does not affect the finalized three-domain classification.")
    else:
        # Try tree_id first; generated_id only as legacy fallback.
        subset = pd.DataFrame()
        if "tree_id" in spectral_df.columns:
            subset = spectral_df[spectral_df["tree_id"] == selected_tree_id]
        elif "generated_id" in spectral_df.columns and "generated_id" in selected_tree.index:
            subset = spectral_df[spectral_df["generated_id"] == selected_tree.get("generated_id")]

        band_cols = [c for c in spectral_df.columns if str(c).startswith("Band_")]
        if not subset.empty and band_cols:
            y = subset.iloc[0][band_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            x = np.arange(1, len(band_cols) + 1)
            fig, ax = plt.subplots(figsize=(10, 3.6))
            ax.plot(x, y)
            ax.set_xlabel("Band number")
            ax.set_ylabel("Reflectance")
            ax.set_title(f"Tree {selected_tree_id} post-QC consensus canopy spectrum")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("No matching post-QC consensus spectrum is available for this tree (this does not affect its final rule status).")


# -----------------------------------------------------------------------------
# TAB 3 — ORCHARD SUMMARY
# -----------------------------------------------------------------------------
with tab_summary:
    st.header("Orchard-Level Three-Domain Summary")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Trees", len(gdf))
    k2.metric("3 supported domains", int((gdf["SUPPORTED_DOMAIN_COUNT"] == 3).sum()))
    k3.metric("2 supported domains", int((gdf["SUPPORTED_DOMAIN_COUNT"] == 2).sum()))
    k4.metric("≥1 supported domain", int((gdf["SUPPORTED_DOMAIN_COUNT"] >= 1).sum()))

    left, right = st.columns(2)
    with left:
        st.subheader("Supported-domain count")
        domain_counts = gdf["SUPPORTED_DOMAIN_COUNT"].value_counts().sort_index().rename_axis("Supported domains").reset_index(name="Trees")
        st.dataframe(domain_counts, hide_index=True, use_container_width=True)

        st.subheader("Cross-domain patterns")
        pattern_counts = gdf["CROSS_DOMAIN_PATTERN"].value_counts().rename_axis("Pattern").reset_index(name="Trees")
        st.dataframe(pattern_counts, hide_index=True, use_container_width=True)

    with right:
        st.subheader("Field inspection tiers")
        tier_counts = gdf["FIELD_INSPECTION_TIER"].value_counts().rename_axis("Tier").reset_index(name="Trees")
        st.dataframe(tier_counts, hide_index=True, use_container_width=True)

        st.subheader("Assessment completeness")
        completeness = gdf["MULTIDOMAIN_COMPLETENESS"].value_counts().rename_axis("Completeness").reset_index(name="Trees")
        st.dataframe(completeness, hide_index=True, use_container_width=True)

    st.warning(
        "Do not interpret 0 supported domains as confirmed healthy. Trees in DATA_LIMITED_REVIEW may have incomplete evidence, "
        "and NO_SUPPORTED_ANOMALY means only that no finalized supported-evidence condition was met."
    )

    export_cols = [
        "tree_id", "FIELD_INSPECTION_TIER", "CROSS_DOMAIN_PATTERN", "SUPPORTED_DOMAIN_COUNT",
        "ASSESSABLE_DOMAIN_COUNT", "WATER_EVIDENCE_STATUS", "BIOCHEMICAL_STATUS", "STRUCTURE_STATUS_FINAL",
        "CROSS_DOMAIN_INTERPRETATION",
    ]
    export_cols = [c for c in export_cols if c in gdf.columns]
    summary_csv = gdf[export_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download orchard three-domain summary CSV",
        summary_csv,
        file_name="orchard_three_domain_summary.csv",
        mime="text/csv",
    )


# -----------------------------------------------------------------------------
# TAB 4 — RULES / METHODS
# -----------------------------------------------------------------------------
with tab_methods:
    st.header("Final Deterministic Framework")
    st.code(
        """
TREE
 │
 ├── WATER STATUS
 │     WBI direct block
 │     + SWIR block (NDMI2 + NDSI_RWC counted as ONE block)
 │     + PRI physiological corroboration
 │
 ├── CANOPY BIOCHEMICAL STATUS
 │     Chlorophyll/red-edge sub-block: NDRE + CIred-edge + REP
 │     Pigment/senescence sub-block: PSRI + SIPI
 │     ARI1 corroboration; NDVI context only
 │
 └── STRUCTURE
       LAS H_P95 primary
       Raster CHM P95 measurement corroboration
       H_IQR context only

Then compare the THREE domain-level outcomes.
No raw index is allowed to become an extra final-domain vote.
        """,
        language="text",
    )

    st.subheader("Operational thresholds")
    threshold_rows = [
        ["Biochemical", "NDRE", f"≤ {BIO_THRESHOLDS['NDRE_LOW_MAX']:.6f}", "chlorophyll/red-edge"],
        ["Biochemical", "CIred-edge", f"≤ {BIO_THRESHOLDS['CIRED_LOW_MAX']:.6f}", "chlorophyll/red-edge"],
        ["Biochemical", "REP", f"≤ {BIO_THRESHOLDS['REP_LOW_MAX_NM']:.3f} nm", "red-edge position"],
        ["Biochemical", "PSRI", f"≥ {BIO_THRESHOLDS['PSRI_HIGH_MIN']:.6f}", "pigment/senescence"],
        ["Biochemical", "SIPI", f"≥ {BIO_THRESHOLDS['SIPI_HIGH_MIN']:.6f}", "pigment/senescence"],
        ["Biochemical", "ARI1", f"≥ {BIO_THRESHOLDS['ARI1_HIGH_MIN']:.6f}", "corroborative only"],
        ["Structure", "LAS H_P95", f"≤ {STRUCTURE_THRESHOLDS['H_P95_P25_M']:.3f} m", "operational low stature"],
        ["Structure", "Raster CHM P95", f"≤ {STRUCTURE_THRESHOLDS['RASTER_CHM_P95_P25_M']:.3f} m", "measurement corroboration"],
        ["Structure", "H_IQR", f"≥ {STRUCTURE_THRESHOLDS['H_IQR_P75_M']:.3f} m", "context only"],
    ]
    st.dataframe(pd.DataFrame(threshold_rows, columns=["Domain", "Indicator", "Operational condition", "Role"]), hide_index=True, use_container_width=True)

    st.subheader("Water-domain logic")
    st.markdown(
        "- **Supported water anomaly:** WBI abnormal **and** SWIR water block concordantly abnormal.\n"
        "- **Strong internal water evidence:** supported direct-water condition **plus** PRI abnormal.\n"
        "- NDMI2 and NDSI_RWC are highly redundant and are therefore counted as **one SWIR block**, not two votes.\n"
        "- Missing/failed direct-water evidence is never converted to normal."
    )

    st.subheader("Biochemical-domain logic")
    st.markdown(
        "- Chlorophyll/red-edge support requires **NDRE low + CIred-edge low + REP low**.\n"
        "- Pigment/senescence support requires **PSRI high + SIPI high**.\n"
        "- Both sub-blocks together create **CONCORDANT_BIOCHEMICAL_DECLINE**.\n"
        "- ARI1 high upgrades this to **STRONG_CONCORDANT_BIOCHEMICAL_DECLINE**.\n"
        "- NDVI remains context and is not an independent final biochemical vote."
    )

    st.subheader("Structural-domain logic")
    st.markdown(
        f"- LAS H_P95 ≤ **{STRUCTURE_THRESHOLDS['H_P95_P25_M']:.3f} m** = orchard-relative low stature.\n"
        f"- Raster CHM P95 ≤ **{STRUCTURE_THRESHOLDS['RASTER_CHM_P95_P25_M']:.3f} m** corroborates the structural measurement.\n"
        "- H_IQR is retained only as vertical-complexity context.\n"
        "- Structural status describes relative canopy stature and does not identify a causal stress."
    )

    st.subheader("Cross-domain interpretation boundary")
    st.info(
        "Water, Biochemical and Structure are assessed independently first. Their agreement strengthens internal evidence, "
        "but cross-domain concordance does not prove that one domain caused another."
    )


# =============================================================================
# 10. OPTIONAL LLM — STRICTLY DOWNSTREAM
# =============================================================================

with st.sidebar.expander("🤖 LLM Field Assistant", expanded=False):
    st.markdown("Natural-language support only. The deterministic three-domain result above is calculated first.")

    if not GEMINI_AVAILABLE:
        st.info("google-generativeai is not installed; the three-domain app remains fully functional.")
    else:
        api_key = st.text_input("Gemini API key", type="password")
        model_name = st.text_input("Gemini model", value=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

        active_name = SCENARIOS[selected_scenario][0]
        target_count = len(target_gdf) if selected_scenario not in {"GAP_ANALYSIS", "OVERVIEW"} else len(gdf)
        selected_context = (
            f"Tree {selected_tree_id}: water={selected_tree.get('WATER_EVIDENCE_STATUS', 'NA')}; "
            f"biochemical={selected_tree.get('BIOCHEMICAL_STATUS', 'NA')}; "
            f"structure={selected_tree.get('STRUCTURE_STATUS_FINAL', 'NA')}; "
            f"pattern={selected_tree.get('CROSS_DOMAIN_PATTERN', 'NA')}; "
            f"inspection tier={selected_tree.get('FIELD_INSPECTION_TIER', 'NA')}."
        )

        system_prompt = f"""
You are a precision-agriculture decision-support assistant for a UAV hyperspectral citrus orchard study.

ACTIVE MAP LAYER: {active_name}
TARGET COUNT: {target_count}
SELECTED TREE CONTEXT: {selected_context}

Critical constraints:
- Treat WATER, BIOCHEMICAL and STRUCTURE as the only final major domains.
- Do not count correlated indices as additional independent domain votes.
- Cross-domain agreement is corroboration, not causal proof.
- Do not call a tree healthy merely because no supported anomaly was detected.
- Do not infer disease identity, drought cause, nutrient deficiency, root rot, pest, heat or frost without field evidence.
- Structural low stature is relative orchard stature, not proof of stress.
- PRI, NDVI and H_IQR are supporting/context variables according to the deterministic framework.
- Keep recommendations framed as field-inspection priorities and measurements to verify.
"""

        if st.button("Generate selected-tree inspection plan"):
            if not api_key:
                st.warning("Enter an API key first.")
            else:
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(model_name, system_instruction=system_prompt)
                    response = model.generate_content(
                        "Give a concise field inspection plan for the selected tree. Separate remote-sensing evidence from measurements that must be confirmed in the field."
                    )
                    st.markdown(response.text)
                except Exception as exc:
                    st.error(f"LLM API error: {exc}")
