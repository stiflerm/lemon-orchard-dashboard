"""
Orchard Diagnostic Intelligence v4 — User-Centred Three-Domain DSS
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
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from shapely.geometry import Point
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from streamlit_folium import st_folium

try:
    from scipy.stats import mannwhitneyu, pearsonr
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

APP_TITLE = "🍋 Orchard Intelligence — Water • Biochemistry • Structure"
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



WATER_SENSITIVITY_THRESHOLDS = {
    "P20": {
        "WBI_LOW_MAX": 0.9732224384098488,
        "PRI_LOW_MAX": -0.0905354641709804,
        "NDMI2_HIGH_MIN": -0.1563689890814745,
        "NDSI_RWC_LOW_MAX": 0.23842041377519282,
    },
    "P25": {
        "WBI_LOW_MAX": 0.9780581745028392,
        "PRI_LOW_MAX": -0.08910375378375263,
        "NDMI2_HIGH_MIN": -0.17834779593229538,
        "NDSI_RWC_LOW_MAX": 0.25159298573596856,
    },
    "P30": {
        "WBI_LOW_MAX": 0.9835634253103394,
        "PRI_LOW_MAX": -0.08797404072436948,
        "NDMI2_HIGH_MIN": -0.19210124039554718,
        "NDSI_RWC_LOW_MAX": 0.26483841514836876,
    },
}

BIO_SENSITIVITY_THRESHOLDS = {
    "P20": {"NDRE": 0.18991320677124718, "CIRED": 0.5020010800931471, "REP": 725.0965727700146,
            "PSRI": 0.24515380157382927, "SIPI": 1.6691380599575552, "ARI1": 4.560504481288227},
    "P25": {"NDRE": 0.2047129778647948, "CIRED": 0.5487572048699663, "REP": 725.3890241067692,
            "PSRI": 0.2317542293068353, "SIPI": 1.6044129002460532, "ARI1": 4.409562434704582},
    "P30": {"NDRE": 0.21786641970173387, "CIRED": 0.6123779812211838, "REP": 725.6065279573306,
            "PSRI": 0.21685585761922752, "SIPI": 1.56515408797565, "ARI1": 4.274624848018448},
}

GROUND_SPECTRA_CANDIDATES = [
    DATA_DIR / "ground_spectroradiometer.csv",
    DATA_DIR / "SVC_ground_validation.csv",
    DATA_DIR / "SVC_5spectra_analysis_ready.csv",
    APP_DIR / "ground_spectroradiometer.csv",
    APP_DIR / "SVC_ground_validation.csv",
]

WAVELENGTH_MAP_CANDIDATES = [
    DATA_DIR / "band_wavelengths.csv",
    DATA_DIR / "wavelength_mapping.csv",
    DATA_DIR / "band_mapping_by_strip.csv",
    APP_DIR / "band_wavelengths.csv",
    APP_DIR / "wavelength_mapping.csv",
]

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

# =============================================================================
# 4. USER-CENTRED SCENARIOS / MAP HELPERS
# =============================================================================

VIEW_OPTIONS = {
    "WATER": "💧 Water status",
    "BIOCHEMICAL": "🧪 Canopy biochemical status",
    "STRUCTURE": "🌳 Tree structure",
    "COMBINED": "🔗 Combined multi-domain priority",
    "SCREENING": "🔎 Screening / incomplete evidence",
    "GAPS": "🍋 Planting-gap inventory",
}

VIEW_COLORS = {
    "WATER": "#1E90FF",
    "BIOCHEMICAL": "#9B59B6",
    "STRUCTURE": "#2ECC71",
    "COMBINED_3": "#E74C3C",
    "COMBINED_2": "#F39C12",
    "SCREENING": "#3498DB",
    "DATA_LIMITED": "#95A5A6",
    "GAPS": "#FFFFFF",
}

WATER_TOOLTIP = [
    ("tree_id", "Tree ID"),
    ("WATER_EVIDENCE_STATUS", "Water result"),
    ("WBI_VALUE", "WBI"),
    ("NDMI2_VALUE", "NDMI2"),
    ("NDSI_RWC_VALUE", "NDSI-RWC"),
    ("PRI_VALUE", "PRI support"),
    ("WATER_MEASUREMENT_QUALITY", "Measurement quality"),
]

BIO_TOOLTIP = [
    ("tree_id", "Tree ID"),
    ("BIOCHEMICAL_STATUS", "Biochemical result"),
    ("NDRE_VALUE", "NDRE"),
    ("CIRED_EDGE_VALUE", "CI red-edge"),
    ("REP_D1_NM_VALUE", "REP (nm)"),
    ("PSRI_VALUE", "PSRI"),
    ("SIPI_VALUE", "SIPI"),
    ("ARI1_VALUE", "ARI1 support"),
    ("NDVI_VALUE", "NDVI context"),
]

STRUCT_TOOLTIP = [
    ("tree_id", "Tree ID"),
    ("STRUCTURE_STATUS_FINAL", "Structure result"),
    ("H_P95_m", "LAS H-P95 (m)"),
    ("RASTER_CHM_P95_m", "Raster CHM P95 (m)"),
    ("H_IQR_m", "Vertical variability (m)"),
    ("STRUCTURE_EVIDENCE_STRENGTH", "Evidence strength"),
]

COMBINED_TOOLTIP = [
    ("tree_id", "Tree ID"),
    ("SUPPORTED_DOMAIN_COUNT", "Supported domains"),
    ("CROSS_DOMAIN_PATTERN", "Cross-domain pattern"),
    ("FIELD_INSPECTION_TIER", "Inspection priority"),
    ("WATER_EVIDENCE_STATUS", "Water"),
    ("BIOCHEMICAL_STATUS", "Biochemical"),
    ("STRUCTURE_STATUS_FINAL", "Structure"),
]


def human_pattern(value):
    mapping = {
        "WATER_BIOCHEMICAL_STRUCTURE": "Water + biochemical + structure",
        "WATER_BIOCHEMICAL_ONLY": "Water + biochemical",
        "WATER_STRUCTURE_ONLY": "Water + structure",
        "BIOCHEMICAL_STRUCTURE_ONLY": "Biochemical + structure",
        "WATER_ONLY": "Water only",
        "BIOCHEMICAL_ONLY": "Biochemical only",
        "STRUCTURE_ONLY": "Structure only",
        "NO_SUPPORTED_MAJOR_DOMAIN_ANOMALY": "No supported major-domain anomaly",
    }
    return mapping.get(str(value), str(value).replace("_", " ").title())


def human_tier(value):
    mapping = {
        "HIGH_MULTI_DOMAIN_PRIORITY": "High multi-domain priority",
        "MULTI_DOMAIN_PRIORITY": "Two-domain priority",
        "SINGLE_DOMAIN_PRIORITY": "Single-domain priority",
        "SCREENING_PRIORITY": "Screening priority",
        "DATA_LIMITED_REVIEW": "Data-limited review",
        "NO_SUPPORTED_ANOMALY": "No supported anomaly",
    }
    return mapping.get(str(value), str(value).replace("_", " ").title())


def make_target_mask(gdf, view, include_screening=False, combined_mode="2+ domains", screening_mode="Screening evidence"):
    if view == "WATER":
        mask = as_bool(gdf["WATER_SUPPORTED"])
        if include_screening and "WATER_SCREENING_ONLY" in gdf.columns:
            mask |= as_bool(gdf["WATER_SCREENING_ONLY"])
        return mask
    if view == "BIOCHEMICAL":
        mask = as_bool(gdf["BIOCHEMICAL_SUPPORTED"])
        if include_screening and "BIOCHEMICAL_SCREENING_ONLY" in gdf.columns:
            mask |= as_bool(gdf["BIOCHEMICAL_SCREENING_ONLY"])
        return mask
    if view == "STRUCTURE":
        return as_bool(gdf["STRUCTURE_LOW_STATURE"])
    if view == "COMBINED":
        if combined_mode == "3 domains":
            return pd.to_numeric(gdf["SUPPORTED_DOMAIN_COUNT"], errors="coerce").eq(3)
        if combined_mode == "Exactly 2 domains":
            return pd.to_numeric(gdf["SUPPORTED_DOMAIN_COUNT"], errors="coerce").eq(2)
        return pd.to_numeric(gdf["SUPPORTED_DOMAIN_COUNT"], errors="coerce").ge(2)
    if view == "SCREENING":
        if screening_mode == "Data-limited review":
            return gdf["FIELD_INSPECTION_TIER"].eq("DATA_LIMITED_REVIEW")
        return gdf["FIELD_INSPECTION_TIER"].eq("SCREENING_PRIORITY")
    return pd.Series(False, index=gdf.index)


def tooltip_spec(view):
    if view == "WATER":
        return WATER_TOOLTIP
    if view == "BIOCHEMICAL":
        return BIO_TOOLTIP
    if view == "STRUCTURE":
        return STRUCT_TOOLTIP
    return COMBINED_TOOLTIP


def scenario_metrics(view):
    if view == "WATER":
        return ["WBI_VALUE", "NDMI2_VALUE", "NDSI_RWC_VALUE", "PRI_VALUE", "NDVI_VALUE"]
    if view == "BIOCHEMICAL":
        return ["NDRE_VALUE", "CIRED_EDGE_VALUE", "REP_D1_NM_VALUE", "PSRI_VALUE", "SIPI_VALUE", "ARI1_VALUE", "NDVI_VALUE"]
    if view == "STRUCTURE":
        return ["H_P95_m", "RASTER_CHM_P95_m", "H_IQR_m"]
    return [
        "WBI_VALUE", "NDMI2_VALUE", "NDSI_RWC_VALUE", "PRI_VALUE",
        "NDRE_VALUE", "CIRED_EDGE_VALUE", "REP_D1_NM_VALUE", "PSRI_VALUE", "SIPI_VALUE", "ARI1_VALUE",
        "NDVI_VALUE", "H_P95_m", "RASTER_CHM_P95_m", "H_IQR_m",
    ]


def add_target_layer(m, target_gdf, view):
    if target_gdf.empty:
        return
    fields_aliases = [(f, a) for f, a in tooltip_spec(view) if f in target_gdf.columns]
    fields = [x[0] for x in fields_aliases]
    aliases = [x[1] + ":" for x in fields_aliases]
    tooltip = folium.GeoJsonTooltip(fields=fields, aliases=aliases, localize=True, sticky=True)
    popup = folium.GeoJsonPopup(fields=fields, aliases=aliases, localize=True, labels=True)

    if view == "COMBINED":
        def style_fn(feature):
            n = feature.get("properties", {}).get("SUPPORTED_DOMAIN_COUNT", 0)
            color = VIEW_COLORS["COMBINED_3"] if int(n or 0) == 3 else VIEW_COLORS["COMBINED_2"]
            return {"fillColor": color, "color": "#FFFFFF", "weight": 2.2, "fillOpacity": 0.78}
    elif view == "SCREENING":
        def style_fn(feature):
            tier = feature.get("properties", {}).get("FIELD_INSPECTION_TIER", "")
            color = VIEW_COLORS["DATA_LIMITED"] if tier == "DATA_LIMITED_REVIEW" else VIEW_COLORS["SCREENING"]
            return {"fillColor": color, "color": "#FFFFFF", "weight": 2.0, "fillOpacity": 0.72}
    elif view == "STRUCTURE":
        def style_fn(feature):
            corroborated = str(feature.get("properties", {}).get("STRUCTURE_CORROBORATED", "False")).lower() in {"true", "1"}
            color = VIEW_COLORS["STRUCTURE"] if corroborated else "#F1C40F"
            return {"fillColor": color, "color": "#FFFFFF", "weight": 2.0, "fillOpacity": 0.75}
    else:
        color = VIEW_COLORS[view]
        def style_fn(_):
            return {"fillColor": color, "color": "#FFFFFF", "weight": 2.0, "fillOpacity": 0.75}

    folium.GeoJson(
        target_gdf,
        style_function=style_fn,
        tooltip=tooltip,
        popup=popup,
        name="Highlighted trees",
    ).add_to(m)


def make_navigation_points(source_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if source_gdf is None or source_gdf.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
    work = source_gdf.copy()
    try:
        projected = work.to_crs(work.estimate_utm_crs()) if work.crs.is_geographic else work.copy()
        points = projected.copy()
        points.geometry = projected.geometry.centroid
        return points.to_crs(epsg=4326)
    except Exception:
        out = work.to_crs(epsg=4326).copy()
        out.geometry = out.geometry.representative_point()
        return out


def build_tree_navigation_kml(source_gdf: gpd.GeoDataFrame, layer_title: str, color_hex="#FF0000") -> bytes:
    points = make_navigation_points(source_gdf)
    if points.empty:
        return b""
    h = color_hex.lstrip("#")
    rr, gg, bb = h[0:2], h[2:4], h[4:6]
    kml_color = f"ff{bb}{gg}{rr}"
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
        f'<name>{html.escape(layer_title)}</name>',
        '<Style id="treeTarget"><IconStyle>', f'<color>{kml_color}</color><scale>1.15</scale>',
        '<Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>',
        '</IconStyle></Style>',
    ]
    details = [x[0] for x in COMBINED_TOOLTIP] + [x[0] for x in WATER_TOOLTIP + BIO_TOOLTIP + STRUCT_TOOLTIP]
    details = list(dict.fromkeys(details))
    for _, row in points.iterrows():
        rp = row.geometry
        desc = [f"<b>Tree ID:</b> {html.escape(str(row.get('tree_id', 'NA')))}"]
        for field in details:
            if field in row.index and pd.notna(row.get(field)):
                value = row.get(field)
                if isinstance(value, (float, np.floating)):
                    value = f"{float(value):.4f}"
                desc.append(f"<b>{html.escape(field)}:</b> {html.escape(str(value))}")
        parts += [
            '<Placemark>', f'<name>Tree {html.escape(str(row.get("tree_id", "NA")))}</name>',
            '<styleUrl>#treeTarget</styleUrl>', f'<description><![CDATA[{"<br>".join(desc)}]]></description>',
            f'<Point><coordinates>{rp.x:.8f},{rp.y:.8f},0</coordinates></Point>', '</Placemark>'
        ]
    parts.append('</Document></kml>')
    return "\n".join(parts).encode("utf-8")


# =============================================================================
# 5. GAP ANALYSIS — INVENTORY ONLY
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
                    x, y = unrotate(gx, a["Row_Center_Y"], grid_angle_degrees)
                    gaps.append(Point(x, y))
    gaps_gdf = gpd.GeoDataFrame({"geometry": gaps}, crs=work.crs)
    if len(gaps_gdf):
        buffers = work.geometry.buffer(2.0).unary_union
        gaps_gdf = gaps_gdf[~gaps_gdf.intersects(buffers)]
    gaps_wgs84 = gaps_gdf.to_crs(epsg=4326)
    total_gaps = len(gaps_wgs84)
    ideal = len(work) + total_gaps
    return gaps_wgs84, len(work), total_gaps, (100 * total_gaps / ideal if ideal else 0.0)


# =============================================================================
# 6. VALIDATION / ROBUSTNESS HELPERS
# =============================================================================

def jaccard(a, b):
    aa, bb = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    union = np.logical_or(aa, bb).sum()
    return np.logical_and(aa, bb).sum() / union if union else np.nan


def spectral_angle_deg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return np.nan
    a, b = a[mask], b[mask]
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den <= 0:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / den, -1, 1))))


def spectral_rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2))) if mask.sum() else np.nan


def rank_biserial_from_u(u, n1, n2):
    return (2.0 * u) / (n1 * n2) - 1.0 if n1 and n2 else np.nan


def sensitivity_domain_masks(df, scheme):
    wt = WATER_SENSITIVITY_THRESHOLDS[scheme]
    bt = BIO_SENSITIVITY_THRESHOLDS[scheme]
    def num(c): return pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)
    WBI, NDMI, NDSI = num("WBI_VALUE"), num("NDMI2_VALUE"), num("NDSI_RWC_VALUE")
    water = WBI.notna() & NDMI.notna() & NDSI.notna() & (WBI <= wt["WBI_LOW_MAX"]) & (NDMI >= wt["NDMI2_HIGH_MIN"]) & (NDSI <= wt["NDSI_RWC_LOW_MAX"])
    NDRE, CI, REP = num("NDRE_VALUE"), num("CIRED_EDGE_VALUE"), num("REP_D1_NM_VALUE")
    PSRI, SIPI = num("PSRI_VALUE"), num("SIPI_VALUE")
    bio = NDRE.notna() & CI.notna() & REP.notna() & PSRI.notna() & SIPI.notna() & (NDRE <= bt["NDRE"]) & (CI <= bt["CIRED"]) & (REP <= bt["REP"]) & (PSRI >= bt["PSRI"]) & (SIPI >= bt["SIPI"])
    h95 = num("H_P95_m")
    s_thr = {"P20": STRUCTURE_THRESHOLDS["H_P95_P20_M"], "P25": STRUCTURE_THRESHOLDS["H_P95_P25_M"], "P30": STRUCTURE_THRESHOLDS["H_P95_P30_M"]}[scheme]
    structure = h95.notna() & (h95 <= s_thr)
    return water.fillna(False), bio.fillna(False), structure.fillna(False)


def sensitivity_target_mask(df, view, scheme, combined_mode="2+ domains"):
    W, B, S = sensitivity_domain_masks(df, scheme)
    if view == "WATER": return W
    if view == "BIOCHEMICAL": return B
    if view == "STRUCTURE": return S
    if view == "COMBINED":
        count = W.astype(int) + B.astype(int) + S.astype(int)
        if combined_mode == "3 domains": return count.eq(3)
        if combined_mode == "Exactly 2 domains": return count.eq(2)
        return count.ge(2)
    return pd.Series(False, index=df.index)


def comparison_statistics(df, target_mask, reference_mask, metrics):
    rows = []
    for c in metrics:
        if c not in df.columns: continue
        x = pd.to_numeric(df.loc[target_mask, c], errors="coerce").dropna().to_numpy(float)
        y = pd.to_numeric(df.loc[reference_mask, c], errors="coerce").dropna().to_numpy(float)
        if not len(x) or not len(y): continue
        row = {"Metric": c, "Target n": len(x), "Reference n": len(y), "Target median": np.median(x), "Reference median": np.median(y), "Median difference": np.median(x)-np.median(y)}
        if SCIPY_AVAILABLE:
            try:
                u, p = mannwhitneyu(x, y, alternative="two-sided")
                row["Mann–Whitney p"] = p
                row["Rank-biserial"] = rank_biserial_from_u(u, len(x), len(y))
            except Exception:
                pass
        rows.append(row)
    return pd.DataFrame(rows)


def pca_feature_space(df, features, target_mask, reference_mask):
    available = [c for c in features if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() >= 3]
    if len(available) < 3: return pd.DataFrame(), None, []
    X = df[available].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)
    group = np.where(target_mask, "Highlighted", np.where(reference_mask, "Internal reference", "Other"))
    out = pd.DataFrame({"PC1": pcs[:,0], "PC2": pcs[:,1], "Group": group, "tree_id": df["tree_id"].values})
    return out, pca.explained_variance_ratio_*100, available


def morans_i_knn(gdf, target_mask, k=4, permutations=199, seed=42):
    y = np.asarray(target_mask, dtype=int)
    prevalence = y.mean() if len(y) else np.nan
    if len(y) < max(5, k+1) or prevalence in (0,1): return np.nan, np.nan, prevalence, np.nan
    try:
        projected = gdf.to_crs(gdf.estimate_utm_crs())
        cent = projected.geometry.centroid
        coords = np.c_[cent.x, cent.y]
    except Exception:
        return np.nan, np.nan, prevalence, np.nan
    n = len(y); k_eff=min(k,n-1)
    neigh=NearestNeighbors(n_neighbors=k_eff+1).fit(coords).kneighbors(return_distance=False)[:,1:]
    def calc(vals):
        z=vals-vals.mean(); den=np.sum(z*z)
        if den==0: return np.nan
        return sum(np.mean(z[i]*z[neigh[i]]) for i in range(n))/den
    obs=calc(y.astype(float))
    idx=np.where(y==1)[0]
    nbr=float(np.mean([y[neigh[i]].mean() for i in idx])) if len(idx) else np.nan
    lift=nbr/prevalence if prevalence else np.nan
    rng=np.random.default_rng(seed)
    perms=np.asarray([calc(rng.permutation(y).astype(float)) for _ in range(permutations)])
    p=(np.sum(np.abs(perms)>=abs(obs))+1)/(len(perms)+1)
    return obs,p,prevalence,lift


def load_wavelength_map():
    path = first_existing(WAVELENGTH_MAP_CANDIDATES)
    if path is None: return {}, None
    try:
        d=pd.read_csv(path)
        cols={c.lower():c for c in d.columns}
        wave=next((c for k,c in cols.items() if "wavelength" in k),None)
        band=next((c for k,c in cols.items() if k in {"band","band_name","band_col","column"}),None)
        if wave and band:
            return {str(r[band]):float(r[wave]) for _,r in d.dropna(subset=[wave]).iterrows()}, path.name
    except Exception:
        pass
    return {}, path.name


def ground_wide_to_long(df):
    id_col = "tree_id" if "tree_id" in df.columns else ("spectrum_id" if "spectrum_id" in df.columns else None)
    if id_col is None: return pd.DataFrame(), None
    rows=[]
    for c in df.columns:
        if c == id_col: continue
        m=re.search(r"(\d{3,4}(?:\.\d+)?)", str(c))
        if not m: continue
        wl=float(m.group(1))
        vals=pd.to_numeric(df[c], errors="coerce")
        for idx,v in vals.items():
            if pd.notna(v): rows.append({id_col:df.loc[idx,id_col],"wavelength_nm":wl,"reflectance":float(v)})
    return pd.DataFrame(rows), id_col


def normalize_ground_long(df):
    cols={c.lower():c for c in df.columns}
    wave=next((c for k,c in cols.items() if k in {"wavelength_nm","wavelength","nm"} or "wavelength" in k),None)
    refl=next((c for k,c in cols.items() if k in {"reflectance","refl","value"} or "reflectance" in k),None)
    id_col="tree_id" if "tree_id" in df.columns else ("spectrum_id" if "spectrum_id" in df.columns else None)
    if wave and refl and id_col:
        out=df[[id_col,wave,refl]].rename(columns={wave:"wavelength_nm",refl:"reflectance"}).copy()
        out["wavelength_nm"]=pd.to_numeric(out["wavelength_nm"],errors="coerce")
        out["reflectance"]=pd.to_numeric(out["reflectance"],errors="coerce")
        return out.dropna(),id_col
    return ground_wide_to_long(df)


def spectrum_value(group, target_nm, tol=6.0):
    if group.empty: return np.nan
    w=group["wavelength_nm"].to_numpy(float); r=group["reflectance"].to_numpy(float)
    i=int(np.argmin(np.abs(w-target_nm)))
    return float(r[i]) if abs(w[i]-target_nm)<=tol else np.nan


def compute_ground_indices(long_df, id_col, reflectance_scale="Auto"):
    if long_df.empty: return pd.DataFrame()
    d=long_df.copy()
    if reflectance_scale == "Percent (0–100)": d["reflectance"]=d["reflectance"]/100.0
    elif reflectance_scale == "Auto":
        med=np.nanmedian(d["reflectance"].to_numpy(float))
        if np.isfinite(med) and med>2: d["reflectance"]=d["reflectance"]/100.0
    rows=[]
    for sid,g in d.groupby(id_col):
        R=lambda x:spectrum_value(g,x)
        vals={x:R(x) for x in [445,500,531,550,570,670,680,700,705,750,800,900,970,1100,1222,2200,2264]}
        def sd(a,b): return (a-b)/(a+b) if np.isfinite(a) and np.isfinite(b) and abs(a+b)>1e-12 else np.nan
        row={id_col:sid}
        row.update({
            "GROUND_NDVI":sd(vals[800],vals[670]),
            "GROUND_NDRE":sd(vals[750],vals[705]),
            "GROUND_CIRED_EDGE":(vals[750]/vals[705]-1) if np.isfinite(vals[750]) and np.isfinite(vals[705]) and abs(vals[705])>1e-12 else np.nan,
            "GROUND_WBI":(vals[900]/vals[970]) if np.isfinite(vals[900]) and np.isfinite(vals[970]) and abs(vals[970])>1e-12 else np.nan,
            "GROUND_PRI":sd(vals[531],vals[570]),
            "GROUND_PSRI":((vals[680]-vals[500])/vals[750]) if np.isfinite(vals[680]) and np.isfinite(vals[500]) and np.isfinite(vals[750]) and abs(vals[750])>1e-12 else np.nan,
            "GROUND_SIPI":((vals[800]-vals[445])/(vals[800]-vals[680])) if np.isfinite(vals[800]) and np.isfinite(vals[445]) and np.isfinite(vals[680]) and abs(vals[800]-vals[680])>1e-12 else np.nan,
            "GROUND_ARI1":(1/vals[550]-1/vals[700]) if np.isfinite(vals[550]) and np.isfinite(vals[700]) and vals[550]!=0 and vals[700]!=0 else np.nan,
            "GROUND_NDMI2":sd(vals[2200],vals[1100]),
            "GROUND_NDSI_RWC":sd(vals[1222],vals[2264]),
        })
        # REP from maximum positive derivative 680–750 nm
        gg=g[(g["wavelength_nm"]>=680)&(g["wavelength_nm"]<=750)].sort_values("wavelength_nm")
        if len(gg)>=3:
            w=gg["wavelength_nm"].to_numpy(float); r=gg["reflectance"].to_numpy(float)
            der=np.gradient(r,w); row["GROUND_REP_D1_NM"]=float(w[np.nanargmax(der)])
        else: row["GROUND_REP_D1_NM"]=np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 7. SIMPLE VISUAL HELPERS
# =============================================================================

def status_card(title, value, subtitle, icon=""):
    st.markdown(
        f"""<div style='padding:1rem;border:1px solid rgba(128,128,128,.35);border-radius:14px;height:145px;'>
        <div style='font-size:1.1rem;font-weight:700'>{icon} {html.escape(str(title))}</div>
        <div style='font-size:2rem;font-weight:800;margin:.35rem 0'>{html.escape(str(value))}</div>
        <div style='opacity:.75;font-size:.9rem'>{html.escape(str(subtitle))}</div>
        </div>""", unsafe_allow_html=True
    )


def indicator_histogram(df, column, threshold=None, direction=None, title=None):
    if column not in df.columns: return None
    s=pd.to_numeric(df[column],errors="coerce").dropna()
    if s.empty: return None
    fig=px.histogram(x=s, nbins=28, labels={"x":column,"y":"Trees"}, title=title or column)
    if threshold is not None and np.isfinite(threshold):
        fig.add_vline(x=threshold,line_dash="dash",annotation_text=f"Operational threshold {threshold:.3f}")
    fig.update_layout(height=310,margin=dict(l=10,r=10,t=50,b=10),showlegend=False)
    return fig


# =============================================================================
# 8. APP LOAD / CONTROLS
# =============================================================================

st.set_page_config(page_title="Orchard Three-Domain DSS", layout="wide")
st.title(APP_TITLE)
st.caption("Choose a physiological/structural question, see the affected trees on the map, and hover/click a crown to see the evidence used. The system is a screening and field-prioritization tool, not a disease diagnosis.")

geometry_gdf = load_tree_geometry()
rule_df, rule_source = load_final_rule_database()
spectral_df = load_spectral_data()
try:
    gdf = merge_geometry_and_rules(geometry_gdf, rule_df)
except Exception as exc:
    st.error(f"Could not align canopy geometry and final database: {exc}")
    st.stop()

st.sidebar.header("What do you want to inspect?")
view = st.sidebar.radio("Choose a scenario", options=list(VIEW_OPTIONS), format_func=lambda k: VIEW_OPTIONS[k], label_visibility="collapsed")
st.sidebar.caption("The map highlights only trees meeting the selected rule. Hover a highlighted crown for the measurements used.")

include_screening=False
combined_mode="2+ domains"
screening_mode="Screening evidence"
if view in {"WATER","BIOCHEMICAL"}:
    include_screening=st.sidebar.checkbox("Also show weaker screening-level cases", value=False)
if view=="COMBINED":
    combined_mode=st.sidebar.selectbox("Combined view", ["2+ domains","3 domains","Exactly 2 domains"])
if view=="SCREENING":
    screening_mode=st.sidebar.selectbox("Review layer", ["Screening evidence","Data-limited review"])

if view=="GAPS":
    with st.sidebar.expander("Planting-gap settings", expanded=False):
        expected_tree_spacing=st.number_input("Expected tree spacing (m)",0.5,20.0,DEFAULT_TREE_SPACING_M,0.1)
        row_distance_threshold=st.number_input("Row clustering threshold (m)",0.5,10.0,DEFAULT_ROW_DISTANCE_THRESHOLD_M,0.1)
        grid_angle_degrees=st.number_input("Grid rotation angle (deg)",-180.0,180.0,DEFAULT_GRID_ANGLE_DEG,1.0)
        max_empty_space_m=st.number_input("Maximum gap considered (m)",2.0,100.0,DEFAULT_MAX_EMPTY_SPACE_M,1.0)
    gaps_gdf,total_trees,total_gaps,yield_loss_percentage=calculate_gaps(gdf,expected_tree_spacing,row_distance_threshold,grid_angle_degrees,max_empty_space_m)
    target_mask=pd.Series(False,index=gdf.index); target_gdf=gdf.iloc[0:0].copy()
else:
    gaps_gdf=gpd.GeoDataFrame(columns=["geometry"],geometry="geometry",crs="EPSG:4326")
    target_mask=make_target_mask(gdf,view,include_screening,combined_mode,screening_mode)
    target_gdf=gdf[target_mask].copy()

st.sidebar.markdown("---")
st.sidebar.caption(rule_source)
st.sidebar.metric("Highlighted trees", len(target_gdf) if view!="GAPS" else len(gaps_gdf))
if view!="GAPS":
    st.sidebar.caption(f"{100*len(target_gdf)/len(gdf):.1f}% of {len(gdf)} orchard trees")


# =============================================================================
# 9. MAIN TABS
# =============================================================================

tab_map, tab_evidence, tab_summary, tab_validation, tab_methods = st.tabs([
    "🗺️ Map & Scenarios",
    "🧭 Evidence Explained",
    "📊 Orchard Summary",
    "🧪 Validation & Robustness",
    "📘 Methods",
])

with tab_map:
    if view=="WATER":
        st.header("💧 Water-status screening")
        st.write("Highlighted trees meet the finalized water rule. Hover a tree to see WBI, SWIR water indices and PRI support.")
    elif view=="BIOCHEMICAL":
        st.header("🧪 Canopy biochemical screening")
        st.write("Highlighted trees have concordant chlorophyll/red-edge and pigment/senescence evidence.")
    elif view=="STRUCTURE":
        st.header("🌳 Structural stature")
        st.write("Highlighted trees have orchard-relative low LAS H-P95. Green = raster-corroborated; yellow = LAS-only.")
    elif view=="COMBINED":
        st.header("🔗 Cross-domain priority")
        st.write("Red = all three domains; orange = exactly two domains. Agreement increases inspection priority, not causal certainty.")
    elif view=="SCREENING":
        st.header("🔎 Screening / incomplete evidence")
        st.write("These trees do not meet a supported major-domain rule but still deserve review because evidence is partial, discordant or incomplete.")
    else:
        st.header("🍋 Planting-gap inventory")

    if view!="GAPS":
        c1,c2,c3=st.columns(3)
        c1.metric("Highlighted",len(target_gdf))
        c2.metric("Share of orchard",f"{100*len(target_gdf)/len(gdf):.1f}%")
        if view=="COMBINED": c3.metric("3-domain trees",int((gdf["SUPPORTED_DOMAIN_COUNT"]==3).sum()))
        else: c3.metric("Total trees",len(gdf))

    center=[gdf.geometry.centroid.y.mean(),gdf.geometry.centroid.x.mean()]
    m=folium.Map(location=center,zoom_start=18,max_zoom=22,tiles="CartoDB positron")
    folium.GeoJson(gdf,style_function=lambda _:{"fillColor":"#D0D0D0","color":"#777777","weight":0.7,"fillOpacity":0.05},name="All tree crowns").add_to(m)
    if view=="GAPS":
        for _,row in gaps_gdf.iterrows():
            folium.CircleMarker([row.geometry.y,row.geometry.x],radius=5,color="#C0392B",fill=True,fill_opacity=.9,tooltip="Calculated planting gap").add_to(m)
    else:
        add_target_layer(m,target_gdf,view)
    folium.LayerControl(collapsed=True).add_to(m)
    map_event=st_folium(m,height=650,use_container_width=True)
    st.caption("Hover = quick evidence. Click = persistent popup with the same evidence. No Tree-ID selection is required.")

    if view!="GAPS" and not target_gdf.empty:
        col1,col2=st.columns(2)
        with col1:
            st.download_button("Download highlighted trees (GeoJSON)",target_gdf.to_json(),file_name=f"{view.lower()}_targets.geojson",mime="application/geo+json")
        with col2:
            color = VIEW_COLORS.get(view, VIEW_COLORS["COMBINED_2"])
            kml=build_tree_navigation_kml(target_gdf,f"{VIEW_OPTIONS[view]} targets",color)
            st.download_button("📍 Download highlighted trees (KML)",kml,file_name=f"{view.lower()}_targets.kml",mime="application/vnd.google-earth.kml+xml")
    elif view=="GAPS" and len(gaps_gdf):
        st.metric("Calculated gaps",total_gaps)
        st.metric("Estimated planting-capacity gap",f"{yield_loss_percentage:.1f}%")

with tab_evidence:
    st.header("What the highlighted trees mean")
    if view=="WATER":
        cols=st.columns(3)
        with cols[0]: status_card("Direct VNIR water evidence","WBI","Low WBI contributes one direct water block","💧")
        with cols[1]: status_card("SWIR water evidence","NDMI2 + NDSI-RWC","These two correlated indices count as ONE block","🌊")
        with cols[2]: status_card("Physiology support","PRI","Corroborates but cannot create the supported water class alone","⚡")
        st.info("Supported water anomaly = abnormal WBI AND concordant SWIR block. PRI can strengthen the internal evidence.")
        charts=st.columns(2)
        with charts[0]:
            fig=indicator_histogram(gdf,"WBI_VALUE",WATER_SENSITIVITY_THRESHOLDS["P25"]["WBI_LOW_MAX"],title="WBI across orchard")
            if fig: st.plotly_chart(fig,use_container_width=True)
        with charts[1]:
            fig=indicator_histogram(gdf,"NDMI2_VALUE",WATER_SENSITIVITY_THRESHOLDS["P25"]["NDMI2_HIGH_MIN"],title="NDMI2 across orchard")
            if fig: st.plotly_chart(fig,use_container_width=True)
        supported=as_bool(gdf["WATER_SUPPORTED"])
        x1,x2,x3=st.columns(3)
        x1.metric("Supported water trees",int(supported.sum()))
        x2.metric("Also biochemical",int((supported & as_bool(gdf["BIOCHEMICAL_SUPPORTED"])).sum()))
        x3.metric("Also low stature",int((supported & as_bool(gdf["STRUCTURE_LOW_STATURE"])).sum()))
    elif view=="BIOCHEMICAL":
        cols=st.columns(3)
        with cols[0]: status_card("Chlorophyll / red-edge block","NDRE + CIred + REP","All three must support the same direction","🍃")
        with cols[1]: status_card("Pigment / senescence block","PSRI + SIPI","Both must support the pigment response","🧪")
        with cols[2]: status_card("Additional support","ARI1","Strengthens concordant biochemical decline; NDVI is context","🔬")
        st.info("A supported biochemical domain requires BOTH the chlorophyll/red-edge block and the pigment/senescence block.")
        charts=st.columns(2)
        with charts[0]:
            fig=indicator_histogram(gdf,"NDRE_VALUE",BIO_THRESHOLDS["NDRE_LOW_MAX"],title="NDRE across orchard")
            if fig: st.plotly_chart(fig,use_container_width=True)
        with charts[1]:
            fig=indicator_histogram(gdf,"PSRI_VALUE",BIO_THRESHOLDS["PSRI_HIGH_MIN"],title="PSRI across orchard")
            if fig: st.plotly_chart(fig,use_container_width=True)
        supported=as_bool(gdf["BIOCHEMICAL_SUPPORTED"])
        x1,x2,x3=st.columns(3)
        x1.metric("Supported biochemical trees",int(supported.sum()))
        x2.metric("Also water",int((supported & as_bool(gdf["WATER_SUPPORTED"])).sum()))
        x3.metric("Also low stature",int((supported & as_bool(gdf["STRUCTURE_LOW_STATURE"])).sum()))
    elif view=="STRUCTURE":
        cols=st.columns(3)
        with cols[0]: status_card("Primary stature","LAS H-P95",f"Low stature ≤ {STRUCTURE_THRESHOLDS['H_P95_P25_M']:.3f} m","🌳")
        with cols[1]: status_card("Independent measurement check","Raster CHM P95",f"Corroboration threshold ≤ {STRUCTURE_THRESHOLDS['RASTER_CHM_P95_P25_M']:.3f} m","🛰️")
        with cols[2]: status_card("Canopy complexity context","H-IQR","Shown for context; not a stress vote","📐")
        fig=indicator_histogram(gdf,"H_P95_m",STRUCTURE_THRESHOLDS["H_P95_P25_M"],title="LAS H-P95 across orchard")
        if fig: st.plotly_chart(fig,use_container_width=True)
        s=as_bool(gdf["STRUCTURE_LOW_STATURE"]); cor=as_bool(gdf["STRUCTURE_CORROBORATED"])
        x1,x2,x3=st.columns(3)
        x1.metric("Low-stature trees",int(s.sum()))
        x2.metric("Raster-corroborated",int((s&cor).sum()))
        x3.metric("LAS-only",int((s&~cor).sum()))
        st.warning("Low stature describes relative canopy size, not a proven cause or disease.")
    elif view=="COMBINED":
        c1,c2,c3=st.columns(3)
        c1.metric("All 3 domains",int((gdf["SUPPORTED_DOMAIN_COUNT"]==3).sum()))
        c2.metric("Exactly 2 domains",int((gdf["SUPPORTED_DOMAIN_COUNT"]==2).sum()))
        c3.metric("Exactly 1 domain",int((gdf["SUPPORTED_DOMAIN_COUNT"]==1).sum()))
        patt=gdf["CROSS_DOMAIN_PATTERN"].map(human_pattern).value_counts().reset_index()
        patt.columns=["Pattern","Trees"]
        fig=px.bar(patt,x="Trees",y="Pattern",orientation="h",title="How the three domains combine")
        fig.update_layout(height=430,margin=dict(l=10,r=10,t=50,b=10),yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig,use_container_width=True)
        st.info("Water, biochemical and structure are evaluated independently first. Their agreement strengthens inspection priority but does not prove a causal chain.")
    elif view=="SCREENING":
        tier=gdf["FIELD_INSPECTION_TIER"].map(human_tier).value_counts().reset_index(); tier.columns=["Review category","Trees"]
        fig=px.bar(tier,x="Review category",y="Trees",title="Why some trees are not placed in a supported domain")
        st.plotly_chart(fig,use_container_width=True)
        st.info("Screening priority = partial evidence. Data-limited review = missing/inconclusive evidence. Neither category should be called normal.")
    else:
        st.info("Gap analysis is an orchard inventory function and is intentionally separate from the physiological/structural rule engine.")

with tab_summary:
    st.header("Orchard at a glance")
    w=int(as_bool(gdf["WATER_SUPPORTED"]).sum()); b=int(as_bool(gdf["BIOCHEMICAL_SUPPORTED"]).sum()); s=int(as_bool(gdf["STRUCTURE_LOW_STATURE"]).sum())
    c1,c2,c3,c4=st.columns(4)
    c1.metric("💧 Water evidence",w,f"{100*w/len(gdf):.1f}% of orchard")
    c2.metric("🧪 Biochemical evidence",b,f"{100*b/len(gdf):.1f}% of orchard")
    c3.metric("🌳 Low stature",s,f"{100*s/len(gdf):.1f}% of orchard")
    c4.metric("🔴 All 3 domains",int((gdf["SUPPORTED_DOMAIN_COUNT"]==3).sum()),"highest cross-domain priority")

    left,right=st.columns(2)
    with left:
        dc=gdf["SUPPORTED_DOMAIN_COUNT"].value_counts().sort_index().reset_index(); dc.columns=["Supported domains","Trees"]
        fig=px.pie(dc,values="Trees",names="Supported domains",hole=.55,title="How many domains support each tree?")
        st.plotly_chart(fig,use_container_width=True)
    with right:
        domain_df=pd.DataFrame({"Domain":["Water","Biochemical","Structure"],"Trees":[w,b,s]})
        fig=px.bar(domain_df,x="Domain",y="Trees",text="Trees",title="Supported evidence by domain")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig,use_container_width=True)

    left,right=st.columns(2)
    with left:
        patt=gdf["CROSS_DOMAIN_PATTERN"].map(human_pattern).value_counts().reset_index(); patt.columns=["Pattern","Trees"]
        fig=px.bar(patt,x="Trees",y="Pattern",orientation="h",title="Cross-domain combinations")
        fig.update_layout(height=430,yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig,use_container_width=True)
    with right:
        comp=gdf["MULTIDOMAIN_COMPLETENESS"].replace({"COMPLETE_3_OF_3":"Complete: 3/3 domains","PARTIAL_2_OF_3":"Partial: 2/3 domains","LIMITED_0_OR_1_OF_3":"Limited: 0–1/3 domains"}).value_counts().reset_index(); comp.columns=["Completeness","Trees"]
        fig=px.pie(comp,values="Trees",names="Completeness",hole=.55,title="Data completeness")
        st.plotly_chart(fig,use_container_width=True)
    st.warning("No supported anomaly ≠ confirmed healthy. A tree may be data-limited, outside these rule thresholds, younger/pruned, or affected by a factor not represented in the three domains.")

    with st.expander("Show exact orchard counts / download"):
        exact=pd.DataFrame({
            "Inspection tier":gdf["FIELD_INSPECTION_TIER"].map(human_tier).value_counts(),
        }).reset_index().rename(columns={"index":"Category","Inspection tier":"Trees"})
        st.dataframe(exact,hide_index=True,use_container_width=True)
        cols=["tree_id","FIELD_INSPECTION_TIER","CROSS_DOMAIN_PATTERN","SUPPORTED_DOMAIN_COUNT","ASSESSABLE_DOMAIN_COUNT","WATER_EVIDENCE_STATUS","BIOCHEMICAL_STATUS","STRUCTURE_STATUS_FINAL","CROSS_DOMAIN_INTERPRETATION"]
        st.download_button("Download orchard summary CSV",gdf[[c for c in cols if c in gdf.columns]].to_csv(index=False).encode(),"orchard_three_domain_summary.csv","text/csv")

with tab_validation:
    st.header("Validation & robustness — research view")
    st.caption("This page preserves the internal validation metrics from the earlier research app and adds an external ground-spectroradiometer module. Internal robustness is not the same as field accuracy.")
    if view in {"SCREENING","GAPS"}:
        st.info("For rule robustness, choose Water, Biochemical, Structure or Combined multi-domain priority in the sidebar.")
    else:
        reference_mask = gdf["FIELD_INSPECTION_TIER"].eq("NO_SUPPORTED_ANOMALY") & gdf["MULTIDOMAIN_COMPLETENESS"].eq("COMPLETE_3_OF_3")
        current_mask = target_mask.astype(bool)
        a,b,c=st.columns(3)
        a.metric("Highlighted trees",int(current_mask.sum()))
        b.metric("Internal reference trees",int(reference_mask.sum()))
        c.metric("Reference share",f"{100*reference_mask.mean():.1f}%")

        st.subheader("1. Multi-domain / multi-indicator concordance")
        if view=="WATER":
            temp=pd.DataFrame({"Water status":gdf["WATER_EVIDENCE_STATUS"].value_counts()}).reset_index(); st.dataframe(temp,hide_index=True,use_container_width=True)
        elif view=="BIOCHEMICAL":
            temp=pd.DataFrame({"Biochemical status":gdf["BIOCHEMICAL_STATUS"].value_counts()}).reset_index(); st.dataframe(temp,hide_index=True,use_container_width=True)
        elif view=="STRUCTURE":
            temp=pd.DataFrame({"Structure status":gdf["STRUCTURE_STATUS_FINAL"].value_counts()}).reset_index(); st.dataframe(temp,hide_index=True,use_container_width=True)
        else:
            temp=gdf["SUPPORTED_DOMAIN_COUNT"].value_counts().sort_index().reset_index(); temp.columns=["Supported domains","Trees"]; st.dataframe(temp,hide_index=True,use_container_width=True)

        st.subheader("2. Spectral consistency against the internal reference")
        band_cols=[c for c in spectral_df.columns if str(c).startswith("Band_")] if not spectral_df.empty else []
        if not spectral_df.empty and band_cols and current_mask.any() and reference_mask.any():
            key="tree_id" if "tree_id" in spectral_df.columns else ("generated_id" if "generated_id" in spectral_df.columns and "generated_id" in gdf.columns else None)
            if key:
                target_ids=gdf.loc[current_mask,key].tolist(); ref_ids=gdf.loc[reference_mask,key].tolist()
                t=spectral_df[spectral_df[key].isin(target_ids)]; r=spectral_df[spectral_df[key].isin(ref_ids)]
                if len(t) and len(r):
                    tm=t[band_cols].apply(pd.to_numeric,errors="coerce").mean().to_numpy(float); rm=r[band_cols].apply(pd.to_numeric,errors="coerce").mean().to_numpy(float)
                    sam=spectral_angle_deg(tm,rm); rms=spectral_rmse(tm,rm)
                    q1,q2=st.columns(2); q1.metric("Spectral angle",f"{sam:.3f}°" if np.isfinite(sam) else "NA"); q2.metric("Reflectance RMSE",f"{rms:.5f}" if np.isfinite(rms) else "NA")
                    x=np.arange(1,len(band_cols)+1)
                    fig=go.Figure(); fig.add_trace(go.Scatter(x=x,y=tm,name="Highlighted mean")); fig.add_trace(go.Scatter(x=x,y=rm,name="Internal reference mean")); fig.update_layout(title="Consensus canopy spectra",xaxis_title="Band number",yaxis_title="Reflectance",height=370)
                    st.plotly_chart(fig,use_container_width=True)
                else: st.info("No post-QC consensus spectra overlap both groups.")
            else: st.info("Consensus spectrum table has no usable tree identifier.")
        else: st.info("Post-QC consensus spectra are unavailable or one comparison group is empty.")

        st.subheader("3. Threshold sensitivity and Jaccard robustness")
        schemes=["P20","P25","P30"]
        runs={s:sensitivity_target_mask(gdf,view,s,combined_mode) for s in schemes}
        sens=pd.DataFrame({"Scheme":schemes,"Highlighted trees":[int(runs[s].sum()) for s in schemes]})
        st.plotly_chart(px.bar(sens,x="Scheme",y="Highlighted trees",text="Highlighted trees",title="How target count changes when thresholds move"),use_container_width=True)
        jac=pd.DataFrame(index=schemes,columns=schemes,dtype=float)
        for i in schemes:
            for j in schemes: jac.loc[i,j]=jaccard(runs[i],runs[j])
        st.write("Jaccard similarity (1.0 = identical target set)"); st.dataframe(jac.round(3),use_container_width=True)
        if runs["P25"].any():
            stability=np.c_[runs["P20"],runs["P25"],runs["P30"]].mean(axis=1)
            stable=float(np.mean(stability[runs["P25"].to_numpy()] == 1.0)); st.metric("P25 targets retained in all three schemes",f"{100*stable:.1f}%")

        st.subheader("4. Highlighted vs internal-reference distributions")
        stats_df=comparison_statistics(gdf,current_mask,reference_mask,scenario_metrics(view))
        if not stats_df.empty: st.dataframe(stats_df.round(5),hide_index=True,use_container_width=True)
        else: st.info("Not enough observations for a group comparison.")

        st.subheader("5. Structural corroboration")
        if current_mask.any():
            low=as_bool(gdf["STRUCTURE_LOW_STATURE"]); cor=as_bool(gdf["STRUCTURE_CORROBORATED"])
            c1,c2=st.columns(2); c1.metric("Highlighted trees also low stature",int((current_mask&low).sum())); c2.metric("...with raster corroboration",int((current_mask&low&cor).sum()))
        st.caption("Structural agreement strengthens internal plausibility but does not establish the cause of the spectral anomaly.")

        st.subheader("6. Multivariate feature-space corroboration (PCA)")
        features=scenario_metrics("COMBINED")
        pca_df,explained,used=pca_feature_space(gdf,features,current_mask,reference_mask)
        if not pca_df.empty:
            fig=px.scatter(pca_df,x="PC1",y="PC2",color="Group",hover_data=["tree_id"],title=f"PCA feature space — PC1 {explained[0]:.1f}%, PC2 {explained[1]:.1f}%")
            st.plotly_chart(fig,use_container_width=True)
            with st.expander("Features used in PCA"): st.write(used)
        else: st.info("Insufficient numeric features for PCA.")

        st.subheader("7. Spatial coherence (exploratory kNN Moran's I)")
        I,p,prev,lift=morans_i_knn(gdf,current_mask,k=4,permutations=199)
        if np.isfinite(I):
            c1,c2,c3=st.columns(3); c1.metric("Moran's I",f"{I:.3f}"); c2.metric("Permutation pseudo-p",f"{p:.3f}"); c3.metric("Flagged-neighbor lift",f"{lift:.2f}×" if np.isfinite(lift) else "NA")
        else: st.info("Spatial coherence requires a non-trivial mix of highlighted and non-highlighted trees.")

    st.markdown("---")
    st.subheader("8. Ground spectroradiometer validation")
    st.write("Use the processed SVC/XHR ground spectra here. This is the external validation layer only when spectra are tree-matched and collected close enough to the UAV acquisition to be biologically comparable.")
    auto_ground=first_existing(GROUND_SPECTRA_CANDIDATES)
    upload=st.file_uploader("Upload processed ground spectroradiometer CSV",type=["csv"],key="ground_spec")
    ground_raw=None
    if upload is not None:
        ground_raw=pd.read_csv(upload)
    elif auto_ground is not None:
        ground_raw=pd.read_csv(auto_ground); st.caption(f"Loaded ground spectra: {auto_ground.name}")
    if ground_raw is None:
        st.info("No processed ground spectroradiometer CSV is loaded yet. The internal validation sections above remain fully functional.")
        st.caption("Accepted formats: long form [tree_id/spectrum_id, wavelength_nm, reflectance] or wide form with wavelength-named columns. For UAV-vs-ground quantitative validation, a tree_id mapping is required.")
    else:
        ground_long,id_col=normalize_ground_long(ground_raw)
        if ground_long.empty:
            st.error("Could not identify spectrum ID, wavelength and reflectance columns in this file.")
        else:
            scale=st.selectbox("Ground reflectance scale",["Auto","Fraction (0–1)","Percent (0–100)"],index=0)
            ground_idx=compute_ground_indices(ground_long,id_col,scale)
            st.success(f"Parsed {ground_idx[id_col].nunique()} ground spectra.")
            if id_col != "tree_id":
                st.warning("The file has spectrum_id but no tree_id. Upload a mapping CSV with columns spectrum_id,tree_id to compare ground spectra with UAV trees.")
                map_upload=st.file_uploader("Optional spectrum-to-tree mapping CSV",type=["csv"],key="ground_map")
                if map_upload is not None:
                    mp=pd.read_csv(map_upload)
                    if {"spectrum_id","tree_id"}.issubset(mp.columns): ground_idx=ground_idx.merge(mp[["spectrum_id","tree_id"]],on="spectrum_id",how="left")
            if "tree_id" in ground_idx.columns:
                merged=ground_idx.merge(gdf.drop(columns="geometry"),on="tree_id",how="inner")
                st.metric("Tree-matched ground spectra",len(merged))
                pairs=[("GROUND_WBI","WBI_VALUE","WBI"),("GROUND_PRI","PRI_VALUE","PRI"),("GROUND_NDVI","NDVI_VALUE","NDVI"),("GROUND_NDRE","NDRE_VALUE","NDRE"),("GROUND_CIRED_EDGE","CIRED_EDGE_VALUE","CIred-edge"),("GROUND_REP_D1_NM","REP_D1_NM_VALUE","REP"),("GROUND_PSRI","PSRI_VALUE","PSRI"),("GROUND_SIPI","SIPI_VALUE","SIPI"),("GROUND_ARI1","ARI1_VALUE","ARI1"),("GROUND_NDMI2","NDMI2_VALUE","NDMI2"),("GROUND_NDSI_RWC","NDSI_RWC_VALUE","NDSI-RWC")]
                rows=[]
                for gc,uc,label in pairs:
                    if gc not in merged.columns or uc not in merged.columns: continue
                    x=pd.to_numeric(merged[gc],errors="coerce"); y=pd.to_numeric(merged[uc],errors="coerce"); mask=x.notna()&y.notna()
                    if mask.sum()<2: continue
                    xv=x[mask].to_numpy(float); yv=y[mask].to_numpy(float); r=np.corrcoef(xv,yv)[0,1] if len(xv)>=2 else np.nan
                    rows.append({"Index":label,"n":len(xv),"Pearson r":r,"MAE":np.mean(np.abs(yv-xv)),"RMSE":np.sqrt(np.mean((yv-xv)**2)),"Bias (UAV-ground)":np.mean(yv-xv)})
                metrics=pd.DataFrame(rows)
                if not metrics.empty:
                    st.dataframe(metrics.round(4),hide_index=True,use_container_width=True)
                    choice=st.selectbox("Plot ground vs UAV index",metrics["Index"].tolist())
                    gc,uc,_=next(p for p in pairs if p[2]==choice)
                    plot=merged[["tree_id",gc,uc]].dropna().rename(columns={gc:"Ground",uc:"UAV"})
                    fig=px.scatter(plot,x="Ground",y="UAV",hover_data=["tree_id"],title=f"{choice}: ground spectroradiometer vs UAV crown")
                    if len(plot):
                        lo=min(plot["Ground"].min(),plot["UAV"].min()); hi=max(plot["Ground"].max(),plot["UAV"].max()); fig.add_shape(type="line",x0=lo,y0=lo,x1=hi,y1=hi,line=dict(dash="dash"))
                    st.plotly_chart(fig,use_container_width=True)
                else: st.info("Matched trees exist, but not enough paired index values were available for quantitative comparison.")

                # Optional full-spectrum ground-vs-UAV comparison when exact UAV wavelength mapping is available.
                wl_map, wl_source = load_wavelength_map()
                band_cols_uav = [c for c in spectral_df.columns if str(c).startswith("Band_")] if not spectral_df.empty else []
                spec_key = "tree_id" if "tree_id" in spectral_df.columns else ("generated_id" if "generated_id" in spectral_df.columns and "generated_id" in gdf.columns else None)
                if wl_map and band_cols_uav and spec_key:
                    # If spectral table uses generated_id, map canonical tree_id to generated_id from final database.
                    id_bridge = gdf[["tree_id", spec_key]].drop_duplicates() if spec_key != "tree_id" else None
                    spectral_metrics=[]
                    for tree_id in merged["tree_id"].dropna().unique():
                        ground_group = ground_long[ground_long[id_col].eq(tree_id)] if id_col == "tree_id" else pd.DataFrame()
                        if ground_group.empty and id_col == "spectrum_id" and "spectrum_id" in ground_idx.columns:
                            sids = ground_idx.loc[ground_idx.get("tree_id").eq(tree_id), "spectrum_id"].tolist()
                            ground_group = ground_long[ground_long[id_col].isin(sids)]
                        if ground_group.empty:
                            continue
                        lookup_id = tree_id
                        if spec_key != "tree_id":
                            bridge = id_bridge[id_bridge["tree_id"].eq(tree_id)]
                            if bridge.empty: continue
                            lookup_id = bridge.iloc[0][spec_key]
                        urow = spectral_df[spectral_df[spec_key].eq(lookup_id)]
                        if urow.empty: continue
                        usable=[]
                        for bc in band_cols_uav:
                            if bc not in wl_map: continue
                            gv=spectrum_value(ground_group,float(wl_map[bc]),tol=6.0)
                            uv=pd.to_numeric(pd.Series([urow.iloc[0][bc]]),errors="coerce").iloc[0]
                            if pd.notna(gv) and pd.notna(uv): usable.append((gv,float(uv)))
                        if len(usable)>=20:
                            ga=np.array([x[0] for x in usable],float); ua=np.array([x[1] for x in usable],float)
                            spectral_metrics.append({"tree_id":tree_id,"bands compared":len(usable),"SAM_deg":spectral_angle_deg(ga,ua),"RMSE":spectral_rmse(ga,ua),"MAE":float(np.mean(np.abs(ua-ga))),"Bias_UAV_minus_ground":float(np.mean(ua-ga))})
                    if spectral_metrics:
                        st.write("**Full-spectrum ground vs UAV agreement**")
                        st.caption(f"UAV wavelength mapping: {wl_source}")
                        sm=pd.DataFrame(spectral_metrics)
                        st.dataframe(sm.round(5),hide_index=True,use_container_width=True)
                        c1,c2=st.columns(2); c1.metric("Median SAM",f"{sm['SAM_deg'].median():.3f}°"); c2.metric("Median spectral RMSE",f"{sm['RMSE'].median():.5f}")
                else:
                    st.caption("Full-spectrum ground-vs-UAV SAM/RMSE becomes available when an exact UAV band-to-wavelength mapping CSV is provided. Index-level ground validation above does not require that mapping.")
            else:
                st.info("Ground spectra can be visualized, but external UAV comparison requires a tree_id mapping.")
            with st.expander("Preview computed ground indices"):
                st.dataframe(ground_idx.head(50),hide_index=True,use_container_width=True)

with tab_methods:
    st.header("How the system decides")
    st.markdown("""
    **Water:** WBI is one direct block; NDMI2 + NDSI-RWC form one concordant SWIR block; PRI is corroboration only.  
    **Biochemical:** NDRE + CI red-edge + REP form the chlorophyll/red-edge block; PSRI + SIPI form the pigment/senescence block; ARI1 corroborates.  
    **Structure:** LAS H-P95 is primary; raster CHM P95 corroborates the measurement; H-IQR is context.  
    **Final integration:** only the three domain outcomes are combined. Raw indices are never counted as extra domains.
    """)
    st.subheader("Operational thresholds")
    rows=[
        ["Water","WBI",f"≤ {WATER_SENSITIVITY_THRESHOLDS['P25']['WBI_LOW_MAX']:.6f}","direct water block"],
        ["Water","NDMI2",f"≥ {WATER_SENSITIVITY_THRESHOLDS['P25']['NDMI2_HIGH_MIN']:.6f}","SWIR block"],
        ["Water","NDSI-RWC",f"≤ {WATER_SENSITIVITY_THRESHOLDS['P25']['NDSI_RWC_LOW_MAX']:.6f}","SWIR block"],
        ["Water","PRI",f"≤ {WATER_SENSITIVITY_THRESHOLDS['P25']['PRI_LOW_MAX']:.6f}","corroboration only"],
        ["Biochemical","NDRE",f"≤ {BIO_THRESHOLDS['NDRE_LOW_MAX']:.6f}","chlorophyll/red-edge"],
        ["Biochemical","CI red-edge",f"≤ {BIO_THRESHOLDS['CIRED_LOW_MAX']:.6f}","chlorophyll/red-edge"],
        ["Biochemical","REP",f"≤ {BIO_THRESHOLDS['REP_LOW_MAX_NM']:.3f} nm","red-edge position"],
        ["Biochemical","PSRI",f"≥ {BIO_THRESHOLDS['PSRI_HIGH_MIN']:.6f}","pigment/senescence"],
        ["Biochemical","SIPI",f"≥ {BIO_THRESHOLDS['SIPI_HIGH_MIN']:.6f}","pigment/senescence"],
        ["Biochemical","ARI1",f"≥ {BIO_THRESHOLDS['ARI1_HIGH_MIN']:.6f}","corroboration"],
        ["Structure","LAS H-P95",f"≤ {STRUCTURE_THRESHOLDS['H_P95_P25_M']:.3f} m","primary stature"],
        ["Structure","Raster CHM P95",f"≤ {STRUCTURE_THRESHOLDS['RASTER_CHM_P95_P25_M']:.3f} m","measurement corroboration"],
    ]
    st.dataframe(pd.DataFrame(rows,columns=["Domain","Indicator","Operational condition","Role"]),hide_index=True,use_container_width=True)
    st.info("The app is downstream of the finalized processing workflow. It does not rerun strip QC, index QC, SWIR consensus or LAS processing.")


# =============================================================================
# 10. OPTIONAL LLM — SCENARIO LEVEL, DOWNSTREAM ONLY
# =============================================================================
with st.sidebar.expander("🤖 Field assistant",expanded=False):
    st.caption("Optional natural-language support. It cannot change the deterministic classification.")
    if not GEMINI_AVAILABLE:
        st.info("google-generativeai is not installed; all scientific functions remain available.")
    else:
        api_key=st.text_input("Gemini API key",type="password")
        model_name=st.text_input("Gemini model",value=os.getenv("GEMINI_MODEL","gemini-2.5-flash"))
        if st.button("Explain current scenario"):
            if not api_key: st.warning("Enter an API key first.")
            else:
                prompt=f"Current view: {VIEW_OPTIONS[view]}; highlighted trees: {len(target_gdf)} of {len(gdf)}. Explain what this means to a field user. Do not diagnose disease or claim causality."
                try:
                    genai.configure(api_key=api_key); model=genai.GenerativeModel(model_name); response=model.generate_content(prompt); st.markdown(response.text)
                except Exception as exc: st.error(f"LLM API error: {exc}")
