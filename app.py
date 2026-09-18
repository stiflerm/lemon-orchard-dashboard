"""
Orchard Diagnostic Intelligence v5 — Supported-Evidence Three-Domain DSS
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
  orchard_diagnostic_intelligence_v5_supported_evidence.py
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
NOT ground-truth diagnostic accuracy. The public interface intentionally shows
only supported findings; upstream QC continues to control which measurements are usable.
"""

from __future__ import annotations

import html
import math
import os
import re
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



# =============================================================================
# 4. USER-CENTRED SCENARIOS / MAP HELPERS — SIMPLIFIED FINAL INTERFACE
# =============================================================================

# The public interface intentionally shows ONLY supported findings.
# QC and missing-data handling remain in the backend database but are not exposed as user-facing classes.
VIEW_OPTIONS = {
    "WATER": "💧 Water anomaly",
    "BIOCHEMICAL": "🧪 Biochemical anomaly",
    "STRUCTURE": "🌳 Low canopy stature",
    "WB": "🔗 Water + Biochemical",
    "WBS": "🔴 Water + Biochemical + Low Canopy Stature",
    "GAPS": "🍋 Orchard inventory — planting gaps",
}

VIEW_COLORS = {
    "WATER": "#1E88E5",
    "BIOCHEMICAL": "#8E44AD",
    "STRUCTURE": "#2E8B57",
    "WB": "#F39C12",
    "WBS": "#C0392B",
    "GAPS": "#C0392B",
}


def add_user_labels(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create plain-language display fields without changing backend scientific statuses."""
    out = gdf.copy()

    water_supported = as_bool(out["WATER_SUPPORTED"]) if "WATER_SUPPORTED" in out.columns else pd.Series(False, index=out.index)
    bio_supported = as_bool(out["BIOCHEMICAL_SUPPORTED"]) if "BIOCHEMICAL_SUPPORTED" in out.columns else pd.Series(False, index=out.index)
    structure_low = as_bool(out["STRUCTURE_LOW_STATURE"]) if "STRUCTURE_LOW_STATURE" in out.columns else pd.Series(False, index=out.index)
    structure_cor = as_bool(out["STRUCTURE_CORROBORATED"]) if "STRUCTURE_CORROBORATED" in out.columns else pd.Series(False, index=out.index)

    out["Water finding"] = np.where(water_supported, "Water anomaly", "")
    if "WATER_EVIDENCE_STATUS" in out.columns:
        strong = out["WATER_EVIDENCE_STATUS"].eq("STRONG_INTERNAL_WATER_EVIDENCE") & water_supported
        out.loc[strong, "Water finding"] = "Water anomaly + PRI support"

    out["Biochemical finding"] = np.where(bio_supported, "Biochemical anomaly", "")
    if "BIOCHEMICAL_STATUS" in out.columns:
        strong = out["BIOCHEMICAL_STATUS"].eq("STRONG_CONCORDANT_BIOCHEMICAL_DECLINE") & bio_supported
        out.loc[strong, "Biochemical finding"] = "Biochemical anomaly + ARI1 support"

    out["Structure finding"] = np.where(structure_low, "Low canopy stature", "")
    out["Raster support"] = np.where(structure_low & structure_cor, "Yes", np.where(structure_low, "No", ""))

    out["Combined finding"] = ""
    out.loc[water_supported & bio_supported, "Combined finding"] = "Water + Biochemical"
    out.loc[water_supported & bio_supported & structure_low, "Combined finding"] = "Water + Biochemical + Low Canopy Stature"
    return out


WATER_TOOLTIP = [
    ("tree_id", "Tree ID"),
    ("Water finding", "Result"),
    ("WBI_VALUE", "WBI"),
    ("NDMI2_VALUE", "NDMI2"),
    ("NDSI_RWC_VALUE", "NDSI-RWC"),
    ("PRI_VALUE", "PRI"),
]

BIO_TOOLTIP = [
    ("tree_id", "Tree ID"),
    ("Biochemical finding", "Result"),
    ("NDRE_VALUE", "NDRE"),
    ("CIRED_EDGE_VALUE", "CI red-edge"),
    ("REP_D1_NM_VALUE", "REP (nm)"),
    ("PSRI_VALUE", "PSRI"),
    ("SIPI_VALUE", "SIPI"),
    ("ARI1_VALUE", "ARI1"),
    ("NDVI_VALUE", "NDVI (context)"),
]

STRUCT_TOOLTIP = [
    ("tree_id", "Tree ID"),
    ("Structure finding", "Result"),
    ("H_P95_m", "LAS H-P95 (m)"),
    ("RASTER_CHM_P95_m", "Raster CHM P95 (m)"),
    ("Raster support", "Raster support"),
    ("H_IQR_m", "H-IQR (context)"),
]

COMBINED_TOOLTIP = [
    ("tree_id", "Tree ID"),
    ("Combined finding", "Combined result"),
    ("Water finding", "Water"),
    ("Biochemical finding", "Biochemical"),
    ("Structure finding", "Structure"),
    ("WBI_VALUE", "WBI"),
    ("NDMI2_VALUE", "NDMI2"),
    ("NDSI_RWC_VALUE", "NDSI-RWC"),
    ("NDRE_VALUE", "NDRE"),
    ("CIRED_EDGE_VALUE", "CI red-edge"),
    ("REP_D1_NM_VALUE", "REP (nm)"),
    ("PSRI_VALUE", "PSRI"),
    ("SIPI_VALUE", "SIPI"),
    ("ARI1_VALUE", "ARI1"),
    ("H_P95_m", "LAS H-P95 (m)"),
    ("Raster support", "Raster support"),
]


def human_pattern(value):
    mapping = {
        "WATER_BIOCHEMICAL_STRUCTURE": "Water + Biochemical + Low Canopy Stature",
        "WATER_BIOCHEMICAL_ONLY": "Water + Biochemical",
        "WATER_STRUCTURE_ONLY": "Water + Low Canopy Stature",
        "BIOCHEMICAL_STRUCTURE_ONLY": "Biochemical + Low Canopy Stature",
        "WATER_ONLY": "Water only",
        "BIOCHEMICAL_ONLY": "Biochemical only",
        "STRUCTURE_ONLY": "Low Canopy Stature only",
        "NO_SUPPORTED_MAJOR_DOMAIN_ANOMALY": "No supported combination",
    }
    return mapping.get(str(value), str(value).replace("_", " ").title())


def make_target_mask(gdf, view):
    W = as_bool(gdf["WATER_SUPPORTED"])
    B = as_bool(gdf["BIOCHEMICAL_SUPPORTED"])
    S = as_bool(gdf["STRUCTURE_LOW_STATURE"])

    if view == "WATER":
        return W
    if view == "BIOCHEMICAL":
        return B
    if view == "STRUCTURE":
        return S
    if view == "WB":
        # Exactly the two supported spectral domains, with structure valid and not low.
        structure_no_low = gdf.get("STRUCTURE_STATUS_FINAL", pd.Series("", index=gdf.index)).eq("NO_LOW_STATURE_EVIDENCE")
        return W & B & structure_no_low
    if view == "WBS":
        return W & B & S
    return pd.Series(False, index=gdf.index)


def comparison_reference_mask(gdf, view):
    """Scenario-specific valid comparison group. Never labelled 'healthy'."""
    water_no = gdf.get("WATER_EVIDENCE_STATUS", pd.Series("", index=gdf.index)).eq("NO_CONCORDANT_WATER_ANOMALY")
    bio_no = gdf.get("BIOCHEMICAL_STATUS", pd.Series("", index=gdf.index)).eq("NO_CONCORDANT_BIOCHEMICAL_DECLINE")
    struct_no = gdf.get("STRUCTURE_STATUS_FINAL", pd.Series("", index=gdf.index)).eq("NO_LOW_STATURE_EVIDENCE")
    if view == "WATER":
        return water_no
    if view == "BIOCHEMICAL":
        return bio_no
    if view == "STRUCTURE":
        return struct_no
    if view == "WB":
        return water_no & bio_no & struct_no
    if view == "WBS":
        return water_no & bio_no & struct_no
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
        return ["WBI_VALUE", "NDMI2_VALUE", "NDSI_RWC_VALUE", "PRI_VALUE"]
    if view == "BIOCHEMICAL":
        return ["NDRE_VALUE", "CIRED_EDGE_VALUE", "REP_D1_NM_VALUE", "PSRI_VALUE", "SIPI_VALUE", "ARI1_VALUE"]
    if view == "STRUCTURE":
        return ["H_P95_m", "RASTER_CHM_P95_m", "H_IQR_m"]
    return [
        "WBI_VALUE", "NDMI2_VALUE", "NDSI_RWC_VALUE", "PRI_VALUE",
        "NDRE_VALUE", "CIRED_EDGE_VALUE", "REP_D1_NM_VALUE", "PSRI_VALUE", "SIPI_VALUE", "ARI1_VALUE",
        "H_P95_m", "RASTER_CHM_P95_m", "H_IQR_m",
    ]


def add_target_layer(m, target_gdf, view):
    if target_gdf.empty:
        return
    fields_aliases = [(f, a) for f, a in tooltip_spec(view) if f in target_gdf.columns]
    fields = [x[0] for x in fields_aliases]
    aliases = [x[1] + ":" for x in fields_aliases]
    tooltip = folium.GeoJsonTooltip(fields=fields, aliases=aliases, localize=True, sticky=True)
    popup = folium.GeoJsonPopup(fields=fields, aliases=aliases, localize=True, labels=True)
    color = VIEW_COLORS[view]

    folium.GeoJson(
        target_gdf,
        style_function=lambda _: {"fillColor": color, "color": "#FFFFFF", "weight": 2.2, "fillOpacity": 0.78},
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
    details = list(dict.fromkeys([x[0] for x in COMBINED_TOOLTIP + WATER_TOOLTIP + BIO_TOOLTIP + STRUCT_TOOLTIP]))
    for _, row in points.iterrows():
        rp = row.geometry
        desc = [f"<b>Tree ID:</b> {html.escape(str(row.get('tree_id', 'NA')))}"]
        for field in details:
            if field == "tree_id":
                continue
            if field in row.index and pd.notna(row.get(field)) and str(row.get(field)) != "":
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




# =============================================================================
# 7. USER-FRIENDLY VISUAL / VALIDATION HELPERS
# =============================================================================

def status_card(title, value, subtitle, icon=""):
    st.markdown(
        f"""<div style='padding:1rem;border:1px solid rgba(128,128,128,.35);border-radius:14px;min-height:145px;'>
        <div style='font-size:1.05rem;font-weight:700'>{icon} {html.escape(str(title))}</div>
        <div style='font-size:1.9rem;font-weight:800;margin:.35rem 0'>{html.escape(str(value))}</div>
        <div style='opacity:.78;font-size:.9rem'>{html.escape(str(subtitle))}</div>
        </div>""", unsafe_allow_html=True
    )


def count_bar(labels, counts, title):
    d = pd.DataFrame({"Evidence step": labels, "Trees": counts})
    fig = px.bar(d, x="Trees", y="Evidence step", orientation="h", text="Trees", title=title)
    fig.update_traces(textposition="outside")
    fig.update_layout(height=max(300, 70 * len(d) + 100), margin=dict(l=10, r=20, t=55, b=10), yaxis={'categoryorder':'array','categoryarray':labels[::-1]})
    return fig


def numeric_suffix(value):
    m = re.search(r"(-?\d+(?:\.\d+)?)$", str(value))
    return float(m.group(1)) if m else float("inf")


def spectral_axis_from_mapping(band_cols):
    """Return x-axis values and whether they are true wavelengths."""
    wl_map, source = load_wavelength_map()
    if wl_map:
        values = []
        ok = True
        for c in band_cols:
            if c not in wl_map:
                ok = False
                break
            try:
                values.append(float(wl_map[c]))
            except Exception:
                ok = False
                break
        if ok and len(values) == len(band_cols):
            return np.asarray(values, float), "Wavelength (nm)", True, source

    # Safe fallback only when the column suffix itself is clearly a wavelength.
    suffixes = np.asarray([numeric_suffix(c) for c in band_cols], float)
    if len(suffixes) and np.isfinite(suffixes).all() and suffixes.min() >= 350 and suffixes.max() <= 2500:
        return suffixes, "Wavelength (nm)", True, "Band-column wavelength labels"

    return np.arange(1, len(band_cols) + 1), "Band number", False, source


def add_rule_regions(fig, view, wavelength_axis=True):
    """Shade wavelength regions used by the selected rule when true wavelength mapping exists."""
    if not wavelength_axis:
        return fig

    if view in {"WATER", "WB", "WBS"}:
        for x0, x1, label in [
            (526, 536, "PRI 531"),
            (565, 575, "PRI 570"),
            (895, 905, "WBI 900"),
            (965, 975, "WBI 970"),
        ]:
            fig.add_vrect(x0=x0, x1=x1, opacity=0.10, line_width=0, annotation_text=label, annotation_position="top left")

    if view in {"BIOCHEMICAL", "WB", "WBS"}:
        fig.add_vrect(x0=680, x1=750, opacity=0.12, line_width=0,
                      annotation_text="Red-edge 680–750 nm: NDRE / CIred / REP / PSRI",
                      annotation_position="top left")
        for x0, x1, label in [
            (440, 450, "SIPI 445"),
            (495, 505, "PSRI 500"),
            (545, 555, "ARI1 550"),
            (795, 805, "SIPI 800"),
        ]:
            fig.add_vrect(x0=x0, x1=x1, opacity=0.08, line_width=0, annotation_text=label, annotation_position="bottom left")
    return fig


def sensitivity_target_mask(df, view, scheme, combined_mode=None):
    """Override old helper with the simplified final public scenarios."""
    W, B, S = sensitivity_domain_masks(df, scheme)
    if view == "WATER":
        return W
    if view == "BIOCHEMICAL":
        return B
    if view == "STRUCTURE":
        return S
    if view == "WB":
        # Exact two-domain sensitivity: Water + Biochemical with a valid structural value above the low-stature threshold.
        h95 = pd.to_numeric(df.get("H_P95_m"), errors="coerce")
        return W & B & h95.notna() & ~S
    if view == "WBS":
        return W & B & S
    return pd.Series(False, index=df.index)


def threshold_table_for_view(view):
    rows = []
    for scheme in ["P20", "P25", "P30"]:
        if view in {"WATER", "WB", "WBS"}:
            w = WATER_SENSITIVITY_THRESHOLDS[scheme]
            rows += [
                [scheme, "Water", "WBI", f"≤ {w['WBI_LOW_MAX']:.6f}"],
                [scheme, "Water", "NDMI2", f"≥ {w['NDMI2_HIGH_MIN']:.6f}"],
                [scheme, "Water", "NDSI-RWC", f"≤ {w['NDSI_RWC_LOW_MAX']:.6f}"],
            ]
        if view in {"BIOCHEMICAL", "WB", "WBS"}:
            b = BIO_SENSITIVITY_THRESHOLDS[scheme]
            rows += [
                [scheme, "Biochemical", "NDRE", f"≤ {b['NDRE']:.6f}"],
                [scheme, "Biochemical", "CI red-edge", f"≤ {b['CIRED']:.6f}"],
                [scheme, "Biochemical", "REP", f"≤ {b['REP']:.3f} nm"],
                [scheme, "Biochemical", "PSRI", f"≥ {b['PSRI']:.6f}"],
                [scheme, "Biochemical", "SIPI", f"≥ {b['SIPI']:.6f}"],
            ]
        if view in {"STRUCTURE", "WB", "WBS"}:
            s_thr = {"P20": STRUCTURE_THRESHOLDS["H_P95_P20_M"], "P25": STRUCTURE_THRESHOLDS["H_P95_P25_M"], "P30": STRUCTURE_THRESHOLDS["H_P95_P30_M"]}[scheme]
            rows.append([scheme, "Structure", "LAS H-P95", f"≤ {s_thr:.3f} m"])
    return pd.DataFrame(rows, columns=["Sensitivity scheme", "Domain", "Indicator", "Condition"])


def relevant_cross_domain_counts(gdf):
    patterns = [
        "WATER_BIOCHEMICAL_ONLY",
        "WATER_STRUCTURE_ONLY",
        "BIOCHEMICAL_STRUCTURE_ONLY",
        "WATER_BIOCHEMICAL_STRUCTURE",
    ]
    labels = {
        "WATER_BIOCHEMICAL_ONLY": "Water + Biochemical",
        "WATER_STRUCTURE_ONLY": "Water + Low Canopy Stature",
        "BIOCHEMICAL_STRUCTURE_ONLY": "Biochemical + Low Canopy Stature",
        "WATER_BIOCHEMICAL_STRUCTURE": "Water + Biochemical + Low Canopy Stature",
    }
    vc = gdf["CROSS_DOMAIN_PATTERN"].value_counts() if "CROSS_DOMAIN_PATTERN" in gdf.columns else pd.Series(dtype=int)
    return pd.DataFrame({"Combination": [labels[p] for p in patterns], "Trees": [int(vc.get(p, 0)) for p in patterns]})



# =============================================================================
# 8. APP LOAD / CONTROLS
# =============================================================================

st.set_page_config(page_title="Orchard Three-Domain DSS", layout="wide")
st.title("🍋 Orchard Intelligence — Water • Biochemistry • Structure")
st.caption(
    "Choose a supported finding, see the corresponding trees on the map, and hover/click a crown to see the measurements used. "
    "Only supported rule outcomes are shown in the public interface; QC remains active in the backend."
)

geometry_gdf = load_tree_geometry()
rule_df, rule_source = load_final_rule_database()
spectral_df = load_spectral_data()
try:
    gdf = merge_geometry_and_rules(geometry_gdf, rule_df)
    gdf = add_user_labels(gdf)
except Exception as exc:
    st.error(f"Could not align canopy geometry and final database: {exc}")
    st.stop()

st.sidebar.header("What do you want to inspect?")
view = st.sidebar.radio(
    "Choose a scenario",
    options=list(VIEW_OPTIONS),
    format_func=lambda k: VIEW_OPTIONS[k],
    label_visibility="collapsed",
)
st.sidebar.caption("The map highlights only trees that meet the selected supported rule.")

if view == "GAPS":
    with st.sidebar.expander("Advanced planting-gap settings", expanded=False):
        st.caption("These settings affect only the orchard-inventory gap calculation, not Water, Biochemical, or Structure results.")
        expected_tree_spacing = st.number_input("Expected tree spacing (m)", 0.5, 20.0, DEFAULT_TREE_SPACING_M, 0.1,
                                                help="Typical distance between neighboring trees along a row.")
        row_distance_threshold = st.number_input("Row grouping tolerance (m)", 0.5, 10.0, DEFAULT_ROW_DISTANCE_THRESHOLD_M, 0.1,
                                                 help="How close tree centers must be across the row direction to be grouped into the same planting row.")
        grid_angle_degrees = st.number_input("Orchard-row rotation angle (deg)", -180.0, 180.0, DEFAULT_GRID_ANGLE_DEG, 1.0,
                                             help="Rotates the orchard coordinate system so planting rows can be analysed as approximately horizontal lines.")
        max_empty_space_m = st.number_input("Maximum gap considered (m)", 2.0, 100.0, DEFAULT_MAX_EMPTY_SPACE_M, 1.0,
                                            help="Larger open spaces are ignored so orchard boundaries are not mistaken for missing trees.")
    gaps_gdf, total_trees, total_gaps, yield_loss_percentage = calculate_gaps(
        gdf, expected_tree_spacing, row_distance_threshold, grid_angle_degrees, max_empty_space_m
    )
    target_mask = pd.Series(False, index=gdf.index)
    target_gdf = gdf.iloc[0:0].copy()
else:
    gaps_gdf = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
    target_mask = make_target_mask(gdf, view)
    target_gdf = gdf[target_mask].copy()

st.sidebar.markdown("---")
st.sidebar.caption(rule_source)
st.sidebar.metric("Highlighted trees", len(target_gdf) if view != "GAPS" else len(gaps_gdf))
if view != "GAPS":
    st.sidebar.caption(f"{100 * len(target_gdf) / len(gdf):.1f}% of {len(gdf)} orchard trees")


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
    if view == "WATER":
        st.header("💧 Water anomaly")
        st.write("Highlighted trees satisfy the finalized Water rule: WBI and the SWIR water block agree. PRI is shown only as additional physiological support.")
    elif view == "BIOCHEMICAL":
        st.header("🧪 Biochemical anomaly")
        st.write("Highlighted trees show agreement between the chlorophyll/red-edge block and the pigment/senescence block. ARI1 can provide additional support.")
    elif view == "STRUCTURE":
        st.header("🌳 Low canopy stature")
        st.write("Highlighted trees have orchard-relative low LAS H-P95. Raster CHM support is shown in the tree popup as an independent measurement check.")
    elif view == "WB":
        st.header("🔗 Water + Biochemical")
        st.write("Highlighted trees have supported Water and Biochemical anomalies, while the available structural result does not meet the low-stature rule.")
    elif view == "WBS":
        st.header("🔴 Water + Biochemical + Low Canopy Stature")
        st.write("Highlighted trees show supported evidence in all three domains. This is the strongest cross-domain field-inspection group, but it does not establish causality.")
    else:
        st.header("🍋 Orchard inventory — planting gaps")
        st.write("This is a separate orchard-layout tool. It estimates possible missing planting positions from tree spacing and row geometry; it is not a health classification.")

    if view != "GAPS":
        c1, c2, c3 = st.columns(3)
        c1.metric("Highlighted trees", len(target_gdf))
        c2.metric("Share of orchard", f"{100 * len(target_gdf) / len(gdf):.1f}%")
        c3.metric("Total mapped trees", len(gdf))

    center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=18, max_zoom=22, tiles="CartoDB positron")
    folium.GeoJson(
        gdf,
        style_function=lambda _: {"fillColor": "#D0D0D0", "color": "#777777", "weight": 0.7, "fillOpacity": 0.05},
        name="All tree crowns",
    ).add_to(m)

    if view == "GAPS":
        for _, row in gaps_gdf.iterrows():
            folium.CircleMarker(
                [row.geometry.y, row.geometry.x], radius=5, color="#C0392B", fill=True, fill_opacity=.9,
                tooltip="Calculated planting gap"
            ).add_to(m)
    else:
        add_target_layer(m, target_gdf, view)

    folium.LayerControl(collapsed=True).add_to(m)
    st_folium(m, height=650, use_container_width=True)
    st.caption("Hover = quick evidence. Click = persistent popup. No Tree-ID selection is required.")

    if view != "GAPS" and not target_gdf.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download highlighted trees (GeoJSON)", target_gdf.to_json(),
                file_name=f"{view.lower()}_targets.geojson", mime="application/geo+json"
            )
        with col2:
            kml = build_tree_navigation_kml(target_gdf, f"{VIEW_OPTIONS[view]} targets", VIEW_COLORS[view])
            st.download_button(
                "📍 Download highlighted trees (KML)", kml,
                file_name=f"{view.lower()}_targets.kml", mime="application/vnd.google-earth.kml+xml"
            )
    elif view == "GAPS" and len(gaps_gdf):
        c1, c2 = st.columns(2)
        c1.metric("Calculated planting gaps", total_gaps)
        c2.metric("Estimated planting-capacity gap", f"{yield_loss_percentage:.1f}%")


with tab_evidence:
    st.header("Why these trees are highlighted")

    if view == "WATER":
        wbi = pd.to_numeric(gdf.get("WBI_VALUE"), errors="coerce") <= WATER_SENSITIVITY_THRESHOLDS["P25"]["WBI_LOW_MAX"]
        ndmi = pd.to_numeric(gdf.get("NDMI2_VALUE"), errors="coerce") >= WATER_SENSITIVITY_THRESHOLDS["P25"]["NDMI2_HIGH_MIN"]
        ndsi = pd.to_numeric(gdf.get("NDSI_RWC_VALUE"), errors="coerce") <= WATER_SENSITIVITY_THRESHOLDS["P25"]["NDSI_RWC_LOW_MAX"]
        swir = ndmi & ndsi
        supported = as_bool(gdf["WATER_SUPPORTED"])
        pri = pd.to_numeric(gdf.get("PRI_VALUE"), errors="coerce") <= WATER_SENSITIVITY_THRESHOLDS["P25"]["PRI_LOW_MAX"]
        pri_supported = supported & pri

        c1, c2, c3 = st.columns(3)
        with c1: status_card("Direct water evidence", "WBI", "One direct spectral water block", "💧")
        with c2: status_card("SWIR water evidence", "NDMI2 + NDSI-RWC", "These two correlated indices form ONE SWIR block", "🌊")
        with c3: status_card("Additional support", "PRI", "Can strengthen the result but cannot create it alone", "⚡")
        st.markdown("### Decision rule")
        st.markdown("**WBI abnormal**  +  **SWIR block abnormal**  →  **Water anomaly**  →  *PRI may add physiological support*")
        fig = count_bar(
            ["WBI abnormal", "SWIR block abnormal", "Water anomaly: WBI + SWIR agree", "Water anomaly + PRI support"],
            [int(wbi.fillna(False).sum()), int(swir.fillna(False).sum()), int(supported.sum()), int(pri_supported.fillna(False).sum())],
            "How the Water evidence builds"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif view == "BIOCHEMICAL":
        chl = as_bool(gdf.get("BIO_CHL_RE_SUPPORTED", pd.Series(False, index=gdf.index)))
        pigment = as_bool(gdf.get("BIO_PIGMENT_SUPPORTED", pd.Series(False, index=gdf.index)))
        supported = as_bool(gdf["BIOCHEMICAL_SUPPORTED"])
        ari = as_bool(gdf.get("BIO_ARI1_HIGH", pd.Series(False, index=gdf.index)))
        strong = supported & ari

        c1, c2, c3 = st.columns(3)
        with c1: status_card("Chlorophyll / red-edge", "NDRE + CIred + REP", "All three support the same low red-edge/chlorophyll response", "🍃")
        with c2: status_card("Pigment / senescence", "PSRI + SIPI", "Both support the pigment/senescence response", "🍂")
        with c3: status_card("Additional support", "ARI1", "Shown as additional support after the two main blocks agree", "🔬")
        st.markdown("### Decision rule")
        st.markdown("**Chlorophyll/red-edge block abnormal**  +  **Pigment/senescence block abnormal**  →  **Biochemical anomaly**  →  *ARI1 may add support*")
        fig = count_bar(
            ["Chlorophyll / red-edge block", "Pigment / senescence block", "Biochemical anomaly: both blocks agree", "Biochemical anomaly + ARI1 support"],
            [int(chl.sum()), int(pigment.sum()), int(supported.sum()), int(strong.sum())],
            "How the Biochemical evidence builds"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif view == "STRUCTURE":
        low = as_bool(gdf["STRUCTURE_LOW_STATURE"])
        cor = as_bool(gdf["STRUCTURE_CORROBORATED"])
        supported_raster = low & cor
        las_only = low & ~cor
        c1, c2, c3 = st.columns(3)
        with c1: status_card("Primary structural measure", "LAS H-P95", f"Low canopy stature ≤ {STRUCTURE_THRESHOLDS['H_P95_P25_M']:.3f} m", "🌳")
        with c2: status_card("Independent raster check", "Raster CHM P95", "Used as measurement support, not a second biological vote", "🛰️")
        with c3: status_card("Canopy variability", "H-IQR", "Context only; it does not create the low-stature class", "📐")
        st.markdown("### Decision rule")
        st.markdown(f"**LAS H-P95 ≤ {STRUCTURE_THRESHOLDS['H_P95_P25_M']:.3f} m**  →  **Low canopy stature**. Raster CHM can independently support that measurement.")
        fig = count_bar(
            ["Low canopy stature (LAS H-P95)", "Low stature + raster support", "Low stature from LAS only"],
            [int(low.sum()), int(supported_raster.sum()), int(las_only.sum())],
            "Structural evidence"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Low canopy stature is an orchard-relative structural finding; it is not itself a disease diagnosis.")

    elif view == "WB":
        W = as_bool(gdf["WATER_SUPPORTED"]); B = as_bool(gdf["BIOCHEMICAL_SUPPORTED"]); S = as_bool(gdf["STRUCTURE_LOW_STATURE"])
        st.markdown("### Two-domain combination")
        st.markdown("**Water anomaly** + **Biochemical anomaly** + *no low-stature result* → **Water + Biochemical**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Water anomaly", int(W.sum()))
        c2.metric("Biochemical anomaly", int(B.sum()))
        c3.metric("Water + Biochemical highlighted", int(target_mask.sum()))
        st.info("This view is intentionally separate from the three-domain view below.")

    elif view == "WBS":
        W = as_bool(gdf["WATER_SUPPORTED"]); B = as_bool(gdf["BIOCHEMICAL_SUPPORTED"]); S = as_bool(gdf["STRUCTURE_LOW_STATURE"])
        st.markdown("### Three-domain combination")
        st.markdown("**Water anomaly** + **Biochemical anomaly** + **Low canopy stature** → **Three-domain cross-domain anomaly**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Water", int(W.sum()))
        c2.metric("Biochemical", int(B.sum()))
        c3.metric("Low stature", int(S.sum()))
        c4.metric("All three together", int(target_mask.sum()))
        st.caption("Agreement across three domains strengthens field-inspection priority but does not prove that one domain caused another.")

    else:
        st.info("Planting-gap analysis is an orchard inventory function and is intentionally separate from the three-domain health-screening framework.")


with tab_summary:
    st.header("Orchard at a glance")
    W = as_bool(gdf["WATER_SUPPORTED"])
    B = as_bool(gdf["BIOCHEMICAL_SUPPORTED"])
    S = as_bool(gdf["STRUCTURE_LOW_STATURE"])
    exact_wb = make_target_mask(gdf, "WB")
    triple = make_target_mask(gdf, "WBS")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💧 Water anomaly", int(W.sum()))
    c2.metric("🧪 Biochemical anomaly", int(B.sum()))
    c3.metric("🌳 Low canopy stature", int(S.sum()))
    c4.metric("🔗 Water + Biochemical", int(exact_wb.sum()))
    c5.metric("🔴 All three", int(triple.sum()))

    left, right = st.columns(2)
    with left:
        domain_df = pd.DataFrame({"Supported finding": ["Water anomaly", "Biochemical anomaly", "Low canopy stature"],
                                  "Trees": [int(W.sum()), int(B.sum()), int(S.sum())]})
        fig = px.bar(domain_df, x="Supported finding", y="Trees", text="Trees", title="Supported findings by domain")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        cross = relevant_cross_domain_counts(gdf)
        fig = px.bar(cross, x="Trees", y="Combination", orientation="h", text="Trees",
                     title="Cross-domain combinations (2 or more supported domains)")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Only supported cross-domain combinations are shown here. Single-domain and unsupported/background trees are not included in this chart.")

    with st.expander("Download supported orchard results"):
        cols = [
            "tree_id", "Water finding", "Biochemical finding", "Structure finding", "Raster support",
            "CROSS_DOMAIN_PATTERN", "WBI_VALUE", "NDMI2_VALUE", "NDSI_RWC_VALUE", "PRI_VALUE",
            "NDRE_VALUE", "CIRED_EDGE_VALUE", "REP_D1_NM_VALUE", "PSRI_VALUE", "SIPI_VALUE", "ARI1_VALUE",
            "H_P95_m", "RASTER_CHM_P95_m"
        ]
        out = gdf[[c for c in cols if c in gdf.columns]].copy()
        supported_any = W | B | S
        out = out.loc[supported_any].copy()
        st.download_button("Download supported results CSV", out.to_csv(index=False).encode(), "orchard_supported_results.csv", "text/csv")


with tab_validation:
    st.header("Validation & robustness — research view")
    st.caption(
        "This page tests whether the finalized supported rules are spectrally interpretable, stable to reasonable threshold changes, "
        "and numerically distinct from valid comparison trees. These are robustness tests, not disease-diagnostic accuracy."
    )

    if view == "GAPS":
        st.info("Select Water, Biochemical, Low Canopy Stature, Water + Biochemical, or the three-domain scenario to run research validation.")
    else:
        current_mask = target_mask.astype(bool)
        reference_mask = comparison_reference_mask(gdf, view).astype(bool)

        # ---------------------------------------------------------------------
        # 1. Spectral / structural signature comparison
        # ---------------------------------------------------------------------
        st.subheader("1. Signature comparison: highlighted trees vs valid comparison trees")
        c1, c2 = st.columns(2)
        c1.metric("Highlighted trees", int(current_mask.sum()))
        c2.metric("Comparison trees", int(reference_mask.sum()))
        st.caption("Comparison trees have valid measurements for the selected scenario but do not meet that anomaly rule. They are not labelled 'healthy'.")

        if view == "STRUCTURE":
            metric = "H_P95_m"
            target_h = pd.to_numeric(gdf.loc[current_mask, metric], errors="coerce").dropna()
            reference_h = pd.to_numeric(gdf.loc[reference_mask, metric], errors="coerce").dropna()
            comp = pd.concat([
                pd.DataFrame({"LAS H-P95 (m)": target_h.values, "Group": "Low canopy stature"}),
                pd.DataFrame({"LAS H-P95 (m)": reference_h.values, "Group": "Comparison"}),
            ], ignore_index=True)
            if not comp.empty:
                fig = px.box(comp, x="Group", y="LAS H-P95 (m)", points="all", title="Structural height comparison")
                fig.add_hline(y=STRUCTURE_THRESHOLDS["H_P95_P25_M"], line_dash="dash", annotation_text="Operational P25-derived threshold")
                st.plotly_chart(fig, use_container_width=True)
        else:
            band_cols = sorted([c for c in spectral_df.columns if str(c).startswith("Band_")], key=numeric_suffix) if not spectral_df.empty else []
            if not spectral_df.empty and band_cols and current_mask.any() and reference_mask.any():
                key = "tree_id" if "tree_id" in spectral_df.columns else ("generated_id" if "generated_id" in spectral_df.columns and "generated_id" in gdf.columns else None)
                if key:
                    target_ids = gdf.loc[current_mask, key].tolist()
                    ref_ids = gdf.loc[reference_mask, key].tolist()
                    t = spectral_df[spectral_df[key].isin(target_ids)]
                    r = spectral_df[spectral_df[key].isin(ref_ids)]
                    if len(t) and len(r):
                        tm = t[band_cols].apply(pd.to_numeric, errors="coerce").mean().to_numpy(float)
                        rm = r[band_cols].apply(pd.to_numeric, errors="coerce").mean().to_numpy(float)
                        x, x_title, true_wavelength, wl_source = spectral_axis_from_mapping(band_cols)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=x, y=tm, mode="lines", name="Highlighted trees — mean"))
                        fig.add_trace(go.Scatter(x=x, y=rm, mode="lines", name="Comparison trees — mean"))
                        fig = add_rule_regions(fig, view, true_wavelength)
                        fig.update_layout(title="Consensus canopy spectral signatures", xaxis_title=x_title, yaxis_title="Reflectance", height=460)
                        st.plotly_chart(fig, use_container_width=True)
                        if true_wavelength:
                            st.caption(f"Rule-sensitive wavelength regions are highlighted. Wavelength source: {wl_source or 'resolved mapping'}.")
                        else:
                            st.info("The spectral curves are available, but exact wavelength highlighting requires a band-to-wavelength mapping file. No wavelength positions are guessed.")

                        sam = spectral_angle_deg(tm, rm)
                        rms = spectral_rmse(tm, rm)
                        m1, m2 = st.columns(2)
                        m1.metric("Spectral angle (SAM)", f"{sam:.3f}°" if np.isfinite(sam) else "NA")
                        m2.metric("Spectral RMSE", f"{rms:.5f}" if np.isfinite(rms) else "NA")
                        st.caption("SAM summarizes spectral-shape difference; RMSE summarizes overall reflectance separation. Neither is an accuracy percentage.")

                        if view in {"WATER", "WB", "WBS"}:
                            st.markdown("**Water-specific note:** the VNIR curve highlights PRI (531/570 nm) and WBI (900/970 nm). NDMI2 and NDSI-RWC use SWIR wavelengths outside this VNIR consensus curve and are therefore assessed numerically below rather than falsely drawn here.")
                    else:
                        st.info("No post-QC consensus spectra overlap both the highlighted and comparison groups.")
                else:
                    st.info("The consensus spectrum table has no tree identifier that can be safely matched to the final database.")
            else:
                st.info("Post-QC consensus spectra are unavailable, or one comparison group is empty.")

        # ---------------------------------------------------------------------
        # 2. Threshold sensitivity / Jaccard
        # ---------------------------------------------------------------------
        st.subheader("2. Threshold robustness")
        st.write("The operational classification uses **frozen P25-derived orchard-relative thresholds**. They are not universal citrus hard thresholds. Robustness is tested by repeating the same logic with P20 and P30 alternatives derived from the same QC-approved orchard population.")
        schemes = ["P20", "P25", "P30"]
        runs = {s: sensitivity_target_mask(gdf, view, s) for s in schemes}
        sens = pd.DataFrame({"Threshold scheme": ["P20 (stricter)", "P25 (operational)", "P30 (more inclusive)"],
                             "Highlighted trees": [int(runs[s].sum()) for s in schemes]})
        st.plotly_chart(px.bar(sens, x="Threshold scheme", y="Highlighted trees", text="Highlighted trees",
                               title="How many trees are retained when the percentile-derived threshold moves"), use_container_width=True)

        jac = pd.DataFrame(index=schemes, columns=schemes, dtype=float)
        for i in schemes:
            for j in schemes:
                jac.loc[i, j] = jaccard(runs[i], runs[j])
        j1, j2 = st.columns(2)
        j1.metric("P20 ↔ P25 Jaccard", f"{jac.loc['P20','P25']:.3f}" if np.isfinite(jac.loc['P20','P25']) else "NA")
        j2.metric("P25 ↔ P30 Jaccard", f"{jac.loc['P25','P30']:.3f}" if np.isfinite(jac.loc['P25','P30']) else "NA")
        st.caption("Jaccard = overlap of the selected tree sets: 1.0 means identical sets; lower values mean stronger sensitivity to threshold choice.")
        with st.expander("Show full Jaccard matrix and the P20/P25/P30 threshold values"):
            st.dataframe(jac.round(3), use_container_width=True)
            st.dataframe(threshold_table_for_view(view), hide_index=True, use_container_width=True)

        # ---------------------------------------------------------------------
        # 3. Indicator separation
        # ---------------------------------------------------------------------
        st.subheader("3. Indicator separation")
        metrics = [m for m in scenario_metrics(view) if m in gdf.columns]
        if metrics and current_mask.any() and reference_mask.any():
            choice = st.selectbox("Choose an indicator to compare", metrics, key=f"indicator_compare_{view}")
            target_vals = pd.to_numeric(gdf.loc[current_mask, choice], errors="coerce").dropna()
            ref_vals = pd.to_numeric(gdf.loc[reference_mask, choice], errors="coerce").dropna()
            plot_df = pd.concat([
                pd.DataFrame({"Value": target_vals.values, "Group": "Highlighted"}),
                pd.DataFrame({"Value": ref_vals.values, "Group": "Comparison"}),
            ], ignore_index=True)
            if not plot_df.empty:
                fig = px.box(plot_df, x="Group", y="Value", points="all", title=f"{choice}: highlighted vs comparison trees")
                st.plotly_chart(fig, use_container_width=True)
            stats_df = comparison_statistics(gdf, current_mask, reference_mask, metrics)
            with st.expander("Research statistics for all indicators"):
                if not stats_df.empty:
                    st.dataframe(stats_df.round(5), hide_index=True, use_container_width=True)
                    st.caption("Mann–Whitney p tests distributional difference; rank-biserial gives effect direction/size. These do not establish causality.")
                else:
                    st.info("Not enough observations for group statistics.")
        else:
            st.info("Not enough observations for indicator comparison.")

        # ---------------------------------------------------------------------
        # 4. Structural cross-check
        # ---------------------------------------------------------------------
        st.subheader("4. Structural cross-check")
        low = as_bool(gdf["STRUCTURE_LOW_STATURE"])
        cor = as_bool(gdf["STRUCTURE_CORROBORATED"])
        if view == "STRUCTURE":
            c1, c2 = st.columns(2)
            c1.metric("Low canopy stature trees", int(current_mask.sum()))
            c2.metric("With raster support", int((current_mask & cor).sum()))
            st.caption("Raster CHM P95 is an independent measurement check; it is not counted as a separate biological domain.")
        elif view == "WB":
            st.info("This scenario is deliberately defined as Water + Biochemical without low canopy stature, so structural overlap is not part of this selected group.")
        else:
            c1, c2 = st.columns(2)
            c1.metric("Highlighted trees also low stature", int((current_mask & low).sum()))
            c2.metric("...with raster support", int((current_mask & low & cor).sum()))
            st.caption("Structural agreement strengthens cross-domain corroboration but does not identify the causal stressor.")

        # ---------------------------------------------------------------------
        # 5. Advanced research analysis
        # ---------------------------------------------------------------------
        with st.expander("5. Advanced research analysis — PCA and spatial coherence", expanded=False):
            st.markdown("**PCA feature-space comparison**")
            features = scenario_metrics("WBS")
            pca_df, explained, used = pca_feature_space(gdf, features, current_mask, reference_mask)
            if not pca_df.empty:
                fig = px.scatter(pca_df, x="PC1", y="PC2", color="Group", hover_data=["tree_id"],
                                 title=f"PCA feature space — PC1 {explained[0]:.1f}%, PC2 {explained[1]:.1f}%")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Exploratory only: separation indicates multivariate feature-space distinction, not biological class identity.")
                with st.expander("Features used in PCA"):
                    st.write(used)
            else:
                st.info("Insufficient numeric features for PCA.")

            st.markdown("**Spatial coherence (exploratory kNN Moran's I)**")
            I, p, prev, lift = morans_i_knn(gdf, current_mask, k=4, permutations=199)
            if np.isfinite(I):
                c1, c2, c3 = st.columns(3)
                c1.metric("Moran's I", f"{I:.3f}")
                c2.metric("Permutation pseudo-p", f"{p:.3f}")
                c3.metric("Flagged-neighbor lift", f"{lift:.2f}×" if np.isfinite(lift) else "NA")
                st.caption("Positive spatial coherence means highlighted trees tend to occur near one another; it does not identify why.")
            else:
                st.info("Spatial coherence requires a non-trivial mix of highlighted and non-highlighted trees.")

        # ---------------------------------------------------------------------
        # 6. Ground validation placeholder
        # ---------------------------------------------------------------------
        st.subheader("6. Ground spectroradiometer validation")
        st.info("Ground spectroradiometer validation will be added in the next step using your actual processed SVC/XHR field spectra and tree mapping. No generic ground-data assumptions are applied in this version.")


with tab_methods:
    st.header("How the system decides")
    st.markdown("""
    **Water anomaly** — WBI is one direct block; NDMI2 + NDSI-RWC form one correlated SWIR block; PRI is additional support only.  
    **Biochemical anomaly** — NDRE + CI red-edge + REP form the chlorophyll/red-edge block; PSRI + SIPI form the pigment/senescence block; ARI1 is additional support.  
    **Low canopy stature** — LAS H-P95 is primary; raster CHM P95 is an independent measurement check; H-IQR is context.  
    **Cross-domain integration** — Water + Biochemical is shown as a two-domain combination, while Water + Biochemical + Low Canopy Stature is the three-domain combination. Raw indices are never counted as extra domains.
    """)

    st.subheader("Operational thresholds")
    st.info("These are frozen **P25-derived orchard-relative thresholds** calculated from the QC-approved study population. They are operational thresholds for this dataset, not universal citrus hard thresholds.")
    rows = [
        ["Water", "WBI", f"≤ {WATER_SENSITIVITY_THRESHOLDS['P25']['WBI_LOW_MAX']:.6f}", "Direct water block"],
        ["Water", "NDMI2", f"≥ {WATER_SENSITIVITY_THRESHOLDS['P25']['NDMI2_HIGH_MIN']:.6f}", "SWIR block"],
        ["Water", "NDSI-RWC", f"≤ {WATER_SENSITIVITY_THRESHOLDS['P25']['NDSI_RWC_LOW_MAX']:.6f}", "SWIR block"],
        ["Water", "PRI", f"≤ {WATER_SENSITIVITY_THRESHOLDS['P25']['PRI_LOW_MAX']:.6f}", "Additional support only"],
        ["Biochemical", "NDRE", f"≤ {BIO_THRESHOLDS['NDRE_LOW_MAX']:.6f}", "Chlorophyll / red-edge"],
        ["Biochemical", "CI red-edge", f"≤ {BIO_THRESHOLDS['CIRED_LOW_MAX']:.6f}", "Chlorophyll / red-edge"],
        ["Biochemical", "REP", f"≤ {BIO_THRESHOLDS['REP_LOW_MAX_NM']:.3f} nm", "Chlorophyll / red-edge"],
        ["Biochemical", "PSRI", f"≥ {BIO_THRESHOLDS['PSRI_HIGH_MIN']:.6f}", "Pigment / senescence"],
        ["Biochemical", "SIPI", f"≥ {BIO_THRESHOLDS['SIPI_HIGH_MIN']:.6f}", "Pigment / senescence"],
        ["Biochemical", "ARI1", f"≥ {BIO_THRESHOLDS['ARI1_HIGH_MIN']:.6f}", "Additional support only"],
        ["Structure", "LAS H-P95", f"≤ {STRUCTURE_THRESHOLDS['H_P95_P25_M']:.3f} m", "Primary low-stature rule"],
        ["Structure", "Raster CHM P95", f"≤ {STRUCTURE_THRESHOLDS['RASTER_CHM_P95_P25_M']:.3f} m", "Independent measurement support"],
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Domain", "Indicator", "Operational condition", "Role"]), hide_index=True, use_container_width=True)

    st.subheader("Upstream quality control")
    st.write("Quality-control and missing-data checks remain active in the scientific processing chain, but the public map presents only supported findings that pass the finalized rule framework.")
    st.info("The app is downstream of the finalized processing workflow. It does not rerun strip QC, index QC, SWIR consensus, or LAS processing.")


# =============================================================================
# 10. OPTIONAL LLM — DOWNSTREAM EXPLANATION ONLY
# =============================================================================
with st.sidebar.expander("🤖 Field assistant", expanded=False):
    st.caption("Optional natural-language explanation. It cannot change the deterministic classification.")
    if not GEMINI_AVAILABLE:
        st.info("google-generativeai is not installed; all scientific functions remain available.")
    else:
        api_key = st.text_input("Gemini API key", type="password")
        model_name = st.text_input("Gemini model", value=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
        if st.button("Explain current scenario"):
            if not api_key:
                st.warning("Enter an API key first.")
            else:
                prompt = (
                    f"Current view: {VIEW_OPTIONS[view]}; highlighted trees: {len(target_gdf)} of {len(gdf)}. "
                    "Explain this supported finding to a field user in simple language. Do not diagnose disease, do not infer causality, "
                    "and do not introduce unsupported categories."
                )
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    st.markdown(response.text)
                except Exception as exc:
                    st.error(f"Assistant error: {exc}")

