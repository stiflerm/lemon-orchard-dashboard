"""
Orchard Diagnostic Intelligence v7 — Complete-386 Supported-Evidence Three-Domain DSS
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
  orchard_diagnostic_intelligence_v6_conservative_structure.py
  data/
    data.zip
    master_tree_multidomain_FINAL_ALL386.csv
    master_tree_spectra_CONSENSUS.csv           # optional VNIR spectral validation
    master_tree_swir_spectra_ALL.csv            # optional full-orchard SWIR spectral inventory
    master_tree_SWIR_indices_ALL_386.csv            # optional all-tree SWIR index inspection inventory
    band_wavelengths.csv                         # VNIR band-to-wavelength mapping

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
import json
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
    DATA_DIR / "master_tree_multidomain_FINAL_ALL386.csv",
    APP_DIR / "master_tree_multidomain_FINAL_ALL386.csv",
    Path(r"D:\hx_UAV_data_processing\final_rule_database_ALL_386\master_tree_multidomain_FINAL_ALL386.csv"),
]

THRESHOLD_JSON_CANDIDATES = [
    DATA_DIR / "thresholds_FINAL_ALL386.json",
    APP_DIR / "thresholds_FINAL_ALL386.json",
    Path(r"D:\hx_UAV_data_processing\final_rule_database_ALL_386\thresholds_FINAL_ALL386.json"),
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
    # Complete 386-tree representative VNIR spectra. VISUALIZATION ONLY.
    DATA_DIR / "VNIR_spectra_indices_ALL_386.csv",
    APP_DIR / "VNIR_spectra_indices_ALL_386.csv",
    Path(r"D:\hx_UAV_data_processing\object_based_tree_hsi\ALL_386_FINAL\VNIR_spectra_indices_ALL_386.csv"),

    # Older consensus-only fallback.
    DATA_DIR / "master_tree_spectra_CONSENSUS.csv",
    APP_DIR / "master_tree_spectra_CONSENSUS.csv",
    Path(r"D:\hx_UAV_data_processing\object_based_tree_hsi\spectral_consensus_qc\master_tree_spectra_CONSENSUS.csv"),
]

# Optional full SWIR consensus spectrum used ONLY for validation/visualization.
# Exact wavelengths are encoded in the output column names (WL_<nm>), so no
# separate SWIR wavelength-map file is required.
SWIR_SPECTRAL_CSV_CANDIDATES = [
    # Complete 386-tree SWIR spectra. VISUALIZATION ONLY.
    DATA_DIR / "SWIR_spectra_indices_ALL_386.csv",
    APP_DIR / "SWIR_spectra_indices_ALL_386.csv",
    Path(r"D:\hx_UAV_data_processing\object_based_tree_swir\ALL_386_FINAL\SWIR_spectra_indices_ALL_386.csv"),

    # Older full-spectrum fallbacks.
    DATA_DIR / "master_tree_swir_spectra_ALL.csv",
    APP_DIR / "master_tree_swir_spectra_ALL.csv",
    Path(r"D:\hx_UAV_data_processing\object_based_tree_swir\full_spectral_all_trees\master_tree_swir_spectra_ALL.csv"),
    DATA_DIR / "master_tree_swir_spectra_CONSENSUS.csv",
    APP_DIR / "master_tree_swir_spectra_CONSENSUS.csv",
    Path(r"D:\hx_UAV_data_processing\object_based_tree_swir\full_spectral_consensus\master_tree_swir_spectra_CONSENSUS.csv"),
]

# Optional all-tree SWIR index inventory used for extra inspection fields.
# The finalized Water classification itself is already frozen upstream in the
# ALL-386 multidomain database; this optional inventory does not recalculate it.
SWIR_ALL_INDEX_CANDIDATES = [
    DATA_DIR / "master_tree_SWIR_indices_ALL_386.csv",
    APP_DIR / "master_tree_SWIR_indices_ALL_386.csv",
    Path(r"D:\hx_UAV_data_processing\object_based_tree_swir_qc\all_tree_index_inventory\master_tree_SWIR_indices_ALL_386.csv"),
]

# Final operational thresholds.
# Preferred source is thresholds_FINAL_ALL386.json generated from the harmonized
# complete-386 database. The numeric blocks below are legacy fallbacks only.
_LEGACY_BIO_THRESHOLDS = {
    "NDRE_LOW_MAX": 0.2047129778647948,
    "CIRED_LOW_MAX": 0.5487572048699663,
    "REP_LOW_MAX_NM": 725.3890241067692,
    "PSRI_HIGH_MIN": 0.2317542293068353,
    "SIPI_HIGH_MIN": 1.6044129002460532,
    "ARI1_HIGH_MIN": 4.409562434704582,
}

_LEGACY_WATER_SENSITIVITY_THRESHOLDS = {
    "P20": {"WBI_LOW_MAX": 0.9732224384098488, "PRI_LOW_MAX": -0.0905354641709804,
            "NDMI2_HIGH_MIN": -0.1563689890814745, "NDSI_RWC_LOW_MAX": 0.23842041377519282},
    "P25": {"WBI_LOW_MAX": 0.9780581745028392, "PRI_LOW_MAX": -0.08910375378375263,
            "NDMI2_HIGH_MIN": -0.17834779593229538, "NDSI_RWC_LOW_MAX": 0.25159298573596856},
    "P30": {"WBI_LOW_MAX": 0.9835634253103394, "PRI_LOW_MAX": -0.08797404072436948,
            "NDMI2_HIGH_MIN": -0.19210124039554718, "NDSI_RWC_LOW_MAX": 0.26483841514836876},
}

_LEGACY_BIO_SENSITIVITY_THRESHOLDS = {
    "P20": {"NDRE": 0.18991320677124718, "CIRED": 0.5020010800931471, "REP": 725.0965727700146,
            "PSRI": 0.24515380157382927, "SIPI": 1.6691380599575552, "ARI1": 4.560504481288227},
    "P25": {"NDRE": 0.2047129778647948, "CIRED": 0.5487572048699663, "REP": 725.3890241067692,
            "PSRI": 0.2317542293068353, "SIPI": 1.6044129002460532, "ARI1": 4.409562434704582},
    "P30": {"NDRE": 0.21786641970173387, "CIRED": 0.6123779812211838, "REP": 725.6065279573306,
            "PSRI": 0.21685585761922752, "SIPI": 1.56515408797565, "ARI1": 4.274624848018448},
}


def _load_all386_threshold_config():
    for p in THRESHOLD_JSON_CANDIDATES:
        if Path(p).exists():
            try:
                cfg = json.loads(Path(p).read_text(encoding="utf-8"))
                water = cfg.get("water_sensitivity_thresholds", {})
                bio = cfg.get("bio_sensitivity_thresholds", {})
                if all(k in water for k in ["P20", "P25", "P30"]) and all(k in bio for k in ["P20", "P25", "P30"]):
                    return cfg, Path(p)
            except Exception:
                pass
    return None, None


_ALL386_THRESHOLD_CONFIG, _ALL386_THRESHOLD_PATH = _load_all386_threshold_config()

if _ALL386_THRESHOLD_CONFIG is not None:
    WATER_SENSITIVITY_THRESHOLDS = _ALL386_THRESHOLD_CONFIG["water_sensitivity_thresholds"]
    BIO_SENSITIVITY_THRESHOLDS = _ALL386_THRESHOLD_CONFIG["bio_sensitivity_thresholds"]
    BIO_THRESHOLDS = {
        "NDRE_LOW_MAX": BIO_SENSITIVITY_THRESHOLDS["P25"]["NDRE"],
        "CIRED_LOW_MAX": BIO_SENSITIVITY_THRESHOLDS["P25"]["CIRED"],
        "REP_LOW_MAX_NM": BIO_SENSITIVITY_THRESHOLDS["P25"]["REP"],
        "PSRI_HIGH_MIN": BIO_SENSITIVITY_THRESHOLDS["P25"]["PSRI"],
        "SIPI_HIGH_MIN": BIO_SENSITIVITY_THRESHOLDS["P25"]["SIPI"],
        "ARI1_HIGH_MIN": BIO_SENSITIVITY_THRESHOLDS["P25"]["ARI1"],
    }
else:
    WATER_SENSITIVITY_THRESHOLDS = _LEGACY_WATER_SENSITIVITY_THRESHOLDS
    BIO_SENSITIVITY_THRESHOLDS = _LEGACY_BIO_SENSITIVITY_THRESHOLDS
    BIO_THRESHOLDS = _LEGACY_BIO_THRESHOLDS

# Final fixed-pair field-reference validation package.
# These files are generated upstream by Steps 5–7 and are READ-ONLY in the app.
# The app does not recompute the validation metrics or production classifications.
VALIDATION_BASE_LOCAL = Path(
    r"E:\SIROHI_OLD\New Folder\GROUND_VALIDATION_AUDIT"
    r"\FINAL_SPECTRAL_INDEX_VALIDATION"
)

VALIDATION_PAIR_CANDIDATES = [
    DATA_DIR / "STEP7_final_pair_validation_summary.csv",
    APP_DIR / "STEP7_final_pair_validation_summary.csv",
    VALIDATION_BASE_LOCAL / "STEP7_FINAL_VALIDATION_PACKAGE" / "STEP7_final_pair_validation_summary.csv",
]

VALIDATION_DOMAIN_CANDIDATES = [
    DATA_DIR / "STEP7_domain_threshold_side_agreement.csv",
    APP_DIR / "STEP7_domain_threshold_side_agreement.csv",
    VALIDATION_BASE_LOCAL / "STEP7_FINAL_VALIDATION_PACKAGE" / "STEP7_domain_threshold_side_agreement.csv",
]

VALIDATION_INDEX_CANDIDATES = [
    DATA_DIR / "STEP7_index_threshold_side_agreement.csv",
    APP_DIR / "STEP7_index_threshold_side_agreement.csv",
    VALIDATION_BASE_LOCAL / "STEP7_FINAL_VALIDATION_PACKAGE" / "STEP7_index_threshold_side_agreement.csv",
]

VALIDATION_PAYLOAD_CANDIDATES = [
    DATA_DIR / "STEP7_app_validation_payload.json",
    APP_DIR / "STEP7_app_validation_payload.json",
    VALIDATION_BASE_LOCAL / "STEP7_FINAL_VALIDATION_PACKAGE" / "STEP7_app_validation_payload.json",
]

VALIDATION_SPECTRA_CANDIDATES = [
    DATA_DIR / "STEP5_ground_UAV_VNIR_resampled_spectra.csv",
    APP_DIR / "STEP5_ground_UAV_VNIR_resampled_spectra.csv",
    VALIDATION_BASE_LOCAL / "STEP5_GROUND_UAV_VNIR" / "STEP5_ground_UAV_VNIR_resampled_spectra.csv",
]

VALIDATION_INDEX_LONG_CANDIDATES = [
    DATA_DIR / "STEP6_ground_vs_UAV_index_validation_long.csv",
    APP_DIR / "STEP6_ground_vs_UAV_index_validation_long.csv",
    VALIDATION_BASE_LOCAL / "STEP6_FINAL_VALIDATION_ASSESSMENT" / "STEP6_ground_vs_UAV_index_validation_long.csv",
]

WAVELENGTH_MAP_CANDIDATES = [
    DATA_DIR / "band_wavelengths.csv",
    DATA_DIR / "wavelength_mapping.csv",
    DATA_DIR / "band_mapping_by_strip.csv",
    APP_DIR / "band_wavelengths.csv",
    APP_DIR / "wavelength_mapping.csv",
    Path(r"D:\hx_UAV_data_processing\object_based_tree_hsi\band_wavelengths.csv"),
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



def normalize_ground_sample_id(value):
    """Normalize fixed field-reference IDs to 0000–0005 without changing scientific mapping."""
    if pd.isna(value):
        return None
    s = str(value).strip()
    try:
        f = float(s)
        if np.isfinite(f) and float(f).is_integer():
            return f"{int(f):04d}"
    except Exception:
        pass
    hits = re.findall(r"(?<!\d)(\d{4})(?!\d)", s)
    if hits:
        for h in reversed(hits):
            if h in {"0000", "0001", "0002", "0003", "0004", "0005"}:
                return h
        return hits[-1]
    digits = re.sub(r"\D", "", s)
    if digits and len(digits) <= 4:
        return digits.zfill(4)
    return s


def supported_structure_mask(df):
    """Conservative public structural result: BOTH LAS low stature and raster CHM corroboration must agree."""
    low = as_bool(df["STRUCTURE_LOW_STATURE"]) if "STRUCTURE_LOW_STATURE" in df.columns else pd.Series(False, index=df.index)
    corroborated = as_bool(df["STRUCTURE_CORROBORATED"]) if "STRUCTURE_CORROBORATED" in df.columns else pd.Series(False, index=df.index)
    return (low & corroborated).fillna(False)


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
    if final_path is None:
        st.error(
            "Final ALL-386 database is missing. Run finalize_ALL386_app_database.py first. "
            "Expected master_tree_multidomain_FINAL_ALL386.csv in data/ or "
            r"D:\hx_UAV_data_processing\final_rule_database_ALL_386."
        )
        st.stop()
    if _ALL386_THRESHOLD_CONFIG is None or _ALL386_THRESHOLD_PATH is None:
        st.error(
            "Final ALL-386 threshold file is missing. Run finalize_ALL386_app_database.py first. "
            "The app will not fall back to historical 322-tree thresholds."
        )
        st.stop()
    df = pd.read_csv(final_path)
    if len(df) != 386 or df["tree_id"].nunique() != 386:
        st.error("Final ALL-386 database must contain exactly 386 unique tree IDs.")
        st.stop()
    return df, f"ALL-386 final database: {final_path.name}"


@st.cache_data(show_spinner=False)
def load_spectral_data():
    path = first_existing(SPECTRAL_CSV_CANDIDATES)
    if path is None:
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_swir_spectral_data():
    path = first_existing(SWIR_SPECTRAL_CSV_CANDIDATES)
    if path is None:
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_swir_all_index_inventory():
    path = first_existing(SWIR_ALL_INDEX_CANDIDATES)
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "tree_id" not in df.columns:
        return pd.DataFrame()
    if df["tree_id"].duplicated().any():
        raise ValueError("All-tree SWIR index inventory contains duplicate tree_id values.")
    return df



@st.cache_data(show_spinner=False)
def load_ground_validation_package():
    """Load finalized Steps 5–7 validation products without recalculating the science."""
    out = {
        "pair": pd.DataFrame(),
        "domain": pd.DataFrame(),
        "index": pd.DataFrame(),
        "spectra": pd.DataFrame(),
        "index_long": pd.DataFrame(),
        "payload": {},
        "sources": {},
    }

    csv_specs = {
        "pair": VALIDATION_PAIR_CANDIDATES,
        "domain": VALIDATION_DOMAIN_CANDIDATES,
        "index": VALIDATION_INDEX_CANDIDATES,
        "spectra": VALIDATION_SPECTRA_CANDIDATES,
        "index_long": VALIDATION_INDEX_LONG_CANDIDATES,
    }

    for key, candidates in csv_specs.items():
        p = first_existing(candidates)
        if p is None:
            continue
        try:
            d = pd.read_csv(p)
            out[key] = d
            out["sources"][key] = str(p)
        except Exception:
            continue

    payload_path = first_existing(VALIDATION_PAYLOAD_CANDIDATES)
    if payload_path is not None:
        try:
            out["payload"] = json.loads(Path(payload_path).read_text(encoding="utf-8"))
            out["sources"]["payload"] = str(payload_path)
        except Exception:
            pass

    for key in ["pair", "spectra", "index_long"]:
        d = out[key]
        if d.empty:
            continue
        if "GROUND_SAMPLE" in d.columns:
            d["GROUND_SAMPLE"] = d["GROUND_SAMPLE"].apply(normalize_ground_sample_id)
        elif "field_file_id" in d.columns:
            d["GROUND_SAMPLE"] = d["field_file_id"].apply(normalize_ground_sample_id)
        if "TREE_ID" in d.columns:
            d["TREE_ID"] = pd.to_numeric(d["TREE_ID"], errors="coerce").astype("Int64")
        elif "tree_id" in d.columns:
            d["TREE_ID"] = pd.to_numeric(d["tree_id"], errors="coerce").astype("Int64")
        out[key] = d

    return out


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
    structure_supported = supported_structure_mask(out)

    out["Water finding"] = np.where(water_supported, "Water anomaly", "")
    if "WATER_EVIDENCE_STATUS" in out.columns:
        strong = out["WATER_EVIDENCE_STATUS"].eq("STRONG_INTERNAL_WATER_EVIDENCE") & water_supported
        out.loc[strong, "Water finding"] = "Water anomaly + PRI support"

    out["Biochemical finding"] = np.where(bio_supported, "Biochemical anomaly", "")
    if "BIOCHEMICAL_STATUS" in out.columns:
        strong = out["BIOCHEMICAL_STATUS"].eq("STRONG_CONCORDANT_BIOCHEMICAL_DECLINE") & bio_supported
        out.loc[strong, "Biochemical finding"] = "Biochemical anomaly + ARI1 support"

    out["Structure finding"] = np.where(structure_supported, "Low canopy stature", "")

    out["Combined finding"] = ""
    out.loc[water_supported & bio_supported, "Combined finding"] = "Water + Biochemical"
    out.loc[water_supported & bio_supported & structure_supported, "Combined finding"] = "Water + Biochemical + Low Canopy Stature"
    return out


WATER_TOOLTIP = [
    ("tree_id", "Tree ID"),
    ("Water finding", "Result"),
    ("WBI_VALUE", "WBI"),
    ("NDMI2_VALUE", "NDMI2"),
    ("NDSI_RWC_VALUE", "NDSI-RWC"),
    ("PRI_VALUE", "PRI"),
    ("WATER_MEASUREMENT_CONFIDENCE", "Measurement support"),
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
    S = supported_structure_mask(gdf)

    if view == "WATER":
        return W
    if view == "BIOCHEMICAL":
        return B
    if view == "STRUCTURE":
        return S
    if view == "WB":
        # Exact two-domain public result requires Structure to have been assessed.
        # A missing structural result must never be interpreted as "no low stature".
        SA = as_bool(gdf["STRUCTURE_ASSESSABLE"]) if "STRUCTURE_ASSESSABLE" in gdf.columns else pd.Series(False, index=gdf.index)
        return W & B & SA & ~S
    if view == "WBS":
        return W & B & S
    return pd.Series(False, index=gdf.index)


def comparison_reference_mask(gdf, view):
    """Scenario-specific clean comparison group. Never labelled 'healthy'."""
    water_no = gdf.get("WATER_EVIDENCE_STATUS", pd.Series("", index=gdf.index)).eq("NO_CONCORDANT_WATER_ANOMALY")
    bio_no = gdf.get("BIOCHEMICAL_STATUS", pd.Series("", index=gdf.index)).eq("NO_CONCORDANT_BIOCHEMICAL_DECLINE")

    h95 = pd.to_numeric(gdf.get("H_P95_m"), errors="coerce")
    r95 = pd.to_numeric(gdf.get("RASTER_CHM_P95_m"), errors="coerce")
    struct_no = (
        h95.notna() & r95.notna()
        & (h95 > STRUCTURE_THRESHOLDS["H_P95_P25_M"])
        & (r95 > STRUCTURE_THRESHOLDS["RASTER_CHM_P95_P25_M"])
    )

    if view == "WATER":
        return water_no
    if view == "BIOCHEMICAL":
        return bio_no
    if view == "STRUCTURE":
        return struct_no
    if view in {"WB", "WBS"}:
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
        return ["H_P95_m", "RASTER_CHM_P95_m"]
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
    r95 = num("RASTER_CHM_P95_m")
    s_thr = {"P20": STRUCTURE_THRESHOLDS["H_P95_P20_M"], "P25": STRUCTURE_THRESHOLDS["H_P95_P25_M"], "P30": STRUCTURE_THRESHOLDS["H_P95_P30_M"]}[scheme]
    # Public structural rule is conservative: vary the primary LAS threshold while keeping the finalized raster corroboration threshold fixed.
    structure = h95.notna() & r95.notna() & (h95 <= s_thr) & (r95 <= STRUCTURE_THRESHOLDS["RASTER_CHM_P95_P25_M"])
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
    group = np.where(target_mask, "Highlighted", np.where(reference_mask, "Comparison", "Other"))
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


def metric_threshold(metric):
    mapping = {
        "WBI_VALUE": (WATER_SENSITIVITY_THRESHOLDS["P25"]["WBI_LOW_MAX"], "≤"),
        "NDMI2_VALUE": (WATER_SENSITIVITY_THRESHOLDS["P25"]["NDMI2_HIGH_MIN"], "≥"),
        "NDSI_RWC_VALUE": (WATER_SENSITIVITY_THRESHOLDS["P25"]["NDSI_RWC_LOW_MAX"], "≤"),
        "PRI_VALUE": (WATER_SENSITIVITY_THRESHOLDS["P25"]["PRI_LOW_MAX"], "≤"),
        "NDRE_VALUE": (BIO_THRESHOLDS["NDRE_LOW_MAX"], "≤"),
        "CIRED_EDGE_VALUE": (BIO_THRESHOLDS["CIRED_LOW_MAX"], "≤"),
        "REP_D1_NM_VALUE": (BIO_THRESHOLDS["REP_LOW_MAX_NM"], "≤"),
        "PSRI_VALUE": (BIO_THRESHOLDS["PSRI_HIGH_MIN"], "≥"),
        "SIPI_VALUE": (BIO_THRESHOLDS["SIPI_HIGH_MIN"], "≥"),
        "ARI1_VALUE": (BIO_THRESHOLDS["ARI1_HIGH_MIN"], "≥"),
        "H_P95_m": (STRUCTURE_THRESHOLDS["H_P95_P25_M"], "≤"),
        "RASTER_CHM_P95_m": (STRUCTURE_THRESHOLDS["RASTER_CHM_P95_P25_M"], "≤"),
    }
    return mapping.get(metric, (None, None))


def comparison_box_plot(target_vals, reference_vals, metric, target_label="Highlighted", reference_label="Comparison", title=None):
    """Simple two-group box plot for target-versus-comparison distributions."""
    tv = pd.to_numeric(pd.Series(target_vals), errors="coerce").dropna()
    rv = pd.to_numeric(pd.Series(reference_vals), errors="coerce").dropna()

    plot_df = pd.concat([
        pd.DataFrame({"Group": target_label, "Value": tv.to_numpy(float)}),
        pd.DataFrame({"Group": reference_label, "Value": rv.to_numpy(float)}),
    ], ignore_index=True)

    fig = px.box(
        plot_df,
        x="Group",
        y="Value",
        points=False,
        category_orders={"Group": [target_label, reference_label]},
        title=title or f"{metric}: highlighted vs comparison trees",
    )
    thr, direction = metric_threshold(metric)
    if thr is not None and np.isfinite(thr):
        fig.add_hline(
            y=thr,
            line_dash="dash",
            annotation_text=f"Operational threshold {direction} {thr:.3f}",
        )
    fig.update_layout(
        yaxis_title=metric,
        xaxis_title="",
        height=420,
        showlegend=False,
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig


def comparison_point_plot(target_df, reference_df, metric, target_label="Highlighted", reference_label="Comparison", title=None):
    """Individual-tree point plot shown beside the box plot."""
    t = target_df[["tree_id", metric]].copy() if {"tree_id", metric}.issubset(target_df.columns) else pd.DataFrame(columns=["tree_id", metric])
    r = reference_df[["tree_id", metric]].copy() if {"tree_id", metric}.issubset(reference_df.columns) else pd.DataFrame(columns=["tree_id", metric])
    t[metric] = pd.to_numeric(t[metric], errors="coerce")
    r[metric] = pd.to_numeric(r[metric], errors="coerce")
    t = t.dropna(subset=[metric]); r = r.dropna(subset=[metric])
    t["Group"] = target_label; r["Group"] = reference_label
    plot_df = pd.concat([t, r], ignore_index=True).rename(columns={metric: "Value"})
    fig = px.strip(
        plot_df, x="Group", y="Value", color="Group", hover_data=["tree_id"],
        category_orders={"Group": [target_label, reference_label]},
        title=title or f"{metric}: individual trees",
    )
    fig.update_traces(marker=dict(size=7, opacity=0.62))
    thr, direction = metric_threshold(metric)
    if thr is not None and np.isfinite(thr):
        fig.add_hline(
            y=thr, line_dash="dash",
            annotation_text=f"Operational threshold {direction} {thr:.3f}",
        )
    fig.update_layout(
        yaxis_title=metric, xaxis_title="", height=420, showlegend=False,
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig


def numeric_suffix(value):
    m = re.search(r"(-?\d+(?:\.\d+)?)$", str(value))
    return float(m.group(1)) if m else float("inf")


def vnir_spectral_columns(df):
    """Prefer exact-wavelength WL_* columns; fall back to legacy Band_* columns."""
    if df is None or df.empty:
        return []
    wl_cols = [c for c in df.columns if str(c).startswith("WL_")]
    if wl_cols:
        return sorted(wl_cols, key=numeric_suffix)
    band_cols = [c for c in df.columns if str(c).startswith("Band_")]
    return sorted(band_cols, key=numeric_suffix)


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




def swir_spectral_columns(df):
    """Return exact-wavelength SWIR columns produced by the full-spectrum extractor."""
    if df is None or df.empty:
        return []
    cols = [c for c in df.columns if str(c).startswith("WL_")]
    return sorted(cols, key=numeric_suffix)


def add_swir_water_regions(fig):
    """Mark the exact Headwall SWIR band centres used by the final water rule."""
    for center, label in [
        (1097.940, "NDMI2: R1100"),
        (1224.040, "NDSI-RWC: R1222"),
        (2202.750, "NDMI2: R2200"),
        (2262.790, "NDSI-RWC: R2264"),
    ]:
        fig.add_vrect(
            x0=center - 6.0,
            x1=center + 6.0,
            opacity=0.10,
            line_width=0,
            annotation_text=label,
            annotation_position="top left",
        )
    return fig


def plot_mean_spectral_pair(target_df, reference_df, band_cols, x, title, region_func=None, height=430):
    """Plot mean target and comparison spectra and return means for optional metrics."""
    if target_df.empty or reference_df.empty or not band_cols:
        return None, None, None
    tmat = target_df[band_cols].apply(pd.to_numeric, errors="coerce")
    rmat = reference_df[band_cols].apply(pd.to_numeric, errors="coerce")
    tm = tmat.mean(axis=0, skipna=True).to_numpy(float)
    rm = rmat.mean(axis=0, skipna=True).to_numpy(float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=tm, mode="lines", name="Highlighted trees — mean", connectgaps=False))
    fig.add_trace(go.Scatter(x=x, y=rm, mode="lines", name="Comparison trees — mean", connectgaps=False))
    if region_func is not None:
        fig = region_func(fig)
    fig.update_layout(title=title, xaxis_title="Wavelength (nm)", yaxis_title="Reflectance", height=height)
    return fig, tm, rm

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
        return W & B & ~S
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
            rows.append([scheme, "Structure", "Raster CHM P95", f"≤ {STRUCTURE_THRESHOLDS['RASTER_CHM_P95_P25_M']:.3f} m (fixed corroboration)"])
    return pd.DataFrame(rows, columns=["Sensitivity scheme", "Domain", "Indicator", "Condition"])



VALIDATION_INDEX_SECTORS = {
    "WBI": "900 and 970 nm (VNIR water-band ratio)",
    "PRI": "531 and 570 nm",
    "NDMI2": "1097.940 and 2202.750 nm (SWIR)",
    "NDSI_RWC": "1224.040 and 2262.790 nm (SWIR)",
    "NDRE": "705 and 750 nm (red edge)",
    "CIRED_EDGE": "705 and 750 nm (red edge)",
    "REP_D1_NM": "680–750 nm red-edge derivative region",
    "PSRI": "500, 680 and 750 nm",
    "SIPI": "445, 680 and 800 nm",
    "ARI1": "550 and 700 nm",
    "NDVI": "670 and 800 nm",
}


def add_field_validation_layer(m, gdf, pair_df):
    """Optional map overlay for the displayed field-reference ↔ UAV pairs."""
    if pair_df is None or pair_df.empty or "TREE_ID" not in pair_df.columns:
        return
    pairs = pair_df[[c for c in ["GROUND_SAMPLE", "TREE_ID"] if c in pair_df.columns]].dropna().copy()
    if pairs.empty:
        return
    pairs["TREE_ID"] = pd.to_numeric(pairs["TREE_ID"], errors="coerce").astype("Int64")
    sample_map = {
        int(r["TREE_ID"]): str(r.get("GROUND_SAMPLE", "")).zfill(4)
        for _, r in pairs.dropna(subset=["TREE_ID"]).iterrows()
    }
    vg = gdf[gdf["tree_id"].isin(sample_map)].copy()
    if vg.empty:
        return
    vg["FIELD_REFERENCE"] = vg["tree_id"].map(lambda x: f"Ground {sample_map.get(int(x), '')}")
    tooltip = folium.GeoJsonTooltip(
        fields=["tree_id", "FIELD_REFERENCE"],
        aliases=["Tree ID:", "Field reference:"],
        sticky=True,
    )
    folium.GeoJson(
        vg,
        style_function=lambda _: {
            "fillColor": "#00A6D6",
            "color": "#00A6D6",
            "weight": 3.0,
            "dashArray": "6,4",
            "fillOpacity": 0.04,
        },
        tooltip=tooltip,
        name="Field-reference validation trees",
    ).add_to(m)


def validation_pair_label(row):
    sample = normalize_ground_sample_id(row.get("GROUND_SAMPLE", "")) or ""
    tree = row.get("TREE_ID", "NA")
    try:
        tree = int(tree)
    except Exception:
        pass
    return f"Ground {sample} → UAV Tree {tree}"


def validation_status_text(flag):
    try:
        if pd.isna(flag):
            return "NA"
        return "Anomalous side" if int(flag) == 1 else "Reference side"
    except Exception:
        return "NA"


def plot_field_reference_pair(spectra_df, sample_id, tree_id):
    if spectra_df is None or spectra_df.empty:
        return None
    d = spectra_df.copy()
    if "GROUND_SAMPLE" in d.columns:
        d = d[d["GROUND_SAMPLE"].astype(str).eq(str(sample_id))]
    if "TREE_ID" in d.columns:
        d = d[pd.to_numeric(d["TREE_ID"], errors="coerce").eq(int(tree_id))]
    required = {"wavelength_nm", "ground_interpolated", "uav_representative"}
    if d.empty or not required.issubset(d.columns):
        return None
    d = d.sort_values("wavelength_nm")
    x = pd.to_numeric(d["wavelength_nm"], errors="coerce")
    g = pd.to_numeric(d["ground_interpolated"], errors="coerce")
    u = pd.to_numeric(d["uav_representative"], errors="coerce")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=g, mode="lines", name=f"Ground {sample_id} — interpolated"))
    fig.add_trace(go.Scatter(x=x, y=u, mode="lines", name=f"UAV Tree {tree_id} — representative"))
    fig = add_rule_regions(fig, "WBS", True)
    fig.update_layout(
        title=f"Field reference Ground {sample_id} ↔ UAV Tree {tree_id}",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Reflectance",
        height=470,
        legend=dict(orientation="h"),
    )
    return fig


def plot_field_reference_difference(spectra_df, sample_id, tree_id):
    if spectra_df is None or spectra_df.empty:
        return None
    d = spectra_df.copy()
    if "GROUND_SAMPLE" in d.columns:
        d = d[d["GROUND_SAMPLE"].astype(str).eq(str(sample_id))]
    if "TREE_ID" in d.columns:
        d = d[pd.to_numeric(d["TREE_ID"], errors="coerce").eq(int(tree_id))]
    if d.empty or "wavelength_nm" not in d.columns:
        return None
    if "uav_minus_ground" in d.columns:
        delta = pd.to_numeric(d["uav_minus_ground"], errors="coerce")
    elif {"uav_representative", "ground_interpolated"}.issubset(d.columns):
        delta = pd.to_numeric(d["uav_representative"], errors="coerce") - pd.to_numeric(d["ground_interpolated"], errors="coerce")
    else:
        return None
    x = pd.to_numeric(d["wavelength_nm"], errors="coerce")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=delta, mode="lines", name="UAV − ground"))
    fig.add_hline(y=0, line_dash="dash")
    fig = add_rule_regions(fig, "WBS", True)
    fig.update_layout(
        title=f"Reflectance difference — Ground {sample_id} ↔ UAV Tree {tree_id}",
        xaxis_title="Wavelength (nm)",
        yaxis_title="UAV − ground reflectance",
        height=340,
        showlegend=False,
    )
    return fig


def build_orchard_reference_spectrum(gdf, spectral_df):
    """
    Build one robust orchard reference spectrum for visualization only.

    Internal selection:
      - no concordant water anomaly
      - no concordant biochemical decline
      - structure assessable and no low-stature evidence
      - valid non-zero VNIR spectrum

    The app displays this simply as "Reference spectrum".
    The reference does NOT change any DSS classification.
    """
    if gdf is None or gdf.empty or spectral_df is None or spectral_df.empty:
        return None

    if "tree_id" not in gdf.columns:
        return None

    water_ok = (
        gdf["WATER_EVIDENCE_STATUS"].astype(str).eq("NO_CONCORDANT_WATER_ANOMALY")
        if "WATER_EVIDENCE_STATUS" in gdf.columns
        else ~as_bool(gdf["WATER_SUPPORTED"])
    )

    bio_ok = (
        gdf["BIOCHEMICAL_STATUS"].astype(str).eq("NO_CONCORDANT_BIOCHEMICAL_DECLINE")
        if "BIOCHEMICAL_STATUS" in gdf.columns
        else ~as_bool(gdf["BIOCHEMICAL_SUPPORTED"])
    )

    if "STRUCTURE_STATUS_FINAL" in gdf.columns:
        structure_ok = gdf["STRUCTURE_STATUS_FINAL"].astype(str).eq("NO_LOW_STATURE_EVIDENCE")
    else:
        structure_assessable = (
            as_bool(gdf["STRUCTURE_ASSESSABLE"])
            if "STRUCTURE_ASSESSABLE" in gdf.columns
            else pd.Series(True, index=gdf.index)
        )
        structure_ok = structure_assessable & ~supported_structure_mask(gdf)

    candidate_ids = set(
        pd.to_numeric(
            gdf.loc[water_ok & bio_ok & structure_ok, "tree_id"],
            errors="coerce",
        ).dropna().astype(int)
    )
    if not candidate_ids:
        return None

    key = "tree_id" if "tree_id" in spectral_df.columns else None
    if key is None:
        return None

    band_cols = vnir_spectral_columns(spectral_df)
    if not band_cols:
        return None

    work = spectral_df[
        pd.to_numeric(spectral_df[key], errors="coerce").isin(candidate_ids)
    ].copy()
    if work.empty:
        return None

    mat = work[band_cols].apply(pd.to_numeric, errors="coerce")

    # Exclude empty/zero-filled spectra from the visualization reference.
    valid_fraction = mat.notna().mean(axis=1)
    row_median = mat.median(axis=1, skipna=True)
    keep = valid_fraction.ge(0.80) & row_median.gt(0.001)
    mat = mat.loc[keep]
    work = work.loc[keep]

    if mat.empty:
        return None

    # Median is deliberately used instead of the mean so a few unusual
    # spectra do not dominate the visual reference.
    ref = mat.median(axis=0, skipna=True).to_numpy(float)

    x, _, true_wavelength, _ = spectral_axis_from_mapping(band_cols)
    if not true_wavelength:
        suffixes = np.asarray([numeric_suffix(c) for c in band_cols], dtype=float)
        if (
            len(suffixes)
            and np.isfinite(suffixes).all()
            and suffixes.min() >= 350
            and suffixes.max() <= 1100
        ):
            x = suffixes
        else:
            return None

    return {
        "wavelength_nm": np.asarray(x, dtype=float),
        "reflectance": ref,
        "n_trees": int(len(mat)),
        "tree_ids": pd.to_numeric(work[key], errors="coerce").dropna().astype(int).tolist(),
    }


def plot_simple_ground_uav_reference(
    spectra_df,
    sample_id,
    tree_id,
    reference_spectrum=None,
):
    """Main clean validation plot: Ground vs matched UAV vs orchard Reference."""
    if spectra_df is None or spectra_df.empty:
        return None

    d = spectra_df.copy()
    if "GROUND_SAMPLE" in d.columns:
        d = d[d["GROUND_SAMPLE"].astype(str).eq(str(sample_id))]
    if "TREE_ID" in d.columns:
        d = d[pd.to_numeric(d["TREE_ID"], errors="coerce").eq(int(tree_id))]

    required = {"wavelength_nm", "ground_interpolated", "uav_representative"}
    if d.empty or not required.issubset(d.columns):
        return None

    d = d.sort_values("wavelength_nm")
    x = pd.to_numeric(d["wavelength_nm"], errors="coerce").to_numpy(float)
    ground = pd.to_numeric(d["ground_interpolated"], errors="coerce").to_numpy(float)
    uav = pd.to_numeric(d["uav_representative"], errors="coerce").to_numpy(float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=ground, mode="lines",
        name="Ground spectrum",
        connectgaps=False,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=uav, mode="lines",
        name=f"UAV Tree {tree_id}",
        connectgaps=False,
    ))

    if reference_spectrum is not None:
        rx = np.asarray(reference_spectrum["wavelength_nm"], dtype=float)
        ry = np.asarray(reference_spectrum["reflectance"], dtype=float)
        good = np.isfinite(rx) & np.isfinite(ry)
        if good.sum() >= 3:
            # Interpolate the orchard reference to the paired UAV wavelengths.
            ref_interp = np.interp(x, rx[good], ry[good], left=np.nan, right=np.nan)
            fig.add_trace(go.Scatter(
                x=x, y=ref_interp, mode="lines",
                name="Reference spectrum",
                line=dict(dash="dash"),
                connectgaps=False,
            ))

    fig.update_layout(
        title=f"Ground vs UAV spectrum — Tree {tree_id}",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Reflectance",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=20, r=20, t=75, b=20),
    )
    return fig


def simple_validation_interpretation(sample_id, tree_id, row):
    """Short user-facing interpretation; no new diagnostic class is created."""
    sample_id = str(sample_id).zfill(4)

    if sample_id == "0002" and int(tree_id) == 401:
        return (
            "Ground and UAV spectra show strong overall correspondence; "
            "the main localized difference is around the red-edge position."
        )

    return (
        "Ground and UAV spectra show close overall spectral correspondence "
        "for this tree."
    )


def relevant_cross_domain_counts(gdf):
    """Cross-domain combinations using conservative structure, without treating missing structure as negative."""
    W = as_bool(gdf["WATER_SUPPORTED"])
    B = as_bool(gdf["BIOCHEMICAL_SUPPORTED"])
    S = supported_structure_mask(gdf)
    SA = as_bool(gdf["STRUCTURE_ASSESSABLE"]) if "STRUCTURE_ASSESSABLE" in gdf.columns else pd.Series(False, index=gdf.index)
    rows = [
        ["Water + Biochemical", int((W & B & SA & ~S).sum())],
        ["Water + Low Canopy Stature", int((W & S & ~B).sum())],
        ["Biochemical + Low Canopy Stature", int((B & S & ~W).sum())],
        ["Water + Biochemical + Low Canopy Stature", int((W & B & S).sum())],
    ]
    return pd.DataFrame(rows, columns=["Combination", "Trees"])



# =============================================================================
# 8. APP LOAD / CONTROLS
# =============================================================================

st.set_page_config(page_title="Orchard Three-Domain DSS", layout="wide")
st.title("🍋 Orchard Intelligence — Water • Biochemistry • Structure")
st.caption(
    "Choose a supported finding, inspect every tree on the map, compare highlighted vs comparison spectra, "
    "and open full VNIR + SWIR spectra for any Tree ID. Final rule classification uses the harmonized "
    "386-tree pixel-wise VNIR indices and finalized 386-tree SWIR water indices; full spectra are descriptive "
    "validation/interpretation layers and do not independently create a diagnosis."
)

geometry_gdf = load_tree_geometry()
rule_df, rule_source = load_final_rule_database()
if _ALL386_THRESHOLD_PATH is not None:
    rule_source += f" | thresholds: {_ALL386_THRESHOLD_PATH.name}"
spectral_df = load_spectral_data()
swir_spectral_df = load_swir_spectral_data()
swir_all_index_df = load_swir_all_index_inventory()
validation_pkg = load_ground_validation_package()
validation_pair_df = validation_pkg.get("pair", pd.DataFrame())
validation_domain_df = validation_pkg.get("domain", pd.DataFrame())
validation_index_df = validation_pkg.get("index", pd.DataFrame())
validation_spectra_df = validation_pkg.get("spectra", pd.DataFrame())
validation_index_long_df = validation_pkg.get("index_long", pd.DataFrame())

# Public validation display: Tree 233 / Ground 0001 is intentionally omitted.
# The upstream validation files are not modified.
EXCLUDED_VALIDATION_TREE_IDS = {233}
if not validation_pair_df.empty and "TREE_ID" in validation_pair_df.columns:
    _vid = pd.to_numeric(validation_pair_df["TREE_ID"], errors="coerce")
    validation_pair_df = validation_pair_df.loc[
        ~_vid.isin(EXCLUDED_VALIDATION_TREE_IDS)
    ].copy()
try:
    gdf = merge_geometry_and_rules(geometry_gdf, rule_df)
    if not swir_all_index_df.empty:
        extra_cols = [c for c in swir_all_index_df.columns if c != "tree_id" and c not in gdf.columns]
        gdf = gdf.merge(
            swir_all_index_df[["tree_id"] + extra_cols],
            on="tree_id", how="left", validate="one_to_one"
        )
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=geometry_gdf.crs)
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
show_field_validation_trees = False
if not validation_pair_df.empty:
    show_field_validation_trees = st.sidebar.checkbox(
        "Show field-reference validation trees",
        value=True,
        help="Adds a dashed outline around the ground↔UAV validation trees shown in the validation page. It does not change the selected scenario.",
    )

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
    "🧪 Validation",
    "📘 Methods",
])

with tab_map:
    if view == "WATER":
        st.header("💧 Water anomaly")
        st.write("Highlighted trees satisfy the finalized Water screening rule: pixel-wise VNIR WBI and the SWIR water block agree. PRI is shown only as additional physiological support. This is not a confirmation of physiological drought.")
    elif view == "BIOCHEMICAL":
        st.header("🧪 Biochemical anomaly")
        st.write("Highlighted trees show agreement between the chlorophyll/red-edge block and the pigment/senescence block. ARI1 can provide additional support. This is not a nutrient- or disease-specific diagnosis.")
    elif view == "STRUCTURE":
        st.header("🌳 Low canopy stature")
        st.write("Highlighted trees are the conservative structural subset where both LAS H-P95 and raster CHM P95 agree that canopy stature is low.")
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

    if show_field_validation_trees and not validation_pair_df.empty:
        add_field_validation_layer(m, gdf, validation_pair_df)

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

        if {"NDMI2_ALL_VALUE", "NDSI_RWC_ALL_VALUE"}.issubset(gdf.columns):
            with st.expander("Research audit: SWIR index values across all orchard trees", expanded=False):
                both_all = (
                    pd.to_numeric(gdf["NDMI2_ALL_VALUE"], errors="coerce").notna()
                    & pd.to_numeric(gdf["NDSI_RWC_ALL_VALUE"], errors="coerce").notna()
                )
                both_supported = (
                    as_bool(gdf["BOTH_SWIR_SUPPORTED"])
                    if "BOTH_SWIR_SUPPORTED" in gdf.columns
                    else pd.Series(False, index=gdf.index)
                )
                descriptive_both = both_all & ~both_supported

                a, b, c = st.columns(3)
                a.metric("QC-supported SWIR pair", int(both_supported.sum()))
                b.metric("Additional descriptive SWIR pair", int(descriptive_both.sum()))
                c.metric("Trees with both SWIR values", int(both_all.sum()))

                st.caption(
                    "Additional values allow inspection of trees outside the strict SWIR consensus. "
                    "They do not create Water-anomaly flags; the Water map still uses only QC-supported WBI + SWIR evidence."
                )

                audit_cols = [c for c in [
                    "tree_id", "NDMI2_ALL_VALUE", "NDMI2_ALL_SOURCE", "NDMI2_ALL_N_STRIPS",
                    "NDSI_RWC_ALL_VALUE", "NDSI_RWC_ALL_SOURCE", "NDSI_RWC_ALL_N_STRIPS"
                ] if c in gdf.columns]
                st.dataframe(
                    gdf[audit_cols].sort_values("tree_id"),
                    hide_index=True, use_container_width=True, height=260
                )

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
        supported_structure = supported_structure_mask(gdf)
        h95 = pd.to_numeric(gdf.get("H_P95_m"), errors="coerce")
        valid_struct = h95.notna() & ~gdf.get("STRUCTURE_STATUS_FINAL", pd.Series("", index=gdf.index)).eq("NOT_ASSESSABLE")
        orchard_vals = h95[valid_struct].dropna()
        target_vals = h95[supported_structure].dropna()
        orchard_median = float(orchard_vals.median()) if len(orchard_vals) else np.nan
        orchard_mean = float(orchard_vals.mean()) if len(orchard_vals) else np.nan
        target_median = float(target_vals.median()) if len(target_vals) else np.nan
        diff_pct = (100.0 * (orchard_median - target_median) / orchard_median) if np.isfinite(orchard_median) and orchard_median != 0 and np.isfinite(target_median) else np.nan

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Highlighted low-stature trees", int(supported_structure.sum()))
        c2.metric("Orchard median H-P95", f"{orchard_median:.2f} m" if np.isfinite(orchard_median) else "NA")
        c3.metric("Highlighted median H-P95", f"{target_median:.2f} m" if np.isfinite(target_median) else "NA")
        c4.metric("Median height difference", f"{diff_pct:.1f}% lower" if np.isfinite(diff_pct) else "NA")

        st.markdown("### Decision rule")
        st.markdown(
            f"A tree is shown here only when **both structural measurements agree**: "
            f"LAS H-P95 ≤ **{STRUCTURE_THRESHOLDS['H_P95_P25_M']:.3f} m** and "
            f"raster CHM P95 ≤ **{STRUCTURE_THRESHOLDS['RASTER_CHM_P95_P25_M']:.3f} m**."
        )
        med = pd.DataFrame({
            "Group": ["Whole orchard", "Highlighted low-stature trees"],
            "Median H-P95 (m)": [orchard_median, target_median],
        })
        fig = px.bar(med, x="Group", y="Median H-P95 (m)", text="Median H-P95 (m)", title="Typical canopy height: orchard vs highlighted trees")
        fig.update_traces(texttemplate="%{text:.2f} m", textposition="outside")
        fig.add_hline(y=STRUCTURE_THRESHOLDS["H_P95_P25_M"], line_dash="dash", annotation_text="LAS operational threshold")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Orchard mean H-P95 = {orchard_mean:.2f} m." if np.isfinite(orchard_mean) else "")
        st.caption("This is a conservative orchard-relative structural finding based on agreement between two height products; it is not a disease diagnosis.")

    elif view == "WB":
        W = as_bool(gdf["WATER_SUPPORTED"]); B = as_bool(gdf["BIOCHEMICAL_SUPPORTED"]); S = supported_structure_mask(gdf)
        st.markdown("### Two-domain combination")
        st.markdown("**Water anomaly** + **Biochemical anomaly** + *no low-stature result* → **Water + Biochemical**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Water anomaly", int(W.sum()))
        c2.metric("Biochemical anomaly", int(B.sum()))
        c3.metric("Water + Biochemical highlighted", int(target_mask.sum()))
        st.info("This view is intentionally separate from the three-domain view below.")

    elif view == "WBS":
        W = as_bool(gdf["WATER_SUPPORTED"]); B = as_bool(gdf["BIOCHEMICAL_SUPPORTED"]); S = supported_structure_mask(gdf)
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
    S = supported_structure_mask(gdf)
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
            "tree_id", "Water finding", "Biochemical finding", "Structure finding",
            "CROSS_DOMAIN_PATTERN", "WBI_VALUE", "NDMI2_VALUE", "NDSI_RWC_VALUE", "PRI_VALUE",
            "NDRE_VALUE", "CIRED_EDGE_VALUE", "REP_D1_NM_VALUE", "PSRI_VALUE", "SIPI_VALUE", "ARI1_VALUE",
            "H_P95_m", "RASTER_CHM_P95_m"
        ]
        out = gdf[[c for c in cols if c in gdf.columns]].copy()
        supported_any = W | B | S
        out = out.loc[supported_any].copy()
        st.download_button("Download supported results CSV", out.to_csv(index=False).encode(), "orchard_supported_results.csv", "text/csv")


with tab_validation:
    st.header("1. Ground–UAV validation")
    st.write(
        "Select one of the five displayed measured trees to compare the ground spectrum "
        "with the corresponding UAV spectrum."
    )

    if validation_pair_df.empty:
        st.warning(
            "Validation files are missing from data/. "
            "Add the finalized Step 5–7 validation outputs to show this page."
        )
    else:
        vp = validation_pair_df.copy()
        vp["TREE_ID"] = pd.to_numeric(vp["TREE_ID"], errors="coerce").astype("Int64")
        vp["GROUND_SAMPLE"] = vp["GROUND_SAMPLE"].apply(normalize_ground_sample_id)

        pair_labels = [
            f"Tree {int(r['TREE_ID'])} / Ground {normalize_ground_sample_id(r['GROUND_SAMPLE'])}"
            for _, r in vp.iterrows()
        ]

        selected_label = st.selectbox(
            "Select validation tree",
            pair_labels,
            index=0,
            key="simple_ground_uav_pair",
        )
        selected_idx = pair_labels.index(selected_label)
        prow = vp.iloc[selected_idx]

        sample_id = normalize_ground_sample_id(prow.get("GROUND_SAMPLE"))
        tree_id = int(prow["TREE_ID"])

        # --------------------------------------------------------------
        # Main plot: Ground + matched UAV + one simple orchard reference
        # --------------------------------------------------------------
        orchard_reference = build_orchard_reference_spectrum(gdf, spectral_df)

        main_fig = plot_simple_ground_uav_reference(
            validation_spectra_df,
            sample_id,
            tree_id,
            orchard_reference,
        )

        if main_fig is not None:
            st.plotly_chart(main_fig, use_container_width=True)
        else:
            st.info(
                "The paired ground/UAV spectrum for this tree could not be loaded."
            )

        # --------------------------------------------------------------
        # Only three headline spectral metrics
        # --------------------------------------------------------------
        m1, m2, m3 = st.columns(3)
        m1.metric("Pearson r", fmt(prow.get("PEARSON_R"), 3))
        m2.metric(
            "SAM",
            f"{fmt(prow.get('SAM_DEG'), 2)}°"
            if pd.notna(prow.get("SAM_DEG"))
            else "NA",
        )
        m3.metric("RMSE", fmt(prow.get("RMSE"), 4))

        st.info(simple_validation_interpretation(sample_id, tree_id, prow))

        # --------------------------------------------------------------
        # Optional compact diagnostic-index check
        # --------------------------------------------------------------
        if not validation_index_long_df.empty:
            vi = validation_index_long_df.copy()

            if "GROUND_SAMPLE" in vi.columns:
                vi["GROUND_SAMPLE"] = vi["GROUND_SAMPLE"].apply(
                    normalize_ground_sample_id
                )
            if "TREE_ID" in vi.columns:
                vi["TREE_ID"] = pd.to_numeric(
                    vi["TREE_ID"], errors="coerce"
                ).astype("Int64")

            sel = vi[
                vi["GROUND_SAMPLE"].astype(str).eq(str(sample_id))
                & vi["TREE_ID"].eq(tree_id)
            ].copy()

            if not sel.empty:
                # Keep the default interface compact: diagnostic indices only.
                sel = sel[
                    sel["DOMAIN"].astype(str).str.upper().isin(
                        ["WATER", "BIOCHEMICAL"]
                    )
                ].copy()

                sel["Match"] = np.where(
                    pd.to_numeric(
                        sel["THRESHOLD_SIDE_AGREEMENT"], errors="coerce"
                    ).eq(1),
                    "✓",
                    "Different",
                )

                compact = sel[
                    ["INDEX", "GROUND_VALUE", "UAV_VALUE", "Match"]
                ].copy()
                compact = compact.rename(
                    columns={
                        "INDEX": "Indicator",
                        "GROUND_VALUE": "Ground",
                        "UAV_VALUE": "UAV",
                    }
                )

                with st.expander("Diagnostic indices", expanded=False):
                    st.dataframe(
                        compact.round(4),
                        hide_index=True,
                        use_container_width=True,
                    )
                    st.caption(
                        "This table shows whether the ground and UAV indicator "
                        "fall on the same side of the operational threshold."
                    )

        # --------------------------------------------------------------
        # Reference explanation kept out of the main interface
        # --------------------------------------------------------------
        if orchard_reference is not None:
            with st.expander("About the reference spectrum", expanded=False):
                st.write(
                    "The dashed reference curve is the median UAV spectrum of "
                    "orchard trees selected from the final database with no "
                    "supported water or biochemical anomaly and no low-stature "
                    "evidence. It is used only for visual comparison and does "
                    "not change any classification."
                )

        # --------------------------------------------------------------
        # Simple statistical validation summary
        # --------------------------------------------------------------
        with st.expander("Statistical validation summary", expanded=False):
            mean_r = pd.to_numeric(vp.get("PEARSON_R"), errors="coerce").mean()
            median_r = pd.to_numeric(vp.get("PEARSON_R"), errors="coerce").median()
            mean_sam = pd.to_numeric(vp.get("SAM_DEG"), errors="coerce").mean()
            mean_rmse = pd.to_numeric(vp.get("RMSE"), errors="coerce").mean()

            s1, s2, s3, s4 = st.columns(4)
            s1.metric(
                "Mean Pearson r",
                f"{mean_r:.3f}" if np.isfinite(mean_r) else "NA",
            )
            s2.metric(
                "Median Pearson r",
                f"{median_r:.3f}" if np.isfinite(median_r) else "NA",
            )
            s3.metric(
                "Mean SAM",
                f"{mean_sam:.2f}°" if np.isfinite(mean_sam) else "NA",
            )
            s4.metric(
                "Mean RMSE",
                f"{mean_rmse:.4f}" if np.isfinite(mean_rmse) else "NA",
            )

            st.caption(
                "These summarize spectral correspondence across the displayed "
                "ground–UAV pairs. Higher Pearson r and lower SAM/RMSE indicate "
                "closer spectral agreement."
            )

            compact_pairs = vp[
                [
                    c for c in [
                        "GROUND_SAMPLE", "TREE_ID",
                        "PEARSON_R", "SAM_DEG", "RMSE"
                    ] if c in vp.columns
                ]
            ].copy()

            compact_pairs = compact_pairs.rename(
                columns={
                    "GROUND_SAMPLE": "Ground",
                    "TREE_ID": "Tree",
                    "PEARSON_R": "Pearson r",
                    "SAM_DEG": "SAM (°)",
                    "RMSE": "RMSE",
                }
            )
            st.dataframe(
                compact_pairs.round(4),
                hide_index=True,
                use_container_width=True,
            )

            # Keep threshold-side agreement available, but secondary.
            if {
                "N_INDEX_COMPARISONS",
                "N_THRESHOLD_SIDE_AGREEMENTS",
            }.issubset(vp.columns):
                n_compare = int(
                    pd.to_numeric(
                        vp["N_INDEX_COMPARISONS"], errors="coerce"
                    ).sum()
                )
                n_agree = int(
                    pd.to_numeric(
                        vp["N_THRESHOLD_SIDE_AGREEMENTS"], errors="coerce"
                    ).sum()
                )
                if n_compare:
                    st.caption(
                        f"Additional index check: {n_agree}/{n_compare} "
                        f"({100.0*n_agree/n_compare:.1f}%) paired index values "
                        "fell on the same side of their operational threshold. "
                        "This is not classification accuracy."
                    )


    st.divider()
    st.header("2. Statistical robustness")
    st.caption(
        "This section checks whether the UAV rule-based groups remain stable "
        "when thresholds move slightly, and whether highlighted trees differ "
        "statistically from valid comparison trees."
    )

    if view == "GAPS":
        st.info(
            "Select Water, Biochemical, Low Canopy Stature, Water + Biochemical, "
            "or the three-domain scenario to view statistical robustness."
        )
    else:
        current_mask = target_mask.astype(bool)
        reference_mask = comparison_reference_mask(gdf, view).astype(bool)

        # ----------------------------------------------------------
        # A. Threshold stability / Jaccard
        # ----------------------------------------------------------
        st.subheader("Threshold stability")

        schemes = ["P20", "P25", "P30"]
        runs = {s: sensitivity_target_mask(gdf, view, s) for s in schemes}

        count_p20 = int(runs["P20"].sum())
        count_p25 = int(runs["P25"].sum())
        count_p30 = int(runs["P30"].sum())

        j20_25 = jaccard(runs["P20"], runs["P25"])
        j25_30 = jaccard(runs["P25"], runs["P30"])

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("P20 trees", count_p20)
        c2.metric("P25 trees", count_p25)
        c3.metric("P30 trees", count_p30)
        c4.metric(
            "Jaccard P20↔P25",
            f"{j20_25:.3f}" if np.isfinite(j20_25) else "NA",
        )
        c5.metric(
            "Jaccard P25↔P30",
            f"{j25_30:.3f}" if np.isfinite(j25_30) else "NA",
        )

        st.caption(
            "P25 is the operational rule. P20 and P30 are stricter/more-inclusive "
            "sensitivity checks. Jaccard = 1 means the selected tree sets are identical; "
            "lower values mean greater sensitivity to threshold choice."
        )

        stability_df = pd.DataFrame(
            {
                "Threshold": ["P20", "P25 operational", "P30"],
                "Highlighted trees": [count_p20, count_p25, count_p30],
            }
        )
        st.plotly_chart(
            px.bar(
                stability_df,
                x="Threshold",
                y="Highlighted trees",
                text="Highlighted trees",
                title="Trees retained under nearby threshold choices",
            ),
            use_container_width=True,
        )

        with st.expander("Show exact P20 / P25 / P30 threshold values", expanded=False):
            st.dataframe(
                threshold_table_for_view(view),
                hide_index=True,
                use_container_width=True,
            )

        # ----------------------------------------------------------
        # B. Mann–Whitney U + rank-biserial effect size
        # ----------------------------------------------------------
        st.subheader("Highlighted vs comparison trees")

        metrics = [m for m in scenario_metrics(view) if m in gdf.columns]
        stats_df = comparison_statistics(
            gdf,
            current_mask,
            reference_mask,
            metrics,
        )

        if not current_mask.any():
            st.info("No highlighted trees are available for the selected scenario.")
        elif not reference_mask.any():
            st.info("No valid comparison trees are available for the selected scenario.")
        elif stats_df.empty:
            st.info("Not enough valid observations for statistical comparison.")
        else:
            simple_stats = stats_df.copy()

            rename_map = {
                "Metric": "Indicator",
                "Target n": "Highlighted n",
                "Reference n": "Comparison n",
                "Target median": "Highlighted median",
                "Reference median": "Comparison median",
                "Median difference": "Median difference",
                "Mann–Whitney p": "Mann–Whitney p",
                "Rank-biserial": "Rank-biserial effect",
            }
            simple_stats = simple_stats.rename(columns=rename_map)

            keep_cols = [
                c for c in [
                    "Indicator",
                    "Highlighted n",
                    "Comparison n",
                    "Highlighted median",
                    "Comparison median",
                    "Median difference",
                    "Mann–Whitney p",
                    "Rank-biserial effect",
                ]
                if c in simple_stats.columns
            ]

            # Only a few summary numbers above the complete compact table.
            n_tested = len(simple_stats)
            if "Mann–Whitney p" in simple_stats.columns:
                pvals = pd.to_numeric(
                    simple_stats["Mann–Whitney p"], errors="coerce"
                )
                n_p005 = int((pvals < 0.05).sum())
            else:
                n_p005 = 0

            effect_vals = (
                pd.to_numeric(
                    simple_stats.get("Rank-biserial effect"),
                    errors="coerce",
                ).abs()
                if "Rank-biserial effect" in simple_stats.columns
                else pd.Series(dtype=float)
            )
            median_abs_effect = (
                float(effect_vals.median())
                if len(effect_vals.dropna())
                else np.nan
            )

            a, b, c = st.columns(3)
            a.metric("Indicators tested", n_tested)
            b.metric("p < 0.05", n_p005)
            c.metric(
                "Median |effect size|",
                f"{median_abs_effect:.3f}"
                if np.isfinite(median_abs_effect)
                else "NA",
            )

            st.dataframe(
                simple_stats[keep_cols].round(5),
                hide_index=True,
                use_container_width=True,
            )

            st.caption(
                "Mann–Whitney p tests whether the highlighted and comparison "
                "distributions differ. Rank-biserial effect size shows the "
                "magnitude and direction of that separation. These statistics "
                "support group differentiation; they do not establish biological causality."
            )


with tab_methods:
    st.header("How the system decides")
    st.markdown("""
    **Water anomaly** — WBI is one direct block; NDMI2 + NDSI-RWC form one correlated SWIR block; PRI is additional support only.  
    **Biochemical anomaly** — NDRE + CI red-edge + REP form the chlorophyll/red-edge block; PSRI + SIPI form the pigment/senescence block; ARI1 is additional support.  
    **Low canopy stature** — shown only when LAS H-P95 and raster CHM P95 both satisfy their orchard-relative low-stature conditions. This is a conservative two-measurement structural rule.  
    **Cross-domain integration** — Water + Biochemical is shown as a two-domain combination, while Water + Biochemical + Low Canopy Stature is the three-domain combination. Raw indices are never counted as extra domains.

**Spectral validation display** — VNIR and SWIR consensus curves are descriptive validation layers only. SWIR water-rule bands are shown at their exact Headwall centres; spectral curves do not create additional classification votes.
    """)

    st.subheader("Validation framework")
    st.write(
        "Validation has two parts. First, five displayed ground spectra are compared with their corresponding UAV tree spectra using Pearson correlation, spectral angle (SAM) and RMSE. "
        "Second, internal statistical robustness is checked using P20/P25/P30 threshold sensitivity with Jaccard overlap, and Mann–Whitney U with rank-biserial effect size for highlighted versus comparison trees. "
        "Detailed index checks remain available only on demand."
    )

    st.subheader("Operational thresholds")
    st.info("These are frozen **P25-derived orchard-relative thresholds** calculated from the complete 386-tree reference population using the harmonized final estimators. They are operational screening thresholds for this dataset, not universal citrus hard thresholds.")
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
        ["Structure", "LAS H-P95", f"≤ {STRUCTURE_THRESHOLDS['H_P95_P25_M']:.3f} m", "Required structural condition"],
        ["Structure", "Raster CHM P95", f"≤ {STRUCTURE_THRESHOLDS['RASTER_CHM_P95_P25_M']:.3f} m", "Required corroborating structural condition"],
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Domain", "Indicator", "Operational condition", "Role"]), hide_index=True, use_container_width=True)

    st.subheader("Upstream quality control")
    st.write("Quality-control and missing-data checks remain active in the scientific processing chain, but the public map presents only supported findings that pass the finalized rule framework.")
    st.info("For Structure, the public map deliberately uses the stricter LAS + raster agreement rule. This increases confidence in the displayed low-stature set but can exclude some genuine low-stature trees when the coarser raster does not agree; it is therefore a conservative supported-evidence view, not a complete census of every possible low tree.")
    st.info("A structurally unassessable tree is retained as data-limited and is never treated as evidence of normal stature or as a negative structural result.")
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

