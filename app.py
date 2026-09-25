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