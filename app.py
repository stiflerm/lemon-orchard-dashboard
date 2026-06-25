import zipfile
import tempfile
import os
import streamlit as st
import geopandas as gpd
import folium
import streamlit.components.v1 as components
import base64
import io
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
import matplotlib.pyplot as plt
import math
from shapely.geometry import Point
from sklearn.cluster import AgglomerativeClustering
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Orchard Diagnostic Intelligence", layout="wide")
st.title("🍋 Orchard Diagnostic Intelligence Platform")

# --- 2. DATA INGESTION & CACHING ---
@st.cache_data
def load_and_process_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    physical_path = os.path.join(current_dir, "data", "data.zip")
    
    if not os.path.exists(physical_path):
        st.error(f"File not found on server: {physical_path}")
        st.stop()
        
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(physical_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
        
    shp_file = None
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.endswith(".shp"):
                shp_file = os.path.join(root, file)
                break
                
    if not shp_file:
        st.error("No .shp file was found inside data.zip.")
        st.stop()
        
    gdf = gpd.read_file(shp_file).drop_duplicates(subset=['tree_id'])
    
    # Standard 2D Statistical Thresholds
    ndvi_mean = gdf['NDVI_mn'].mean()
    lai_mean = gdf['LAI_mn'].mean()
    radius_mean = gdf['Radius_m'].mean()
    psri_mean = gdf['PSRI_mn'].mean()
    
    wbi_25 = gdf['WBI_mn'].quantile(0.25)
    ndvi_25 = gdf['NDVI_mn'].quantile(0.25)
    mcari_25 = gdf['MCARI_mn'].quantile(0.25)
    lai_25 = gdf['LAI_mn'].quantile(0.25)
    psri_75 = gdf['PSRI_mn'].quantile(0.75)
    ndvi_mi_25 = gdf['NDVI_mi'].quantile(0.25)
    ndvi_sd_75 = gdf['NDVI_sd'].quantile(0.75)
    cri1_sd_75 = gdf['CRI1_sd'].quantile(0.75)
    pri_25 = gdf['PRI_mn'].quantile(0.25)
    
    # 3D CHM Thresholds
    if 'CHM_max' in gdf.columns:
        chm_max_25 = gdf['CHM_max'].quantile(0.25)
    else:
        gdf['CHM_max'] = 1.0 
        chm_max_25 = 0.5

    # Apply Logic Flags
    # Flag A: WBI vs NDVI
    gdf['Flag_A'] = (gdf['WBI_mn'] < wbi_25) & (gdf['NDVI_mn'] > ndvi_25) 
    # Flag B: NDVI vs MCARI
    gdf['Flag_B'] = (gdf['NDVI_mn'] > ndvi_mean) & (gdf['MCARI_mn'] < mcari_25) 
    # Flag C: Radius vs LAI vs PSRI (Reverted to 2D Logic)
    gdf['Flag_C'] = (gdf['Radius_m'] > radius_mean) & (gdf['LAI_mn'] < lai_25) & (gdf['PSRI_mn'] > psri_75)
    # Flag E: Variance Logic
    gdf['Flag_D'] = (gdf['NDVI_mn'] > ndvi_25) & (gdf['NDVI_sd'] > ndvi_sd_75) & (gdf['CRI1_sd'] > cri1_sd_75) & (gdf['NDVI_mi'] < ndvi_mi_25) 
    # Flag F: PRI Drop
    gdf['Flag_F'] = (gdf['PRI_mn'] < pri_25) & (gdf['LAI_mn'] > lai_25) & (gdf['WBI_mn'] > wbi_25) & (gdf['PSRI_mn'] > psri_mean) 
    # Flag H: NEW - CHM Stunted Growth Logic
    gdf['Flag_H'] = (gdf['CHM_max'] < chm_max_25) & (gdf['NDVI_mn'] > ndvi_mean)
    
    return gdf.to_crs(epsg=4326)

gdf = load_and_process_data()

# --- 3. SPECTRAL PLOTTING ENGINE ---
def plot_spectral_signature(target_gdf, all_gdf):
    bands = ['Blue_mn', 'Green_mn', 'Red_mn', 'RedEdge_mn', 'NIR_mn']
    wavelengths = [450, 560, 650, 730, 840] 
    
    available_bands = [b for b in bands if b in all_gdf.columns]
    if len(available_bands) != len(bands):
        st.warning("⚠️ **Spectral Bands Missing:** To view spectral charts, ensure your shapefile contains the exact columns: `Blue_mn`, `Green_mn`, `Red_mn`, `RedEdge_mn`, `NIR_mn`.")
        return None
        
    baseline_spectra = all_gdf[bands].mean().values
    target_spectra = target_gdf[bands].mean().values
    
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(wavelengths, baseline_spectra, color='lightgreen', linestyle='--', label='Healthy Baseline', linewidth=2)
    ax.plot(wavelengths, target_spectra, color='red', marker='o', label='Flagged Targets', linewidth=2)
    ax.set_title("Spectral Signature Verification", fontsize=10)
    ax.set_xlabel("Wavelength (nm)", fontsize=8)
    ax.set_ylabel("Reflectance", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    return fig

# --- 4. SIDEBAR CONTROLS ---
st.sidebar.header("Diagnostic Controls")

scenario_dict = {
    'Flag_A': (
        'A: Target Irrigation (Drought)', 'blue', 'Focuses on canopies with low water absorption but stable physical structure.',
        '''**1. Inputs Used:** WBI_min (Water Band Index Minimum), NDVI_mean.
**2. Purpose:** WBI monitors plant water status; NDVI_mean tracks chlorophyll-driven vigor.
**3. Scientific Conclusion:** Targets trees with minimal water status but high chlorophyll vigor. This identifies "early-stage physiological drought" where the tree maintains greenness but has depleted its internal water reserves, signaling the need for immediate irrigation before irreversible wilting occurs.'''
    ),
    'Flag_B': (
        'B: Target Fertilizer (Hidden Hunger)', 'purple', 'Identifies physically large canopies with low chlorophyll/nitrogen concentration.',
        '''**1. Inputs Used:** NDVI_mean, MCARI_min (Modified Chlorophyll Absorption).
**2. Purpose:** NDVI_mean captures total biomass; MCARI_min isolates chlorophyll concentration from structural interference.
**3. Scientific Conclusion:** Targets trees with high biomass but minimal chlorophyll. This mismatch indicates "hidden hunger" where trees are structurally mature but nitrogen-deficient, requiring variable-rate fertilization.'''
    ),
    'Flag_C': (
        'C: Inspect Root Rot (Decline)', 'red', 'Flags mature trees exhibiting systemic thinning and active leaf breakdown.',
        '''**1. Inputs Used:** Radius_m, LAI_mean, PSRI_max (Plant Senescence Reflectance).
**2. Purpose:** Radius tracks growth stage; LAI monitors canopy leaf area; PSRI_max captures stress-induced pigment changes.
**3. Scientific Conclusion:** Targets mature trees (Radius) with low leaf area and high senescence (PSRI). This signature is indicative of root-zone decay, where reduced nutrient uptake forces leaf shedding and cellular senescence.'''
    ),
    'Flag_D': (
        'E: Spot-Spray (Localized Pests)', 'darkred', 'Identifies asymmetric intra-canopy stress signature.',
        '''**1. Inputs Used:** NDVI_mean, NDVI_std (Standard Deviation), NDVI_min, CRI1_std (Carotenoid Variance).
**2. Purpose:** NDVI_min detects local pockets of necrosis; standard deviation metrics quantify internal spectral variance.
**3. Scientific Conclusion:** Targets trees with high average health but high internal variance. This asymmetry—where one sector of the canopy is healthy while another exhibits necrosis—is the pathognomonic spectral signature of localized pest or pathogen infestation.'''
    ),
    'Flag_F': (
        'F: Acute Heat/Frost Shock', 'cyan', 'Detects sudden photosynthetic shutdown via PRI drop.',
        '''**1. Inputs Used:** PRI_min (Photochemical Reflectance Index), LAI_mean, WBI_mean, PSRI_mean.
**2. Purpose:** PRI captures light-use efficiency; other indices ensure structure/water levels remain nominal.
**3. Scientific Conclusion:** Targets trees with high structure/water but suppressed PRI. This indicates the light-harvesting mechanism (xanthophyll cycle) has shut down due to acute thermal stress, despite the canopy appearing physically intact.'''
    ),
    'Flag_H': (
        'H: Stunted Growth (Height Anomaly)', 'magenta', 'Isolates trees with limited vertical mass vs. high vigor.',
        '''**1. Inputs Used:** CHM_max (Canopy Height Model Peak), NDVI_mean.
**2. Purpose:** CHM_max measures structural height; NDVI_mean confirms vegetative metabolism.
**3. Scientific Conclusion:** Targets trees with high photosynthetic metabolism but bottom-tier structural height. This suggests localized soil compaction (hardpan) or root restriction preventing vertical development, regardless of top-tier health.'''
    ),
    'GAP_ANALYSIS': (
        '🍋 Geometric Gap & Yield Analysis', 'red', 'Calculates missing trees and yield loss.',
        '''**1. Inputs Used:** Geometric Centroids, X/Y Proximity.
**2. Purpose:** Measures deviation from standardized 5.5m orchard planting grids.
**3. Scientific Conclusion:** Extrapolates gaps where spatial distance exceeds planting protocol thresholds, accounting for geometric row alignment to verify precise yield loss locations.'''
    )
}

selected_scenario = st.sidebar.selectbox("Select Target Scenario", options=list(scenario_dict.keys()), format_func=lambda x: scenario_dict[x][0])
st.sidebar.markdown("---")
st.sidebar.header("Temporal Analysis")
st.sidebar.info("Currently viewing static baseline flight.")

# --- 5. PRE-PROCESSING ---
if selected_scenario == 'GAP_ANALYSIS':
    gap_calc_gdf = gdf.copy()
    utm_crs = gap_calc_gdf.estimate_utm_crs()
    gap_calc_gdf = gap_calc_gdf.to_crs(utm_crs)

    gap_calc_gdf['centroid'] = gap_calc_gdf.geometry.centroid
    raw_x = gap_calc_gdf['centroid'].apply(lambda p: p.x)
    raw_y = gap_calc_gdf['centroid'].apply(lambda p: p.y)
    mean_x, mean_y = raw_x.mean(), raw_y.mean()

    def rotate_coords(x, y, angle_deg):
        rx = (x - mean_x) * math.cos(math.radians(angle_deg)) + (y - mean_y) * math.sin(math.radians(angle_deg))
        ry = -(x - mean_x) * math.sin(math.radians(angle_deg)) + (y - mean_y) * math.cos(math.radians(angle_deg))
        return rx, ry

    def unrotate_coords(rx, ry, angle_deg):
        x = rx * math.cos(math.radians(angle_deg)) - ry * math.sin(math.radians(angle_deg))
        y = rx * math.sin(math.radians(angle_deg)) + ry * math.cos(math.radians(angle_deg))
        return x + mean_x, y + mean_y

    expected_tree_spacing, row_distance_threshold, grid_angle_degrees, max_empty_space_m = 5.5, 2.5, 75.0, 20.0         

    rotated_coords = [rotate_coords(x, y, grid_angle_degrees) for x, y in zip(raw_x, raw_y)]
    gap_calc_gdf['x'], gap_calc_gdf['y'] = [c[0] for c in rotated_coords], [c[1] for c in rotated_coords]

    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=row_distance_threshold, linkage='average')
    gap_calc_gdf['Row_ID'] = clustering.fit_predict(np.array(gap_calc_gdf['y'].tolist()).reshape(-1, 1))
    
    row_centers = gap_calc_gdf.groupby('Row_ID')['y'].mean().to_dict()
    gap_calc_gdf['Row_Center_Y'] = gap_calc_gdf['Row_ID'].map(row_centers)

    gaps = []
    for row_id, group in gap_calc_gdf.groupby('Row_ID'):
        group = group.sort_values(by='x').reset_index(drop=True)
        for i in range(len(group) - 1):
            tree_A, tree_B = group.iloc[i], group.iloc[i + 1]
            dist = tree_B['x'] - tree_A['x']
            
            if (expected_tree_spacing * 1.5) < dist <= max_empty_space_m:
                missing_count = int(np.round(dist / expected_tree_spacing)) - 1
                for j in range(1, missing_count + 1):
                    gap_x_rotated = tree_A['x'] + (j * (dist / (missing_count + 1)))
                    real_x, real_y = unrotate_coords(gap_x_rotated, tree_A['Row_Center_Y'], grid_angle_degrees)
                    gaps.append(Point(real_x, real_y))

    gaps_gdf = gpd.GeoDataFrame(geometry=gaps, crs=gap_calc_gdf.crs)
    
    # 2D Collision Filter (Prevents overlapping points without relying on CHM)
    tree_buffers = gap_calc_gdf.geometry.buffer(2.0).unary_union 
    gaps_gdf = gaps_gdf[~gaps_gdf.intersects(tree_buffers)]
    
    gaps_folium_gdf = gaps_gdf.to_crs(epsg=4326)
    total_trees, total_gaps = len(gap_calc_gdf), len(gaps_folium_gdf)
    ideal_capacity = total_trees + total_gaps
    yield_loss_percentage = (total_gaps / ideal_capacity) * 100 if ideal_capacity > 0 else 0.0

else:
    target_gdf = gdf[gdf[selected_scenario] == True]
    color = scenario_dict[selected_scenario][1]

# --- 6. MAIN VISUALIZATION LAYOUT ---
col1, col2 = st.columns([3, 1])

with col1:
    toggle_col1, toggle_col2 = st.columns(2)
    with toggle_col1:
        show_lai = st.checkbox("Load LAI UAV Overlay", value=True)
    with toggle_col2:
        show_canopies = st.checkbox("Show Map Vector Overlay", value=True)
    
    m = folium.Map(location=[gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()], zoom_start=18, max_zoom=22, tiles='CartoDB dark_matter')
    
    if show_lai:
        tiff_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "LAI_1.tif")
        if os.path.exists(tiff_path):
            with rasterio.open(tiff_path) as src:
                minx, miny, maxx, maxy = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
                scale = min(1.0, 1500.0 / src.width)
                lai_data = src.read(1, out_shape=(int(src.height * scale), int(src.width * scale)), resampling=rasterio.enums.Resampling.nearest)
                lai_data = np.nan_to_num(lai_data, nan=-9999.0)
                masked_data = np.ma.masked_where((lai_data == (src.nodata or -9999.0)) | (lai_data <= 0.1), lai_data)
                
                vmin_val, vmax_val = np.percentile(masked_data.compressed(), [5, 95]) if len(masked_data.compressed()) > 0 else (0.0, 3.0)
                colored_image = (plt.cm.RdYlGn(plt.Normalize(vmin=vmin_val, vmax=vmax_val)(masked_data)) * 255).astype(np.uint8)
                colored_image[..., 3] = np.where(masked_data.mask, 0, 255) 
                
                img_buffer = io.BytesIO()
                plt.imsave(img_buffer, colored_image, format='png')
                img_buffer.seek(0)
                
                folium.raster_layers.ImageOverlay(
                    image=f"data:image/png;base64,{base64.b64encode(img_buffer.read()).decode()}",
                    bounds=[[miny, minx], [maxy, maxx]], opacity=0.9, name='LAI UAV Index Map'
                ).add_to(m)

    if show_canopies:
        if selected_scenario == 'GAP_ANALYSIS':
            folium.GeoJson(gdf, style_function=lambda x: {'fillColor': 'none', 'color': '#00FFCC', 'weight': 1.5, 'fillOpacity': 0.0}, name="Orchard Canopies").add_to(m)
            for idx, row in gaps_folium_gdf.iterrows():
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x], radius=5, color='#000000', weight=2.0, fill=True, fill_color='#FFFFFF', fill_opacity=1.0, tooltip="Calculated Crop Gap"
                ).add_to(m)
        else:
            if not target_gdf.empty:
                folium.GeoJson(target_gdf, style_function=lambda x: {'fillColor': color, 'color': 'white', 'weight': 2.0, 'fillOpacity': 0.7}, tooltip=folium.GeoJsonTooltip(fields=['tree_id', 'NDVI_mn'], aliases=['Tree ID:', 'NDVI:'])).add_to(m)

    folium.LayerControl().add_to(m)
    components.html(m._repr_html_(), height=650)

with col2:
    st.header("Scenario Details")
    st.subheader(scenario_dict[selected_scenario][0])
    st.write(scenario_dict[selected_scenario][2])
    
    with st.expander("🔬 View Scientific Rationale"):
        st.markdown(scenario_dict[selected_scenario][3])
    
    if selected_scenario == 'GAP_ANALYSIS':
        st.metric(label="Total Orchard Trees", value=total_trees)
        st.metric(label="Calculated Crop Gaps", value=total_gaps)
        st.metric(label="Total Yield Loss", value=f"{yield_loss_percentage:.2f}%")
        
        st.markdown("---")
        st.header("📥 Export Gaps")
        if not gaps_folium_gdf.empty:
            st.download_button(label=f"Download {total_gaps} Gaps (GeoJSON)", data=gaps_folium_gdf.to_json(), file_name="calculated_orchard_gaps.geojson", mime="application/geo+json")
    else:
        target_count, total_trees = len(target_gdf), len(gdf)
        st.metric(label="Targeted Trees", value=target_count, delta=f"{(target_count / total_trees * 100) if total_trees > 0 else 0:.1f}% of block", delta_color="inverse")
            
        if target_count > 0:
            fig = plot_spectral_signature(target_gdf, gdf)
            if fig is not None:
                st.pyplot(fig)
        
        st.markdown("---")
        st.header("📥 Export Targets")
        if not target_gdf.empty:
            st.download_button(label=f"Download {target_count} Targets (GeoJSON)", data=target_gdf.to_json(), file_name=f"field_targets_{selected_scenario}.geojson", mime="application/geo+json")