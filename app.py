import zipfile
import tempfile
import os
import streamlit as st
import geopandas as gpd
import folium
import streamlit.components.v1 as components
import json
import base64
import io
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
import matplotlib.pyplot as plt

# --- MATHEMATICAL & GEOMETRIC DEPENDENCIES ---
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
        
    # Extract the zip file into the server's temporary directory
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(physical_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
        
    # Dynamically find the .shp file anywhere inside the extracted folder
    shp_file = None
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.endswith(".shp"):
                shp_file = os.path.join(root, file)
                break
                
    if not shp_file:
        st.error("No .shp file was found inside data.zip.")
        st.stop()
        
    # Read the file directly from the temporary unzipped location
    gdf = gpd.read_file(shp_file)
    gdf = gdf.drop_duplicates(subset=['tree_id'])
    
    # Calculate thresholds
    ndvi_mean = gdf['NDVI_mn'].mean()
    lai_mean = gdf['LAI_mn'].mean()
    radius_mean = gdf['Radius_m'].mean()
    psri_mean = gdf['PSRI_mn'].mean()
    wbi_25 = gdf['WBI_mn'].quantile(0.25)
    ndvi_25 = gdf['NDVI_mn'].quantile(0.25)
    mcari_25 = gdf['MCARI_mn'].quantile(0.25)
    lai_25 = gdf['LAI_mn'].quantile(0.25)
    radius_25 = gdf['Radius_m'].quantile(0.25)
    ndvi_mi_25 = gdf['NDVI_mi'].quantile(0.25)
    ndvi_sd_25 = gdf['NDVI_sd'].quantile(0.25)
    lci_25 = gdf['LCI_mn'].quantile(0.25)
    ndvi_75 = gdf['NDVI_mn'].quantile(0.75)
    psri_75 = gdf['PSRI_mn'].quantile(0.75)
    ndvi_sd_75 = gdf['NDVI_sd'].quantile(0.75)
    cri1_sd_75 = gdf['CRI1_sd'].quantile(0.75)
    cri1_75 = gdf['CRI1_mn'].quantile(0.75)
    ndvi_mx_75 = gdf['NDVI_mx'].quantile(0.75)
    pri_25 = gdf['PRI_mn'].quantile(0.25)

    # Apply Logic Flags
    gdf['Flag_A'] = (gdf['WBI_mn'] < wbi_25) & (gdf['NDVI_mn'] > ndvi_25)
    gdf['Flag_B'] = (gdf['NDVI_mn'] > ndvi_mean) & (gdf['MCARI_mn'] < mcari_25)
    gdf['Flag_C'] = (gdf['Radius_m'] > radius_mean) & (gdf['LAI_mn'] < lai_25) & (gdf['PSRI_mn'] > psri_75)
    gdf['Flag_E'] = (gdf['NDVI_mn'] > ndvi_25) & (gdf['NDVI_sd'] > ndvi_sd_75) & (gdf['CRI1_sd'] > cri1_sd_75) & (gdf['NDVI_mi'] < ndvi_mi_25)
    gdf['Flag_F'] = (gdf['PRI_mn'] < pri_25) & (gdf['LAI_mn'] > lai_25) & (gdf['WBI_mn'] > wbi_25) & (gdf['PSRI_mn'] > psri_mean)
    gdf['Flag_G'] = (gdf['NDVI_mn'] < ndvi_25) & (gdf['LAI_mn'] > lai_25) & (gdf['CRI1_mn'] > cri1_75)
    gdf['Flag_H'] = (gdf['WBI_mn'] < wbi_25) & (gdf['LCI_mn'] < lci_25) & (gdf['PSRI_mn'] > psri_75)
    gdf['Flag_I'] = (gdf['Radius_m'] < radius_mean) & (gdf['LAI_mn'] < lai_mean) & (gdf['NDVI_mn'] > ndvi_75) & (gdf['NDVI_sd'] < ndvi_sd_25)
    gdf['Flag_J'] = (gdf['Radius_m'] < radius_25) & (gdf['LAI_mn'] < lai_25) & (gdf['NDVI_mx'] > ndvi_mx_75) & (gdf['NDVI_sd'] > ndvi_sd_75)
    
    return gdf.to_crs(epsg=4326)

gdf = load_and_process_data()

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.header("Diagnostic Controls")

# Flag_D has been removed from UI dictionary execution
scenario_dict = {
    'Flag_A': ('A: Target Irrigation (Drought)', 'blue', 'Focuses on canopies with low water absorption but stable physical structure.'),
    'Flag_B': ('B: Target Fertilizer (Hidden Hunger)', 'purple', 'Identifies physically large canopies with low chlorophyll/nitrogen concentration.'),
    'Flag_C': ('C: Inspect Root Rot (Decline)', 'red', 'Flags mature trees exhibiting systemic thinning and active leaf breakdown.'),
    'Flag_E': ('E: Spot-Spray (Localized Pests)', 'darkred', 'Finds trees with extreme internal variance indicating localized damage on specific branches.'),
    'Flag_F': ('F: Acute Heat/Frost Shock', 'cyan', 'Detects pre-visual shock via PRI drop while structure and hydration remain stable.'),
    'Flag_G': ('G: Harvest Signal (Fruit/Flowers)', 'gold', 'Highlights canopies showing heavy fruit load or blooming via carotenoid spikes.'),
    'Flag_H': ('H: Soil Salinity / Osmotic Stress', 'brown', 'Identifies secondary chlorosis and dehydration caused by salt accumulation.'),
    'Flag_I': ('I: Pruning Verified', 'green', 'Confirms healthy, vigorous canopies that have recently reduced in physical volume.'),
    'Flag_J': ('J: Trunk Weeds (Young Trees)', 'magenta', 'Flags young saplings with artificially high health scores due to surrounding weed competition.'),
    'GAP_ANALYSIS': ('🍋 Geometric Gap & Yield Analysis', 'red', 'Calculates missing trees and yield loss percentage based on spatial canopy architecture.')
}

selected_scenario = st.sidebar.selectbox(
    "Select Target Scenario", 
    options=list(scenario_dict.keys()), 
    format_func=lambda x: scenario_dict[x][0]
)

st.sidebar.markdown("---")
st.sidebar.header("Temporal Analysis")
st.sidebar.info("Currently viewing static baseline flight.")
st.sidebar.slider("Select Flight Date", min_value=1, max_value=2, value=1, disabled=True)


# --- 4. PRE-PROCESSING FOR INTEGRATED SCENARIOS ---
if selected_scenario == 'GAP_ANALYSIS':
    # Copy baseline data and project to local metric zone for calculation
    gap_calc_gdf = gdf.copy()
    utm_crs = gap_calc_gdf.estimate_utm_crs()
    gap_calc_gdf = gap_calc_gdf.to_crs(utm_crs)

    # Extract centroids and compute pivot center
    gap_calc_gdf['centroid'] = gap_calc_gdf.geometry.centroid
    raw_x = gap_calc_gdf['centroid'].apply(lambda p: p.x)
    raw_y = gap_calc_gdf['centroid'].apply(lambda p: p.y)
    mean_x, mean_y = raw_x.mean(), raw_y.mean()

    def rotate_coords(x, y, angle_deg):
        x_centered = x - mean_x
        y_centered = y - mean_y
        angle_rad = math.radians(angle_deg)
        rx = x_centered * math.cos(angle_rad) + y_centered * math.sin(angle_rad)
        ry = -x_centered * math.sin(angle_rad) + y_centered * math.cos(angle_rad)
        return rx, ry

    def unrotate_coords(rx, ry, angle_deg):
        angle_rad = math.radians(angle_deg)
        x = rx * math.cos(angle_rad) - ry * math.sin(angle_rad)
        y = rx * math.sin(angle_rad) + ry * math.cos(angle_rad)
        return x + mean_x, y + mean_y

    # Calibrated parameters
    expected_tree_spacing = 5.5      
    row_distance_threshold = 2.5     
    grid_angle_degrees = 75.0        
    max_empty_space_m = 20.0         

    # Rotate coordinates
    rotated_coords = [rotate_coords(x, y, grid_angle_degrees) for x, y in zip(raw_x, raw_y)]
    gap_calc_gdf['x'] = [c[0] for c in rotated_coords]  
    gap_calc_gdf['y'] = [c[1] for c in rotated_coords]  

    # Delineate rows using clustering
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=row_distance_threshold, linkage='average' 
    )
    y_matrix = np.array(gap_calc_gdf['y'].tolist()).reshape(-1, 1)
    gap_calc_gdf['Row_ID'] = clustering.fit_predict(y_matrix)

    row_centers = gap_calc_gdf.groupby('Row_ID')['y'].mean().to_dict()
    gap_calc_gdf['Row_Center_Y'] = gap_calc_gdf['Row_ID'].map(row_centers)

    # Calculate gaps
    gaps = []
    for row_id, group in gap_calc_gdf.groupby('Row_ID'):
        group = group.sort_values(by='x').reset_index(drop=True)
        for i in range(len(group) - 1):
            tree_A = group.iloc[i]
            tree_B = group.iloc[i + 1]
            dist = tree_B['x'] - tree_A['x']
            
            if (expected_tree_spacing * 1.5) < dist <= max_empty_space_m:
                missing_count = int(np.round(dist / expected_tree_spacing)) - 1
                for j in range(1, missing_count + 1):
                    gap_x_rotated = tree_A['x'] + (j * (dist / (missing_count + 1)))
                    gap_y_rotated = tree_A['Row_Center_Y']
                    real_x, real_y = unrotate_coords(gap_x_rotated, gap_y_rotated, grid_angle_degrees)
                    gaps.append(Point(real_x, real_y))

    gaps_gdf = gpd.GeoDataFrame(geometry=gaps, crs=gap_calc_gdf.crs)
    # CRITICAL: Transform points back to EPSG:4326 for unified Folium visualization
    gaps_folium_gdf = gaps_gdf.to_crs(epsg=4326)

    total_trees = len(gap_calc_gdf)
    total_gaps = len(gaps_folium_gdf)
    ideal_capacity = total_trees + total_gaps
    yield_loss_percentage = (total_gaps / ideal_capacity) * 100 if ideal_capacity > 0 else 0.0
else:
    target_gdf = gdf[gdf[selected_scenario] == True]
    color = scenario_dict[selected_scenario][1]


# --- 5. MAIN VISUALIZATION LAYOUT ---
col1, col2 = st.columns([3, 1])

with col1:
    toggle_col1, toggle_col2 = st.columns(2)
    with toggle_col1:
        show_lai = st.checkbox("Load LAI UAV Overlay", value=True)
    with toggle_col2:
        show_canopies = st.checkbox("Show Map Vector Overlay", value=True)
    
    center_y = gdf.geometry.centroid.y.mean()
    center_x = gdf.geometry.centroid.x.mean()
    
    m = folium.Map(location=[center_y, center_x], zoom_start=18, max_zoom=22, tiles='CartoDB dark_matter')
    
    # Base Raster Ingestion (LAI Overlay Shared Across All Scenarios)
    if show_lai:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        tiff_path = os.path.join(current_dir, "data", "LAI_1.tif")
        
        if os.path.exists(tiff_path):
            with rasterio.open(tiff_path) as src:
                minx, miny, maxx, maxy = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
                image_bounds = [[miny, minx], [maxy, maxx]]
                
                scale = min(1.0, 1500.0 / src.width)
                out_shape = (int(src.height * scale), int(src.width * scale))
                
                lai_data = src.read(1, out_shape=out_shape, resampling=rasterio.enums.Resampling.nearest)
                lai_data = np.nan_to_num(lai_data, nan=-9999.0)
                
                nodata = src.nodata if src.nodata is not None else -9999.0
                masked_data = np.ma.masked_where((lai_data == nodata) | (lai_data <= 0.1), lai_data)
                
                valid_pixels = masked_data.compressed()
                if len(valid_pixels) > 0:
                    vmin_val, vmax_val = np.percentile(valid_pixels, [5, 95])
                else:
                    vmin_val, vmax_val = 0.0, 3.0
                    
                cmap = plt.cm.RdYlGn
                norm = plt.Normalize(vmin=vmin_val, vmax=vmax_val)
                colored_image = cmap(norm(masked_data))
                colored_image = (colored_image * 255).astype(np.uint8)
                colored_image[..., 3] = np.where(masked_data.mask, 0, 255) 
                
                img_buffer = io.BytesIO()
                plt.imsave(img_buffer, colored_image, format='png')
                img_buffer.seek(0)
                encoded_string = base64.b64encode(img_buffer.read()).decode()
                image_url = f"data:image/png;base64,{encoded_string}"
                
                folium.raster_layers.ImageOverlay(
                    image=image_url,
                    bounds=image_bounds,
                    opacity=0.9,
                    name='LAI UAV Index Map'
                ).add_to(m)
        else:
            st.error(f"TIFF file not found at: {tiff_path}")

    # Vector Overlay Mapping
    if show_canopies:
        if selected_scenario == 'GAP_ANALYSIS':
            # Render clean, open canopy boundaries to verify alignment on the raster
            folium.GeoJson(
                gdf,
                style_function=lambda x: {
                    'fillColor': 'none',
                    'color': '#00FFCC', 
                    'weight': 1.5,
                    'fillOpacity': 0.0
                },
                name="Orchard Canopies"
            ).add_to(m)

            # Overlay individual gap points directly onto the Folium/LAI map layout
            if not gaps_folium_gdf.empty:
                for idx, row in gaps_folium_gdf.iterrows():
                    folium.CircleMarker(
                        location=[row.geometry.y, row.geometry.x],
                        radius=5,
                        color='#FF0033',
                        weight=2.0,
                        fill=True,
                        fill_color='#FFFFFF',
                        fill_opacity=1.0,
                        tooltip=f"Calculated Crop Gap (Row Center Location)"
                    ).add_to(m)
        else:
            # Standard Scenario Vector Logic
            if not target_gdf.empty:
                target_json = target_gdf.copy()
                tooltip = folium.GeoJsonTooltip(
                    fields=['tree_id', 'NDVI_mn', 'WBI_mn', 'LAI_mn', 'MCARI_mn'],
                    aliases=['Tree ID:', 'NDVI:', 'WBI:', 'LAI:', 'MCARI:'],
                    localize=True
                )
                folium.GeoJson(
                    target_json,
                    style_function=lambda x: {
                        'fillColor': color,
                        'color': 'white', 
                        'weight': 2.0,
                        'fillOpacity': 0.7
                    },
                    name="Targeted Trees",
                    tooltip=tooltip
                ).add_to(m)

    folium.LayerControl().add_to(m)
    components.html(m._repr_html_(), height=650)

with col2:
    st.header("Scenario Details")
    st.subheader(scenario_dict[selected_scenario][0])
    st.write(scenario_dict[selected_scenario][2])
    
    if selected_scenario == 'GAP_ANALYSIS':
        st.metric(label="Total Orchard Trees", value=total_trees)
        st.metric(label="Calculated Crop Gaps", value=total_gaps)
        st.metric(label="Total Yield Loss", value=f"{yield_loss_percentage:.2f}%")
        
        st.markdown("---")
        st.header("📥 Export Gaps")
        if not gaps_folium_gdf.empty:
            geojson_data = gaps_folium_gdf.to_json()
            st.download_button(
                label=f"Download {total_gaps} Gaps (GeoJSON)",
                data=geojson_data,
                file_name="calculated_orchard_gaps.geojson",
                mime="application/geo+json",
                help="Download missing tree point coordinates for precision field replanting operations."
            )
    else:
        total_trees = len(gdf)
        target_count = len(target_gdf)
        pct_block = (target_count / total_trees) * 100 if total_trees > 0 else 0
        
        st.metric(label="Targeted Trees", value=target_count, delta=f"{pct_block:.1f}% of block", delta_color="inverse")
        
        st.markdown("---")
        st.header("📥 Export Targets")
        if not target_gdf.empty:
            geojson_data = target_gdf.to_json()
            st.download_button(
                label=f"Download {target_count} Targets (GeoJSON)",
                data=geojson_data,
                file_name=f"field_targets_{selected_scenario}.geojson",
                mime="application/geo+json",
                help="Download these specific tree polygons for use in QGIS, Google Earth, or mobile field apps."
            )
        else:
            st.info("No targets found in this scenario to export.")
        
        st.markdown("---")
        st.header("🤖 Agentic Insights")
        st.warning("LLM Backend Disconnected")
        st.write(f"*Mock Insight generated for {scenario_dict[selected_scenario][0]}...*")
        st.info(f"Analysis indicates {target_count} targets. Spatial clustering detected in the primary zones. Recommend immediate field verification within 48 hours to validate findings.")