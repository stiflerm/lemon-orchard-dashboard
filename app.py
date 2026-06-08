import zipfile
import tempfile
import os
import streamlit as st
import geopandas as gpd
import folium
import streamlit.components.v1 as components
import json
import os
import base64
import io
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
import matplotlib.pyplot as plt

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
    gdf['Flag_D'] = (gdf['NDVI_mn'] > ndvi_75) & (gdf['LAI_mn'] < lai_25) & (gdf['Radius_m'] < radius_25)
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

scenario_dict = {
    'Flag_A': ('A: Target Irrigation (Drought)', 'blue', 'Focuses on canopies with low water absorption but stable physical structure.'),
    'Flag_B': ('B: Target Fertilizer (Hidden Hunger)', 'purple', 'Identifies physically large canopies with low chlorophyll/nitrogen concentration.'),
    'Flag_C': ('C: Inspect Root Rot (Decline)', 'red', 'Flags mature trees exhibiting systemic thinning and active leaf breakdown.'),
    'Flag_D': ('D: False Positives (Weeds/Interference)', 'orange', 'Identifies small, dense polygons that are likely inter-row weeds rather than true trees.'),
    'Flag_E': ('E: Spot-Spray (Localized Pests)', 'darkred', 'Finds trees with extreme internal variance indicating localized damage on specific branches.'),
    'Flag_F': ('F: Acute Heat/Frost Shock', 'cyan', 'Detects pre-visual shock via PRI drop while structure and hydration remain stable.'),
    'Flag_G': ('G: Harvest Signal (Fruit/Flowers)', 'gold', 'Highlights canopies showing heavy fruit load or blooming via carotenoid spikes.'),
    'Flag_H': ('H: Soil Salinity / Osmotic Stress', 'brown', 'Identifies secondary chlorosis and dehydration caused by salt accumulation.'),
    'Flag_I': ('I: Pruning Verified', 'green', 'Confirms healthy, vigorous canopies that have recently reduced in physical volume.'),
    'Flag_J': ('J: Trunk Weeds (Young Trees)', 'magenta', 'Flags young saplings with artificially high health scores due to surrounding weed competition.')
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

# --- 4. MAIN LAYOUT ---
col1, col2 = st.columns([3, 1])

with col1:
    toggle_col1, toggle_col2 = st.columns(2)
    with toggle_col1:
        show_lai = st.checkbox("Load LAI UAV Overlay", value=True)
    with toggle_col2:
        show_canopies = st.checkbox("Show Targeted Canopies", value=True)
    
    center_y = gdf.geometry.centroid.y.mean()
    center_x = gdf.geometry.centroid.x.mean()
    
    m = folium.Map(location=[center_y, center_x], zoom_start=18, max_zoom=22, tiles='CartoDB dark_matter')
    
    # ==========================================
    # TIFF PROCESSING PIPELINE (Multi-band)
    # ==========================================
    if show_lai:
        # Construct absolute path for the new file
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
                    name='Raw TIFF UAV Overlay (Band 4)'
                ).add_to(m)
        else:
            st.error(f"TIFF file not found at: {tiff_path}")

    # Filter targets
    target_gdf = gdf[gdf[selected_scenario] == True]
    color = scenario_dict[selected_scenario][1]

    # Render Polygons
    if show_canopies and not target_gdf.empty:
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
    
    total_trees = len(gdf)
    target_count = len(target_gdf)
    pct_block = (target_count / total_trees) * 100 if total_trees > 0 else 0
    
    st.metric(label="Targeted Trees", value=target_count, delta=f"{pct_block:.1f}% of block", delta_color="inverse")
    
    # --- NEW EXPORT TOOL ---
    st.markdown("---")
    st.header("📥 Export Targets")
    if not target_gdf.empty:
        # Convert the filtered GeoDataFrame to a GeoJSON string
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