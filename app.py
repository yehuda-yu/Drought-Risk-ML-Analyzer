import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
import rasterio
from rasterio.io import MemoryFile
from rasterio.windows import Window
from rasterio.plot import plotting_extent
import pickle
import os
from matplotlib.transforms import Bbox
import cartopy.crs as ccrs
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as PathEffects
import geemap
import ee
import folium
from streamlit_folium import folium_static
from datetime import datetime, timedelta
import tempfile
import json
from streamlit.components.v1 import html

# --------------------------------------------------------------------------------
# Page and UI Configuration
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Drought Risk Assessment Application",
    page_icon="🌍",
    # layout="wide"
)

# Custom CSS for a refined UI
st.markdown("""
<style>
    .main {
        padding: 2rem;
        font-family: "Helvetica Neue", Arial, sans-serif;
        color: #2C3E50;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
    h1, h2, h3 {
        color: #2C3E50;
        margin-bottom: 1rem;
    }
    h1 {
        font-size: 2.2rem;
        margin-bottom: 1.5rem;
    }
    h2 {
        font-size: 1.8rem;
        margin-top: 2rem;
    }
    h3 {
        font-size: 1.4rem;
    }
    .stAlert {
        background-color: #f8f9fa !important;
        padding: 1rem !important;
        border-radius: 4px !important;
        margin: 1rem 0 !important;
    }
    .citation-box {
    background-color: #f2f2f2;
    border-left: 4px solid #4CAF50;
    padding: 1rem;
    margin-bottom: 2rem;
    color: #2C3E50; /* Set a dark text color here */
}

.citation-box p {
    margin: 0;
    font-size: 0.95rem;
    color: #2C3E50; /* Ensure text within p tags is also dark */
}
    .upload-instructions {
        font-size: 0.9rem;
        color: #555;
        margin-bottom: 1rem;
    }
    .satellite-selector {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# JavaScript callback for handling drawn features
draw_callback = """
<script>
document.addEventListener('DOMContentLoaded', function() {
    const map = document.querySelector('#map');
    if (map) {
        map.addEventListener('draw:created', function(e) {
            const type = e.layerType;
            const layer = e.layer;
            if (type === 'rectangle') {
                const bounds = layer.getBounds();
                const coordinates = [
                    bounds.getWest(),
                    bounds.getSouth(),
                    bounds.getEast(),
                    bounds.getNorth()
                ];
                window.Streamlit.setComponentValue({
                    'type': type,
                    'coordinates': coordinates
                });
            }
        });
    }
});
</script>
"""

# Initialize session state for drawn features if it doesn't exist
if 'drawn_features' not in st.session_state:
    st.session_state.drawn_features = None

# --------------------------------------------------------------------------------
# Model Loading and Utilities
# --------------------------------------------------------------------------------
@st.cache_resource
def load_venus_model():
    """
    Load the trained Venus satellite model and corresponding scaler from a pickle file.

    The model is a Support Vector Machine (SVM) designed for drought risk 
    assessment using multi-band Venµs satellite imagery.
    """
    model_file = 'model-svm.pkl'
    if not os.path.exists(model_file):
        st.error(f"Model file {model_file} not found.")
        return None, None
    try:
        with open(model_file, 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['scaler']
    except Exception as e:
        st.error(f"Error loading Venus model: {str(e)}")
        return None, None

@st.cache_resource
def load_sentinel2_model():
    """
    Load the trained Sentinel-2 satellite model and corresponding scaler from a pickle file.

    The model is a Support Vector Machine (SVM) designed for drought risk 
    assessment using multi-band Sentinel-2 satellite imagery.
    """
    model_file = 'S2_svm_classification_model.pkl'
    if not os.path.exists(model_file):
        st.error(f"Model file {model_file} not found.")
        return None, None
    try:
        with open(model_file, 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['scaler']
    except Exception as e:
        st.error(f"Error loading Sentinel-2 model: {str(e)}")
        return None, None

def get_rgb_image(src, satellite_type):
    """
    Extract RGB bands from the GeoTIFF file and normalize them to [0,1].
    This yields a visually interpretable composite image of the region.
    
    Parameters:
    - src: Rasterio data source
    - satellite_type: Either 'venus' or 'sentinel2' to determine appropriate bands
    """
    try:
        if satellite_type == "venus":
            # Venus bands 7, 4, 3 for RGB
            red = src.read(7).astype(np.float32)
            green = src.read(4).astype(np.float32)
            blue = src.read(3).astype(np.float32)
        else:  # sentinel2
            # Sentinel-2 bands 4, 3, 2 for RGB (common RGB mapping)
            red = src.read(4).astype(np.float32)
            green = src.read(3).astype(np.float32)
            blue = src.read(2).astype(np.float32)

        rgb = np.dstack((red, green, blue))
        rgb_min, rgb_max = np.nanmin(rgb), np.nanmax(rgb)
        if rgb_max > rgb_min:
            rgb = (rgb - rgb_min) / (rgb_max - rgb_min)
        else:
            # Fallback if no variation in pixel values
            rgb = np.zeros_like(rgb)

        return rgb
    except Exception as e:
        st.error(f"Error creating RGB image: {str(e)}")
        return None

def predict_geotiff(model, scaler, uploaded_file, satellite_type, chunk_size=256):
    """
    Predict drought risk probabilities on a given GeoTIFF using the trained model.

    Parameters:
    - model: Trained SVM model for drought risk assessment.
    - scaler: Scaler used to normalize input features.
    - uploaded_file: The uploaded GeoTIFF file.
    - satellite_type: Either 'venus' or 'sentinel2'
    - chunk_size: Size of chunks to process large images incrementally.

    Returns:
    - rgb_image: Normalized RGB composite image array.
    - probability_predictions: 2D numpy array of drought risk probabilities.
    - meta: Metadata associated with the input GeoTIFF.
    """
    try:
        with MemoryFile(uploaded_file.read()) as memfile:
            with memfile.open() as src:
                height, width, band_count = src.height, src.width, src.count

                # Get minimum required bands based on satellite type
                min_required_bands = 11 if satellite_type == "venus" else 8  # Sentinel-2 needs at least 8 bands
                
                if band_count < min_required_bands:
                    st.error(f"Image has {band_count} bands, but at least {min_required_bands} are required for {satellite_type} data.")
                    return None, None, None

                # Get RGB image
                rgb_image = get_rgb_image(src, satellite_type)
                if rgb_image is None:
                    return None, None, None

                probability_predictions = np.zeros((height, width), dtype=np.float32)

                # Progress UI
                progress_text = st.empty()
                progress_bar = st.progress(0)
                total_chunks = ((height + chunk_size - 1) // chunk_size) * ((width + chunk_size - 1) // chunk_size)
                chunk_count = 0

                # Process image in chunks to avoid memory overload
                for y in range(0, height, chunk_size):
                    for x in range(0, width, chunk_size):
                        window = Window(x, y, min(chunk_size, width - x), min(chunk_size, height - y))
                        data = src.read(window=window)

                        # For Venus: exclude the first band as per the model's expected input structure
                        # For Sentinel-2: use all bands
                        if satellite_type == "venus":
                            data = data[1:, :, :]
                        
                        features = data.reshape(data.shape[0], -1).T

                        if np.isnan(features).any() or np.isinf(features).any():
                            st.error("Invalid (NaN or infinite) values found in input data.")
                            return None, None, None

                        # Normalize features
                        features_normalized = scaler.transform(features)

                        # Get decision values and convert to probabilities
                        decision_values = model.decision_function(features_normalized)
                        probabilities = 1 / (1 + np.exp(-decision_values))
                        probabilities = probabilities.reshape((window.height, window.width))

                        probability_predictions[y:y+window.height, x:x+window.width] = probabilities

                        # Update progress
                        chunk_count += 1
                        progress = chunk_count / total_chunks
                        progress_bar.progress(progress)
                        progress_text.text(f"Processing: {progress:.1%} complete")

                progress_text.text("Processing complete!")
                return rgb_image, probability_predictions, src.meta
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None, None

def plot_predictions(rgb_image, probability_predictions, satellite_type, colormap='drought', threshold=0.5, meta=None, image_date=None):
    """
    Plotting visualizations using Matplotlib:
    1. RGB composite image.
    2. Probability map of drought risk.
    3. Statistical analysis (histogram, summary stats).
    4. Overlay: highlight areas above threshold on the RGB image.

    Parameters:
    - rgb_image: NumPy array of shape (H, W, 3) with normalized RGB data.
    - probability_predictions: 2D NumPy array of drought risk probabilities.
    - satellite_type: Either 'venus' or 'sentinel2' to display in titles
    - colormap: Colormap name or 'drought' for custom colormap.
    - threshold: Probability threshold for high-risk areas.
    - meta: Metadata associated with the GeoTIFF (for spatial reference)
    """
    from matplotlib.colors import LinearSegmentedColormap

    # Format satellite name for display
    satellite_display = "Venµs" if satellite_type == "venus" else "Sentinel-2"

    # Custom drought colormap if requested
    if colormap == 'drought':
        colors = ['#313695', '#4575B4', '#74ADD1', '#ABD9E9', '#E0F3F8',
                  '#FFFFBF', '#FEE090', '#FDAE61', '#F46D43', '#D73027', '#A50026']
        cmap = LinearSegmentedColormap.from_list("drought", colors)
    else:
        cmap = plt.get_cmap(colormap)

    # Create tabs for visualization
    tabs = st.tabs(["RGB Image", "Probability Map", "Statistical Analysis", "Overlay"])

    # Function to add professional north arrow to plots
    def add_north_arrow(ax, pos=(0.95, 0.05), size=0.1, color='#999999'):
        """Add a more professional north arrow to the plot"""
        # Main arrow
        arrow_start = (pos[0], pos[1])
        arrow_end = (pos[0], pos[1] + size)
        
        # Create arrow with custom styling
        arrow = FancyArrowPatch(
            arrow_start, arrow_end,
            transform=ax.transAxes,
            arrowstyle='-|>', 
            lw=1.2,
            mutation_scale=15, 
            color=color,
            path_effects=[PathEffects.withStroke(linewidth=2, foreground='white')]
        )
        ax.add_patch(arrow)
        
        # Add the "N" character with professional styling
        txt = ax.text(
            pos[0], pos[1] + size + 0.01, 'N', 
            transform=ax.transAxes,
            ha='center', 
            va='bottom',
            fontsize=10, 
            fontweight='bold',
            color=color,
            path_effects=[PathEffects.withStroke(linewidth=2, foreground='white')]
        )
    
    # Function to add coordinate grid to plots (simplified)
    def add_coordinate_grid(ax, meta=None, grid_alpha=0.15, label_size=8):
        """Add coordinate grid to the plot without labels"""
        if meta is None or 'transform' not in meta or 'crs' not in meta:
            return  # Can't add grid without transform and CRS
        
        try:
            # Try to use cartopy to add a proper coordinate grid without labels
            gl = ax.gridlines(crs=ccrs.PlateCarree(), alpha=grid_alpha, 
                             linestyle='--', color='gray', draw_labels=False)
        except Exception:
            # Fallback: add a simple grid
            ax.grid(alpha=grid_alpha, linestyle='--', color='gray')
            
    # TAB 1: RGB Composite
    with tabs[0]:
        st.subheader(f"{satellite_display} RGB Composite")
        
        # Enhance RGB image brightness
        brightened_rgb = np.clip(rgb_image * 1.3, 0, 1)  # Increase brightness by 30%
        
        # Create figure with enhanced cartographic elements
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(brightened_rgb)
        ax.set_title(f"{satellite_display} RGB Composite", fontsize=14)
        ax.axis('off')
        
        # Add north arrow only
        add_north_arrow(ax, color='#999999')
        
        st.pyplot(fig)
        plt.close()

    # TAB 2: Probability Map
    with tabs[1]:
        st.subheader(f"Drought Risk Probability Map ({satellite_display})")
        
        # Create figure with both map and colorbar
        fig = plt.figure(figsize=(12, 10), facecolor='white')
        plt.tight_layout()
        gs = gridspec.GridSpec(1, 1)
        
        # Standard non-georeferenced plot as fallback option
        try:
            if meta is not None and 'crs' in meta and 'transform' in meta:
                try:
                    # Get the projection from the metadata
                    projection = ccrs.PlateCarree()
                    
                    # Create map with coordinate reference system
                    ax = fig.add_subplot(gs[0, 0], projection=projection)
                    
                    # Get the extent in the target projection - handle potential issues
                    try:
                        extent = plotting_extent(
                            meta['transform'], 
                            (probability_predictions.shape[1], probability_predictions.shape[0])
                        )
                        
                        # Plot with proper geographic extent
                        im = ax.imshow(
                            probability_predictions, 
                            cmap=cmap,
                            extent=extent,
                            transform=ccrs.PlateCarree(),
                            origin='upper'
                        )
                        
                        # Add grid lines without labels
                        try:
                            gl = ax.gridlines(crs=ccrs.PlateCarree(), alpha=0.15, 
                                             linestyle='--', color='gray', draw_labels=False)
                        except:
                            # Silently fail if gridlines can't be added
                            pass
                            
                        # Remove all spines and frames
                        for spine in ax.spines.values():
                            spine.set_visible(False)
                            
                    except Exception:
                        # If extent calculation fails, fall back to simple plot
                        raise ValueError("Could not calculate plot extent")
                        
                except Exception:
                    # Fall back to regular plot if any part of georeferencing fails
                    ax = fig.add_subplot(gs[0, 0])
                    im = ax.imshow(probability_predictions, cmap=cmap)
                    ax.axis('off')
                    # Remove all frames
                    for spine in ax.spines.values():
                        spine.set_visible(False)
            else:
                # Standard non-georeferenced plot
                ax = fig.add_subplot(gs[0, 0])
                im = ax.imshow(probability_predictions, cmap=cmap)
                ax.axis('off')
                # Remove all frames
                for spine in ax.spines.values():
                    spine.set_visible(False)
        except Exception:
            # Ultimate fallback - ensure we always show something
            ax = fig.add_subplot(gs[0, 0])
            im = ax.imshow(probability_predictions, cmap=cmap)
            ax.axis('off')
            # Remove all frames
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        # Add north arrow only
        add_north_arrow(ax)
        
        # Add colorbar
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('Drought Risk Probability', fontsize=12)
        
        # Add title
        plt.title(f"Drought Risk Probability Map ({satellite_display})", fontsize=14)
        
        # Remove any extra spacing that could cause black borders
        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
        
        st.pyplot(fig)
        plt.close()

    # TAB 3: Statistical Analysis
    with tabs[2]:
        st.subheader("Statistical Analysis")
        binary_predictions = np.where(probability_predictions >= threshold, 1, 0)
        high_risk_count = np.sum(binary_predictions == 1)
        low_risk_count = np.sum(binary_predictions == 0)
        total_pixels = high_risk_count + low_risk_count

        high_risk_percentage = (high_risk_count / total_pixels) * 100
        low_risk_percentage = (low_risk_count / total_pixels) * 100
        
        # Add area calculation if spatial reference is available
        area_info = ""
        if meta is not None and 'transform' in meta:
            try:
                pixel_size_x = abs(meta['transform'][0])
                pixel_size_y = abs(meta['transform'][4])
                pixel_area_m2 = pixel_size_x * pixel_size_y
                total_area_km2 = (total_pixels * pixel_area_m2) / 1_000_000
                high_risk_area_km2 = (high_risk_count * pixel_area_m2) / 1_000_000
                low_risk_area_km2 = (low_risk_count * pixel_area_m2) / 1_000_000
                
                # Format area information as regular text, not code blocks
                area_info = f"""
                **Total Area:** {total_area_km2:.2f} km²
                
                **High Risk Area:** {high_risk_area_km2:.2f} km² ({high_risk_percentage:.2f}%)
                
                **Low Risk Area:** {low_risk_area_km2:.2f} km² ({low_risk_percentage:.2f}%)
                """
            except Exception:
                # Fallback if calculation fails
                pass

        # Create a clean layout with proper text formatting
        st.markdown(f"""
        **Satellite Data Source:** {satellite_display}
        
        **Total Pixels Analyzed:** {total_pixels:,}

        **High Risk Areas (Probability ≥ {threshold}):** {high_risk_percentage:.2f}%

        **Low Risk Areas (Probability < {threshold}):** {low_risk_percentage:.2f}%
        """)
        
        # Add area information separately to ensure proper formatting
        if area_info:
            st.markdown(area_info)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(probability_predictions.flatten(), bins=50, color='skyblue', edgecolor='black')
        ax.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.2f}')
        ax.set_title(f"Distribution of Drought Risk Probabilities ({satellite_display})", fontsize=14)
        ax.set_xlabel("Probability")
        ax.set_ylabel("Frequency")
        ax.legend()
        
        # Add a second y-axis showing percentage
        ax2 = ax.twinx()
        ax2.set_ylabel('Percentage of Total Area (%)')
        ax2.set_ylim(0, 100 * ax.get_ylim()[1] / total_pixels)
        
        st.pyplot(fig)
        plt.close()

    # TAB 4: Overlay
    with tabs[3]:
        st.subheader(f"{satellite_display} RGB + Forecast Overlay (High-Risk Areas)")
        alpha = st.slider("Set Forecast Layer Transparency", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

        # Create overlay with brightened RGB image
        brightened_rgb = np.clip(rgb_image * 1.3, 0, 1)  # Increase brightness by 30%
        overlay = brightened_rgb.copy()
        mask = probability_predictions >= threshold
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * np.array([1, 0, 0])  # blend with red

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(overlay, origin='upper')
        ax.set_title(f"{satellite_display} RGB + Drought Risk Overlay", fontsize=14)
        ax.axis('off')
        
        # Add legend for the overlay
        red_patch = mpatches.Patch(color='red', alpha=alpha, label=f'High Risk (≥{threshold:.2f})')
        ax.legend(handles=[red_patch], loc='lower right')
        
        # Add north arrow only
        add_north_arrow(ax, color='#999999')
        
        st.pyplot(fig)
        plt.close()

    # Set the main title for the entire figure
    if image_date:
        fig.suptitle(f"Drought Risk Analysis ({satellite_type.capitalize()}) - {image_date.strftime('%Y-%m-%d')}", 
                     fontsize=22, y=1.05, color='#2C3E50', fontweight='bold')
    else:
        fig.suptitle(f"Drought Risk Analysis ({satellite_type.capitalize()})", 
                     fontsize=22, y=1.05, color='#2C3E50', fontweight='bold')

# Initialize Earth Engine
@st.cache_resource
def initialize_gee():
    """Initialize Google Earth Engine and validate credentials."""
    try:
        ee.Initialize()
        return True
    except Exception as e:
        return False

def setup_gee():
    """
    Setup and authenticate Google Earth Engine.
    Returns instructions if authentication is needed.
    """
    is_initialized = initialize_gee()
    if not is_initialized:
        st.error("Google Earth Engine authentication required!")
        st.markdown("""
        ### Google Earth Engine Setup Instructions:
        1. Go to [Google Earth Engine](https://earthengine.google.com/) and sign up
        2. Create a Google Cloud Project:
           - Visit [Google Cloud Console](https://console.cloud.google.com)
           - Create a new project or select an existing one
           - Enable the Earth Engine API for your project
        3. Install the Earth Engine CLI:
           ```
           pip install earthengine-api --upgrade
           ```
        4. Authenticate in your terminal:
           ```
           earthengine authenticate
           ```
        5. Once authenticated, restart this application
        """)
        return False
    return True

def get_s2_collection(aoi, start_date, end_date):
    """
    Get Sentinel-2 image collection for the specified area and date range.
    
    Args:
        aoi: Earth Engine Geometry
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
    
    Returns:
        ee.Image: Processed Sentinel-2 image
    """
    # Import Sentinel-2 Surface Reflectance collection
    s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                    .filterBounds(aoi)
                    .filterDate(start_date, end_date)
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
    
    if s2_collection.size().getInfo() == 0:
        return None
    
    # Select the least cloudy image
    image = ee.Image(s2_collection.sort('CLOUDY_PIXEL_PERCENTAGE').first())
    
    # Select the bands we need and rename them
    selected_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    image = image.select(selected_bands)
    
    return image

def download_s2_image(image, aoi, output_path):
    """
    Download Sentinel-2 image as GeoTIFF.
    
    Args:
        image: ee.Image object
        aoi: Earth Engine Geometry
        output_path: Path to save the GeoTIFF
    
    Returns:
        str: Path to downloaded file or None if failed
    """
    try:
        # Get the projection and transform
        proj = image.projection()
        scale = 10  # 10m resolution
        
        # Export image to GeoTIFF
        url = image.getDownloadURL({
            'scale': scale,
            'crs': proj,
            'region': aoi,
            'format': 'GEO_TIFF'
        })
        
        # Download the image
        import requests
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return output_path
        return None
    except Exception as e:
        st.error(f"Error downloading image: {str(e)}")
        return None

def main():
    # --------------------------------------------------------------------------------
    # Title and Citation Instructions
    # --------------------------------------------------------------------------------
    st.title("🌍 Satellite-Based Drought Risk Assessment")

    # Citation box
    st.markdown("""
    <div class="citation-box">
    <p><strong>Citation:</strong> If you use this application or the model's outputs in your research, please cite:</p>
    <p><em>Yungstein, Y., Fishman, N., Lerner, G., Mulero, G., Michael, Y., Yaakobi, A., Obersteiner, S., Rez, L., Klein, T., & Helman, D. (2025). Early detection of drought-stressed stands in Mediterranean forests using machine learning classification models and a rainfall exclusion experiment</em></p>
    </div>
    """, unsafe_allow_html=True)

    # Introductory Description
    st.markdown("""
    This application leverages advanced machine learning methods to estimate drought risk from satellite imagery. 
    It integrates trained Support Vector Machine models that interpret multi-band geospatial data to produce pixel-wise probability maps of drought vulnerability.

    ### Key Features:
    - **Multi-Satellite Support**: Process imagery from both Venµs and Sentinel-2 satellites.
    - **High-Quality Visualization**: RGB composites from appropriate satellite bands.
    - **Drought Probability Mapping**: Pixel-level probability assessments of drought risk.
    - **Robust Statistical Analysis**: Histograms, thresholds, and summary statistics for intuitive interpretation.
    - **Overlay Functionality**: Superimpose drought risk areas over RGB images for contextual insights.
    """)

    # --------------------------------------------------------------------------------
    # Satellite Selection
    # --------------------------------------------------------------------------------
    st.header("Select Satellite Data Source")
    
    st.markdown("""
    <div class="satellite-selector">
    Choose the satellite data source that matches your input GeoTIFF file:
    </div>
    """, unsafe_allow_html=True)
    
    satellite_options = ["Venµs", "Sentinel-2"]
    selected_satellites = st.multiselect(
        "Select one or more satellite data sources:",
        options=satellite_options,
        default=["Venµs"],
        help="Select the satellite platform(s) that your data comes from. You can select multiple options to compare results."
    )
    
    if not selected_satellites:
        st.warning("Please select at least one satellite data source to proceed.")
        return

    # Model Loading
    models = {}
    if "Venµs" in selected_satellites:
        venus_model, venus_scaler = load_venus_model()
        if venus_model is None or venus_scaler is None:
            st.error("Venus model failed to load. Please ensure the model file (model-svm.pkl) is present and valid.")
            return
        models["venus"] = {"model": venus_model, "scaler": venus_scaler}
    
    if "Sentinel-2" in selected_satellites:
        s2_model, s2_scaler = load_sentinel2_model()
        if s2_model is None or s2_scaler is None:
            st.error("Sentinel-2 model failed to load. Please ensure the model file (S2_svm_classification_model.pkl) is present and valid.")
            return
        models["sentinel2"] = {"model": s2_model, "scaler": s2_scaler}

        # Google Earth Engine Integration for Sentinel-2
        if setup_gee():
            st.header("Download Sentinel-2 Data")
            st.markdown("""
            You can download Sentinel-2 data directly using Google Earth Engine. Draw a polygon on the map
            to select your area of interest, then specify the date range for image acquisition.
            """)

            # Create a map centered on a default location
            m = geemap.Map(center=[31.5, 35], zoom=7)
            
            # Add drawing controls
            draw_control = folium.plugins.Draw(
                export=False,
                position='topleft',
                draw_options={
                    'polyline': False,
                    'rectangle': True,
                    'circle': False,
                    'circlemarker': False,
                    'marker': False
                }
            )
            m.add_child(draw_control)
            
            # Display the map
            st.markdown("### Draw Area of Interest")
            st.markdown("Use the rectangle tool (☐) on the left side of the map to draw your area of interest.")
            map_container = folium_static(m)
            
            # Add JavaScript callback
            html(draw_callback)
            
            # Handle drawn features callback
            if st.session_state.drawn_features:
                st.success("Area of interest selected!")
                
            # Date selection
            st.markdown("### Select Date Range")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
            with col2:
                end_date = st.date_input("End Date", datetime.now())

            # Download button
            if st.session_state.get('drawn_features'):
                if st.button("Download Sentinel-2 Data"):
                    with st.spinner("Downloading Sentinel-2 data..."):
                        # Convert the drawn features to GEE geometry
                        aoi = ee.Geometry.Rectangle(st.session_state.drawn_features['coordinates'])
                        
                        # Get the image
                        image = get_s2_collection(aoi, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                        
                        if image is not None:
                            # Create a temporary file
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
                                # Download the image
                                downloaded_path = download_s2_image(image, aoi, tmp_file.name)
                                
                                if downloaded_path:
                                    st.success("Sentinel-2 data downloaded successfully!")
                                    # Automatically use the downloaded file for analysis
                                    with open(downloaded_path, 'rb') as f:
                                        uploaded_file = f
                                else:
                                    st.error("Failed to download Sentinel-2 data. Please try again or upload your own file.")
                        else:
                            st.error("No suitable Sentinel-2 images found for the selected date range. Please try different dates or reduce cloud coverage threshold.")

    # File Upload Section for manual upload
    st.header("Upload Satellite GeoTIFF")
    if "Sentinel-2" in selected_satellites and not st.session_state.get('uploaded_file'):
        st.markdown("You can either use the downloaded Sentinel-2 data above or upload your own GeoTIFF file.")
    
    upload_instructions = {
        "venus": "Please upload a multi-band GeoTIFF file (≥11 bands) from the Venµs satellite.",
        "sentinel2": "Please upload a multi-band GeoTIFF file (≥8 bands) from the Sentinel-2 satellite."
    }
    
    if len(selected_satellites) == 1:
        satellite_key = "venus" if selected_satellites[0] == "Venµs" else "sentinel2"
        st.markdown(
            f"<div class='upload-instructions'>{upload_instructions[satellite_key]}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='upload-instructions'>Please upload a multi-band GeoTIFF file from one of your selected satellites. The appropriate model will be applied based on your selection.</div>",
            unsafe_allow_html=True
        )
    
    uploaded_file = st.file_uploader(
        "Choose a GeoTIFF file",
        type=['tif', 'tiff'],
        help="Upload a multi-band GeoTIFF from one of your selected satellite platforms."
    )

    # Processing and Visualization
    if uploaded_file is not None:
        # If multiple satellites selected, let user specify which one matches the uploaded file
        satellite_for_file = None
        if len(selected_satellites) > 1:
            satellite_for_file = st.radio(
                "Which satellite platform does this file represent?",
                selected_satellites
            )
            satellite_key = "venus" if satellite_for_file == "Venµs" else "sentinel2"
        else:
            satellite_key = "venus" if selected_satellites[0] == "Venµs" else "sentinel2"
        
        # Date input for the image
        image_date = st.date_input("Select the date of the image")
        
        # Process the file with the appropriate model
        with st.spinner(f"Analyzing your {satellite_key} satellite data..."):
            rgb_image, probability_predictions, meta = predict_geotiff(
                models[satellite_key]["model"], 
                models[satellite_key]["scaler"], 
                uploaded_file,
                satellite_key
            )
        
        if rgb_image is not None and probability_predictions is not None:
            st.header("Visualization & Analysis Settings")

            colormap_option = st.selectbox(
                "Select a colormap for the drought risk probability map:",
                options=['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'drought']
            )

            threshold = st.slider(
                "Set the drought risk threshold:",
                min_value=0.0, max_value=1.0, value=0.5, step=0.01
            )

            st.header("Analysis Results")
            plot_predictions(rgb_image, probability_predictions, satellite_key, colormap=colormap_option, threshold=threshold, meta=meta, image_date=image_date)

            # Download Section
            st.header("Download Results")
            st.markdown("**Export your predictions for further analysis or integration into GIS tools.**")

            col1, col2 = st.columns(2)

            # Convert predictions to CSV
            predictions_df = pd.DataFrame({
                'probability': probability_predictions.flatten()
            })
            csv_data = predictions_df.to_csv(index=False)
            
            with col1:
                st.download_button(
                    label="📊 Download Predictions (CSV)",
                    data=csv_data,
                    file_name=f"drought_predictions_{satellite_key}_{image_date.strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    help="Download all pixel-level probability predictions as CSV."
                )

            # Save predictions as GeoTIFF
            with MemoryFile() as memfile:
                with memfile.open(
                    driver='GTiff',
                    height=meta['height'],
                    width=meta['width'],
                    count=1,
                    dtype='float32',
                    crs=meta['crs'],
                    transform=meta['transform']
                ) as dst:
                    dst.write(probability_predictions, 1)
                geotiff_data = memfile.read()
            
            with col2:
                st.download_button(
                    label="🗺️ Download Predictions (GeoTIFF)",
                    data=geotiff_data,
                    file_name=f"drought_predictions_{satellite_key}_{image_date.strftime('%Y%m%d')}.tif",
                    mime="application/octet-stream",
                    help="Download the georeferenced predictions for use in GIS applications."
                )

            # Help / Instructions
            with st.expander("Need Help?"):
                st.markdown(f"""
                **Instructions:**
                
                1. **Select Satellite**: Choose the satellite platform(s) whose data you'll be using.
                2. **Upload Data**: Click "Browse files" and select a multi-band GeoTIFF from your selected satellite.
                3. **Set Visualization Parameters**: Choose a colormap and adjust the drought risk threshold.
                4. **Explore Results**: 
                   - **RGB Image**: View satellite imagery in natural-color form.
                   - **Probability Map**: Examine spatial distribution of drought risk.
                   - **Statistical Analysis**: Gain quantitative insight via histograms and summary statistics.
                   - **Overlay**: Visualize high-risk areas superimposed on the RGB image.
                5. **Download Results**: Export predictions in CSV or GeoTIFF formats.
                
                **Contact:** For further information, please contact Dr. David Helman (David.Helman@mail.huji.ac.il).
                """)

    # Sidebar Information & Citation
    st.sidebar.title("About the Models")
    st.sidebar.markdown("""
    **Model Origin:**  
    These models are part of ongoing research aiming to enhance drought forecasting capabilities through high-resolution satellite imagery.

    **Supported Satellites:**  
    - **Venµs**: Vegetation and Environment monitoring on a New Micro-Satellite
    - **Sentinel-2**: Copernicus Programme satellite with multi-spectral imaging capabilities""")

if __name__ == "__main__":
    main()
