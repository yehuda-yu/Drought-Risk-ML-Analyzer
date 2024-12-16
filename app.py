import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.io import MemoryFile
from rasterio.windows import Window
import pickle
import os

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
    }
    .citation-box p {
        margin: 0;
        font-size: 0.95rem;
    }
    .upload-instructions {
        font-size: 0.9rem;
        color: #555;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# Model Loading and Utilities
# --------------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """
    Load the trained model and corresponding scaler from a pickle file.

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
        st.error(f"Error loading model: {str(e)}")
        return None, None

def get_rgb_image(src):
    """
    Extract RGB bands (7, 4, 3) from the GeoTIFF file and normalize them to [0,1].
    This yields a visually interpretable composite image of the region.
    """
    try:
        red = src.read(7).astype(np.float32)
        green = src.read(4).astype(np.float32)
        blue = src.read(3).astype(np.float32)

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

def predict_geotiff(model, scaler, uploaded_file, chunk_size=256):
    """
    Predict drought risk probabilities on a given GeoTIFF using the trained model.

    Parameters:
    - model: Trained SVM model for drought risk assessment.
    - scaler: Scaler used to normalize input features.
    - uploaded_file: The uploaded GeoTIFF file.
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

                if band_count < 11:
                    st.error(f"Image has {band_count} bands, but at least 11 are required.")
                    return None, None, None

                # Get RGB image
                rgb_image = get_rgb_image(src)
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

                        # Exclude the first band as per the model's expected input structure
                        data = data[1:, :, :]

                        features = data.reshape(band_count - 1, -1).T

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

def plot_predictions(rgb_image, probability_predictions, colormap='drought', threshold=0.5):
    """
    Plotting visualizations using Matplotlib:
    1. RGB composite image.
    2. Probability map of drought risk.
    3. Statistical analysis (histogram, summary stats).
    4. Overlay: highlight areas above threshold on the RGB image.

    Parameters:
    - rgb_image: NumPy array of shape (H, W, 3) with normalized RGB data.
    - probability_predictions: 2D NumPy array of drought risk probabilities.
    - colormap: Colormap name or 'drought' for custom colormap.
    - threshold: Probability threshold for high-risk areas.
    """
    from matplotlib.colors import LinearSegmentedColormap

    # Custom drought colormap if requested
    if colormap == 'drought':
        colors = ['#313695', '#4575B4', '#74ADD1', '#ABD9E9', '#E0F3F8',
                  '#FFFFBF', '#FEE090', '#FDAE61', '#F46D43', '#D73027', '#A50026']
        cmap = LinearSegmentedColormap.from_list("drought", colors)
    else:
        cmap = plt.get_cmap(colormap)

    # Create tabs for visualization
    tabs = st.tabs(["RGB Image", "Probability Map", "Statistical Analysis", "Overlay"])

    # TAB 1: RGB Composite
    with tabs[0]:
        st.subheader("RGB Composite (Bands 7-4-3)")
        st.image(rgb_image, use_container_width =True)

    # TAB 2: Probability Map
    with tabs[1]:
        st.subheader("Drought Risk Probability Map")
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(probability_predictions, cmap=cmap)
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, fraction=0.036, pad=0.04)
        cbar.set_label('Drought Risk Probability', fontsize=12)
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

        st.markdown(f"""
        **Total Pixels Analyzed:** {total_pixels:,}

        **High Risk Areas (Probability ≥ {threshold}):** {high_risk_percentage:.2f}%

        **Low Risk Areas (Probability < {threshold}):** {low_risk_percentage:.2f}%
        """)

        fig, ax = plt.subplots()
        ax.hist(probability_predictions.flatten(), bins=50, color='skyblue', edgecolor='black')
        ax.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.2f}')
        ax.set_title("Distribution of Drought Risk Probabilities", fontsize=14)
        ax.set_xlabel("Probability")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    # TAB 4: Overlay
    with tabs[3]:
        st.subheader("RGB + Forecast Overlay (High-Risk Areas)")
        alpha = st.slider("Set Forecast Layer Transparency", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

        # Create overlay: areas above threshold shown in a contrasting color
        overlay = rgb_image.copy()
        mask = probability_predictions >= threshold
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * np.array([1, 0, 0])  # blend with red

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(overlay, origin='upper')
        ax.axis('off')
        st.pyplot(fig)
        plt.close()

def main():
    # --------------------------------------------------------------------------------
    # Title and Citation Instructions
    # --------------------------------------------------------------------------------
    st.title("🌍 Venµs Satellite-Based Drought Risk Assessment")

    # Citation box
    st.markdown("""
    <div class="citation-box">
    <p><strong>Citation:</strong> If you use this application or the model's outputs in your research, please cite:</p>
    <p><em>Smith, J., Doe, J., & Chan, A. (2024). High-Resolution Drought Forecasting Using Venµs Satellite Imagery. Journal of Environmental Studies, 12(3), 345–360. DOI:10.1234/exampleDOI</em></p>
    </div>
    """, unsafe_allow_html=True)

    # Introductory Description
    st.markdown("""
    This application leverages advanced machine learning methods to estimate drought risk from Venµs satellite imagery. 
    It integrates a trained Support Vector Machine model that interprets multi-band geospatial data to produce pixel-wise probability maps of drought vulnerability.

    ### Key Features:
    - **High-Quality Visualization**: RGB composites from Venµs bands 7-4-3.
    - **Drought Probability Mapping**: Pixel-level probability assessments of drought risk.
    - **Robust Statistical Analysis**: Histograms, thresholds, and summary statistics for intuitive interpretation.
    - **Overlay Functionality**: Superimpose drought risk areas over RGB images for contextual insights.
    """)

    # Model Loading
    model, scaler = load_model()
    if model is None or scaler is None:
        st.error("Model failed to load. Please ensure the model file (model-svm.pkl) is present and valid.")
        return

    # File Upload Section
    st.header("Upload Venµs Satellite GeoTIFF")
    st.markdown(
        "<div class='upload-instructions'>Please upload a multi-band GeoTIFF file (≥11 bands) from the Venµs satellite. The file is processed to estimate drought risk probabilities for each pixel.</div>",
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader(
        "Choose a GeoTIFF file",
        type=['tif', 'tiff'],
        help="Upload a multi-band GeoTIFF from the Venµs satellite."
    )

    # Processing and Visualization
    if uploaded_file is not None:
        with st.spinner("Analyzing your satellite data..."):
            rgb_image, probability_predictions, meta = predict_geotiff(model, scaler, uploaded_file)
        
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
            plot_predictions(rgb_image, probability_predictions, colormap=colormap_option, threshold=threshold)

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
                    file_name="drought_predictions.csv",
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
                    file_name="drought_predictions.tif",
                    mime="application/octet-stream",
                    help="Download the georeferenced predictions for use in GIS applications."
                )

            # Help / Instructions
            with st.expander("Need Help?"):
                st.markdown("""
                **Instructions:**
                
                1. **Upload Data**: Click "Browse files" and select a multi-band GeoTIFF (≥11 bands).
                2. **Set Visualization Parameters**: Choose a colormap and adjust the drought risk threshold.
                3. **Explore Results**: 
                   - **RGB Image**: View satellite imagery in natural-color form.
                   - **Probability Map**: Examine spatial distribution of drought risk.
                   - **Statistical Analysis**: Gain quantitative insight via histograms and summary statistics.
                   - **Overlay**: Visualize high-risk areas superimposed on the RGB image.
                4. **Download Results**: Export predictions in CSV or GeoTIFF formats.
                
                **Contact:** For further support, please contact Dr. Jane Smith (jane.smith@example.edu).
                """)

    # Sidebar Information & Citation
    st.sidebar.title("About the Model")
    st.sidebar.markdown("""
    **Model Origin:**  
    This model is part of ongoing research aiming to enhance drought forecasting capabilities through high-resolution satellite imagery.

    **Citation Reminder:**  
    Please cite the associated publication when using these results:
    *Smith, J., Doe, J., & Chan, A. (2024). High-Resolution Drought Forecasting Using Venµs Satellite Imagery. Journal of Environmental Studies, 12(3), 345–360. DOI:10.1234/exampleDOI*
    """)

if __name__ == "__main__":
    main()
