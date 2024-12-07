import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.io import MemoryFile
from rasterio.windows import Window
import pickle
import os

# Constants
CHUNK_SIZE = 256  # Size of chunks for processing large images

# Set page config
st.set_page_config(
    page_title="Drought Forecasting",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .uploadedFile {
        margin: 2rem 0;
    }
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
    h1 {
        color: #2C3E50;
        margin-bottom: 2rem;
    }
    h2 {
        color: #34495E;
        margin: 1.5rem 0;
    }
    .stAlert {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the model and scaler from a file."""
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
    """Extract RGB bands (7, 4, 3) from the GeoTIFF file and normalize."""
    try:
        red = src.read(7).astype(np.float32)
        green = src.read(4).astype(np.float32)
        blue = src.read(3).astype(np.float32)

        rgb = np.dstack((red, green, blue))
        # Simple normalization to [0,1]
        rgb_min, rgb_max = np.nanmin(rgb), np.nanmax(rgb)
        if rgb_max > rgb_min:
            rgb = (rgb - rgb_min) / (rgb_max - rgb_min)
        else:
            rgb = np.zeros_like(rgb)  # fallback if no variation

        return rgb
    except Exception as e:
        st.error(f"Error creating RGB image: {str(e)}")
        return None

def predict_geotiff(model, scaler, uploaded_file, chunk_size=CHUNK_SIZE):
    """Make predictions on a GeoTIFF file using the given model and scaler, excluding the first band."""
    try:
        with MemoryFile(uploaded_file.read()) as memfile:
            with memfile.open() as src:
                height = src.height
                width = src.width
                band_count = src.count

                if band_count < 11:
                    st.error(f"Image contains only {band_count} bands, but model expects at least 11 bands.")
                    return None, None, None

                # Get RGB image first
                rgb_image = get_rgb_image(src)
                if rgb_image is None:
                    return None, None, None

                # Initialize array to store predictions
                probability_predictions = np.zeros((height, width), dtype=np.float32)

                # Progress bar
                progress_text = st.empty()
                progress_bar = st.progress(0)
                total_chunks = ((height + chunk_size - 1) // chunk_size) * ((width + chunk_size - 1) // chunk_size)
                chunk_count = 0

                for y in range(0, height, chunk_size):
                    for x in range(0, width, chunk_size):
                        window = Window(x, y, min(chunk_size, width - x), min(chunk_size, height - y))
                        data = src.read(window=window)

                        # Exclude the first band
                        data = data[1:, :, :]

                        features = data.reshape(band_count - 1, -1).T

                        # Check for NaN or infinite values
                        if np.isnan(features).any() or np.isinf(features).any():
                            st.error("The input data contains invalid values (NaN or infinite). Please check your data.")
                            return None, None, None

                        # Normalize the features
                        features_normalized = scaler.transform(features)

                        # Predict probability values (decision function for SVM)
                        decision_values = model.decision_function(features_normalized)

                        # Convert decision function to probabilities using a logistic function
                        probabilities = 1 / (1 + np.exp(-decision_values))

                        # Reshape probabilities back to original window size
                        probabilities = probabilities.reshape((window.height, window.width))

                        # Store predictions
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
    """Plot RGB image, probability map, statistical analysis, and an overlay with risk-only coloration."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        import numpy as np

        # Create tabs including the new Overlay tab
        tabs = st.tabs(["RGB Image", "Probability Map", "Statistical Analysis", "Overlay"])

        # Determine colormap
        if colormap == 'drought':
            colors = ['#313695', '#4575B4', '#74ADD1', '#ABD9E9', '#E0F3F8',
                      '#FFFFBF', '#FEE090', '#FDAE61', '#F46D43', '#D73027', '#A50026']
            drought_cmap = LinearSegmentedColormap.from_list("drought", colors)
            cmap = drought_cmap
        else:
            cmap = plt.get_cmap(colormap)

        # TAB 1: RGB Image
        with tabs[0]:
            st.subheader("RGB Composite (Bands 7-4-3)")
            st.image(rgb_image, use_column_width=True)

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
            positive_count = np.sum(binary_predictions == 1)
            negative_count = np.sum(binary_predictions == 0)
            total_pixels = positive_count + negative_count
            risk_percentage = (negative_count / total_pixels) * 100
            no_risk_percentage = (positive_count / total_pixels) * 100

            st.markdown(f"""
            **Total Pixels Analyzed:** {total_pixels}

            **High Risk Areas (probability >= {threshold}):** {risk_percentage:.2f}%

            **Low Risk Areas (probability < {threshold}):** {no_risk_percentage:.2f}%
            """)

            fig, ax = plt.subplots()
            ax.hist(probability_predictions.flatten(), bins=50, color='skyblue', edgecolor='black')
            ax.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
            ax.set_title("Distribution of Drought Risk Probabilities")
            ax.set_xlabel("Probability")
            ax.set_ylabel("Frequency")
            ax.legend()
            st.pyplot(fig)
            plt.close()

        # TAB 4: Overlay (RGB + Forecast, but only dryness shown)
        with tabs[3]:
            st.subheader("RGB + Forecast Overlay (Risk-Only)")
            # Slider for adjusting transparency
            alpha = st.slider("Set Forecast Layer Transparency", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

            # Create a masked array where areas below the threshold are masked out
            dry_mask = probability_predictions < threshold
            probability_predictions_masked = np.ma.array(probability_predictions, mask=dry_mask)

            fig, ax = plt.subplots(figsize=(10, 8))
            # Show RGB base
            ax.imshow(rgb_image, origin='upper')
            # Overlay only drought risk areas; masked areas are transparent by default
            im = ax.imshow(probability_predictions_masked, cmap=cmap, alpha=alpha)
            ax.axis('off')
            st.pyplot(fig)
            plt.close()

    except Exception as e:
        st.error(f"Error plotting predictions: {str(e)}")

def main():
    st.title("🌍 Forest Drought Risk Assessment")
    
    # Add description
    st.markdown("""
    This application uses advanced machine learning to assess drought risk from VenUs satellite imagery. 
    Upload a GeoTIFF file to generate a detailed drought risk assessment.
    
    ### Features:
    - RGB visualization of satellite data
    - Advanced drought risk probability mapping
    - Detailed statistical analysis
    - Overlay view with adjustable transparency
    - Export options for further analysis
    """)
    
    # Load model
    model, scaler = load_model()
    if model is None or scaler is None:
        st.error("Failed to load the model. Please check if the model file exists.")
        return
    
    # File uploader with improved UI
    st.header("Upload Satellite Image")
    uploaded_file = st.file_uploader(
        "Choose a GeoTIFF file (must contain at least 11 bands)",
        type=['tif', 'tiff'],
        help="Upload a multi-band GeoTIFF file from Venus satellite"
    )
    
    if uploaded_file is not None:
        with st.spinner("Processing satellite imagery..."):
            # Process image and get predictions
            rgb_image, probability_predictions, meta = predict_geotiff(
                model, scaler, uploaded_file
            )
            
            if rgb_image is not None and probability_predictions is not None:
                st.header("Visualization Settings")

                # Visualization options
                colormap_option = st.selectbox(
                    "Select a colormap for the drought risk probability map:",
                    options=['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'drought']
                )

                threshold = st.slider(
                    "Select the probability threshold for high-risk areas:",
                    min_value=0.0, max_value=1.0, value=0.5, step=0.01
                )

                st.header("Analysis Results")
                
                # Plot predictions
                plot_predictions(rgb_image, probability_predictions, colormap=colormap_option, threshold=threshold)
                
                # Add download section
                st.header("Download Results")
                col1, col2 = st.columns(2)
                
                # Convert predictions to CSV
                predictions_df = pd.DataFrame({
                    'probability': probability_predictions.flatten()
                })
                csv = predictions_df.to_csv(index=False)
                
                with col1:
                    st.download_button(
                        label="📊 Download Predictions (CSV)",
                        data=csv,
                        file_name="drought_predictions.csv",
                        mime="text/csv",
                        help="Download the raw prediction values in CSV format"
                    )
                
                # Save predictions as GeoTIFF
                with MemoryFile() as memfile:
                    with memfile.open(driver='GTiff',
                                    height=meta['height'],
                                    width=meta['width'],
                                    count=1,
                                    dtype='float32',
                                    crs=meta['crs'],
                                    transform=meta['transform']) as dst:
                        dst.write(probability_predictions, 1)
                    
                    with col2:
                        st.download_button(
                            label="🗺 Download Predictions (GeoTIFF)",
                            data=memfile.read(),
                            file_name="drought_predictions.tif",
                            mime="application/octet-stream",
                            help="Download the predictions as a georeferenced TIFF file"
                        )

                with st.expander("Need Help?"):
                    st.markdown("""
                    **Instructions:**

                    1. **Upload a GeoTIFF file**: Click on 'Browse files' and select your multi-band GeoTIFF file. The file should contain at least 11 bands.

                    2. **Adjust Visualization Settings**: Use the controls to select a colormap and adjust the probability threshold for high-risk areas.

                    3. **View Results**: The results will be displayed in four tabs:
                       - **RGB Image**: Displays the RGB composite (Bands 7, 4, and 3).
                       - **Probability Map**: Shows the drought risk probability map.
                       - **Statistical Analysis**: Provides statistics and histograms of the predictions.
                       - **Overlay**: Overlays the probability map on the RGB image with a slider for transparency.

                    4. **Download Results**: Download the predictions as a CSV file or as a GeoTIFF.

                    **Contact Information**: If you encounter any issues or have questions, please contact John Doe at johndoe@example.com.
                    """)

    # Add creator's information
    st.sidebar.title("About")
    st.sidebar.info("""
    Developed by John Doe.

    Contact: johndoe@example.com

    If you use this application, please cite:
    John Doe (2023). Advanced Drought Risk Assessment Tool. Version 1.0.
    """)

if __name__ == "__main__":
    main()
