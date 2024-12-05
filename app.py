import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.io import MemoryFile
from rasterio.windows import Window
import pickle
from matplotlib.colors import LinearSegmentedColormap

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
    try:
        with open('model-svm.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['scaler']
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def get_rgb_image(src):
    """Extract RGB bands (7, 4, 3) from the GeoTIFF file and enhance brightness."""
    try:
        red = src.read(7)
        green = src.read(4)
        blue = src.read(3)

        rgb = np.dstack((red, green, blue))
        
        # Normalize and enhance each band separately
        for i in range(3):
            band = rgb[:,:,i]
            # Use more aggressive percentile clipping for better contrast
            min_val = np.percentile(band, 1)  
            max_val = np.percentile(band, 99)  
            # Normalize and apply gamma correction for brightness
            normalized = np.clip((band - min_val) / (max_val - min_val), 0, 1)
            rgb[:,:,i] = np.power(normalized, 0.8)  
            
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
                    st.error(f"Image contains only {band_count} bands, but model expects 11 bands.")
                    return None, None, None, None, None, None

                # Get RGB image first
                rgb_image = get_rgb_image(src)
                if rgb_image is None:
                    return None, None, None, None, None, None

                # Initialize arrays to store predictions
                probability_predictions = np.zeros((height, width), dtype=np.float32)
                positive_count = 0
                negative_count = 0

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

                        # Normalize the features
                        features_normalized = scaler.transform(features)

                        # Predict probability values (decision function for SVM)
                        decision_values = model.decision_function(features_normalized)

                        # Convert decision function to probabilities using a logistic function
                        probabilities = 1 / (1 + np.exp(-decision_values))

                        # Binary predictions based on decision threshold
                        binary_pred = np.where(probabilities > 0.5, 1, 0)

                        # Count positive (no risk) and negative (risk) predictions
                        positive_count += np.sum(binary_pred == 1)
                        negative_count += np.sum(binary_pred == 0)

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
                return rgb_image, probability_predictions, src.meta, positive_count, negative_count

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None, None, None, None

def plot_predictions(rgb_image, probability_predictions, positive_count, negative_count):
    """Plot RGB image and probability prediction maps side by side with improved visualization."""
    try:
        # Create custom colormap for drought predictions
        colors = ['#313695', '#4575B4', '#74ADD1', '#ABD9E9', '#E0F3F8', 
                 '#FFFFBF', '#FEE090', '#FDAE61', '#F46D43', '#D73027', '#A50026']
        drought_cmap = LinearSegmentedColormap.from_list("drought", colors)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.patch.set_facecolor('#f0f2f6')

        # RGB image
        ax1.imshow(rgb_image)
        ax1.set_title("RGB Composite (Bands 7-4-3)", pad=20, fontsize=14, fontweight='bold')
        ax1.axis('off')

        # Probability prediction map
        im2 = ax2.imshow(probability_predictions, cmap=drought_cmap)
        ax2.set_title("Drought Risk Probability Map", pad=20, fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Add colorbar with custom styling
        cbar = plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.05)
        cbar.set_label('Drought Risk Probability', fontsize=12, labelpad=10)
        cbar.ax.tick_params(labelsize=10)

        # Add prediction statistics
        total_pixels = positive_count + negative_count
        risk_percentage = (negative_count / total_pixels) * 100
        no_risk_percentage = (positive_count / total_pixels) * 100
        
        stats_text = (f"Analysis Results:\n"
                     f"High Risk Areas: {risk_percentage:.1f}%\n"
                     f"Low Risk Areas: {no_risk_percentage:.1f}%")
        
        plt.figtext(0.02, 0.02, stats_text, fontsize=12, 
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=10))

        # Adjust layout
        plt.tight_layout()
        
        # Display in Streamlit
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.error(f"Error plotting predictions: {str(e)}")

def main():
    st.title("🌍 Advanced Drought Risk Assessment")
    
    # Add description
    st.markdown("""
    This application uses advanced machine learning to assess drought risk from satellite imagery. 
    Upload a multi-band GeoTIFF file to generate a detailed drought risk assessment.
    
    ### Features:
    - RGB visualization of satellite data
    - Advanced drought risk probability mapping
    - Detailed statistical analysis
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
            rgb_image, probability_predictions, meta, positive_count, negative_count = predict_geotiff(
                model, scaler, uploaded_file
            )
            
            if rgb_image is not None and probability_predictions is not None:
                st.header("Analysis Results")
                
                # Plot predictions
                plot_predictions(rgb_image, probability_predictions, positive_count, negative_count)
                
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

if __name__ == "__main__":
    main()
