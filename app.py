import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.io import MemoryFile
from rasterio.windows import Window
import pickle

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
    }
    .uploadedFile {
        margin: 2rem 0;
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
                    return None, None, None, None, None

                # Initialize arrays to store predictions
                binary_predictions = np.zeros((height, width), dtype=np.float32)
                probability_predictions = np.zeros((height, width), dtype=np.float32)
                positive_count = 0
                negative_count = 0

                # Progress bar
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

                        # Reshape predictions back to original window size
                        binary_pred = binary_pred.reshape((window.height, window.width))
                        probabilities = probabilities.reshape((window.height, window.width))

                        # Store predictions
                        binary_predictions[y:y+window.height, x:x+window.width] = binary_pred
                        probability_predictions[y:y+window.height, x:x+window.width] = probabilities

                        # Update progress
                        chunk_count += 1
                        progress_bar.progress(chunk_count / total_chunks)

                return binary_predictions, probability_predictions, src.meta, positive_count, negative_count

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None, None, None, None

def plot_predictions(binary_predictions, probability_predictions, positive_count, negative_count):
    """Plot both binary and probability prediction maps side by side."""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Binary prediction map
        im1 = ax1.imshow(binary_predictions, cmap='binary')
        ax1.set_title("Binary Prediction Map\n(White: No Risk, Black: Risk)")
        plt.colorbar(im1, ax=ax1, ticks=[0, 1])

        # Probability prediction map
        im2 = ax2.imshow(probability_predictions, cmap='RdYlBu_r')
        ax2.set_title("Probability Prediction Map")
        plt.colorbar(im2, ax=ax2, label='Drought Risk Probability')

        # Add prediction statistics
        total_pixels = positive_count + negative_count
        risk_percentage = (negative_count / total_pixels) * 100
        no_risk_percentage = (positive_count / total_pixels) * 100
        
        plt.figtext(0.02, -0.1, f"Statistics:\nRisk Areas: {risk_percentage:.1f}%\nNo Risk Areas: {no_risk_percentage:.1f}%",
                   fontsize=10, ha='left')

        # Adjust layout to prevent text overlap
        plt.tight_layout()
        
        # Display in Streamlit
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.error(f"Error plotting predictions: {str(e)}")

def main():
    st.title("Drought Forecasting Application")
    st.write("Upload a GeoTIFF image to get drought predictions")
    
    # Load model
    model, scaler = load_model()
    if model is None or scaler is None:
        st.error("Failed to load the model. Please check if the model file exists.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a GeoTIFF file", type=['tif', 'tiff'])
    
    if uploaded_file is not None:
        st.write("Processing image...")
        
        # Process image and get predictions
        binary_predictions, probability_predictions, meta, positive_count, negative_count = predict_geotiff(
            model, scaler, uploaded_file
        )
        
        if binary_predictions is not None:
            # Plot predictions
            plot_predictions(binary_predictions, probability_predictions, positive_count, negative_count)
            
            # Add download buttons for predictions
            col1, col2 = st.columns(2)
            
            # Convert predictions to CSV
            predictions_df = pd.DataFrame({
                'binary_prediction': binary_predictions.flatten(),
                'probability': probability_predictions.flatten()
            })
            csv = predictions_df.to_csv(index=False)
            
            with col1:
                st.download_button(
                    label="Download Predictions (CSV)",
                    data=csv,
                    file_name="drought_predictions.csv",
                    mime="text/csv"
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
                        label="Download Predictions (GeoTIFF)",
                        data=memfile.read(),
                        file_name="drought_predictions.tif",
                        mime="application/octet-stream"
                    )

if __name__ == "__main__":
    main()
