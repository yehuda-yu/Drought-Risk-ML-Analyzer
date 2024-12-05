import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import io
import rasterio
from rasterio.io import MemoryFile

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

# Load the SVM model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model-svm.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_image(uploaded_file):
    """Process uploaded GeoTIFF image for model prediction"""
    try:
        # Read the GeoTIFF file using rasterio
        with MemoryFile(uploaded_file.read()) as memfile:
            with memfile.open() as dataset:
                # Read all bands
                img_array = dataset.read()
                # Get metadata
                transform = dataset.transform
                crs = dataset.crs
                
                # Transpose array to (height, width, bands) format
                img_array = np.transpose(img_array, (1, 2, 0))
                
                # Reshape for model input (pixels x bands)
                features = img_array.reshape(-1, img_array.shape[-1])
                
                return features, img_array.shape, transform, crs
    except Exception as e:
        st.error(f"Error reading GeoTIFF file: {str(e)}")
        return None, None, None, None

def generate_forecast_map(predictions, original_shape, transform=None, crs=None):
    """Generate a color-coded drought forecast map"""
    # Reshape predictions to original image dimensions
    forecast_map = predictions.reshape(original_shape[:2])
    
    # Create a color-coded map
    fig = px.imshow(
        forecast_map,
        color_continuous_scale=['green', 'yellow', 'red'],
        title='Drought Risk Forecast',
        labels={'color': 'Drought Risk'}
    )
    
    # Add colorbar title
    fig.update_layout(
        title_x=0.5,
        margin=dict(l=20, r=20, t=40, b=20),
        coloraxis_colorbar_title="Risk Level"
    )
    
    return fig

def display_image(uploaded_file):
    """Display GeoTIFF image"""
    try:
        with MemoryFile(uploaded_file.read()) as memfile:
            with memfile.open() as dataset:
                # Read the first three bands (if available) for display
                display_bands = min(3, dataset.count)
                img_array = dataset.read(list(range(1, display_bands + 1)))
                
                # Normalize the data for display
                img_array = np.transpose(img_array, (1, 2, 0))
                img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255
                img_array = img_array.astype(np.uint8)
                
                if display_bands == 1:
                    plt.figure(figsize=(10, 10))
                    plt.imshow(img_array[:,:,0], cmap='gray')
                    plt.axis('off')
                    st.pyplot(plt)
                else:
                    if display_bands == 2:
                        # For 2 bands, duplicate the second band to create an RGB image
                        img_array = np.dstack((img_array, img_array[:,:,1]))
                    st.image(img_array, caption='Uploaded Venus Satellite Image', use_column_width=True)
                
                return True
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")
        return False

def main():
    st.title("🌍 Drought Forecasting Application")
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        "This application uses machine learning to predict drought risk "
        "from Venus satellite imagery. Upload your GeoTIFF image to generate a forecast."
    )
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to load the model. Please check the model file.")
        return
    
    # File upload
    st.header("Upload Venus Satellite Image")
    uploaded_file = st.file_uploader(
        "Choose a GeoTIFF file",
        type=['tif', 'tiff']
    )
    
    if uploaded_file is not None:
        try:
            # Display uploaded image
            if display_image(uploaded_file):
                # Process image when user clicks button
                if st.button("Generate Forecast"):
                    with st.spinner("Processing image..."):
                        # Reset file pointer
                        uploaded_file.seek(0)
                        
                        # Process image
                        features, original_shape, transform, crs = process_image(uploaded_file)
                        
                        if features is not None:
                            # Generate predictions
                            predictions = model.predict_proba(features)
                            
                            # Generate and display forecast map
                            forecast_fig = generate_forecast_map(
                                predictions[:, 1],  # Probability of drought
                                original_shape,
                                transform,
                                crs
                            )
                            st.plotly_chart(forecast_fig, use_container_width=True)
                            
                            # Add download button for the forecast
                            buf = io.BytesIO()
                            forecast_fig.write_image(buf, format='png')
                            st.download_button(
                                label="Download Forecast Map",
                                data=buf.getvalue(),
                                file_name="drought_forecast.png",
                                mime="image/png"
                            )
                            
                            st.success("Forecast generated successfully!")
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("Please ensure you've uploaded a valid Venus satellite GeoTIFF image.")

if __name__ == "__main__":
    main()
