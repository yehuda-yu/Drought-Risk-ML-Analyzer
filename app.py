import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import io
import rasterio
from rasterio.io import MemoryFile
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import leafmap.foliumap as leafmap
import pickle

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

# Load the model
@st.cache_resource
def load_model():
    try:
        with open('model-svm.pkl', 'rb') as f:
            data = pickle.load(f)
        model = data['model']
        scaler = data['scaler']
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def make_prediction(model, scaler, features):
    """Make prediction using the model and scaler"""
    try:
        # Scale the features
        features_normalized = scaler.transform(features)
        
        # Predict decision values
        decision_values = model.decision_function(features_normalized)
        
        # Convert decision values to probabilities using logistic function
        probabilities = 1 / (1 + np.exp(-decision_values))
        
        return probabilities
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
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
                bounds = dataset.bounds
                crs = dataset.crs
                
                # Print information about the bands
                st.write(f"Number of bands in image: {img_array.shape[0]}")
                
                # Select only the first 11 bands if we have more than 11
                if img_array.shape[0] > 11:
                    st.info("Image contains more bands than model expects. Using first 11 bands.")
                    img_array = img_array[:11]
                elif img_array.shape[0] < 11:
                    st.error(f"Image contains only {img_array.shape[0]} bands, but model expects 11 bands.")
                    return None, None, None, None, None
                
                # Transpose array to (height, width, bands) format
                img_array = np.transpose(img_array, (1, 2, 0))
                
                # Reshape for model input (pixels x bands)
                features = img_array.reshape(-1, img_array.shape[-1])
                
                # Replace NaN values with 0
                features = np.nan_to_num(features)
                
                return features, img_array.shape, transform, bounds, crs
    except Exception as e:
        st.error(f"Error reading GeoTIFF file: {str(e)}")
        return None, None, None, None, None

def generate_forecast_map(predictions, original_shape, transform, bounds, crs):
    """Generate a color-coded drought forecast map using matplotlib"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import io

        # Reshape predictions to original image dimensions
        forecast_map = predictions.reshape(original_shape[:2])

        # Create figure and axes
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot the forecast
        img = ax.imshow(
            forecast_map,
            cmap='RdYlGn_r',  # Red (drought) to Green (healthy)
            origin='upper',
            extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
            aspect='auto'
        )

        # Add colorbar
        cbar = plt.colorbar(img, ax=ax, label='Drought Risk', orientation='horizontal', pad=0.05)

        # Add gridlines
        ax.grid(True, linestyle='--', alpha=0.5)

        # Set labels for axes
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        # Set title
        plt.title('Drought Risk Forecast')

        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        buf.seek(0)
        plt.close(fig)

        return buf
    except Exception as e:
        st.error(f"Error generating map: {str(e)}")
        return None


def create_leafmap(predictions, original_shape, transform, bounds, crs):
    """Create an interactive leafmap visualization"""
    try:
        # Reshape predictions to original image dimensions
        forecast_map = predictions.reshape(original_shape[:2])
        
        # Create a temporary GeoTIFF file for the forecast
        with MemoryFile() as memfile:
            with memfile.open(driver='GTiff',
                            height=original_shape[0],
                            width=original_shape[1],
                            count=1,
                            dtype=forecast_map.dtype,
                            crs=crs,
                            transform=transform) as dataset:
                dataset.write(forecast_map, 1)
            
            # Create leafmap
            m = leafmap.Map()
            
            # Add the forecast layer
            m.add_raster(memfile.name,
                        colormap='RdYlGn_r',
                        layer_name='Drought Risk',
                        nodata=0)
            
            # Add colorbar
            m.add_colorbar(label='Drought Risk')
            
            return m
            
    except Exception as e:
        st.error(f"Error creating interactive map: {str(e)}")
        return None

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
    
    # Load model and scaler
    model, scaler = load_model()
    if model is None or scaler is None:
        st.error("Failed to load the model or scaler. Please check the model file.")
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
                        features, original_shape, transform, bounds, crs = process_image(uploaded_file)
                        
                        if features is not None:
                            # Generate predictions using the model dictionary
                            predictions = make_prediction(model, scaler, features)
                            
                            if predictions is not None:
                                # Create tabs for different visualizations
                                tab1, tab2 = st.tabs(["Static Map", "Interactive Map"])
                                
                                with tab1:
                                    # Generate and display static forecast map
                                    map_buf = generate_forecast_map(
                                        predictions,
                                        original_shape,
                                        transform,
                                        bounds,
                                        crs
                                    )
                                    if map_buf:
                                        st.image(map_buf, caption='Drought Risk Forecast Map', use_column_width=True)
                                        
                                        # Add download button for the forecast
                                        st.download_button(
                                            label="Download Forecast Map",
                                            data=map_buf,
                                            file_name="drought_forecast.png",
                                            mime="image/png"
                                        )
                                
                                with tab2:
                                    # Generate and display interactive map
                                    m = create_leafmap(
                                        predictions,
                                        original_shape,
                                        transform,
                                        bounds,
                                        crs
                                    )
                                    if m:
                                        m.to_streamlit(height=600)
                                
                                st.success("Forecast generated successfully!")
                            else:
                                st.error("Failed to generate predictions. Please check the model compatibility.")
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("Please ensure you've uploaded a valid Venus satellite GeoTIFF image.")

if __name__ == "__main__":
    main()
