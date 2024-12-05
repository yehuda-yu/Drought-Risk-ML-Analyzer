import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import io

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

def process_image(image):
    """Process uploaded image for model prediction"""
    # Convert image to numpy array and ensure correct shape
    img_array = np.array(image)
    
    # Reshape the image for model input
    # Note: Adjust this based on your model's expected input shape
    if len(img_array.shape) == 3:
        features = img_array.reshape(-1, img_array.shape[-1])
    else:
        features = img_array.reshape(-1, 1)
    
    return features

def generate_forecast_map(predictions, original_shape):
    """Generate a color-coded drought forecast map"""
    # Reshape predictions to original image dimensions
    forecast_map = predictions.reshape(original_shape[:2])
    
    # Create a color-coded map
    fig = px.imshow(
        forecast_map,
        color_continuous_scale=['green', 'yellow', 'red'],
        title='Drought Risk Forecast'
    )
    fig.update_layout(
        title_x=0.5,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def main():
    st.title("🌍 Drought Forecasting Application")
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        "This application uses machine learning to predict drought risk "
        "from Venus satellite imagery. Upload your image to generate a forecast."
    )
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to load the model. Please check the model file.")
        return
    
    # File upload
    st.header("Upload Venus Satellite Image")
    uploaded_file = st.file_uploader(
        "Choose a satellite image file",
        type=['tif', 'jpg', 'png']
    )
    
    if uploaded_file is not None:
        try:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Process image when user clicks button
            if st.button("Generate Forecast"):
                with st.spinner("Processing image..."):
                    # Process image
                    features = process_image(image)
                    
                    # Generate predictions
                    predictions = model.predict_proba(features)
                    
                    # Generate and display forecast map
                    forecast_fig = generate_forecast_map(
                        predictions[:, 1],  # Probability of drought
                        np.array(image).shape
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
            st.info("Please ensure you've uploaded a valid Venus satellite image.")

if __name__ == "__main__":
    main()
