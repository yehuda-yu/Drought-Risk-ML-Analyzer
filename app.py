# Drought Risk Assessment Application

## Overview
This Streamlit application provides advanced drought risk assessment using machine learning analysis of satellite imagery. The application processes multi-band GeoTIFF images from Venus and Sentinel-2 satellites to generate detailed drought risk probability maps alongside RGB visualizations.

## Features
- **Multi-Satellite Support**: Process imagery from both Venus and Sentinel-2 satellites
- **RGB Visualization**: Displays natural color composites using appropriate bands for each satellite
- **Drought Risk Mapping**: Generates probability maps showing areas at risk of drought
- **Statistical Analysis**: Provides detailed statistics on risk distribution
- **Comparative Analysis**: Analyze the same area with different satellite data sources
- **Export Options**: Download results in both CSV and GeoTIFF formats

## Requirements
- Python 3.7+
- Required packages:
  ```
  streamlit>=1.24.0
  numpy>=1.21.0
  pandas>=1.3.0
  scikit-learn>=0.24.2
  matplotlib>=3.4.0
  joblib>=1.0.1
  rasterio>=1.3.0
  cartopy
  ```

## Installation
1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the model files (`model-svm.pkl` for Venus and `S2_svm_classification_model.pkl` for Sentinel-2) are present in the application directory

## Usage
1. Run the application:
   ```bash
   streamlit run app.py
   ```
2. Select which satellite data source(s) you want to work with
3. Upload a multi-band GeoTIFF file from your selected satellite
4. View the generated visualizations:
   - RGB composite image
   - Drought risk probability map
   - Statistical analysis
   - High-risk areas overlay
5. Download the results in your preferred format (CSV or GeoTIFF)

## Input Requirements
- File format: GeoTIFF
- For Venus data:
  - Minimum bands: 11
  - RGB visualization uses bands 7-4-3
  - First band is excluded from analysis
- For Sentinel-2 data:
  - Minimum bands: 8
  - RGB visualization uses bands 4-3-2

## Output Formats
- **CSV**: Contains raw probability values for each pixel
- **GeoTIFF**: Georeferenced TIFF file containing the drought risk probabilities

## Visualization Guide
- **RGB Image**: Natural color composite using appropriate bands for each satellite
  - Red areas typically indicate bare soil or urban areas
  - Green areas indicate vegetation
  - Blue areas may indicate water or shadows
- **Risk Map**: Probability of drought risk
  - Blue/Green: Low risk areas
  - Yellow/Orange: Moderate risk areas
  - Red: High risk areas

## Satellite Specifications
### Venus
- High-resolution (5m) multi-spectral imaging
- 12 narrow spectral bands
- Optimized for vegetation monitoring

### Sentinel-2
- Medium-resolution (10m, 20m, 60m) multi-spectral imaging
- 13 spectral bands
- Part of the EU Copernicus Programme

## Notes
- Processing time depends on image size
- Large images are processed in chunks to manage memory usage
- Progress bar indicates processing status

## Support
For issues and questions, please open an issue in the repository.
