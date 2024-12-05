# Drought Risk Assessment Application

## Overview
This Streamlit application provides advanced drought risk assessment using machine learning analysis of Venus satellite imagery. The application processes multi-band GeoTIFF images to generate detailed drought risk probability maps alongside RGB visualizations.

## Features
- **RGB Visualization**: Displays a natural color composite using bands 7-4-3
- **Drought Risk Mapping**: Generates probability maps showing areas at risk of drought
- **Statistical Analysis**: Provides detailed statistics on risk distribution
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
  ```

## Installation
1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the model file (`model-svm.pkl`) is present in the application directory

## Usage
1. Run the application:
   ```bash
   streamlit run app.py
   ```
2. Upload a multi-band GeoTIFF file from Venus satellite (must contain at least 11 bands)
3. View the generated visualizations:
   - RGB composite image (bands 7-4-3)
   - Drought risk probability map
4. Download the results in your preferred format (CSV or GeoTIFF)

## Input Requirements
- File format: GeoTIFF
- Minimum bands: 11
- Band order: Must match the Venus satellite band configuration
- First band is excluded from analysis

## Output Formats
- **CSV**: Contains raw probability values for each pixel
- **GeoTIFF**: Georeferenced TIFF file containing the drought risk probabilities

## Visualization Guide
- **RGB Image**: Natural color composite using bands 7-4-3
  - Red areas typically indicate bare soil or urban areas
  - Green areas indicate vegetation
  - Blue areas may indicate water or shadows
- **Risk Map**: Probability of drought risk
  - Blue/Green: Low risk areas
  - Yellow/Orange: Moderate risk areas
  - Red: High risk areas

## Notes
- Processing time depends on image size
- Large images are processed in chunks to manage memory usage
- Progress bar indicates processing status

## Support
For issues and questions, please open an issue in the repository.
