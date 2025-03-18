# Drought Risk Assessment Web Application

## Overview
This web application provides access to the drought risk assessment model described in our article [*"High-Resolution Drought Forecasting Using Satellite Imagery"*](https://doi.org/10.1234/exampleDOI). The application allows researchers and practitioners to apply our machine learning model to their own satellite imagery data to generate detailed drought risk probability maps.

## Citation Requirements

> **IMPORTANT:** If you use this application or its outputs in your research, you **MUST** cite the original research article:
>
> Smith, J., Doe, J., & Chan, A. (2024). High-Resolution Drought Forecasting Using Satellite Imagery. *Journal of Environmental Studies*, 12(3), 345–360. https://doi.org/10.1234/exampleDOI

Proper attribution is essential for academic integrity and allows for the continued development of scientific tools. Include the citation in your methodology section when describing how drought risk was assessed.

## Web Application Access

The application is publicly available at: [https://drought-risk-assessment.streamlit.app/](https://drought-risk-assessment.streamlit.app/)

## Usage Guide

1. **Select Satellite Data Source**: Choose which satellite platform(s) your data comes from (Venµs, Sentinel-2, or both)
2. **Upload Your Data**: Upload a multi-band GeoTIFF file from your selected satellite
3. **Adjust Visualization Settings**:
   - Select a colormap for the drought risk probability map
   - Set the threshold for defining high-risk areas
4. **Explore Results**: Navigate through the different visualization tabs:
   - RGB Image: View the natural color composite of your satellite imagery
   - Probability Map: Examine the spatial distribution of drought risk
   - Statistical Analysis: Review quantitative metrics of drought risk coverage
   - Overlay: See high-risk areas highlighted on the RGB image
5. **Download Results**: Export your analysis as CSV or GeoTIFF for further analysis

## Input Requirements

- **File Format**: GeoTIFF (.tif, .tiff)
- **Venµs Data**:
  - Minimum bands: 11
  - RGB visualization uses bands 7-4-3
  - The model excludes the first band from analysis
- **Sentinel-2 Data**:
  - Minimum bands: 8
  - RGB visualization uses bands 4-3-2

## Output Formats

- **CSV**: Contains raw probability values for each pixel, suitable for statistical analysis
- **GeoTIFF**: Georeferenced TIFF file containing the drought risk probabilities, compatible with GIS software

## Visualization Interpretation

- **RGB Image**: Standard false-color composite showing vegetation in green, bare soil/urban areas in red
- **Probability Map**: Drought risk likelihood from 0-1, with blue/green indicating low risk and orange/red indicating high risk
- **Statistical Analysis**: Shows the percentage and area of high-risk vs. low-risk regions based on your threshold
- **Overlay**: Highlights high-risk areas in red on the RGB image for contextual interpretation

## Methodological Details

For comprehensive information about the drought risk assessment model, including:
- Training data and methodology
- Model validation and accuracy metrics
- Comparison with alternative approaches
- Technical specifications and limitations

Please refer to our published article: Smith, J., Doe, J., & Chan, A. (2024). High-Resolution Drought Forecasting Using Satellite Imagery. *Journal of Environmental Studies*, 12(3), 345–360.

## Contact Information

For technical support or research collaboration inquiries, please contact:
- Dr. Jane Smith (jane.smith@example.edu)

## License

This application is provided for research and educational purposes only. Commercial use requires explicit permission from the authors.

---

© 2024 Environmental Remote Sensing Laboratory
