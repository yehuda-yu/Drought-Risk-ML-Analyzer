# Drought Risk Assessment Web Application

## Overview
This web application provides access to the drought risk assessment model described in our article [Yungstein, Y., Fishman, N., Lerner, G., Mulero, G., Michael, Y., Yaakobi, A., Obersteiner, S., Rez, L., Klein, T., & Helman, D. (2025). Early detection of drought-stressed stands in Mediterranean forests using machine learning classification models and a rainfall exclusion experiment](https://doi.org/10.1234/exampleDOI). The application allows researchers and practitioners to apply our machine learning model to their own satellite imagery data to generate detailed drought risk probability maps.

## Citation Requirements

> If you use this application or its outputs in your research, you **MUST** cite the original research article:
>
> Yungstein, Y., Fishman, N., Lerner, G., Mulero, G., Michael, Y., Yaakobi, A., Obersteiner, S., Rez, L., Klein, T., & Helman, D. (2025). Early detection of drought-stressed stands in Mediterranean forests using machine learning classification models and a rainfall exclusion experiment

## Web Application Access

The application is publicly available at: https://drought-risk-ml-analyzer.streamlit.app/

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

## Output Formats

- **CSV**: Contains raw probability values for each pixel.
- **GeoTIFF**: Georeferenced TIFF file containing the drought risk probabilities, compatible with GIS software

## Visualization Interpretation

- **RGB Image**: Standard RGB-color composite showing the AOI region.
- **Probability Map**: Drought risk likelihood from 0-1, with 0 indicating low risk and 1 indicating high risk
- **Statistical Analysis**: Shows the percentage and area of high-risk vs. low-risk regions based on your threshold
- **Overlay**: Highlights high-risk areas in red on the RGB image for contextual interpretation

## Methodological Details

For comprehensive information about the drought risk assessment model, including:
- Training data and methodology
- Model validation and accuracy metrics
- Comparison with alternative approaches
- Technical specifications and limitations

Please refer to our published article: Yungstein, Y., Fishman, N., Lerner, G., Mulero, G., Michael, Y., Yaakobi, A., Obersteiner, S., Rez, L., Klein, T., & Helman, D. (2025). Early detection of drought-stressed stands in Mediterranean forests using machine learning classification models and a rainfall exclusion experiment

## Contact Information

For technical support or research collaboration inquiries, please contact:
- Dr. David Helman (davidhelman.weebly.com)

## License

This application is provided for research and educational purposes only. Commercial use requires explicit permission from the authors.

---
