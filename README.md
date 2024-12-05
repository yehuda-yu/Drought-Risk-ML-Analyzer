# Drought Forecasting Application

This Streamlit application provides drought forecasting capabilities using Venus satellite imagery. The application uses a pre-trained machine learning model to analyze satellite images and predict areas at risk of drought.

## Features

- Upload Venus satellite images for analysis
- Interactive visualization of drought risk areas
- Downloadable forecast results
- User-friendly interface with clear instructions
- Professional-grade map visualization

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Upload your Venus satellite image using the file uploader

4. View the generated drought forecast map

5. Download the results if desired

## Technical Details

The application uses a Support Vector Machine (SVM) model trained on Venus satellite imagery data. The model analyzes various spectral bands and features to identify areas at risk of drought.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributors

- [Your Name/Organization]

## Acknowledgments

- Venus satellite imagery providers
- Scientific references and methodologies used in the model development
