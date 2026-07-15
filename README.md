# Rare Earth Mine Detection & Expansion Monitor

A web application for detecting potential rare earth mining sites and monitoring their expansion using satellite imagery and machine learning.

## Features

- **Region Scanning**: Grid-based scanning of Myanmar regions with configurable spacing
- **Mine Probability Classification**: ML model predicts likelihood of mining activity at each location
- **Site Clustering**: Groups nearby flagged points into distinct mine sites
- **Expansion Analysis**: Tracks environmental impact over time using NDVI disturbance metrics
- **Interactive Map**: Visualize results with Folium maps
- **Export Reports**: Download results as CSV files

## Demo

- [Live App](https://your-app-url.streamlit.app) (after deployment)

## Tech Stack

- **Frontend**: Streamlit
- **Satellite Data**: Google Earth Engine (AlphaEarth Embeddings)
- **ML Model**: Random Forest Classifier
- **Maps**: Folium

## Setup

### Prerequisites

- Python 3.10+
- Google Cloud Project with Earth Engine API enabled
- Earth Engine credentials (service account for deployment, or local authentication)

### Installation

```bash
# Clone the repository
git clone https://github.com/AyeCham/SatelliteImageProcessor.git
cd SatelliteImageProcessor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Create a `.env` file in the project root:

```
PROJECT_ID=your-google-cloud-project-id
```

2. For local development, authenticate with Earth Engine:

```bash
earthengine authenticate
```

3. For deployment, create a service account and store credentials as secrets.

### Running Locally

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Deployment

### Streamlit Community Cloud

1. Push code to a public GitHub repository
2. Go to [Streamlit Community Cloud](https://share.streamlit.io)
3. Connect your GitHub repository
4. Add secrets (GEE service account JSON) in the dashboard
5. Deploy

### Google Cloud Run

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/your-project/satellite-image-processor
gcloud run deploy satellite-image-processor --image gcr.io/your-project/satellite-image-processor --platform managed
```

## Project Structure

```
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── dataset/
│   ├── mine_clustered_classifier.joblib    # Trained ML model
│   └── feature_clustered_scaler.joblib     # Feature scaler
└── .env                      # Environment variables (not committed)
```

## How It Works

1. **Data Extraction**: Satellite embeddings (64-band AlphaEarth features) are extracted for each grid point
2. **Classification**: ML model predicts mine probability for each point
3. **Clustering**: DBSCAN groups nearby flagged points into distinct sites
4. **Expansion Analysis**: Sentinel-2 NDVI is computed for detected sites to track environmental impact over time

## Model Training

The ML model was trained on labeled mining sites across Myanmar (Kachin, Shan regions) with negative samples from non-mining areas. See `train_model.py` for training code.

## License

MIT

## Acknowledgments

- Google Earth Engine for satellite data access
- AlphaEarth for embedding features
- Streamlit for the web framework
