# FireSight: AI-Powered Wildfire Forecasting

**GDG @ Penn State Solution Challenge Hackathon 2026**
UN Sustainable Development Goals: **SDG 13 (Climate Action)**

**Live Demo**: https://firesight-tau.vercel.app
**Backend API**: https://firesight-api-278574018619.us-central1.run.app

---

## One-Sentence Pitch

**FireSight is an AI-powered wildfire forecasting platform that uses historical U.S. wildfire-related data and machine learning to predict future wildfire risk and visualize those forecasts through an interactive map overlaid with live NASA satellite fire detections.**

---

## Why FireSight Matters

Wildfires are becoming more frequent, more destructive, and more costly — threatening ecosystems, displacing communities, and accelerating climate change through massive CO2 emissions. Most existing wildfire tools are reactive, showing fires after they occur. FireSight takes a proactive approach by forecasting where wildfire risk will be elevated, helping with preparedness, resource allocation, and long-term risk understanding.

This directly supports:
- **SDG 13 (Climate Action)**: Early wildfire risk identification helps reduce response times and the carbon impact of uncontrolled fires
- **SDG 15 (Life on Land)**: Protecting forests and biodiversity by enabling preemptive action in high-risk areas

---

## How It Works

1. **Data Collection** — ERA5 atmospheric reanalysis data from ECMWF (2020–2025) provides temperature, wind speed, and precipitation at 0.25-degree resolution. NASA FIRMS satellite fire detections provide ground-truth labels for where fires actually occurred.

2. **Feature Engineering** — Daily grid-cell features include mean/max temperature, wind speed, precipitation deficit, month, day of year, and 3/7-day rolling averages. Lag features capture whether nearby cells had recent fire activity.

3. **Model Training & Comparison** — Four models were trained and compared: XGBoost, Random Forest, Extra Trees, and CatBoost. XGBoost was selected as the best performer. Class weighting handles the severe imbalance (~0.1% of grid cells have fire events).

4. **Forecasting** — The trained model generates fire probability projections for June–October 2026 across the entire US grid. Predictions are interpolated to 0.125-degree resolution for smoother visualization.

5. **Live Fire Overlay** — The backend fetches real-time fire detections from NASA FIRMS (VIIRS SNPP satellite) every request, allowing users to see model predictions alongside actual current fires.

---

## Datasets

The datasets are too large for GitHub but are documented here for reproducibility:

| Dataset | Source | Size | Description |
|---|---|---|---|
| ERA5 Reanalysis (2020–2025) | [ECMWF/Copernicus](https://cds.climate.copernicus.eu/) | ~2 GB | Hourly 2m temperature, 10m u/v wind, total precipitation. Accumulated and instantaneous variables in separate NetCDF files. |
| ERA5 Forecast (2026 June–Oct) | ECMWF/Copernicus | ~200 MB | Same variables for the 2026 projection period |
| NASA FIRMS VIIRS (2020–2025) | [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) | ~500 MB | Suomi NPP and NOAA-20 VIIRS active fire detections with latitude, longitude, brightness, confidence, FRP, and acquisition timestamps |

To reproduce: download from the sources above, place ERA5 `.nc` files in `data/raw/era5/` and FIRMS `.csv` files in `data/raw/firms/`.

---

## Project Structure

```
FireSight/
├── backend/                      # FastAPI backend (Google Cloud Run)
│   ├── main.py                   # API: /api/risk-grid + /api/live-fires
│   ├── generate_predictions.py   # Converts model CSV → predictions.json
│   ├── predictions.json          # Pre-computed risk grid (95k cells, 0.125°)
│   ├── Dockerfile                # Cloud Run container
│   └── requirements.txt
├── frontend/                     # React + Vite (Vercel)
│   ├── src/
│   │   ├── App.jsx               # Data fetching + layout
│   │   ├── components/
│   │   │   ├── FireMap.jsx       # Leaflet map with grid + fire markers
│   │   │   ├── RiskGridLayer.jsx # Colored 0.125° grid cells
│   │   │   ├── CursorInfo.jsx    # Lat/lon + fire risk % on hover
│   │   │   └── Legend.jsx        # Color scale legend
│   │   └── main.jsx
│   └── package.json
├── src/                          # ML training pipeline
│   ├── app/
│   │   ├── xgboost_forecast_us.py
│   │   ├── random_forest_forecast_us.py
│   │   ├── extra_trees_forecast_us.py
│   │   └── catboost_forecast_us.py
│   └── visualization/
│       ├── firesight_streamlit_app.py
│       └── visualize_model_comparison.py
└── data/                         # Datasets (not included — see above)
```

---

## Quick Start

### Backend

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Generate predictions from model output:
python generate_predictions.py --from-csv ../data/processed/xgboost_projection_2026_jun_oct.csv
# Or use synthetic demo data for development:
python generate_predictions.py --demo

uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

### Environment Variables

| Variable | Where | Description |
|---|---|---|
| `NASA_API_KEY` | Backend | Free NASA FIRMS API key — [get one here](https://firms.modaps.eosdis.nasa.gov/api/area/) |
| `VITE_API_URL` | Frontend | Backend URL (default: `http://localhost:8000`) |

---

## Deployment

### Backend → Google Cloud Run

```bash
cd backend
gcloud run deploy firesight-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars NASA_API_KEY=your_key \
  --port 8080
```

### Frontend → Vercel

```bash
cd frontend
vercel --prod
```

Set `VITE_API_URL` in Vercel project settings to your Cloud Run URL.

---

## Tech Stack

- **Backend**: Python, FastAPI, httpx, Google Cloud Run
- **Frontend**: React (Vite), Leaflet, react-leaflet, axios, Vercel
- **ML**: XGBoost, Random Forest, Extra Trees, CatBoost, scikit-learn, pandas, xarray
- **Data**: ERA5 reanalysis (ECMWF), NASA FIRMS VIIRS satellite fire detections

---

## AI Disclosure

AI tools (Claude, GitHub Copilot) were used during development for code generation, debugging, and deployment assistance.
