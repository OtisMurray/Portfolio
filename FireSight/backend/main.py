import os, csv, json, io
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx

NASA_API_KEY = os.environ.get("NASA_API_KEY", "")

app = FastAPI(title="FireSight API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Load pre-computed risk grid on startup
predictions_path = Path(__file__).parent / "predictions.json"
risk_grid = json.loads(predictions_path.read_text()) if predictions_path.exists() else []
print(f"Loaded {len(risk_grid)} risk grid cells")


@app.get("/api/risk-grid")
def get_risk_grid():
    if not risk_grid:
        raise HTTPException(503, "Risk grid not available")
    return risk_grid


@app.get("/api/live-fires")
async def get_live_fires():
    if not NASA_API_KEY:
        raise HTTPException(503, "NASA_API_KEY not configured")

    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{NASA_API_KEY}/VIIRS_SNPP_NRT/-124,32,-114,42/1"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url)

    if resp.status_code != 200:
        raise HTTPException(502, "Failed to fetch FIRMS data")

    fires = []
    for row in csv.DictReader(io.StringIO(resp.text)):
        try:
            fires.append({
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
                "brightness": float(row["bright_ti4"]),
            })
        except (KeyError, ValueError):
            continue
    return fires


@app.get("/health")
def health():
    return {"status": "ok", "grid_loaded": len(risk_grid) > 0}
