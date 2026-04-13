from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st


st.set_page_config(
    page_title="FireSight | U.S. Wildfire Risk Map",
    page_icon="🔥",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parents[2]
TEST_DATA_PATH = BASE_DIR / "data/processed/test_predictions_2024_2025.csv"
PROJECTION_DATA_PATH = BASE_DIR / "data/processed/projection_2026_jun_oct.csv"

st.title("🔥 FireSight")
st.caption("Interactive U.S. wildfire risk map using historical test predictions and a 2026 seasonal projection")


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    required_cols = ["lat_cell", "lon_cell", "predicted_fire_probability"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "best_model_name" not in df.columns:
        df["best_model_name"] = "Unknown"

    if "projection_type" not in df.columns:
        df["projection_type"] = "historical_prediction"

    if "date" in df.columns:
        df["month_str"] = df["date"].dt.to_period("M").astype(str)
        df["day_str"] = df["date"].dt.strftime("%Y-%m-%d")
    else:
        df["month_str"] = "Unknown"
        df["day_str"] = "Unknown"

    return df

def keep_us_only(df: pd.DataFrame) -> pd.DataFrame:
    lat = df["lat_cell"]
    lon = df["lon_cell"]

    return df[
        (lat >= 25.0) &
        (lat <= 49.5) &
        (lon >= -124.8) &
        (lon <= -66.0) &

        # remove Baja / far southwest Mexico
        ~((lon < -114.0) & (lat < 32.5)) &

        # remove Sonora / Chihuahua spill
        ~((lon >= -114.0) & (lon < -108.0) & (lat < 31.8)) &

        # remove south of Texas border region
        ~((lon >= -108.0) & (lon < -93.0) & (lat < 29.8)) &

        # remove Gulf spill
        ~((lon >= -93.0) & (lat < 28.8))
    ].copy()

st.sidebar.header("Map Controls")

dataset_choice = st.sidebar.selectbox(
    "Dataset",
    ["2026 Jun–Oct Projection", "2024–2025 Test Predictions"],
    index=0,
)

data_path = PROJECTION_DATA_PATH if dataset_choice == "2026 Jun–Oct Projection" else TEST_DATA_PATH

if not data_path.exists():
    st.error(f"Could not find data file:\n{data_path}")
    st.stop()

try:
    df = load_data(data_path)
except Exception as exc:
    st.error(f"Could not load prediction data: {exc}")
    st.stop()

view_mode_options = ["Monthly"]
if dataset_choice == "2024–2025 Test Predictions":
    view_mode_options.append("Daily")

view_mode = st.sidebar.selectbox("View mode", view_mode_options, index=0)

if view_mode == "Monthly":
    available_months = sorted(df["month_str"].dropna().unique())
    selected_month = st.sidebar.selectbox("Month", available_months, index=0)
else:
    available_dates = sorted(df["date"].dt.date.unique())
    selected_date = st.sidebar.selectbox("Date", available_dates, index=0)

min_prob = st.sidebar.slider(
    "Minimum predicted fire probability",
    min_value=0.0,
    max_value=1.0,
    value=0.30,
    step=0.01,
)

threshold_choice = st.sidebar.selectbox(
    "Threshold overlay",
    ["None", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"],
    index=2,
)

show_heatmap = st.sidebar.checkbox("Show heatmap", value=False)
show_points = st.sidebar.checkbox("Show hotspot points", value=True)

max_points = st.sidebar.slider(
    "Max rendered cells",
    min_value=100,
    max_value=10000,
    value=3000,
    step=100,
)

hotspot_threshold = st.sidebar.slider(
    "Hotspot threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.80,
    step=0.01,
)

point_radius = st.sidebar.slider(
    "Point radius multiplier",
    min_value=50,
    max_value=500,
    value=150,
    step=25,
)

heatmap_radius = st.sidebar.slider(
    "Heatmap radius (pixels)",
    min_value=6,
    max_value=20,
    value=10,
    step=2,
)

heatmap_intensity = st.sidebar.slider(
    "Heatmap intensity",
    min_value=0.05,
    max_value=0.8,
    value=0.12,
    step=0.01,
)

map_style_choice = st.sidebar.selectbox(
    "Base map",
    ["OpenStreetMap", "Road", "Light", "Dark"],
    index=0,
)

map_styles = {
    "OpenStreetMap": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    "Road": "road",
    "Light": "light",
    "Dark": "dark",
}

if view_mode == "Monthly":
    filtered = df[df["month_str"] == selected_month].copy()
else:
    filtered = df[df["date"].dt.date == selected_date].copy()

filtered = filtered[filtered["predicted_fire_probability"] >= min_prob].copy()

if threshold_choice != "None":
    threshold_col = f"pred_at_{threshold_choice.replace('.', '_')}"
    if threshold_col in filtered.columns:
        filtered = filtered[filtered[threshold_col] == 1].copy()

if filtered.empty:
    st.warning("No map points match the current filters. Lower the minimum probability or change the time selection.")
    st.stop()

if view_mode == "Monthly":
    map_df = (
        filtered.groupby(["lat_cell", "lon_cell"], as_index=False)
        .agg(
            predicted_fire_probability=("predicted_fire_probability", "max"),
            mean_predicted_fire_probability=("predicted_fire_probability", "mean"),
            days_flagged=("day_str", "nunique"),
            best_model_name=("best_model_name", "first"),
            projection_type=("projection_type", "first"),
        )
    )
    map_df["label_time"] = selected_month
else:
    map_df = filtered.copy()
    map_df["days_flagged"] = 1
    map_df["label_time"] = map_df["day_str"]

map_df = map_df.nlargest(min(max_points, len(map_df)), "predicted_fire_probability").copy()

# actually remove Mexico / border spillover
map_df = keep_us_only(map_df)

map_df["prob_percent"] = (map_df["predicted_fire_probability"] * 100).round(1)

map_df["r"] = 255
map_df["g"] = (215 - map_df["predicted_fire_probability"] * 165).clip(lower=30).astype(int)
map_df["b"] = (80 - map_df["predicted_fire_probability"] * 60).clip(lower=5).astype(int)
map_df["a"] = (70 + map_df["predicted_fire_probability"] * 120).clip(upper=210).astype(int)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dataset", "Projection" if "Projection" in dataset_choice else "Test")
c2.metric("Time", selected_month if view_mode == "Monthly" else str(selected_date))
c3.metric("Rendered cells", f"{len(map_df):,}")
c4.metric("Mean predicted risk", f"{map_df['predicted_fire_probability'].mean():.3f}")

st.markdown("---")

layers = []

if show_heatmap:
    heat_df = map_df.nlargest(min(max_points, len(map_df)), "predicted_fire_probability").copy()
    layers.append(
        pdk.Layer(
            "HeatmapLayer",
            data=heat_df,
            get_position="[lon_cell, lat_cell]",
            get_weight="predicted_fire_probability",
            opacity=0.35,
            threshold=0.15,
            intensity=0.55,
            radiusPixels=22,
            pickable=False,
        )
    )

if show_points:
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[lon_cell, lat_cell]",
            get_fill_color="[r, g, b, a]",
            get_radius=4,
            radius_min_pixels=2,
            radius_max_pixels=8,
            pickable=True,
            auto_highlight=True,
            stroked=False,
        )
    )

tooltip = {
    "html": """
    <div style='font-family: Arial; font-size: 13px;'>
        <b>Predicted fire probability:</b> {prob_percent}%<br/>
        <b>Time:</b> {label_time}<br/>
        <b>Days flagged:</b> {days_flagged}<br/>
        <b>Latitude:</b> {lat_cell}<br/>
        <b>Longitude:</b> {lon_cell}<br/>
        <b>Best model:</b> {best_model_name}<br/>
        <b>Type:</b> {projection_type}
    </div>
    """,
    "style": {
        "backgroundColor": "rgba(20, 20, 20, 0.92)",
        "color": "white",
        "borderRadius": "8px",
    },
}

view_state = pdk.ViewState(
    latitude=39.5,
    longitude=-98.35,
    zoom=3.7,
    pitch=0,
)

deck = pdk.Deck(
    layers=layers,
    initial_view_state=view_state,
    map_style=map_styles[map_style_choice],
    tooltip=tooltip,
)

st.pydeck_chart(deck, use_container_width=True)

st.subheader("Highest-risk locations")

display_cols = [
    "lat_cell",
    "lon_cell",
    "predicted_fire_probability",
    "days_flagged",
    "best_model_name",
    "projection_type",
]

preview = (
    map_df[
        [
            "lat_cell",
            "lon_cell",
            "predicted_fire_probability",
            "days_flagged",
            "best_model_name",
            "projection_type",
        ]
    ]
    .sort_values("predicted_fire_probability", ascending=False)
    .head(25)
    .rename(
        columns={
            "lat_cell": "latitude",
            "lon_cell": "longitude",
            "predicted_fire_probability": "risk_probability",
        }
    )
)

preview["risk_probability"] = preview["risk_probability"].round(4)
preview["latitude"] = preview["latitude"].round(3)
preview["longitude"] = preview["longitude"].round(3)

st.dataframe(preview, use_container_width=True, hide_index=True)

preview["risk_probability"] = preview["risk_probability"].round(4)
st.dataframe(preview, use_container_width=True, hide_index=True)

csv_data = map_df.to_csv(index=False).encode("utf-8")
download_name = (
    f"firesight_{selected_month.replace('-', '_')}.csv"
    if view_mode == "Monthly"
    else f"firesight_{str(selected_date).replace('-', '_')}.csv"
)

st.download_button(
    label="Download filtered map data as CSV",
    data=csv_data,
    file_name=download_name,
    mime="text/csv",
)

st.markdown("---")
st.caption(
    "FireSight visualizes U.S. wildfire risk using model-based historical predictions and a historically informed 2026 seasonal projection."
)