from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from xgboost import XGBClassifier


# ======================
# PATHS
# ======================
BASE_DIR = Path(__file__).resolve().parents[2]

ERA5_2020_2023_ACCUM_PATH = BASE_DIR / "data/raw/era5/era5_us_2020_2023_accum.nc"
ERA5_2020_2023_INSTANT_PATH = BASE_DIR / "data/raw/era5/era5_us_2020_2023_instant.nc"

ERA5_2024_2025_ACCUM_PATH = BASE_DIR / "data/raw/era5/era5_us_2024_2025_accum.nc"
ERA5_2024_2025_INSTANT_PATH = BASE_DIR / "data/raw/era5/era5_us_2024_2025_instant.nc"

FIRMS_SUOMI_PATH = BASE_DIR / "data/raw/firms/suomi_viirs_us_2020_2025.csv"
FIRMS_NOAA20_PATH = BASE_DIR / "data/raw/firms/noaa20_viirs_us_2020_2025.csv"

OUTPUT_DATASET_PATH = BASE_DIR / "data/processed/firesight_model_dataset_us_2020_2025.parquet"
OUTPUT_MODEL_COMPARISON_PATH = BASE_DIR / "data/processed/model_comparison_test_2024_2025.csv"
OUTPUT_BEST_THRESHOLDS_PATH = BASE_DIR / "data/processed/best_model_threshold_metrics_2024_2025.csv"
OUTPUT_IMPORTANCE_PATH = BASE_DIR / "data/processed/best_model_feature_importance_2024_2025.csv"
OUTPUT_TEST_PRED_PATH = BASE_DIR / "data/processed/test_predictions_2024_2025.csv"
OUTPUT_PROJECTION_PATH = BASE_DIR / "data/processed/projection_2026_jun_oct.csv"

# ======================
# SETTINGS
# ======================
GRID_SIZE = 0.25
RANDOM_STATE = 42
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.85]

TRAIN_YEARS = [2020, 2021, 2022, 2023]
TEST_YEARS = [2024, 2025]
PROJECTION_MONTHS = [6, 7, 8, 9, 10]

USE_CACHED_DATASET = True
MAX_TRAIN_ROWS = 750_000
MAX_TEST_ROWS = 1_500_000

USE_FIRMS_FILTERING = True
KEEP_FIRE_TYPE = 0
DROP_LOW_CONFIDENCE = True
MIN_FRP = 5.0
MIN_FIRE_COUNT_FOR_POSITIVE = 2

COMPARISON_THRESHOLD = 0.8
DEFAULT_MAP_THRESHOLD = 0.8

USE_FIRE_HISTORY_IN_EVAL = True
USE_FIRE_HISTORY_IN_PROJECTION = False

MODEL_NAMES = ["Extra Trees", "XGBoost"]


# ======================
# HELPERS
# ======================
def minmax_normalize(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=np.float32)
    return ((series - min_val) / (max_val - min_val)).astype(np.float32)


def find_var(ds: xr.Dataset, candidates: list[str]) -> str:
    for var in candidates:
        if var in ds.data_vars:
            return var
    raise KeyError(f"Could not find any of these variables: {candidates}")


def floor_to_grid(series: pd.Series, grid_size: float) -> pd.Series:
    return (np.floor(series / grid_size) * grid_size).astype(np.float32)


def evaluate_thresholds(y_true: pd.Series, probs: np.ndarray, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    y_true_np = y_true.to_numpy()

    for threshold in thresholds:
        preds = (probs >= threshold).astype(np.int8)
        rows.append({
            "threshold": threshold,
            "precision": precision_score(y_true_np, preds, zero_division=0),
            "recall": recall_score(y_true_np, preds, zero_division=0),
            "f1": f1_score(y_true_np, preds, zero_division=0),
            "predicted_positives": int(preds.sum()),
            "true_positives": int(((preds == 1) & (y_true_np == 1)).sum()),
        })
    return pd.DataFrame(rows)


def summarize_model_at_threshold(model_name: str, y_true: pd.Series, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(np.int8)
    y_true_np = y_true.to_numpy()
    return {
        "model": model_name,
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true_np, probs),
        "precision": precision_score(y_true_np, preds, zero_division=0),
        "recall": recall_score(y_true_np, preds, zero_division=0),
        "f1": f1_score(y_true_np, preds, zero_division=0),
        "predicted_positives": int(preds.sum()),
        "true_positives": int(((preds == 1) & (y_true_np == 1)).sum()),
    }


def print_model_results(model_name: str, split_name: str, y_true: pd.Series, probs: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
    preds = (probs >= threshold).astype(np.int8)
    auc = roc_auc_score(y_true, probs)

    print(f"\n--- {model_name.upper()} | {split_name.upper()} ---")
    print("ROC-AUC:", auc)
    print("Confusion matrix:\n", confusion_matrix(y_true, preds))
    print(classification_report(y_true, preds, digits=4))

    threshold_df = evaluate_thresholds(y_true, probs, THRESHOLDS)
    print(f"\n--- {model_name.upper()} THRESHOLDS | {split_name.upper()} ---")
    print(threshold_df)

    return threshold_df


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col == "date":
            continue
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(np.float32)
        elif pd.api.types.is_integer_dtype(df[col]):
            if df[col].min() >= 0 and df[col].max() < 255:
                df[col] = df[col].astype(np.uint8)
            elif df[col].min() >= -32768 and df[col].max() <= 32767:
                df[col] = df[col].astype(np.int16)
            else:
                df[col] = df[col].astype(np.int32)
    return df


def maybe_sample(df: pd.DataFrame, max_rows: int, stratify_col: str) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df

    pos = df[df[stratify_col] == 1]
    neg = df[df[stratify_col] == 0]

    pos_frac = len(pos) / len(df)
    n_pos = max(1, int(max_rows * pos_frac))
    n_neg = max_rows - n_pos

    pos_sample = pos.sample(n=min(len(pos), n_pos), random_state=RANDOM_STATE)
    neg_sample = neg.sample(n=min(len(neg), n_neg), random_state=RANDOM_STATE)

    out = pd.concat([pos_sample, neg_sample], ignore_index=True)
    return out.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


def build_era5_features(accum_path: Path, instant_path: Path, label: str) -> pd.DataFrame:
    print(f"\n--- LOADING ERA5: {label} ---")

    ds_accum = xr.open_dataset(accum_path)
    ds_instant = xr.open_dataset(instant_path)

    temp_var = find_var(ds_instant, ["t2m"])
    u_var = find_var(ds_instant, ["u10"])
    v_var = find_var(ds_instant, ["v10"])
    precip_var = find_var(ds_accum, ["tp"])

    df_temp = ds_instant[[temp_var, u_var, v_var]].to_dataframe().reset_index()
    df_precip = ds_accum[[precip_var]].to_dataframe().reset_index()

    time_col_temp = "valid_time" if "valid_time" in df_temp.columns else "time"
    time_col_precip = "valid_time" if "valid_time" in df_precip.columns else "time"

    df_temp["time"] = pd.to_datetime(df_temp[time_col_temp])
    df_precip["time"] = pd.to_datetime(df_precip[time_col_precip])

    df_temp["date"] = df_temp["time"].dt.date
    df_precip["date"] = df_precip["time"].dt.date

    df_temp["temperature_c"] = (df_temp[temp_var] - 273.15).astype(np.float32)
    df_temp["wind_speed"] = np.sqrt(df_temp[u_var] ** 2 + df_temp[v_var] ** 2).astype(np.float32)
    df_precip["precipitation_m"] = df_precip[precip_var].astype(np.float32)

    group_cols = ["latitude", "longitude", "date"]

    daily_temp = (
        df_temp.groupby(group_cols, as_index=False)
        .agg(
            temperature_c=("temperature_c", "mean"),
            temperature_c_max=("temperature_c", "max"),
            wind_speed=("wind_speed", "mean"),
            wind_speed_max=("wind_speed", "max"),
        )
    )

    daily_precip = (
        df_precip.groupby(group_cols, as_index=False)
        .agg(
            precipitation_m=("precipitation_m", "sum"),
        )
    )

    era5_daily = daily_temp.merge(daily_precip, on=["latitude", "longitude", "date"], how="inner")
    era5_daily["lat_cell"] = floor_to_grid(era5_daily["latitude"], GRID_SIZE)
    era5_daily["lon_cell"] = floor_to_grid(era5_daily["longitude"], GRID_SIZE)

    era5_grid = (
        era5_daily.groupby(["lat_cell", "lon_cell", "date"], as_index=False)
        .agg(
            temperature_c=("temperature_c", "mean"),
            temperature_c_max=("temperature_c_max", "max"),
            wind_speed=("wind_speed", "mean"),
            wind_speed_max=("wind_speed_max", "max"),
            precipitation_m=("precipitation_m", "sum"),
        )
    )

    return optimize_dtypes(era5_grid)


def build_firms_labels() -> pd.DataFrame:
    print("\n--- LOADING FIRMS ---")
    firms_1 = pd.read_csv(FIRMS_SUOMI_PATH)
    firms_2 = pd.read_csv(FIRMS_NOAA20_PATH)
    firms = pd.concat([firms_1, firms_2], ignore_index=True)

    if USE_FIRMS_FILTERING:
        firms = firms[firms["type"] == KEEP_FIRE_TYPE].copy()

        if DROP_LOW_CONFIDENCE:
            firms = firms[firms["confidence"] != "l"].copy()

        if "frp" in firms.columns:
            firms = firms[firms["frp"] >= MIN_FRP].copy()

    firms["date"] = pd.to_datetime(firms["acq_date"]).dt.date
    firms["lat_cell"] = floor_to_grid(firms["latitude"], GRID_SIZE)
    firms["lon_cell"] = floor_to_grid(firms["longitude"], GRID_SIZE)

    fire_labels = (
        firms.groupby(["lat_cell", "lon_cell", "date"], as_index=False)
        .size()
        .rename(columns={"size": "fire_count"})
    )

    fire_labels["fire_occurred"] = (fire_labels["fire_count"] >= MIN_FIRE_COUNT_FOR_POSITIVE).astype(np.uint8)
    fire_labels = fire_labels[fire_labels["fire_occurred"] == 1].copy()

    fire_labels["fire_count"] = fire_labels["fire_count"].astype(np.int16)
    return optimize_dtypes(fire_labels)


def add_base_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset.copy()
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset["year"] = dataset["date"].dt.year.astype(np.int16)
    dataset["month"] = dataset["date"].dt.month.astype(np.uint8)
    dataset["day_of_year"] = dataset["date"].dt.dayofyear.astype(np.uint16)
    return dataset


def add_normalized_and_score_features(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset.copy()

    dataset["temp_norm"] = minmax_normalize(dataset["temperature_c"])
    dataset["temp_max_norm"] = minmax_normalize(dataset["temperature_c_max"])
    dataset["wind_norm"] = minmax_normalize(dataset["wind_speed"])
    dataset["wind_max_norm"] = minmax_normalize(dataset["wind_speed_max"])
    dataset["precip_norm"] = minmax_normalize(dataset["precipitation_m"])
    dataset["precip_deficit_norm"] = (1 - dataset["precip_norm"]).astype(np.float32)

    dataset["firesight_risk_score"] = (
        0.30 * dataset["temp_norm"]
        + 0.20 * dataset["temp_max_norm"]
        + 0.20 * dataset["wind_norm"]
        + 0.20 * dataset["wind_max_norm"]
        + 0.10 * dataset["precip_deficit_norm"]
    ).astype(np.float32)

    return dataset


def add_lag_features(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset.sort_values(["lat_cell", "lon_cell", "date"]).copy()
    group_cols = ["lat_cell", "lon_cell"]

    dataset["temp_3d_avg"] = dataset.groupby(group_cols)["temperature_c"].transform(lambda s: s.rolling(3, min_periods=1).mean()).astype(np.float32)
    dataset["temp_7d_avg"] = dataset.groupby(group_cols)["temperature_c"].transform(lambda s: s.rolling(7, min_periods=1).mean()).astype(np.float32)

    dataset["temp_max_3d_avg"] = dataset.groupby(group_cols)["temperature_c_max"].transform(lambda s: s.rolling(3, min_periods=1).mean()).astype(np.float32)
    dataset["temp_max_7d_avg"] = dataset.groupby(group_cols)["temperature_c_max"].transform(lambda s: s.rolling(7, min_periods=1).mean()).astype(np.float32)

    dataset["wind_3d_avg"] = dataset.groupby(group_cols)["wind_speed"].transform(lambda s: s.rolling(3, min_periods=1).mean()).astype(np.float32)
    dataset["wind_7d_avg"] = dataset.groupby(group_cols)["wind_speed"].transform(lambda s: s.rolling(7, min_periods=1).mean()).astype(np.float32)

    dataset["wind_max_3d_avg"] = dataset.groupby(group_cols)["wind_speed_max"].transform(lambda s: s.rolling(3, min_periods=1).mean()).astype(np.float32)
    dataset["wind_max_7d_avg"] = dataset.groupby(group_cols)["wind_speed_max"].transform(lambda s: s.rolling(7, min_periods=1).mean()).astype(np.float32)

    dataset["precip_3d_sum"] = dataset.groupby(group_cols)["precipitation_m"].transform(lambda s: s.rolling(3, min_periods=1).sum()).astype(np.float32)
    dataset["precip_7d_sum"] = dataset.groupby(group_cols)["precipitation_m"].transform(lambda s: s.rolling(7, min_periods=1).sum()).astype(np.float32)

    dataset["fire_count_prev_1d"] = dataset.groupby(group_cols)["fire_count"].shift(1).fillna(0).astype(np.float32)
    dataset["fire_count_prev_3d"] = dataset.groupby(group_cols)["fire_count"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum()).fillna(0).astype(np.float32)
    dataset["fire_count_prev_7d"] = dataset.groupby(group_cols)["fire_count"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).sum()).fillna(0).astype(np.float32)

    dataset["fire_occurred_prev_1d"] = (dataset["fire_count_prev_1d"] > 0).astype(np.uint8)
    dataset["fire_occurred_prev_3d"] = (dataset["fire_count_prev_3d"] > 0).astype(np.uint8)
    dataset["fire_occurred_prev_7d"] = (dataset["fire_count_prev_7d"] > 0).astype(np.uint8)

    return dataset


def build_feature_dataset(era5_df: pd.DataFrame, fire_labels: pd.DataFrame) -> pd.DataFrame:
    dataset = era5_df.merge(
        fire_labels[["lat_cell", "lon_cell", "date", "fire_count", "fire_occurred"]],
        on=["lat_cell", "lon_cell", "date"],
        how="left"
    )

    dataset["fire_count"] = dataset["fire_count"].fillna(0).astype(np.float32)
    dataset["fire_occurred"] = dataset["fire_occurred"].fillna(0).astype(np.uint8)

    dataset = add_base_columns(dataset)
    dataset = add_lag_features(dataset)
    dataset = add_normalized_and_score_features(dataset)

    return optimize_dtypes(dataset)


def load_or_build_dataset() -> pd.DataFrame:
    if USE_CACHED_DATASET and OUTPUT_DATASET_PATH.exists():
        print("\n--- LOADING CACHED DATASET ---")
        dataset = pd.read_parquet(OUTPUT_DATASET_PATH)
        dataset["date"] = pd.to_datetime(dataset["date"])
        print("Loaded cached dataset:", OUTPUT_DATASET_PATH)
        print("Dataset shape:", dataset.shape)
        return dataset

    era5_2020_2023 = build_era5_features(
        ERA5_2020_2023_ACCUM_PATH,
        ERA5_2020_2023_INSTANT_PATH,
        "2020_2023",
    )
    era5_2024_2025 = build_era5_features(
        ERA5_2024_2025_ACCUM_PATH,
        ERA5_2024_2025_INSTANT_PATH,
        "2024_2025",
    )

    era5_hist = pd.concat([era5_2020_2023, era5_2024_2025], ignore_index=True)
    fire_labels = build_firms_labels()

    print("\n--- BUILDING HISTORICAL MODEL DATASET ---")
    dataset = build_feature_dataset(era5_hist, fire_labels)

    OUTPUT_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(OUTPUT_DATASET_PATH, index=False)
    print("Saved dataset to:", OUTPUT_DATASET_PATH)
    print("Dataset shape:", dataset.shape)
    return dataset


def make_model_dict(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / max(pos_count, 1)

    models = {}

    if "Extra Trees" in MODEL_NAMES:
        models["Extra Trees"] = ExtraTreesClassifier(
            n_estimators=250,
            max_depth=16,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if "XGBoost" in MODEL_NAMES:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )

    return models


# ======================
# FEATURE LISTS
# ======================
BASE_FEATURE_COLS = [
    "temperature_c",
    "temperature_c_max",
    "wind_speed",
    "wind_speed_max",
    "precipitation_m",
    "temp_norm",
    "temp_max_norm",
    "wind_norm",
    "wind_max_norm",
    "precip_deficit_norm",
    "firesight_risk_score",
    "month",
    "day_of_year",
    "lat_cell",
    "lon_cell",
    "temp_3d_avg",
    "temp_7d_avg",
    "temp_max_3d_avg",
    "temp_max_7d_avg",
    "wind_3d_avg",
    "wind_7d_avg",
    "wind_max_3d_avg",
    "wind_max_7d_avg",
    "precip_3d_sum",
    "precip_7d_sum",
]

FIRE_HISTORY_FEATURE_COLS = [
    "fire_count_prev_1d",
    "fire_count_prev_3d",
    "fire_count_prev_7d",
    "fire_occurred_prev_1d",
    "fire_occurred_prev_3d",
    "fire_occurred_prev_7d",
]

EVAL_FEATURE_COLS = BASE_FEATURE_COLS + FIRE_HISTORY_FEATURE_COLS if USE_FIRE_HISTORY_IN_EVAL else BASE_FEATURE_COLS
PROJECTION_FEATURE_COLS = BASE_FEATURE_COLS if not USE_FIRE_HISTORY_IN_PROJECTION else BASE_FEATURE_COLS + FIRE_HISTORY_FEATURE_COLS


# ======================
# LOAD DATASET
# ======================
dataset = load_or_build_dataset()
print(dataset["fire_occurred"].value_counts(dropna=False))

# ======================
# EVALUATION
# ======================
train_df = dataset[dataset["year"].isin(TRAIN_YEARS)].copy()
test_df = dataset[dataset["year"].isin(TEST_YEARS)].copy()

train_df = maybe_sample(train_df, MAX_TRAIN_ROWS, "fire_occurred")
test_df = maybe_sample(test_df, MAX_TEST_ROWS, "fire_occurred")

X_train = train_df[EVAL_FEATURE_COLS]
y_train = train_df["fire_occurred"]

X_test = test_df[EVAL_FEATURE_COLS]
y_test = test_df["fire_occurred"]

print("\n--- FAST EVALUATION SPLIT SIZES ---")
print("Train shape:", X_train.shape, "positives:", int(y_train.sum()))
print("Test shape:", X_test.shape, "positives:", int(y_test.sum()))

models = make_model_dict(X_train, y_train)

comparison_rows = []
threshold_results = {}
fitted_models = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    fitted_models[model_name] = model

    probs = model.predict_proba(X_test)[:, 1]
    threshold_df = print_model_results(model_name, "test_2024_2025_fast", y_test, probs)
    threshold_results[model_name] = threshold_df

    comparison_rows.append(
        summarize_model_at_threshold(model_name, y_test, probs, COMPARISON_THRESHOLD)
    )

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(OUTPUT_MODEL_COMPARISON_PATH, index=False)
print("Saved model comparison to:", OUTPUT_MODEL_COMPARISON_PATH)

best_model_name = comparison_df.sort_values("roc_auc", ascending=False).iloc[0]["model"]
best_eval_model = fitted_models[best_model_name]
best_test_probs = best_eval_model.predict_proba(X_test)[:, 1]

print("\n--- BEST FAST EVALUATION MODEL ---")
print(best_model_name)

best_threshold_df = threshold_results[best_model_name]
best_threshold_df.to_csv(OUTPUT_BEST_THRESHOLDS_PATH, index=False)
print("Saved best-model threshold metrics to:", OUTPUT_BEST_THRESHOLDS_PATH)

if hasattr(best_eval_model, "feature_importances_"):
    importances = pd.DataFrame({
        "feature": EVAL_FEATURE_COLS,
        "importance": best_eval_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importances.to_csv(OUTPUT_IMPORTANCE_PATH, index=False)
    print("Saved feature importances to:", OUTPUT_IMPORTANCE_PATH)

test_results = test_df.copy()
test_results["predicted_fire_probability"] = best_test_probs
test_results["best_model_name"] = best_model_name

for threshold in THRESHOLDS:
    col_name = f"pred_at_{str(threshold).replace('.', '_')}"
    test_results[col_name] = (best_test_probs >= threshold).astype(np.uint8)

test_results["predicted_fire_risk_label"] = (best_test_probs >= DEFAULT_MAP_THRESHOLD).astype(np.uint8)
test_results.to_csv(OUTPUT_TEST_PRED_PATH, index=False)
print("Saved test predictions to:", OUTPUT_TEST_PRED_PATH)

# ======================
# FINAL PROJECTION MODEL
# ======================
print("\n--- TRAINING FINAL PROJECTION MODEL ON SAMPLED 2020-2025 ---")
full_train_df = maybe_sample(dataset.copy(), MAX_TRAIN_ROWS * 2, "fire_occurred")

X_full_train = full_train_df[PROJECTION_FEATURE_COLS]
y_full_train = full_train_df["fire_occurred"]

final_models = make_model_dict(X_full_train, y_full_train)
final_projection_model = final_models[best_model_name]
final_projection_model.fit(X_full_train, y_full_train)

projection_source = dataset[dataset["month"].isin(PROJECTION_MONTHS)].copy()

projection_agg = (
    projection_source.groupby(["lat_cell", "lon_cell", "month"], as_index=False)
    .agg(
        temperature_c=("temperature_c", "mean"),
        temperature_c_max=("temperature_c_max", "mean"),
        wind_speed=("wind_speed", "mean"),
        wind_speed_max=("wind_speed_max", "mean"),
        precipitation_m=("precipitation_m", "mean"),
        temp_3d_avg=("temp_3d_avg", "mean"),
        temp_7d_avg=("temp_7d_avg", "mean"),
        temp_max_3d_avg=("temp_max_3d_avg", "mean"),
        temp_max_7d_avg=("temp_max_7d_avg", "mean"),
        wind_3d_avg=("wind_3d_avg", "mean"),
        wind_7d_avg=("wind_7d_avg", "mean"),
        wind_max_3d_avg=("wind_max_3d_avg", "mean"),
        wind_max_7d_avg=("wind_max_7d_avg", "mean"),
        precip_3d_sum=("precip_3d_sum", "mean"),
        precip_7d_sum=("precip_7d_sum", "mean"),
    )
)

projection_rows = []
for month in PROJECTION_MONTHS:
    month_df = projection_agg[projection_agg["month"] == month].copy()
    month_df["date"] = pd.Timestamp(year=2026, month=month, day=15)
    projection_rows.append(month_df)

projection_df = pd.concat(projection_rows, ignore_index=True)
projection_df["year"] = 2026
projection_df["day_of_year"] = pd.to_datetime(projection_df["date"]).dt.dayofyear.astype(np.uint16)

projection_df["fire_count_prev_1d"] = 0
projection_df["fire_count_prev_3d"] = 0
projection_df["fire_count_prev_7d"] = 0
projection_df["fire_occurred_prev_1d"] = 0
projection_df["fire_occurred_prev_3d"] = 0
projection_df["fire_occurred_prev_7d"] = 0

projection_df = add_normalized_and_score_features(projection_df)
projection_df = optimize_dtypes(projection_df)

X_projection = projection_df[PROJECTION_FEATURE_COLS]
projection_probs = final_projection_model.predict_proba(X_projection)[:, 1]

projection_results = projection_df.copy()
projection_results["predicted_fire_probability"] = projection_probs.astype(np.float32)
projection_results["best_model_name"] = best_model_name
projection_results["projection_type"] = "historically_informed_seasonal_projection"

for threshold in THRESHOLDS:
    col_name = f"pred_at_{str(threshold).replace('.', '_')}"
    projection_results[col_name] = (projection_probs >= threshold).astype(np.uint8)

projection_results["predicted_fire_risk_label"] = (projection_probs >= DEFAULT_MAP_THRESHOLD).astype(np.uint8)
projection_results.to_csv(OUTPUT_PROJECTION_PATH, index=False)
print("Saved 2026 Jun-Oct projection to:", OUTPUT_PROJECTION_PATH)