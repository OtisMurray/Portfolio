from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = BASE_DIR / "data/processed/firesight_model_dataset_us_2020_2025.parquet"
OUTPUT_DIR = BASE_DIR / "data/processed"

OUTPUT_COMPARISON_PATH = OUTPUT_DIR / "random_forest_model_comparison_test_2024_2025.csv"
OUTPUT_THRESHOLDS_PATH = OUTPUT_DIR / "random_forest_threshold_metrics_test_2024_2025.csv"
OUTPUT_IMPORTANCE_PATH = OUTPUT_DIR / "random_forest_feature_importance.csv"
OUTPUT_TEST_PRED_PATH = OUTPUT_DIR / "random_forest_test_predictions_2024_2025.csv"
OUTPUT_PROJECTION_PATH = OUTPUT_DIR / "random_forest_projection_2026_jun_oct.csv"

MODEL_NAME = "Random Forest"
RANDOM_STATE = 42
TRAIN_YEARS = [2020, 2021, 2022, 2023]
TEST_YEARS = [2024, 2025]
PROJECTION_MONTHS = [6, 7, 8, 9, 10]

MAX_TRAIN_ROWS = 750_000
MAX_TEST_ROWS = 1_500_000

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
USE_FIRE_HISTORY_IN_EVAL = True
USE_FIRE_HISTORY_IN_PROJECTION = False

BASE_FEATURE_COLS = [
    "temperature_c", "temperature_c_max", "wind_speed", "wind_speed_max",
    "precipitation_m", "temp_norm", "temp_max_norm", "wind_norm",
    "wind_max_norm", "precip_deficit_norm", "firesight_risk_score",
    "month", "day_of_year", "lat_cell", "lon_cell", "temp_3d_avg",
    "temp_7d_avg", "temp_max_3d_avg", "temp_max_7d_avg", "wind_3d_avg",
    "wind_7d_avg", "wind_max_3d_avg", "wind_max_7d_avg", "precip_3d_sum",
    "precip_7d_sum",
]

FIRE_HISTORY_FEATURE_COLS = [
    "fire_count_prev_1d", "fire_count_prev_3d", "fire_count_prev_7d",
    "fire_occurred_prev_1d", "fire_occurred_prev_3d", "fire_occurred_prev_7d",
]

EVAL_FEATURE_COLS = BASE_FEATURE_COLS + FIRE_HISTORY_FEATURE_COLS if USE_FIRE_HISTORY_IN_EVAL else BASE_FEATURE_COLS
PROJECTION_FEATURE_COLS = BASE_FEATURE_COLS + FIRE_HISTORY_FEATURE_COLS if USE_FIRE_HISTORY_IN_PROJECTION else BASE_FEATURE_COLS


def format_seconds(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def maybe_sample(df: pd.DataFrame, max_rows: int, stratify_col: str) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()

    pos = df[df[stratify_col] == 1]
    neg = df[df[stratify_col] == 0]

    pos_frac = len(pos) / len(df)
    n_pos = max(1, int(max_rows * pos_frac))
    n_neg = max_rows - n_pos

    pos_sample = pos.sample(n=min(len(pos), n_pos), random_state=RANDOM_STATE)
    neg_sample = neg.sample(n=min(len(neg), n_neg), random_state=RANDOM_STATE)

    out = pd.concat([pos_sample, neg_sample], ignore_index=True)
    return out.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


def minmax_normalize(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=np.float32)
    return ((series - min_val) / (max_val - min_val)).astype(np.float32)


def add_normalized_and_score_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["temp_norm"] = minmax_normalize(df["temperature_c"])
    df["temp_max_norm"] = minmax_normalize(df["temperature_c_max"])
    df["wind_norm"] = minmax_normalize(df["wind_speed"])
    df["wind_max_norm"] = minmax_normalize(df["wind_speed_max"])
    df["precip_norm"] = minmax_normalize(df["precipitation_m"])
    df["precip_deficit_norm"] = (1 - df["precip_norm"]).astype(np.float32)
    df["firesight_risk_score"] = (
        0.30 * df["temp_norm"]
        + 0.20 * df["temp_max_norm"]
        + 0.20 * df["wind_norm"]
        + 0.20 * df["wind_max_norm"]
        + 0.10 * df["precip_deficit_norm"]
    ).astype(np.float32)
    return df


def evaluate_thresholds(y_true: pd.Series, probs: np.ndarray) -> pd.DataFrame:
    y_true_np = y_true.to_numpy()
    rows = []
    auc = roc_auc_score(y_true_np, probs)

    for threshold in THRESHOLDS:
        preds = (probs >= threshold).astype(np.uint8)
        rows.append({
            "model": MODEL_NAME,
            "threshold": threshold,
            "roc_auc": auc,
            "precision": precision_score(y_true_np, preds, zero_division=0),
            "recall": recall_score(y_true_np, preds, zero_division=0),
            "f1": f1_score(y_true_np, preds, zero_division=0),
            "predicted_positives": int(preds.sum()),
            "true_positives": int(((preds == 1) & (y_true_np == 1)).sum()),
        })
    return pd.DataFrame(rows)


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=24,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def get_importance_df(model, feature_cols: list[str]) -> pd.DataFrame | None:
    if hasattr(model, "feature_importances_"):
        return pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
    return None


if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Could not find dataset: {DATASET_PATH}")

overall_start = time.time()

print("\n--- LOADING DATASET ---")
dataset = pd.read_parquet(DATASET_PATH)
dataset["date"] = pd.to_datetime(dataset["date"])

train_df = dataset[dataset["year"].isin(TRAIN_YEARS)].copy()
test_df = dataset[dataset["year"].isin(TEST_YEARS)].copy()

train_df = maybe_sample(train_df, MAX_TRAIN_ROWS, "fire_occurred")
test_df = maybe_sample(test_df, MAX_TEST_ROWS, "fire_occurred")

X_train = train_df[EVAL_FEATURE_COLS]
y_train = train_df["fire_occurred"].astype(np.uint8)
X_test = test_df[EVAL_FEATURE_COLS]
y_test = test_df["fire_occurred"].astype(np.uint8)

print("Train:", X_train.shape, "| positives:", int(y_train.sum()))
print("Test: ", X_test.shape, "| positives:", int(y_test.sum()))

print(f"\n--- TRAINING {MODEL_NAME.upper()} EVALUATION MODEL ---")
eval_model = build_model()

fit_start = time.time()
eval_model.fit(X_train, y_train)
fit_seconds = time.time() - fit_start
print("Evaluation fit time:", format_seconds(fit_seconds))

pred_start = time.time()
test_probs = eval_model.predict_proba(X_test)[:, 1]
pred_seconds = time.time() - pred_start
print("Evaluation predict time:", format_seconds(pred_seconds))

threshold_df = evaluate_thresholds(y_test, test_probs)
best_row = threshold_df.sort_values(["f1", "precision", "recall"], ascending=False).iloc[0]

comparison_df = pd.DataFrame([{
    "model": MODEL_NAME,
    "best_threshold_by_f1": best_row["threshold"],
    "roc_auc": best_row["roc_auc"],
    "precision": best_row["precision"],
    "recall": best_row["recall"],
    "f1": best_row["f1"],
    "predicted_positives": best_row["predicted_positives"],
    "true_positives": best_row["true_positives"],
    "fit_seconds": round(fit_seconds, 2),
    "predict_seconds": round(pred_seconds, 2),
}])

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
comparison_df.to_csv(OUTPUT_COMPARISON_PATH, index=False)
threshold_df.to_csv(OUTPUT_THRESHOLDS_PATH, index=False)

test_results = test_df.copy()
test_results["predicted_fire_probability"] = test_probs.astype(np.float32)
test_results["best_model_name"] = MODEL_NAME

for threshold in THRESHOLDS:
    col = f"pred_at_{str(threshold).replace('.', '_')}"
    test_results[col] = (test_probs >= threshold).astype(np.uint8)

test_results["predicted_fire_risk_label"] = (test_probs >= float(best_row["threshold"])).astype(np.uint8)
test_results.to_csv(OUTPUT_TEST_PRED_PATH, index=False)

importance_df = get_importance_df(eval_model, EVAL_FEATURE_COLS)
if importance_df is not None:
    importance_df.to_csv(OUTPUT_IMPORTANCE_PATH, index=False)

print("Saved comparison to:", OUTPUT_COMPARISON_PATH)
print("Saved thresholds to:", OUTPUT_THRESHOLDS_PATH)
print("Saved test predictions to:", OUTPUT_TEST_PRED_PATH)

print(f"\n--- TRAINING FINAL {MODEL_NAME.upper()} PROJECTION MODEL ---")
full_train_df = maybe_sample(dataset.copy(), MAX_TRAIN_ROWS * 2, "fire_occurred")
X_full_train = full_train_df[PROJECTION_FEATURE_COLS]
y_full_train = full_train_df["fire_occurred"].astype(np.uint8)

projection_model = build_model()
proj_fit_start = time.time()
projection_model.fit(X_full_train, y_full_train)
proj_fit_seconds = time.time() - proj_fit_start
print("Projection fit time:", format_seconds(proj_fit_seconds))

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

X_projection = projection_df[PROJECTION_FEATURE_COLS]

proj_pred_start = time.time()
projection_probs = projection_model.predict_proba(X_projection)[:, 1]
proj_pred_seconds = time.time() - proj_pred_start
print("Projection predict time:", format_seconds(proj_pred_seconds))

projection_results = projection_df.copy()
projection_results["predicted_fire_probability"] = projection_probs.astype(np.float32)
projection_results["best_model_name"] = MODEL_NAME
projection_results["projection_type"] = "historically_informed_seasonal_projection"

for threshold in THRESHOLDS:
    col = f"pred_at_{str(threshold).replace('.', '_')}"
    projection_results[col] = (projection_probs >= threshold).astype(np.uint8)

projection_results["predicted_fire_risk_label"] = (projection_probs >= float(best_row["threshold"])).astype(np.uint8)
projection_results.to_csv(OUTPUT_PROJECTION_PATH, index=False)

print("Saved projection to:", OUTPUT_PROJECTION_PATH)
print("\nTotal script time:", format_seconds(time.time() - overall_start))