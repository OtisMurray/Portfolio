from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ======================
# PATH SETUP
# ======================
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = BASE_DIR / "data/processed/na_pm25_cells_clean.csv"
OUTPUT_FILE = BASE_DIR / "data/processed/lr_predictions_2023.csv"

# ======================
# LOAD DATA
# ======================
df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])

# Sort for time series
df = df.sort_values(["lat", "lon", "date"])

print("Loaded rows:", len(df))

# ======================
# FEATURE ENGINEERING
# ======================
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year

# Time index (trend)
df["time_index"] = (df["year"] - df["year"].min()) * 12 + df["month"]

# Seasonal encoding
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# Lag features
df["pm25_lag1"] = df.groupby(["lat", "lon"])["pm25"].shift(1)
df["pm25_lag2"] = df.groupby(["lat", "lon"])["pm25"].shift(2)

# Drop missing
df = df.dropna()

print("Rows after lagging:", len(df))

# ======================
# TRAIN / VAL / TEST SPLIT
# ======================
train = df[df["year"] <= 2020]
val   = df[df["year"] == 2021]
test  = df[df["year"] == 2022]

features = [
    "lat", "lon",
    "time_index",
    "month_sin", "month_cos",
    "pm25_lag1", "pm25_lag2"
]

target = "pm25"

X_train, y_train = train[features], train[target]
X_val, y_val     = val[features], val[target]
X_test, y_test   = test[features], test[target]

print("Train:", len(X_train), "Val:", len(X_val), "Test:", len(X_test))

# ======================
# TRAIN MODEL
# ======================
model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained.")

# ======================
# VALIDATION PERFORMANCE
# ======================
val_preds = model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print("Validation RMSE (2021):", val_rmse)

# ======================
# TEST PERFORMANCE
# ======================
test_preds = model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
print("Test RMSE (2022):", test_rmse)

# ======================
# NAIVE BASELINE (IMPORTANT)
# ======================
naive_preds = X_test["pm25_lag1"]
naive_rmse = np.sqrt(mean_squared_error(y_test, naive_preds))
print("Naive RMSE (lag1):", naive_rmse)

# ======================
# FORECAST 2023 (MEMORY SAFE)
# ======================
OUTPUT_FILE = BASE_DIR / "data/processed/lr_predictions_2023.csv"

# Delete old file if it exists
if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

future_steps = 12
current_df = df[df["date"] == df["date"].max()].copy()

print("\n--- FORECASTING 2023 ---")

for step in range(future_steps):

    # Move forward one month
    current_df["date"] = current_df["date"] + pd.DateOffset(months=1)
    current_df["month"] = current_df["date"].dt.month
    current_df["year"] = current_df["date"].dt.year

    # Recompute features
    current_df["time_index"] = (current_df["year"] - df["year"].min()) * 12 + current_df["month"]
    current_df["month_sin"] = np.sin(2 * np.pi * current_df["month"] / 12)
    current_df["month_cos"] = np.cos(2 * np.pi * current_df["month"] / 12)

    # Update lag features
    current_df["pm25_lag2"] = current_df["pm25_lag1"]
    current_df["pm25_lag1"] = current_df["pm25"]

    # Predict
    X_future = current_df[features]
    preds = model.predict(X_future)

    current_df["pm25"] = preds

    # ✅ Save ONLY needed columns (smaller file)
    save_df = current_df[["lat", "lon", "date", "pm25"]]

    save_df.to_csv(
        OUTPUT_FILE,
        mode="a",
        header=not OUTPUT_FILE.exists(),
        index=False
    )

    print(f"Saved step {step+1}/12")

print("\nForecast complete.")
print("Saved to:", OUTPUT_FILE)