from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# ======================
# PATHS
# ======================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = BASE_DIR / "data/processed/na_pm25_cells_clean.csv"
OUTPUT_FILE = BASE_DIR / "data/processed/xgb_predictions_2023.csv"

# ======================
# LOAD DATA
# ======================
print("\n--- LOADING DATA ---")
df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["lat", "lon", "date"])

print("Rows:", len(df))

# ======================
# FEATURE ENGINEERING
# ======================
print("\n--- FEATURE ENGINEERING ---")

df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year

df["time_index"] = (df["year"] - df["year"].min()) * 12 + df["month"]

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# ONLY ONE LAG (more realistic)
df["pm25_lag1"] = df.groupby(["lat", "lon"])["pm25"].shift(1)

df = df.dropna()

print("After features:", len(df))

# ======================
# SPLIT
# ======================
train = df[df["year"] <= 2020]
val   = df[df["year"] == 2021]
test  = df[df["year"] == 2022]

features = [
    "lat", "lon",
    "time_index",
    "month_sin", "month_cos",
    "pm25_lag1"
]

target = "pm25"

X_train, y_train = train[features], train[target]
X_val, y_val     = val[features], val[target]
X_test, y_test   = test[features], test[target]

print("\nTrain:", len(X_train), "Val:", len(X_val), "Test:", len(X_test))

# ======================
# MODEL (STILL STRONG BUT REALISTIC)
# ======================
model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.07,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

print("\n--- TRAINING ---")
model.fit(X_train, y_train)
print("--- DONE ---")

# ======================
# EVALUATION
# ======================
val_rmse = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
naive_rmse = np.sqrt(mean_squared_error(y_test, X_test["pm25_lag1"]))

print("\n--- RESULTS ---")
print("Validation RMSE:", val_rmse)
print("Test RMSE:", test_rmse)
print("Naive RMSE:", naive_rmse)

# ======================
# FORECAST 2023
# ======================
print("\n--- FORECASTING 2023 ---")

# Start from last observed month (Dec 2022)
current_df = df[df["date"] == df["date"].max()].copy()

# Remove old file
if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

future_steps = 12

for step in range(future_steps):

    # Move forward one month
    current_df["date"] += pd.DateOffset(months=1)

    current_df["month"] = current_df["date"].dt.month
    current_df["year"] = current_df["date"].dt.year

    current_df["time_index"] = (current_df["year"] - df["year"].min()) * 12 + current_df["month"]
    current_df["month_sin"] = np.sin(2 * np.pi * current_df["month"] / 12)
    current_df["month_cos"] = np.cos(2 * np.pi * current_df["month"] / 12)

    # Update lag
    current_df["pm25_lag1"] = current_df["pm25"]

    preds = model.predict(current_df[features])
    current_df["pm25"] = preds

    save_df = current_df[["lat", "lon", "date", "pm25"]]

    save_df.to_csv(
        OUTPUT_FILE,
        mode="a",
        header=not OUTPUT_FILE.exists(),
        index=False
    )

    print(f"Saved step {step+1}/12")

print("\nSaved predictions to:", OUTPUT_FILE)