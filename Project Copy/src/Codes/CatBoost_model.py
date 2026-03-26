from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

# ======================
# PATHS
# ======================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = BASE_DIR / "data/processed/na_pm25_cells_clean.csv"
OUTPUT_FILE = BASE_DIR / "data/processed/catboost_predictions_2023.csv"

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
# MODEL
# ======================
model = CatBoostRegressor(
    iterations=500,            
    learning_rate=0.01,           
    depth=9,                      
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=100,
    use_best_model=True,
    early_stopping_rounds=50,    
    allow_writing_files=False,    
    l2_leaf_reg=3               
)

print("\n--- TRAINING CATBOOST ---")
model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val)
)
print("--- DONE ---")

print("Best iteration:", model.get_best_iteration())

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

current_df = df[df["date"] == df["date"].max()].copy()

future_steps = 12
all_preds = []

for step in range(future_steps):
    current_df["date"] += pd.DateOffset(months=1)

    current_df["month"] = current_df["date"].dt.month
    current_df["year"] = current_df["date"].dt.year
    current_df["time_index"] = (
        (current_df["year"] - df["year"].min()) * 12 + current_df["month"]
    )
    current_df["month_sin"] = np.sin(2 * np.pi * current_df["month"] / 12)
    current_df["month_cos"] = np.cos(2 * np.pi * current_df["month"] / 12)

    current_df["pm25_lag1"] = current_df["pm25"]

    preds = model.predict(current_df[features])
    current_df["pm25"] = preds

    all_preds.append(current_df[["lat", "lon", "date", "pm25"]].copy())

final_df = pd.concat(all_preds, ignore_index=True)
final_df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved predictions to:", OUTPUT_FILE)