import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

DATA_PATH = "data/rainfall.csv"
DATE_COL = "Data Acquisition Time"
RAIN_COL = "Manual Daily Rainfall (mm)"
STATION_COL = "Station"
LAGS = [1, 2, 3, 7, 14, 30]
ROLLS = [3, 7, 14, 30]

def make_features(s):
    x = s[[DATE_COL, RAIN_COL]].copy().sort_values(DATE_COL).set_index(DATE_COL)
    for lag in LAGS:
        x[f"lag_{lag}"] = x[RAIN_COL].shift(lag)
    for w in ROLLS:
        x[f"roll_mean_{w}"] = x[RAIN_COL].shift(1).rolling(w).mean()
        x[f"roll_std_{w}"] = x[RAIN_COL].shift(1).rolling(w).std()
        x[f"roll_max_{w}"] = x[RAIN_COL].shift(1).rolling(w).max()
    x["month"] = x.index.month
    x["dayofyear"] = x.index.dayofyear
    x["dayofweek"] = x.index.dayofweek
    x["weekofyear"] = x.index.isocalendar().week.astype(int).values
    x["sin_doy"] = np.sin(2*np.pi*x["dayofyear"]/365.25)
    x["cos_doy"] = np.cos(2*np.pi*x["dayofyear"]/365.25)
    x["sin_month"] = np.sin(2*np.pi*x["month"]/12)
    x["cos_month"] = np.cos(2*np.pi*x["month"]/12)
    return x.dropna()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", required=True, help="Exact station name")
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
    df[RAIN_COL] = pd.to_numeric(df[RAIN_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, STATION_COL, RAIN_COL])
    s = df[df[STATION_COL] == args.station].sort_values(DATE_COL).copy()

    if s.empty:
        raise SystemExit("Station not found.")

    data = make_features(s)
    split = int(len(data) * 0.80)
    X, y = data.drop(columns=[RAIN_COL]), data[RAIN_COL]

    model = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85,
        objective="reg:squarederror", random_state=42, n_jobs=4
    )
    model.fit(X.iloc[:split], y.iloc[:split])
    pred = np.clip(model.predict(X.iloc[split:]), 0, None)

    metrics = {
        "station": args.station,
        "observations": int(len(s)),
        "MAE_mm": float(mean_absolute_error(y.iloc[split:], pred)),
        "RMSE_mm": float(np.sqrt(mean_squared_error(y.iloc[split:], pred))),
        "R2": float(r2_score(y.iloc[split:], pred)),
    }

    Path("models").mkdir(exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in args.station)
    joblib.dump(model, f"models/{safe}_xgboost.joblib")
    Path(f"models/{safe}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Saved: models/{safe}_xgboost.joblib")

if __name__ == "__main__":
    main()
