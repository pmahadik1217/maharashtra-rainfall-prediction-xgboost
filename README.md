# Maharashtra Rainfall Prediction using XGBoost

This project uses the supplied **121,904-row Maharashtra daily rainfall dataset** to build a station-wise rainfall forecasting model.

## Dataset

- 121,904 records
- 658 rainfall stations
- Date range in the supplied file: 2021-01-01 to 2025-07-31
- Target: `Manual Daily Rainfall (mm)`
- Main identifier: `Station`
- Location fields available: latitude, longitude, district, tehsil, block, village, basin, river, etc.

## Model

The default model is **XGBoost Regression**.

Features:
- Previous rainfall: 1, 2, 3, 7, 14 and 30-day lags
- Rolling mean: 3, 7, 14 and 30 days
- Rolling standard deviation
- Rolling maximum
- Month
- Day of year
- Day of week
- Week of year
- Cyclic seasonal features using sine/cosine

### Important methodological choice

The data are station-wise and irregular. Therefore, the app trains a separate time-series model for the selected station instead of mixing all stations into one target series.

The final 20% of each station's usable observations are kept as a chronological test set. No random train/test split is used.

## Run locally

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Linux/macOS:
```bash
source .venv/bin/activate
```

Install:
```bash
pip install -r requirements.txt
```

Start the application:
```bash
streamlit run app.py
```

## Command-line training

Example:

```bash
python train_model.py --station "Dukanwadi"
```

This saves:
- `models/<station>_xgboost.joblib`
- `models/<station>_metrics.json`

## GitHub + Streamlit deployment

1. Create a new GitHub repository, for example `maharashtra-rainfall-prediction-xgboost`.
2. Upload this complete project folder.
3. Keep `data/rainfall.csv` in the repository (the supplied file is about 19 MB, below GitHub's normal 100 MB single-file limit).
4. In Streamlit Community Cloud, choose the GitHub repository and set the main file to:
   `app.py`
5. Deploy.

The app then lets you select a station and forecast 7–90 future days.

## Research workflow

For a PhD/research project, do not report only the future forecast. Also report:
- MAE
- RMSE
- R²
- chronological train/test design
- station-wise performance
- seasonal performance
- comparison against a baseline such as seasonal-naive
- uncertainty intervals
- validation on a genuinely held-out future period

For a stronger research model, compare XGBoost with:
- SARIMA/SARIMAX
- Random Forest
- LightGBM
- LSTM/GRU
- Temporal Fusion Transformer or other deep time-series model

## Limitations

The supplied dataset contains rainfall observations but does not include atmospheric predictors such as temperature, humidity, pressure, wind, ENSO/IOD indices, or radar/satellite precipitation. Therefore, this is primarily a **historical rainfall time-series forecasting model**, not a full numerical weather prediction system.

Also, recursive multi-day forecasts accumulate prediction error. Longer horizons should therefore be interpreted cautiously.
