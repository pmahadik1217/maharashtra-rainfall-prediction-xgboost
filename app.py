import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

st.set_page_config(page_title="Maharashtra Rainfall Prediction", page_icon="🌧️", layout="wide")

DATA_PATH = "data/rainfall.csv"
DATE_COL = "Data Acquisition Time"
RAIN_COL = "Manual Daily Rainfall (mm)"
STATION_COL = "Station"

LAGS = [1, 2, 3, 7, 14, 30]
ROLLS = [3, 7, 14, 30]

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
    df[RAIN_COL] = pd.to_numeric(df[RAIN_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, STATION_COL, RAIN_COL]).copy()
    df[RAIN_COL] = df[RAIN_COL].clip(lower=0)
    return df.sort_values([STATION_COL, DATE_COL])

def make_features(s):
    x = s[[DATE_COL, RAIN_COL]].copy().sort_values(DATE_COL)
    x = x.set_index(DATE_COL)
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
    x["sin_doy"] = np.sin(2 * np.pi * x["dayofyear"] / 365.25)
    x["cos_doy"] = np.cos(2 * np.pi * x["dayofyear"] / 365.25)
    x["sin_month"] = np.sin(2 * np.pi * x["month"] / 12)
    x["cos_month"] = np.cos(2 * np.pi * x["month"] / 12)

    return x.dropna()

def feature_row(history, date):
    vals = list(history)
    row = {}
    for lag in LAGS:
        row[f"lag_{lag}"] = vals[-lag]
    for w in ROLLS:
        a = np.asarray(vals[-w:], dtype=float)
        row[f"roll_mean_{w}"] = a.mean()
        row[f"roll_std_{w}"] = a.std(ddof=1) if len(a) > 1 else 0.0
        row[f"roll_max_{w}"] = a.max()

    row["month"] = date.month
    row["dayofyear"] = date.dayofyear
    row["dayofweek"] = date.dayofweek
    row["weekofyear"] = int(date.isocalendar().week)
    row["sin_doy"] = np.sin(2 * np.pi * row["dayofyear"] / 365.25)
    row["cos_doy"] = np.cos(2 * np.pi * row["dayofyear"] / 365.25)
    row["sin_month"] = np.sin(2 * np.pi * row["month"] / 12)
    row["cos_month"] = np.cos(2 * np.pi * row["month"] / 12)
    return pd.DataFrame([row])

def train_model(s, test_fraction=0.20):
    data = make_features(s)
    if len(data) < 120:
        raise ValueError("This station has too few usable observations after creating lag/rolling features.")

    split = int(len(data) * (1 - test_fraction))
    X = data.drop(columns=[RAIN_COL])
    y = data[RAIN_COL]
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
    )
    model.fit(X_train, y_train)

    pred = np.clip(model.predict(X_test), 0, None)
    metrics = {
        "MAE (mm)": mean_absolute_error(y_test, pred),
        "RMSE (mm)": np.sqrt(mean_squared_error(y_test, pred)),
        "R²": r2_score(y_test, pred),
    }
    test = pd.DataFrame({"date": X_test.index, "actual": y_test.values, "predicted": pred})
    return model, metrics, test

def forecast(model, s, days):
    s = s[[DATE_COL, RAIN_COL]].sort_values(DATE_COL)
    history = s[RAIN_COL].astype(float).tolist()
    last_date = s[DATE_COL].iloc[-1]
    out = []

    for i in range(days):
        d = last_date + pd.Timedelta(days=i + 1)
        row = feature_row(history, d)
        p = float(np.clip(model.predict(row)[0], 0, None))
        out.append((d, p))
        history.append(p)

    return pd.DataFrame(out, columns=["date", "predicted_rainfall_mm"])

st.title("🌧️ Maharashtra Rainfall Prediction using XGBoost")
st.caption("Daily station-wise forecasting using the supplied 2021–2025 rainfall dataset.")

df = load_data()

with st.sidebar:
    st.header("Forecast Settings")
    stations = sorted(df[STATION_COL].dropna().unique())
    station = st.selectbox("Select rainfall station", stations)
    horizon = st.slider("Future prediction days", 7, 90, 30)
    test_fraction = st.slider("Test fraction", 0.10, 0.30, 0.20, 0.05)
    run = st.button("Train & Forecast", type="primary")

s = df[df[STATION_COL] == station].copy().sort_values(DATE_COL)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Observations", f"{len(s):,}")
c2.metric("Start", s[DATE_COL].min().strftime("%d-%m-%Y"))
c3.metric("Last date", s[DATE_COL].max().strftime("%d-%m-%Y"))
c4.metric("Mean rainfall", f"{s[RAIN_COL].mean():.2f} mm")

st.subheader("Historical rainfall")
hist = s[[DATE_COL, RAIN_COL]].rename(columns={DATE_COL: "date", RAIN_COL: "rainfall_mm"}).set_index("date")
st.line_chart(hist)

if run:
    with st.spinner("Training XGBoost model and generating future predictions..."):
        model, metrics, test = train_model(s, test_fraction)

        st.subheader("Model performance on unseen test data")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{metrics['MAE (mm)']:.3f} mm")
        m2.metric("RMSE", f"{metrics['RMSE (mm)']:.3f} mm")
        m3.metric("R²", f"{metrics['R²']:.3f}")

        fig = plt.figure(figsize=(12, 4))
        plt.plot(test["date"], test["actual"], label="Actual")
        plt.plot(test["date"], test["predicted"], label="Predicted")
        plt.xlabel("Date")
        plt.ylabel("Rainfall (mm)")
        plt.title(f"Test-period prediction — {station}")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig)

        future = forecast(model, s, horizon)
        st.subheader(f"Future rainfall forecast — next {horizon} days")
        st.line_chart(future.set_index("date"))

        st.dataframe(future, use_container_width=True)
        csv = future.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download forecast CSV",
            csv,
            file_name=f"forecast_{station.replace('/', '_')}_{horizon}days.csv",
            mime="text/csv",
        )
else:
    st.info("Choose a station and click **Train & Forecast**.")
