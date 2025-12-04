import os
import glob
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- SAFE OPTIONAL IMPORTS ---

HAS_XGBOOST = False
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    pass

HAS_STATSMODELS = False
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except ImportError:
    pass

HAS_PROPHET = False
try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    pass

HAS_TENSORFLOW = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    HAS_TENSORFLOW = True
except ImportError:
    pass


# --- APP CONFIG ---

st.set_page_config(
    page_title="NYC Citi Bike Congestion Intelligence",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* You can customize theme here if you like */
</style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# PART 1: DATA FUNCTIONS
# ==========================================

STATION_INFO_URL = "https://gbfs.citibikenyc.com/gbfs/en/station_information.json"
STATION_STATUS_URL = "https://gbfs.citibikenyc.com/gbfs/en/station_status.json"


@st.cache_data(ttl=60)
def load_live_data():
    """Fetch live Citi Bike station info + status and merge."""
    try:
        info_resp = requests.get(STATION_INFO_URL).json()
        info_df = pd.DataFrame(info_resp["data"]["stations"])

        status_resp = requests.get(STATION_STATUS_URL).json()
        status_df = pd.DataFrame(status_resp["data"]["stations"])

        info_df["station_id"] = info_df["station_id"].astype(str)
        status_df["station_id"] = status_df["station_id"].astype(str)

        merged = pd.merge(
            info_df, status_df, on="station_id", suffixes=("_info", "_status")
        )

        merged["capacity"] = merged["capacity"].fillna(1)
        merged["percent_full"] = (
            merged["num_bikes_available"] / merged["capacity"]
        ) * 100

        def get_color(row):
            if row["is_renting"] == 0 or row["is_installed"] == 0:
                return [128, 128, 128, 200]
            elif row["num_bikes_available"] == 0:
                return [255, 0, 0, 200]
            elif row["num_docks_available"] == 0:
                return [0, 0, 255, 200]
            else:
                return [0, 200, 0, 200]

        merged["color"] = merged.apply(get_color, axis=1)
        merged["lat"] = merged["lat"].astype(float)
        merged["lon"] = merged["lon"].astype(float)

        return merged
    except Exception as e:
        st.error(f"Error fetching live data: {e}")
        return pd.DataFrame()


def load_local_historical_data(folder_path: str = "data"):
    """
    Recursively load all CSV files in `folder_path` and concatenate.
    Used for trip-level historical Citi Bike data.
    """
    all_files = glob.glob(os.path.join(folder_path, "**/*.csv"), recursive=True)
    if not all_files:
        return None, f"No CSV files found under '{folder_path}'."

    frames = []
    for fname in all_files:
        try:
            df = pd.read_csv(fname, index_col=None, header=0, low_memory=False)
            frames.append(df)
        except Exception as e:
            st.warning(f"Skipping file {fname} due to error: {e}")

    if not frames:
        return None, "All found CSV files failed to load."

    combined = pd.concat(frames, axis=0, ignore_index=True)
    return combined, f"Successfully loaded and combined {len(frames)} CSV files."


def generate_synthetic_data(days: int = 60) -> pd.DataFrame:
    """Generate synthetic hourly demand data with peaks and simple temperature."""
    dates = pd.date_range(start="2023-01-01", periods=days * 24, freq="h")
    df = pd.DataFrame({"timestamp": dates})

    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    def get_demand(row):
        base = 15
        if row["is_weekend"] == 0:  # weekday
            if 7 <= row["hour"] <= 9:
                return base + 50 + np.random.normal(0, 8)
            if 16 <= row["hour"] <= 19:
                return base + 60 + np.random.normal(0, 8)
        else:  # weekend
            if 11 <= row["hour"] <= 16:
                return base + 30 + np.random.normal(0, 10)
        return base + np.random.normal(0, 3)

    df["rides_ended"] = df.apply(get_demand, axis=1)
    df["rides_ended"] = df["rides_ended"].clip(lower=0).astype(int)

    df["temperature"] = (
        40
        + np.sin(np.linspace(0, 3 * np.pi, len(df))) * 15
        + np.random.normal(0, 2, len(df))
    )

    return df

def preprocess_dataframe(df: pd.DataFrame):
    """
    Trip-level → hourly time series.

    - Detects an 'end time' column (e.g. ended_at, end_date, dropoff_datetime).
    - Resamples to hourly ride counts.
    - Adds calendar and dummy weather features.
    """
    try:
        if df.empty:
            st.error("Combined historical data is empty.")
            return None

        time_cols_map = {
            "ended_at": "end_time",
            "end_date": "end_time",
            "dropoff_datetime": "end_time",
        }

        end_time_col = next(
            (c for c in df.columns if c.lower() in time_cols_map), None
        )
        if not end_time_col:
            st.error(
                "Historical trip data must include an end time column "
                "(e.g. 'ended_at', 'end_date')."
            )
            return None

        df["end_time"] = pd.to_datetime(df[end_time_col], errors="coerce")
        df = df.dropna(subset=["end_time"])

        # Hourly aggregation
        df_hourly = (
            df.set_index("end_time")
            .resample("H")
            .size()
            .to_frame(name="rides_ended")
            .reset_index()
        )
        data = df_hourly.rename(columns={"end_time": "timestamp"})

        if data.empty:
            st.error(
                "Aggregation to hourly data produced an empty dataset. "
                "Check date formats."
            )
            return None

        data["hour"] = data["timestamp"].dt.hour
        data["dayofweek"] = data["timestamp"].dt.dayofweek
        data["is_weekend"] = data["timestamp"].dt.dayofweek.isin([5, 6]).astype(int)

        # Dummy weather placeholder
        data["temperature"] = 50.0

        return data[
            ["timestamp", "hour", "dayofweek", "is_weekend", "rides_ended", "temperature"]
        ]
    except Exception as e:
        st.error(f"Error processing historical data: {e}")
        st.info(
            "Ensure an 'ended_at' or similar end-time column is present and parsable."
        )
        return None


# ==========================================
# PART 2: MODELING FUNCTIONS
# ==========================================


def calculate_risk(prediction: float, capacity: int = 100):
    """Return congestion risk label and color given predicted demand vs capacity."""
    ratio = prediction / max(capacity, 1)
    if ratio > 0.8:
        return "High 🔥", "red"
    elif ratio > 0.5:
        return "Medium ⚠️", "orange"
    else:
        return "Low 🟢", "green"

def train_and_predict(df, selected_models, forecast_horizon=24):

    """
    Train selected models and return:
    - results: {model_name: {pred, rmse, r2}}
    - y_test: pd.Series
    - test_dates: pd.Series
    """
    results = {}

    df_reg = df.copy()
    df_reg["lag_1h"] = df_reg["rides_ended"].shift(1)
    df_reg["lag_24h"] = df_reg["rides_ended"].shift(24)
    df_reg["rolling_3h"] = df_reg["rides_ended"].rolling(3).mean()
    df_reg = df_reg.dropna()

    features = [
        "hour",
        "dayofweek",
        "is_weekend",
        "temperature",
        "lag_1h",
        "lag_24h",
        "rolling_3h",
    ]
    target = "rides_ended"

    if len(df_reg) < 50:
        st.error(
            "Not enough data after feature engineering. Need at least 50 hourly rows."
        )
        return {}, None, None

    train_size = int(len(df_reg) * 0.9)
    X_train = df_reg[features].iloc[:train_size]
    y_train = df_reg[target].iloc[:train_size]
    X_test = df_reg[features].iloc[train_size:]
    y_test = df_reg[target].iloc[train_size:]
    test_dates = df_reg["timestamp"].iloc[train_size:]

    # 1. Linear Regression
    if "Linear Regression" in selected_models:
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        pred = lr.predict(X_test)
        results["Linear Regression"] = {
            "pred": pred,
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "r2": float(r2_score(y_test, pred)),
        }

    # 2. Random Forest
    if "Random Forest" in selected_models:
        rf = RandomForestRegressor(n_estimators=150, random_state=42)
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        results["Random Forest"] = {
            "pred": pred,
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "r2": float(r2_score(y_test, pred)),
        }

    # 3. XGBoost
    if "XGBoost" in selected_models and HAS_XGBOOST:
        xg = xgb.XGBRegressor(
            objective="reg:squarederror", n_estimators=200, max_depth=5
        )
        xg.fit(X_train, y_train)
        pred = xg.predict(X_test)
        results["XGBoost"] = {
            "pred": pred,
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "r2": float(r2_score(y_test, pred)),
        }

    # 4. ARIMA / SARIMA
    if "ARIMA/SARIMA" in selected_models and HAS_STATSMODELS:
        try:
            train_ts = y_train.values
            model = SARIMAX(train_ts, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
            model_fit = model.fit(disp=False)
            pred = model_fit.forecast(steps=len(y_test))
            results["SARIMA"] = {
                "pred": pred,
                "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                "r2": float(r2_score(y_test, pred)),
                "fit_obj": model_fit,
            }
        except Exception as e:
            st.warning(f"SARIMA failed: {e}")

    # 5. Prophet
    if "Prophet" in selected_models and HAS_PROPHET:
        df_prophet = pd.DataFrame(
            {"ds": df_reg["timestamp"].iloc[:train_size], "y": y_train}
        )
        m = Prophet(daily_seasonality=True, yearly_seasonality=False)
        m.fit(df_prophet)
        future = m.make_future_dataframe(periods=len(y_test), freq="H")
        forecast = m.predict(future)
        pred = forecast["yhat"].tail(len(y_test)).values
        results["Prophet"] = {
            "pred": pred,
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "r2": float(r2_score(y_test, pred)),
        }

    # 6. LSTM-RNN
    if "LSTM-RNN" in selected_models and HAS_TENSORFLOW:
        X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

        model = Sequential()
        model.add(LSTM(50, activation="relu", input_shape=(1, X_train.shape[1])))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train_lstm, y_train, epochs=20, batch_size=32, verbose=0)

        pred = model.predict(X_test_lstm).flatten()
        results["LSTM"] = {
            "pred": pred,
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "r2": float(r2_score(y_test, pred)),
        }

    # 7. Hybrid Ensemble
    base_models_for_ensemble = {
        k: v for k, v in results.items() if k not in ["Hybrid Ensemble"]
    }
    if len(base_models_for_ensemble) > 1:
        preds = [v["pred"] for v in base_models_for_ensemble.values()]
        ensemble_pred = np.mean(preds, axis=0)
        results["Hybrid Ensemble"] = {
            "pred": ensemble_pred,
            "rmse": float(np.sqrt(mean_squared_error(y_test, ensemble_pred))),
            "r2": float(r2_score(y_test, ensemble_pred)),
        }

    return results, y_test, test_dates


# ==========================================
# PART 3: MAIN APP LAYOUT – FULL SYSTEM VIEW
# ==========================================

st.title("🚲 NYC Citi Bike Congestion Intelligence")
st.markdown(
    """
Real-time congestion monitoring + historical ML forecasting for NYC Citi Bike stations.  
The system provides congestion prediction, 24‑hour demand forecasts, model benchmarking, and ARIMA diagnostics.
"""
)

# Initialize session state
for key in [
    "analysis_done",
    "model_results",
    "y_test",
    "test_dates",
    "future_df",
    "best_model_name",
    "arima_fit",
    "data_df",
]:
    if key not in st.session_state:
        st.session_state[key] = None

# ==============================
# SECTION 1: REAL-TIME DASHBOARD
# ==============================

with st.spinner("Fetching live Citi Bike feed..."):
    df_live = load_live_data()

if df_live.empty:
    st.stop()

st.sidebar.title("App Controls")
st.sidebar.subheader("Map Filters")
show_empty = st.sidebar.checkbox("Show Empty (No Bikes)", True)
show_full = st.sidebar.checkbox("Show Full (No Docks)", True)

filtered_live = df_live.copy()
if not show_empty:
    filtered_live = filtered_live[filtered_live["num_bikes_available"] > 0]
if not show_full:
    filtered_live = filtered_live[filtered_live["num_docks_available"] > 0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Stations Online", len(filtered_live))
c2.metric("Total Bikes", int(filtered_live["num_bikes_available"].sum()))
c3.metric("E-Bikes", int(filtered_live["num_ebikes_available"].sum()))
c4.metric("Last Update", time.strftime("%H:%M"))

st.markdown("---")

# ==============================
# SECTION 2: INTERACTIVE MAP
# ==============================

st.subheader("📍 Interactive City Map")

LOCATIONS = {
    "All NYC (Default)": {"lat": 40.730610, "lon": -73.935242, "zoom": 11},
    "Grand Central Terminal": {"lat": 40.7527, "lon": -73.9772, "zoom": 14},
    "Central Park": {"lat": 40.785091, "lon": -73.968285, "zoom": 13},
    "Penn Station": {"lat": 40.750568, "lon": -73.993519, "zoom": 14},
    "Times Square": {"lat": 40.758896, "lon": -73.985130, "zoom": 14},
    "Brooklyn Bridge": {"lat": 40.7061, "lon": -73.9969, "zoom": 14},
    "Wall Street / Financial Dist": {"lat": 40.7074, "lon": -74.0113, "zoom": 14},
    "Williamsburg": {"lat": 40.7128, "lon": -73.9610, "zoom": 13},
    "Downtown Brooklyn": {"lat": 40.6917, "lon": -73.9848, "zoom": 14},
    "Long Island City": {"lat": 40.7447, "lon": -73.9485, "zoom": 14},
    "Jersey City / Hoboken": {"lat": 40.7282, "lon": -74.0776, "zoom": 13},
}

loc_select = st.selectbox("Jump to Region:", list(LOCATIONS.keys()))
view_state = pdk.ViewState(
    latitude=LOCATIONS[loc_select]["lat"],
    longitude=LOCATIONS[loc_select]["lon"],
    zoom=LOCATIONS[loc_select]["zoom"],
)

st.pydeck_chart(
    pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=view_state,
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                filtered_live,
                get_position=["lon", "lat"],
                get_color="color",
                get_radius=80,
                pickable=True,
                opacity=0.8,
                filled=True,
            )
        ],
        tooltip={
            "html": """
<b>{name}</b><br/>
🚲 Bikes: {num_bikes_available}<br/>
🅿️ Docks: {num_docks_available}
"""
        },
    )
)

# ==============================
# SECTION 3: TARGET STATION VIEW
# ==============================

st.subheader("🎯 Select Station for Analysis")

station_names = sorted(df_live["name"].unique())
default_idx = (
    station_names.index("Central Park S & 6th Ave")
    if "Central Park S & 6th Ave" in station_names
    else 0
)
target_station = st.selectbox(
    "Choose a station to analyze congestion patterns:", station_names, index=default_idx
)

capacity_for_risk = 80
if target_station:
    row = df_live[df_live["name"] == target_station].iloc[0]
    cap = int(row["capacity"])
    bikes = int(row["num_bikes_available"])
    docks = int(row["num_docks_available"])
    ebikes = int(row["num_ebikes_available"])
    capacity_for_risk = cap if cap > 0 else 80

    pct_full = bikes / cap if cap > 0 else 0.0
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Capacity", cap)
    m2.metric("Available Bikes", bikes)
    m3.metric("Available Docks", docks)
    m4.metric("E-Bikes", ebikes)
    with m5:
        st.markdown("**Station Fullness**")
        st.progress(pct_full, text=f"{int(pct_full * 100)}% full")

st.markdown("---")

# ==============================
# SECTION 4: ML MODEL STUDIO
# ==============================

st.header("🤖 ML Congestion Forecasting Studio")
st.markdown(
    """
Upload or aggregate historical data, train multiple models, compare performance,  
and generate 24‑hour congestion forecasts with risk scores.
"""
)

cfg_col, run_col = st.columns([2, 1])

with cfg_col:
    st.markdown("#### Data Source")
    st.markdown(
        "- **Primary**: All trip-level CSVs in `./data/` are automatically loaded and aggregated to hourly demand.\n"
        "- **Fallback**: Synthetic 60‑day hourly demand is generated if no valid CSVs are found."
    )

    available_models = ["Linear Regression", "Random Forest"]
    if HAS_XGBOOST:
        available_models.append("XGBoost")
    if HAS_STATSMODELS:
        available_models.append("ARIMA/SARIMA")
    if HAS_PROPHET:
        available_models.append("Prophet")
    if HAS_TENSORFLOW:
        available_models.append("LSTM-RNN")

    selected_models = st.multiselect(
        "Select Models to Train",
        available_models,
        default=["Linear Regression", "Random Forest"],
    )

with run_col:
    st.markdown("#### Pipeline Control")
    st.markdown(
        "Time-based train‑test split (~90% train, 10% test) to avoid look‑ahead bias."
    )
    run_btn = st.button("🚀 Run Full Analysis", use_container_width=True)

with st.expander("⚙️ Model Training Pipeline (Explainable Steps)", expanded=False):
    st.markdown(
        """
1. **Step 1 — Train‑test split**  
   - Time‑ordered split: ~90% history → train, 10% → test.

2. **Step 2 — Each model predicts the future**  
   - For each selected model: `y_pred = model.predict(X_test)` on the holdout horizon.

3. **Step 3 — Metric computation**  
   - Per‑model: RMSE, MAE, and R² on the test set.

4. **Step 4 — Cross‑model comparison**  
   - Models ranked by RMSE; Hybrid Ensemble (mean prediction) if multiple models succeed.
"""
    )

# ==============================
# RUN PIPELINE
# ==============================

if run_btn and selected_models:
    with st.spinner("Preparing historical data pipeline..."):
        raw_df, load_msg = load_local_historical_data("data")
        if raw_df is not None:
            data_df = preprocess_dataframe(raw_df)
            if data_df is not None and not data_df.empty:
                st.success(
                    f"✅ Loaded and processed {len(data_df)} hourly records. {load_msg}"
                )
            else:
                st.warning(
                    "Historical data failed validation or aggregation. Using synthetic data."
                )
                data_df = generate_synthetic_data()
        else:
            st.warning(f"Using synthetic data. {load_msg}")
            data_df = generate_synthetic_data()

    st.session_state["data_df"] = data_df

    if data_df is not None and not data_df.empty:
        with st.spinner(f"Training models: {', '.join(selected_models)}"):
            model_results, y_test, test_dates = train_and_predict(
                data_df, selected_models
            )

        if not model_results:
            st.error("No models produced valid predictions.")
        else:
            st.session_state["model_results"] = model_results
            st.session_state["y_test"] = y_test
            st.session_state["test_dates"] = test_dates
            st.session_state["analysis_done"] = True

            # Build metrics and find best model
            rows = []
            for name, res in model_results.items():
                rows.append(
                    {
                        "Model": name,
                        "RMSE": res["rmse"],
                        "MAE": mean_absolute_error(y_test, res["pred"]),
                        "R²": res["r2"],
                    }
                )
            metrics_df = pd.DataFrame(rows).sort_values("RMSE")
            best_model_name = metrics_df.iloc[0]["Model"]
            st.session_state["best_model_name"] = best_model_name

            # 24H forecast from best model (rolling)
            last_time = data_df["timestamp"].iloc[-1]
            future_times = [last_time + timedelta(hours=i) for i in range(1, 25)]
            best_series = model_results[best_model_name]["pred"]

            if len(best_series) >= 24:
                future_preds = best_series[-24:]
            else:
                last_val = best_series[-1]
                pad_len = 24 - len(best_series)
                future_preds = np.concatenate(
                    [best_series, np.repeat(last_val, pad_len)]
                )

            risk_labels = []
            risk_colors = []
            for val in future_preds:
                lbl, clr = calculate_risk(val, capacity_for_risk)
                risk_labels.append(lbl)
                risk_colors.append(clr)

            future_df = pd.DataFrame(
                {
                    "Time": future_times,
                    "Predicted_Rides": future_preds,
                    "Risk_Label": risk_labels,
                    "Color": risk_colors,
                }
            )
            st.session_state["future_df"] = future_df

            # Refit ARIMA on full series for diagnostics, if available
            if "SARIMA" in model_results and HAS_STATSMODELS:
                try:
                    full_series = data_df["rides_ended"].values
                    arima_model = SARIMAX(
                        full_series, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)
                    )
                    arima_fit = arima_model.fit(disp=False)
                    st.session_state["arima_fit"] = arima_fit
                except Exception:
                    st.session_state["arima_fit"] = None

# ==============================
# SECTION 5: RESULTS – METRICS, FORECASTS, RISK
# ==============================

if st.session_state["analysis_done"] and st.session_state["model_results"]:
    model_results = st.session_state["model_results"]
    y_test = st.session_state["y_test"]
    test_dates = st.session_state["test_dates"]
    best_model_name = st.session_state["best_model_name"]
    future_df = st.session_state["future_df"]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔎 Jump to Section")
    st.sidebar.markdown("- [📊 Evaluation Metrics](#evaluation-metrics)", unsafe_allow_html=True)
    st.sidebar.markdown("- [📈 Forecast Plots](#forecast-plots)", unsafe_allow_html=True)
    st.sidebar.markdown("- [🔮 Future 24-hour Risk](#future-24-hour-risk)", unsafe_allow_html=True)
    st.sidebar.markdown("- [🔢 ARIMA Model Analysis](#arima-model-analysis)", unsafe_allow_html=True)

    with st.expander("📊 Evaluation Metrics", expanded=True):
        st.markdown("#### Evaluation Metrics")
        rows = []
        for name, res in model_results.items():
            rows.append(
                {
                    "Model": name,
                    "RMSE": res["rmse"],
                    "MAE": mean_absolute_error(y_test, res["pred"]),
                    "R²": res["r2"],
                }
            )
        metrics_df = pd.DataFrame(rows).sort_values("RMSE")
        st.dataframe(
            metrics_df.style.highlight_min(subset=["RMSE", "MAE"], color="lightgreen")
            .highlight_max(subset=["R²"], color="lightgreen"),
            use_container_width=True,
        )
        st.info(f"🏆 Best Performing Model: **{best_model_name}**")

    with st.expander("📈 Forecast Plots", expanded=False):
        st.markdown("#### Forecast Plots")
        plot_df = pd.DataFrame({"Timestamp": test_dates, "Actual": y_test.values})
        for name, res in model_results.items():
            plot_df[name] = res["pred"]

        fig_ts = px.line(
            plot_df,
            x="Timestamp",
            y=["Actual"] + [m for m in model_results.keys()],
            title="Test-Set Hourly Demand: Actual vs Model Forecasts",
            color_discrete_map={"Actual": "black", "Hybrid Ensemble": "red"},
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    with st.expander("🔮 Future 24-hour Risk", expanded=False):
        st.markdown("#### Future 24-hour Risk")
        st.markdown(
            "24‑step rolling forecast using the best model, translated into congestion risk levels."
        )

        st.write("**Congestion Risk Outlook (every 2–4 hours)**")
        cols = st.columns(6)
        for i in range(0, 24, 4):
            row = future_df.iloc[i]
            with cols[i // 4]:
                st.markdown(f"**{row['Time'].strftime('%b %d, %I %p')}**")
                st.markdown(row["Risk_Label"])
                st.caption(f"{int(row['Predicted_Rides'])} predicted rides")

        fig_future = px.bar(
            future_df,
            x="Time",
            y="Predicted_Rides",
            color="Risk_Label",
            title=f"24-Hour Demand Forecast ({best_model_name})",
            color_discrete_map={
                "High 🔥": "red",
                "Medium ⚠️": "orange",
                "Low 🟢": "green",
            },
        )
        st.plotly_chart(fig_future, use_container_width=True)

        st.markdown("#### Horizon Drill‑Down")
        h = st.slider("Select horizon (hours ahead)", 1, 24, 1)
        row_h = future_df.iloc[h - 1]
        d1, d2, d3 = st.columns(3)
        d1.metric("Horizon", f"{h} hours ahead")
        d2.metric("Predicted Demand", int(row_h["Predicted_Rides"]))
        d3.metric("Risk", row_h["Risk_Label"])

    # ==============================
    # SECTION 6: ARIMA MODEL ANALYSIS
    # ==============================
    with st.expander("🔢 ARIMA Model Analysis", expanded=False):
            st.markdown("#### ARIMA Model Analysis")

            arima_fit = st.session_state.get("arima_fit", None)
            if arima_fit is None:
                st.info("ARIMA/SARIMA diagnostics are available only if SARIMA was trained.")
            else:
                st.markdown("**Forecast with Confidence Intervals**")
                steps = 48
                arima_forecast = arima_fit.get_forecast(steps=steps)
                mean_fc = arima_forecast.predicted_mean
                conf_int = arima_forecast.conf_int(alpha=0.05)

                # >>> ADD THESE TWO LINES <<<
                mean_fc_arr = np.asarray(mean_fc)
                conf_arr = np.asarray(conf_int)

                last_ts = (
                    st.session_state["data_df"]["timestamp"].iloc[-1]
                    if st.session_state["data_df"] is not None
                    else datetime.now()
                )
                fc_idx = pd.date_range(start=last_ts, periods=steps + 1, freq="H")[1:]
                fc_df = pd.DataFrame(
                    {
                        "Timestamp": fc_idx,
                        "Forecast": mean_fc_arr,
                        "Lower": conf_arr[:, 0],
                        "Upper": conf_arr[:, 1],
                    }
                )

                fig_arima = go.Figure()
                fig_arima.add_trace(
                    go.Scatter(
                        x=fc_df["Timestamp"],
                        y=fc_df["Forecast"],
                        mode="lines",
                        name="ARIMA Forecast",
                    )
                )
                fig_arima.add_trace(
                    go.Scatter(
                        x=fc_df["Timestamp"],
                        y=fc_df["Upper"],
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                    )
                )
                fig_arima.add_trace(
                    go.Scatter(
                    x=fc_df["Timestamp"],
                    y=fc_df["Lower"],
                    mode="lines",
                    fill="tonexty",
                    line=dict(width=0),
                    name="95% CI",
                )
            )
            fig_arima.update_layout(
                title="ARIMA Forecast with 95% Confidence Interval",
                xaxis_title="Time",
                yaxis_title="Predicted Rides",
            )
            st.plotly_chart(fig_arima, use_container_width=True)

            st.markdown("**Residual Diagnostics**")
            residuals = arima_fit.resid
            r_df = pd.DataFrame({"t": range(len(residuals)), "Residual": residuals})

            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown("Residuals over time")
                fig_res = px.line(r_df, x="t", y="Residual", title="ARIMA Residuals")
                st.plotly_chart(fig_res, use_container_width=True)
            with rc2:
                st.markdown("Residual distribution")
                fig_hist = px.histogram(
                    r_df, x="Residual", nbins=30, title="Residuals Histogram"
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("**ACF & PACF (Approximate)**")
            max_lag = min(40, len(residuals) - 1)
            if max_lag > 1:
                res_centered = residuals - np.mean(residuals)
                var = np.var(res_centered)
                acfs, pacfs = [], []
                for lag in range(1, max_lag + 1):
                    cov = np.sum(
                        res_centered[lag:] * res_centered[:-lag]
                    ) / len(res_centered)
                    acf_val = cov / var if var > 0 else 0.0
                    acfs.append(acf_val)
                    pacfs.append(acf_val if lag == 1 else acf_val - acfs[lag - 2])
                lag_idx = list(range(1, max_lag + 1))
                acf_df = pd.DataFrame(
                    {"Lag": lag_idx, "ACF": acfs, "PACF": pacfs}
                )

                ac1, ac2 = st.columns(2)
                with ac1:
                    fig_acf = px.bar(acf_df, x="Lag", y="ACF", title="Approximate ACF")
                    st.plotly_chart(fig_acf, use_container_width=True)
                with ac2:
                    fig_pacf = px.bar(acf_df, x="Lag", y="PACF", title="Approximate PACF")
                    st.plotly_chart(fig_pacf, use_container_width=True)
            else:
                st.info("Not enough residual points for meaningful ACF/PACF.")


else:
    st.info(
        "Use the ML Congestion Forecasting Studio above and click **Run Full Analysis** "
        "to generate forecasts, metrics, and ARIMA diagnostics."
    )
