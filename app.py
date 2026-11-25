import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# ---------------------------------------------------------
# 1. Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Car Price Prediction System")
st.markdown("Predict a vehicle's estimated price using the trained machine-learning model.")

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "final_car_price_model.pkl"
DATA_PATH = PROJECT_DIR / "car_data.csv"
COMPARISON_PATH = PROJECT_DIR / "car_price_model_comparison.csv"

# ---------------------------------------------------------
# 2. Load model and dataset
# ---------------------------------------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH.name}. Put it in the same folder as app.py.")
    st.stop()

if not DATA_PATH.exists():
    st.error(f"Dataset file not found: {DATA_PATH.name}. Put it in the same folder as app.py.")
    st.stop()

model = load_model(MODEL_PATH)
df = load_data(DATA_PATH)

# ---------------------------------------------------------
# 3. Feature columns (must match the training script)
# ---------------------------------------------------------
num_cols = [
    "Year", "Engine HP", "Engine Cylinders", "Number of Doors",
    "highway MPG", "city mpg", "Popularity"
]
nominal_cols = [
    "Make", "Model", "Engine Fuel Type", "Transmission Type",
    "Driven_Wheels", "Vehicle Style", "Market Category"
]
ordinal_cols = ["Vehicle Size"]
feature_cols = num_cols + nominal_cols + ordinal_cols

missing_cols = [col for col in feature_cols if col not in df.columns]
if missing_cols:
    st.error("The dataset is missing required input columns: " + ", ".join(missing_cols))
    st.stop()

# ---------------------------------------------------------
# 4. Find best model and show comparison metrics, if available
# ---------------------------------------------------------
best_model_name = "GradientBoosting"
comparison = None

if COMPARISON_PATH.exists():
    try:
        comparison = pd.read_csv(COMPARISON_PATH, index_col=0)
        if "RMSE" in comparison.columns and not comparison.empty:
            best_model_name = str(comparison["RMSE"].astype(float).idxmin())
    except Exception:
        comparison = None

st.sidebar.success(f"Selected model: {best_model_name}")
st.sidebar.caption("Loaded from final_car_price_model.pkl")

# ---------------------------------------------------------
# 5. Input form
# ---------------------------------------------------------
st.subheader("Enter Vehicle Details")

def options_for(column):
    values = df[column].dropna().astype(str).unique().tolist()
    values.sort()
    return values if values else ["Unknown"]

def numeric_bounds(column):
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return 0.0, 100.0, 0.0
    min_value = float(series.min())
    max_value = float(series.max())
    default_value = float(series.median())
    if min_value == max_value:
        max_value = min_value + 1.0
    return min_value, max_value, default_value

with st.form("car_prediction_form"):
    left, right = st.columns(2)

    with left:
        make = st.selectbox("Make", options_for("Make"))
        model_name = st.selectbox("Model", options_for("Model"))
        year_min, year_max, year_default = numeric_bounds("Year")
        year = st.slider(
            "Year",
            min_value=int(year_min),
            max_value=int(year_max),
            value=int(year_default)
        )
        hp_min, hp_max, hp_default = numeric_bounds("Engine HP")
        engine_hp = st.number_input(
            "Engine HP",
            min_value=float(hp_min),
            max_value=float(hp_max),
            value=float(hp_default)
        )
        cylinders = st.selectbox("Engine Cylinders", options_for("Engine Cylinders"))

    with right:
        doors = st.selectbox("Number of Doors", options_for("Number of Doors"))
        highway_min, highway_max, highway_default = numeric_bounds("highway MPG")
        highway_mpg = st.number_input(
            "Highway MPG",
            min_value=float(highway_min),
            max_value=float(highway_max),
            value=float(highway_default)
        )
        city_min, city_max, city_default = numeric_bounds("city mpg")
        city_mpg = st.number_input(
            "City MPG",
            min_value=float(city_min),
            max_value=float(city_max),
            value=float(city_default)
        )
        popularity_min, popularity_max, popularity_default = numeric_bounds("Popularity")
        popularity = st.number_input(
            "Popularity",
            min_value=float(popularity_min),
            max_value=float(popularity_max),
            value=float(popularity_default)
        )

    fuel = st.selectbox("Engine Fuel Type", options_for("Engine Fuel Type"))
    transmission = st.selectbox("Transmission Type", options_for("Transmission Type"))
    driven_wheels = st.selectbox("Driven Wheels", options_for("Driven_Wheels"))
    vehicle_style = st.selectbox("Vehicle Style", options_for("Vehicle Style"))
    market_category = st.selectbox("Market Category", options_for("Market Category"))
    vehicle_size = st.selectbox("Vehicle Size", options_for("Vehicle Size"))

    submitted = st.form_submit_button("💰 Predict Car Price", use_container_width=True)

# ---------------------------------------------------------
# 6. Prediction
# ---------------------------------------------------------
if submitted:
    input_df = pd.DataFrame([{
        "Year": year,
        "Engine HP": engine_hp,
        "Engine Cylinders": cylinders,
        "Number of Doors": doors,
        "highway MPG": highway_mpg,
        "city mpg": city_mpg,
        "Popularity": popularity,
        "Make": make,
        "Model": model_name,
        "Engine Fuel Type": fuel,
        "Transmission Type": transmission,
        "Driven_Wheels": driven_wheels,
        "Vehicle Style": vehicle_style,
        "Market Category": market_category,
        "Vehicle Size": vehicle_size
    }])

    # Preserve the exact feature order used during training.
    input_df = input_df[feature_cols]

    try:
        prediction = float(model.predict(input_df)[0])
        st.markdown("---")
        st.subheader("Prediction Result")
        st.metric("Estimated MSRP", f"${prediction:,.2f}")
        st.info("This is a machine-learning estimate, not an actual dealer quote.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

# ---------------------------------------------------------
# 7. Model comparison (if CSV is present)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Model Performance Comparison")

if comparison is not None and not comparison.empty:
    display_cols = [col for col in ["best_cv_rmse", "MAE", "MSE", "RMSE", "R2 Score"] if col in comparison.columns]
    if display_cols:
        st.dataframe(
            comparison[display_cols].sort_values(
                by="RMSE", ascending=True
            ) if "RMSE" in display_cols else comparison[display_cols],
            use_container_width=True
        )
        if "RMSE" in comparison.columns:
            best_row = comparison.loc[comparison["RMSE"].astype(float).idxmin()]
            st.success(
                f"Best model based on test RMSE: {best_model_name} | "
                f"RMSE: {float(best_row['RMSE']):,.3f}"
            )
else:
    st.caption("Model comparison CSV was not found or could not be read.")

st.caption("Car Price Prediction • Built with Streamlit, Scikit-learn and MLflow")
