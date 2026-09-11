import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODEL_PATH = Path(__file__).with_name("electricity_theft_model.pkl")
THRESHOLD = 0.50

NUMERIC_FEATURES = ['square_feet', 'year_built', 'number_of_occupants', 'number_of_bedrooms', 'number_of_floors', 'number_of_units', 'electricity_usage_kwh', 'peak_usage_kwh', 'off_peak_usage_kwh', 'avg_daily_usage_kwh']
CLASS_DUMMY_COLUMNS = [np.str_('Class_FullServiceRestaurant'), np.str_('Class_Hospital'), np.str_('Class_LargeHotel'), np.str_('Class_LargeOffice'), np.str_('Class_MediumOffice'), np.str_('Class_MidriseApartment'), np.str_('Class_OutPatient'), np.str_('Class_PrimarySchool'), np.str_('Class_QuickServiceRestaurant'), np.str_('Class_SecondarySchool'), np.str_('Class_SmallHotel'), np.str_('Class_SmallOffice'), np.str_('Class_Stand_aloneRetail'), np.str_('Class_StripMall'), np.str_('Class_SuperMarket'), np.str_('Class_Warehouse')]
FEATURE_NAMES = [np.str_('Electricity_Facility_kW_Hourly_'), np.str_('Fans_Electricity_kW_Hourly_'), np.str_('Cooling_Electricity_kW_Hourly_'), np.str_('Heating_Electricity_kW_Hourly_'), np.str_('InteriorLights_Electricity_kW_Hourly_'), np.str_('InteriorEquipment_Electricity_kW_Hourly_'), np.str_('Gas_Facility_kW_Hourly_'), np.str_('Heating_Gas_kW_Hourly_'), np.str_('InteriorEquipment_Gas_kW_Hourly_'), np.str_('Water_Heater_WaterSystems_Gas_kW_Hourly_'), np.str_('Class_FullServiceRestaurant'), np.str_('Class_Hospital'), np.str_('Class_LargeHotel'), np.str_('Class_LargeOffice'), np.str_('Class_MediumOffice'), np.str_('Class_MidriseApartment'), np.str_('Class_OutPatient'), np.str_('Class_PrimarySchool'), np.str_('Class_QuickServiceRestaurant'), np.str_('Class_SecondarySchool'), np.str_('Class_SmallHotel'), np.str_('Class_SmallOffice'), np.str_('Class_Stand_aloneRetail'), np.str_('Class_StripMall'), np.str_('Class_SuperMarket'), np.str_('Class_Warehouse')]

CLASS_MAP = {'FullServiceRestaurant': np.str_('Class_FullServiceRestaurant'), 'Hospital': np.str_('Class_Hospital'), 'LargeHotel': np.str_('Class_LargeHotel'), 'LargeOffice': np.str_('Class_LargeOffice'), 'MediumOffice': np.str_('Class_MediumOffice'), 'MidriseApartment': np.str_('Class_MidriseApartment'), 'OutPatient': np.str_('Class_OutPatient'), 'PrimarySchool': np.str_('Class_PrimarySchool'), 'QuickServiceRestaurant': np.str_('Class_QuickServiceRestaurant'), 'SecondarySchool': np.str_('Class_SecondarySchool'), 'SmallHotel': np.str_('Class_SmallHotel'), 'SmallOffice': np.str_('Class_SmallOffice'), 'StripMall': np.str_('Class_StripMall'), 'SuperMarket': np.str_('Class_SuperMarket'), 'Warehouse': np.str_('Class_Warehouse'), 'Stand-aloneRetail': 'Class_Stand_aloneRetail'}
CLASS_OPTIONS = ['FullServiceRestaurant', 'Hospital', 'LargeHotel', 'LargeOffice', 'MediumOffice', 'MidriseApartment', 'OutPatient', 'PrimarySchool', 'QuickServiceRestaurant', 'SecondarySchool', 'SmallHotel', 'SmallOffice', 'StripMall', 'SuperMarket', 'Warehouse', 'Stand-aloneRetail']


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Place electricity_theft_model.pkl in the same folder as app.py."
        )
    return joblib.load(MODEL_PATH)


model = load_model()


# ---------------------------------------------------------
# Page
# ---------------------------------------------------------
st.set_page_config(
    page_title="Electricity Theft Detection",
    page_icon="⚡",
    layout="centered",
)

st.title("⚡ Electricity Theft Detection")
st.write(
    "Enter building and electricity-usage information to estimate "
    "whether the model predicts electricity theft."
)

st.info(
    "This prediction is produced by the trained machine-learning model "
    "and should be treated as a decision-support result, not proof of theft."
)


# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
with st.form("prediction_form"):
    st.subheader("Building information")

    square_feet = st.number_input(
        "Square feet",
        min_value=0.0,
        value=1000.0,
        step=50.0,
    )

    year_built = st.number_input(
        "Year built",
        min_value=1800,
        max_value=2100,
        value=2000,
        step=1,
    )

    number_of_occupants = st.number_input(
        "Number of occupants",
        min_value=0,
        value=3,
        step=1,
    )

    number_of_bedrooms = st.number_input(
        "Number of bedrooms",
        min_value=0,
        value=2,
        step=1,
    )

    number_of_floors = st.number_input(
        "Number of floors",
        min_value=0,
        value=1,
        step=1,
    )

    number_of_units = st.number_input(
        "Number of units",
        min_value=0,
        value=1,
        step=1,
    )

    building_class = st.selectbox(
        "Building class",
        options=CLASS_OPTIONS,
        index=0,
    )

    st.subheader("Electricity usage")

    electricity_usage_kwh = st.number_input(
        "Electricity usage (kWh)",
        min_value=0.0,
        value=1000.0,
        step=50.0,
    )

    peak_usage_kwh = st.number_input(
        "Peak usage (kWh)",
        min_value=0.0,
        value=400.0,
        step=25.0,
    )

    off_peak_usage_kwh = st.number_input(
        "Off-peak usage (kWh)",
        min_value=0.0,
        value=600.0,
        step=25.0,
    )

    avg_daily_usage_kwh = st.number_input(
        "Average daily usage (kWh)",
        min_value=0.0,
        value=33.0,
        step=1.0,
    )

    submitted = st.form_submit_button(
        "Predict",
        use_container_width=True,
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if submitted:
    row = {
        "square_feet": square_feet,
        "year_built": year_built,
        "number_of_occupants": number_of_occupants,
        "number_of_bedrooms": number_of_bedrooms,
        "number_of_floors": number_of_floors,
        "number_of_units": number_of_units,
        "electricity_usage_kwh": electricity_usage_kwh,
        "peak_usage_kwh": peak_usage_kwh,
        "off_peak_usage_kwh": off_peak_usage_kwh,
        "avg_daily_usage_kwh": avg_daily_usage_kwh,
    }

    input_df = pd.DataFrame([row])

    # Recreate the notebook's one-hot encoding:
    # Class was encoded with drop_first=True, so the baseline class is
    # represented by all-zero Class_* columns.
    for column in CLASS_DUMMY_COLUMNS:
        input_df[column] = 0

    dummy_column = CLASS_MAP[building_class]
    input_df[dummy_column] = 1

    # IMPORTANT: use exactly the feature names/order stored by the trained model.
    input_df = input_df[FEATURE_NAMES]

    if input_df.shape[1] != 26:
        st.error(
            f"Feature mismatch: the model expects 26 features, "
            f"but the app created {input_df.shape[1]}."
        )
        st.stop()

    try:
        probability = float(model.predict_proba(input_df)[0, 1])
        prediction = int(probability >= THRESHOLD)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    st.subheader("Result")

    if prediction == 1:
        st.error("⚠️ Model prediction: Electricity theft detected")
    else:
        st.success("✅ Model prediction: No electricity theft detected")

    st.metric(
        "Theft probability",
        f"{probability:.2%}",
    )

    st.caption(f"Decision threshold: {THRESHOLD:.2f}")
