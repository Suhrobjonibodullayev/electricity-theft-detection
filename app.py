import joblib
import pandas as pd
import streamlit as st

# ============================================================
# Smart Grid Electricity Theft Detection
# ============================================================

MODEL_PATH = "electricity_theft_model.pkl"
THRESHOLD = 0.50

st.set_page_config(
    page_title="Electricity Theft Detection",
    page_icon="⚡",
    layout="wide",
)

# These are the 10 numeric features used by the original notebook.
NUMERIC_FEATURES = [
    "Electricity:Facility [kW](Hourly)",
    "Fans:Electricity [kW](Hourly)",
    "Cooling:Electricity [kW](Hourly)",
    "Heating:Electricity [kW](Hourly)",
    "InteriorLights:Electricity [kW](Hourly)",
    "InteriorEquipment:Electricity [kW](Hourly)",
    "Gas:Facility [kW](Hourly)",
    "Heating:Gas [kW](Hourly)",
    "InteriorEquipment:Gas [kW](Hourly)",
    "Water Heater:WaterSystems:Gas [kW](Hourly)",
]

# The trained model contains these 16 one-hot Class columns.
CLASS_DUMMY_COLUMNS = [
    "Class_FullServiceRestaurant",
    "Class_Hospital",
    "Class_LargeHotel",
    "Class_LargeOffice",
    "Class_MediumOffice",
    "Class_MidriseApartment",
    "Class_OutPatient",
    "Class_PrimarySchool",
    "Class_QuickServiceRestaurant",
    "Class_SecondarySchool",
    "Class_SmallHotel",
    "Class_SmallOffice",
    "Class_Stand_aloneRetail",
    "Class_StripMall",
    "Class_SuperMarket",
    "Class_Warehouse",
]

CLASS_OPTIONS = [
    "FullServiceRestaurant",
    "Hospital",
    "LargeHotel",
    "LargeOffice",
    "MediumOffice",
    "MidriseApartment",
    "OutPatient",
    "PrimarySchool",
    "QuickServiceRestaurant",
    "SecondarySchool",
    "SmallHotel",
    "SmallOffice",
    "Stand-aloneRetail",
    "StripMall",
    "SuperMarket",
    "Warehouse",
]

CLASS_MAP = {
    "FullServiceRestaurant": "Class_FullServiceRestaurant",
    "Hospital": "Class_Hospital",
    "LargeHotel": "Class_LargeHotel",
    "LargeOffice": "Class_LargeOffice",
    "MediumOffice": "Class_MediumOffice",
    "MidriseApartment": "Class_MidriseApartment",
    "OutPatient": "Class_OutPatient",
    "PrimarySchool": "Class_PrimarySchool",
    "QuickServiceRestaurant": "Class_QuickServiceRestaurant",
    "SecondarySchool": "Class_SecondarySchool",
    "SmallHotel": "Class_SmallHotel",
    "SmallOffice": "Class_SmallOffice",
    "Stand-aloneRetail": "Class_Stand_aloneRetail",
    "StripMall": "Class_StripMall",
    "SuperMarket": "Class_SuperMarket",
    "Warehouse": "Class_Warehouse",
}

# Defaults are representative values from the notebook workflow.
DEFAULTS = {
    "Electricity:Facility [kW](Hourly)": 179.81,
    "Fans:Electricity [kW](Hourly)": 15.38,
    "Cooling:Electricity [kW](Hourly)": 48.98,
    "Heating:Electricity [kW](Hourly)": 0.94,
    "InteriorLights:Electricity [kW](Hourly)": 36.55,
    "InteriorEquipment:Electricity [kW](Hourly)": 47.41,
    "Gas:Facility [kW](Hourly)": 86.58,
    "Heating:Gas [kW](Hourly)": 60.38,
    "InteriorEquipment:Gas [kW](Hourly)": 9.14,
    "Water Heater:WaterSystems:Gas [kW](Hourly)": 17.07,
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ============================================================
# Load trained model
# ============================================================

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        f"'{MODEL_PATH}' topilmadi. "
        "electricity_theft_model.pkl faylini app.py bilan bir xil "
        "GitHub papkasiga joylashtiring."
    )
    st.stop()
except Exception as exc:
    st.error(f"Modelni yuklashda xatolik: {exc}")
    st.stop()


# ============================================================
# Interface
# ============================================================

st.title("⚡ Electricity Theft Detection")

st.write(
    "Enter hourly electricity and gas consumption values and select "
    "the building class. The trained XGBoost model will predict "
    "Normal or Electricity Theft."
)

st.info(
    "Model: XGBoost Classifier | Decision threshold: 0.50"
)

st.subheader("Input data")

col1, col2 = st.columns(2)
values = {}

with col1:
    values[NUMERIC_FEATURES[0]] = st.number_input(
        "Electricity Facility (kW/hour)",
        min_value=0.0,
        value=DEFAULTS[NUMERIC_FEATURES[0]],
        step=0.01,
    )
    values[NUMERIC_FEATURES[1]] = st.number_input(
        "Fans Electricity (kW/hour)",
        min_value=0.0,
        value=DEFAULTS[NUMERIC_FEATURES[1]],
        step=0.01,
    )
    values[NUMERIC_FEATURES[2]] = st.number_input(
        "Cooling Electricity (kW/hour)",
        min_value=0.0,
        value=DEFAULTS[NUMERIC_FEATURES[2]],
        step=0.01,
    )
    values[NUMERIC_FEATURES[3]] = st.number_input(
        "Heating Electricity (kW/hour)",
        min_value=0.0,
        value=DEFAULTS[NUMERIC_FEATURES[3]],
        step=0.01,
    )
    values[NUMERIC_FEATURES[4]] = st.number_input(
        "Interior Lights Electricity (kW/hour)",
        min_value=0.0,
        value=DEFAULTS[NUMERIC_FEATURES[4]],
        step=0.01,
    )

with col2:
    values[NUMERIC_FEATURES[5]] = st.number_input(
        "Interior Equipment Electricity (kW/hour)",
        min_value=0.0,
        value=DEFAULTS[NUMERIC_FEATURES[5]],
        step=0.01,
    )
    values[NUMERIC_FEATURES[6]] = st.number_input(
        "Gas Facility (kW/hour)",
        min_value=0.0,
        value=DEFAULTS[NUMERIC_FEATURES[6]],
        step=0.01,
    )
    values[NUMERIC_FEATURES[7]] = st.number_input(
        "Heating Gas (kW/hour)",
        min_value=0.0,
        value=DEFAULTS[NUMERIC_FEATURES[7]],
        step=0.01,
    )
    values[NUMERIC_FEATURES[8]] = st.number_input(
        "Interior Equipment Gas (kW/hour)",
        min_value=0.0,
        value=DEFAULTS[NUMERIC_FEATURES[8]],
        step=0.01,
    )
    values[NUMERIC_FEATURES[9]] = st.number_input(
        "Water Heater Gas (kW/hour)",
        min_value=0.0,
        value=DEFAULTS[NUMERIC_FEATURES[9]],
        step=0.01,
    )

building_class = st.selectbox(
    "Building Class",
    CLASS_OPTIONS,
    index=0,
)

st.divider()

if st.button("🔍 Detect Theft", type="primary", use_container_width=True):

    # --------------------------------------------------------
    # Build the same 26-feature structure as the trained model
    # --------------------------------------------------------
    input_df = pd.DataFrame([values])

    # Explicitly create every one-hot Class column.
    # Do NOT use get_dummies() on a single row because drop_first=True
    # can drop the selected class depending on that single row.
    for column in CLASS_DUMMY_COLUMNS:
        input_df[column] = 0

    input_df[CLASS_MAP[building_class]] = 1

    # The notebook sanitized column names before XGBoost training.
    input_df.columns = input_df.columns.str.replace(
        r"[^A-Za-z0-9_]+", "_", regex=True
    )

    # IMPORTANT:
    # Get the exact feature names and order from the trained model.
    expected_features = list(model.feature_names_in_)

    # reindex() is safer than input_df[FEATURE_NAMES]:
    # missing columns are created with 0 and the order is guaranteed.
    input_df = input_df.reindex(
        columns=expected_features,
        fill_value=0,
    )

    if input_df.shape[1] != model.n_features_in_:
        st.error(
            f"Feature mismatch: model expects {model.n_features_in_} "
            f"features, but the app created {input_df.shape[1]}."
        )
        st.stop()

    if list(input_df.columns) != expected_features:
        st.error("Feature order mismatch between app input and trained model.")
        st.stop()

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------
    try:
        # The model's columns are already aligned exactly above.
        # Disable XGBoost's redundant feature-name validation because
        # the serialized model can contain feature-name metadata from
        # a different XGBoost serialization version.
        probability = float(
            model.predict_proba(
                input_df,
                validate_features=False,
            )[0, 1]
        )
        prediction = int(probability >= THRESHOLD)
    except Exception as exc:
        st.error(f"Prediction vaqtida xatolik: {exc}")
        st.stop()

    st.subheader("Prediction")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        if prediction == 1:
            st.error("🚨 ELECTRICITY THEFT DETECTED")
        else:
            st.success("✅ NORMAL ELECTRICITY CONSUMPTION")

    with result_col2:
        st.metric(
            "Theft probability",
            f"{probability * 100:.2f}%",
        )

    st.progress(probability)

    if prediction == 1:
        st.warning(
            "Model ushbu yozuvni theft sifatida klassifikatsiya qildi "
            "(threshold = 0.50)."
        )
    else:
        st.info(
            "Model ushbu yozuvni normal sifatida klassifikatsiya qildi "
            "(threshold = 0.50)."
        )

    with st.expander("Modelga yuborilgan 26 ta feature'ni ko'rish"):
        st.dataframe(input_df, use_container_width=True)

with st.sidebar:
    st.header("About the model")
    st.write("**Algorithm:** XGBoost Classifier")
    st.write("**Features:** 26")
    st.write("**Target:** Normal vs Theft")
    st.write("**Decision threshold:** 0.50")
    st.write("**Accuracy:** 95.72%")
    st.write("**ROC-AUC:** 95.50%")
