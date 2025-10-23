import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import time  # To simulate model training time if needed

# --- Configuration ---
DATA_FILE = "wearable_tech_sleep_quality.csv"
TARGET_COLUMN = 'Sleep Quality'
FEATURES = ['Heart Rate', 'Body Temperature', 'Room Temperature', 'Humidity', 'Sleep Stage']
CATEGORICAL_FEATURES = ['Sleep Stage'] # Although numeric after encoding, we treat it as categorical for input

# --- Helper Functions ---

# Function to define sleep quality based on rules (from user's PDF)
def calculate_sleep_quality(row):
    """Applies predefined rules to determine sleep quality."""
    if (60 <= row['Heart Rate'] <= 85) and \
       (35 <= row['Body Temperature'] <= 37) and \
       (40 <= row['Humidity'] <= 60):
        return 'Good'
    elif (row['Heart Rate'] > 90) or \
         (row['Body Temperature'] > 38) or \
         (row['Humidity'] > 70):
        return 'Poor'
    else:
        return 'Average'

# --- Caching Data Loading and Preprocessing ---
@st.cache_data
def load_and_preprocess_data(file_path):
    """Loads data, applies preprocessing steps, and returns processed data and encoder."""
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: Dataset file '{file_path}' not found. Please ensure it's in the same directory.")
        return None, None, None

    # Handle missing values (if any)
    data = data.dropna()

    # Apply feature engineering to create the target variable
    data[TARGET_COLUMN] = data.apply(calculate_sleep_quality, axis=1)

    # Encode categorical features
    encoders = {}
    if 'Sleep Stage' in data.columns:
        le = LabelEncoder()
        data['Sleep Stage'] = le.fit_transform(data['Sleep Stage'])
        encoders['Sleep Stage'] = le # Store encoder
        # Store original class names for the selectbox later
        sleep_stage_classes = le.classes_
    else:
        st.warning("Column 'Sleep Stage' not found. Ensure the CSV has the correct columns.")
        return None, None, None

    return data, encoders, sleep_stage_classes

# --- Caching Model Training ---
@st.cache_resource # Use cache_resource for non-serializable objects like models
def train_model(data):
    """Trains the RandomForestClassifier model and returns the model and scaler."""
    if data is None:
        return None, None, 0.0

    X = data[FEATURES]
    y = data[TARGET_COLUMN]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # Use transform, not fit_transform on test set

    # Train Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    return model, scaler, accuracy

# --- Streamlit App UI ---

st.set_page_config(page_title="Sleep Quality Prediction", layout="wide")

# Title
st.title("🌙 Sleep Quality Prediction using Wearables")
st.markdown("Enter the sensor data below to predict sleep quality.")

# Load data and train model (with caching)
data, encoders, sleep_stage_classes = load_and_preprocess_data(DATA_FILE)

if data is not None and encoders is not None:
    model, scaler, accuracy = train_model(data)

    if model and scaler:
        st.sidebar.header("Enter Sleep Parameters:")

        # --- User Inputs ---
        heart_rate = st.sidebar.slider("Heart Rate (bpm)", min_value=50, max_value=120, value=75)
        body_temp = st.sidebar.slider("Body Temperature (°C)", min_value=34.0, max_value=39.0, value=36.5, step=0.1)
        room_temp = st.sidebar.slider("Room Temperature (°C)", min_value=15, max_value=30, value=22)
        humidity = st.sidebar.slider("Humidity (%)", min_value=20, max_value=80, value=50)

        # Get original class names for the selectbox
        # Use index if classes are available, otherwise default to a reasonable list
        sleep_stage_options = list(sleep_stage_classes) if sleep_stage_classes is not None else ['Awake', 'Light', 'REM', 'Deep']
        sleep_stage_str = st.sidebar.selectbox("Sleep Stage", options=sleep_stage_options, index=1) # Default to 'Light'

        # Prediction Button
        predict_button = st.sidebar.button("Predict Sleep Quality", type="primary")

        # Display Model Accuracy
        st.sidebar.markdown("---")
        st.sidebar.subheader("Model Performance")
        st.sidebar.metric("Accuracy on Test Set", f"{accuracy:.2%}")
        if accuracy == 1.0:
             st.sidebar.warning("Note: 100% accuracy might indicate the rules used to create 'Sleep Quality' are easily learned by the model.")

        # --- Prediction Logic ---
        if predict_button:
            # Prepare input data for prediction
            try:
                # Encode the selected sleep stage using the stored encoder
                sleep_stage_encoded = encoders['Sleep Stage'].transform([sleep_stage_str])[0]

                input_data = pd.DataFrame({
                    'Heart Rate': [heart_rate],
                    'Body Temperature': [body_temp],
                    'Room Temperature': [room_temp],
                    'Humidity': [humidity],
                    'Sleep Stage': [sleep_stage_encoded]
                })

                # Ensure feature order matches training
                input_data = input_data[FEATURES]

                # Scale the input data
                input_data_scaled = scaler.transform(input_data)

                # Make prediction
                prediction = model.predict(input_data_scaled)[0]
                prediction_proba = model.predict_proba(input_data_scaled) # Get probabilities

                # Display Prediction
                st.subheader("Prediction Result:")
                if prediction == 'Good':
                    st.success(f"Predicted Sleep Quality: **Good** 👍")
                elif prediction == 'Average':
                    st.warning(f"Predicted Sleep Quality: **Average** 😐")
                else: # Poor
                    st.error(f"Predicted Sleep Quality: **Poor** 👎")

                # Display Probabilities (Optional)
                st.markdown("---")
                st.subheader("Prediction Probabilities:")
                proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
                st.dataframe(proba_df.style.format("{:.2%}"))


            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.error("Please check the input values and ensure the model trained correctly.")

        else:
             st.info("Adjust the sliders and select a sleep stage on the left, then click 'Predict Sleep Quality'.")

    else:
        st.error("Model training failed. Please check the data and code.")
else:
    # Error handled in load_and_preprocess_data
    pass

st.markdown("---")
st.markdown("*(Based on data from wearable sensors and a Random Forest model)*")
