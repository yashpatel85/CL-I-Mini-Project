import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# -------------------------------------------------------------------
# 1. MODEL TRAINING (PLACEHOLDER)
# -------------------------------------------------------------------
# REPLACE THIS FUNCTION with your own data loading and model training code.
# This function should return your trained model/pipeline and the preprocessor.

@st.cache_resource
def get_model_and_preprocessor():
    """
    This is a placeholder function.
    You should replace this with your actual data loading and model training.
    """
    # Create a mock dataset (replace with loading your CSV)
    data = {
        'Age': [30, 45, 25, 50, 35, 60, 28, 42, 33, 55],
        'Physical Activity Level': [30, 60, 45, 20, 50, 10, 70, 40, 60, 30],
        'Stress Level': ['Low', 'Medium', 'Medium', 'High', 'Low', 'High', 'Low', 'Medium', 'Low', 'Medium'],
        'Heart Rate': [70, 75, 68, 80, 72, 85, 65, 78, 70, 76],
        'Daily Steps': [5000, 8000, 7000, 3000, 9000, 2500, 10000, 6000, 8500, 4500],
        'Sleep Quality': ['Good', 'Good', 'Good', 'Poor', 'Good', 'Poor', 'Good', 'Poor', 'Good', 'Poor']
    }
    df = pd.DataFrame(data)

    X = df.drop('Sleep Quality', axis=1)
    y = df['Sleep Quality']

    # Define feature types
    numeric_features = ['Age', 'Physical Activity Level', 'Heart Rate', 'Daily Steps']
    categorical_features = ['Stress Level']

    # Create a preprocessing pipeline
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create the full model pipeline
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', RandomForestClassifier(random_state=42))])

    # Train the model
    model_pipeline.fit(X, y)
    
    return model_pipeline, X.columns

# Load the model
model, feature_names = get_model_and_preprocessor()

# -------------------------------------------------------------------
# 2. SUGGESTION ENGINE
# -------------------------------------------------------------------
def get_suggestions(prediction, user_inputs):
    """
    Generates personalized suggestions based on prediction and user inputs.
    """
    suggestions = []

    if prediction == 'Good':
        return """
        <div style="background-color:#e0f8e0; border-left: 6px solid #4CAF50; padding: 10px; border-radius: 5px;">
            <h4 style="color: #4CAF50;">Great News! Your Predicted Sleep Quality is Good!</h4>
            <p>Keep up the great work. Here are some tips to maintain this healthy habit:</p>
            <ul>
                <li><strong>Consistency is Key:</strong> Try to stick to your current sleep schedule, even on weekends.</li>
                <li><strong>Stay Active:</strong> Your current physical activity level is clearly helping.</li>
                <li><strong>Mindful Moments:</strong> Continue managing your stress effectively.</li>
            </ul>
        </div>
        """

    # If prediction is 'Poor'
    suggestions.append(
        """
        <div style="background-color:#f8e0e0; border-left: 6px solid #F44336; padding: 10px; border-radius: 5px;">
            <h4 style="color: #F44336;">Your Predicted Sleep Quality is Poor.</h4>
            <p>Don't worry, there are small changes you can make to improve it. Here are some personalized suggestions based on your inputs:</p>
            <ul>
        """
    )
    
    # Personalized tips based on inputs
    if user_inputs['Stress Level'] == 'High':
        suggestions.append("<li><strong>Manage Stress:</strong> Your stress level is high. Consider relaxation techniques like meditation, deep breathing, or yoga before bed. </li>")
    elif user_inputs['Stress Level'] == 'Medium':
        suggestions.append("<li><strong>Wind Down:</strong> Your stress level is medium. Try to create a relaxing bedtime ritual, like reading a book or listening to calm music, to signal to your body it's time to sleep.</li>")

    if user_inputs['Physical Activity Level'] < 30:
        suggestions.append("<li><strong>Get Moving:</strong> Your physical activity is low. Even 20-30 minutes of moderate exercise (like a brisk walk) during the day can significantly improve sleep.</li>")
    
    if user_inputs['Daily Steps'] < 5000:
        suggestions.append("<li><strong>Increase Daily Steps:</strong> Your step count is low. Try to incorporate more walking into your day, like taking the stairs or a short walk after meals.</li>")

    # General tips for poor sleep
    suggestions.append("<li><strong>Sleep Schedule:</strong> Try to go to bed and wake up at the same time every day, even on weekends.</li>")
    suggestions.append("<li><strong>Screen Time:</strong> Avoid screens (phones, TVs, computers) at least an hour before bed. The blue light can interfere with your sleep hormones.</li>")
    suggestions.append("<li><strong>Create a Restful Environment:</strong> Make sure your bedroom is dark, quiet, and cool.</li>")
    
    suggestions.append("</ul></div>")
    return "".join(suggestions)

# -------------------------------------------------------------------
# 3. STREAMLIT UI
# -------------------------------------------------------------------
st.set_page_config(page_title="Sleep Quality Predictor", layout="wide")
st.title("💤 Sleep Quality Predictor")
st.markdown("Enter your wearable data and lifestyle factors below to predict your sleep quality and get personalized tips.")

# Sidebar for user inputs
st.sidebar.header("Enter Your Data:")

def user_input_features():
    age = st.sidebar.slider("Age", 18, 80, 35)
    activity = st.sidebar.slider("Physical Activity Level (minutes/day)", 0, 120, 30)
    stress = st.sidebar.select_slider("Stress Level", options=['Low', 'Medium', 'High'], value='Medium')
    heart_rate = st.sidebar.slider("Resting Heart Rate (bpm)", 50, 100, 70)
    steps = st.sidebar.slider("Daily Steps", 1000, 20000, 5000)

    data = {
        'Age': age,
        'Physical Activity Level': activity,
        'Stress Level': stress,
        'Heart Rate': heart_rate,
        'Daily Steps': steps
    }
    
    # Ensure columns are in the same order as training
    features = pd.DataFrame(data, index=[0])[list(feature_names)]
    return features, data

input_df, user_data = user_input_features()

# Main page layout
col1, col2 = st.columns([1, 1])

# Column 1: Inputs
with col1:
    st.subheader("Your Inputs")
    st.dataframe(input_df.T.rename(columns={0: 'Values'}))

# Column 2: Prediction and Suggestions
with col2:
    st.subheader("Prediction & Suggestions")
    
    if st.button("Predict My Sleep Quality"):
        try:
            # Make prediction
            prediction = model.predict(input_df)[0]
            
            # Get and display suggestions
            suggestions_html = get_suggestions(prediction, user_data)
            st.markdown(suggestions_html, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")