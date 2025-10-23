# Sleep Quality Prediction using Wearable Tech Dataset
# ---------------------------------------------------

# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Step 2: Load Dataset
file_path = "D:/CL - I Mini Project/wearable_tech_sleep_quality (1).csv"  # <-- change to your path
data = pd.read_csv(file_path)
print("✅ Dataset Loaded Successfully!")
print("\nFirst 5 rows:\n", data.head())
print("\nColumns:", list(data.columns))

# Step 3: Check for missing values
print("\nChecking missing values:\n", data.isnull().sum())
data = data.dropna()

# Step 4: Encode categorical columns
if 'Sleep Stage' in data.columns:
    label_encoder = LabelEncoder()
    data['Sleep Stage'] = label_encoder.fit_transform(data['Sleep Stage'])
else:
    print("⚠️ 'Sleep Stage' column not found. Please verify dataset.")

# Step 5: Create a Sleep Quality Label (based on conditions)
# We'll create a target label since dataset doesn’t have sleep_quality directly
def calculate_sleep_quality(row):
    # Define simple rules based on physiological norms
    if (60 <= row['Heart Rate'] <= 85) and (35 <= row['Body Temperature'] <= 37) and (40 <= row['Humidity'] <= 60):
        return "Good"
    elif (50 <= row['Heart Rate'] <= 95) and (34 <= row['Body Temperature'] <= 38):
        return "Average"
    else:
        return "Poor"

data["Sleep Quality"] = data.apply(calculate_sleep_quality, axis=1)

# Step 6: Prepare data for model
features = ['Heart Rate', 'Body Temperature', 'Room Temperature', 'Humidity', 'Sleep Stage']
for col in features:
    if col not in data.columns:
        raise KeyError(f"❌ Missing expected column: {col}")

X = data[features]
y = data['Sleep Quality']

# Normalize numeric data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 8: Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 9: Evaluate Model
y_pred = model.predict(X_test)
print("\nModel Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 10: Ask User Inputs
print("\n--- Sleep Quality Prediction Assistant ---")
heart_rate = float(input("Enter your average Heart Rate (bpm): "))
body_temp = float(input("Enter your Body Temperature (°C): "))
room_temp = float(input("Enter Room Temperature (°C): "))
humidity = float(input("Enter Room Humidity (%): "))
sleep_stage_input = input("Enter your Sleep Stage (Light/Deep/REM): ").capitalize()

# Encode user’s sleep stage using same encoder
if sleep_stage_input not in label_encoder.classes_:
    print(f"⚠️ '{sleep_stage_input}' not recognized, defaulting to 'Light'")
    sleep_stage_encoded = label_encoder.transform(['Light'])[0]
else:
    sleep_stage_encoded = label_encoder.transform([sleep_stage_input])[0]

# Step 11: Predict Sleep Quality
user_data = pd.DataFrame([[heart_rate, body_temp, room_temp, humidity, sleep_stage_encoded]],
                         columns=['Heart Rate', 'Body Temperature', 'Room Temperature', 'Humidity', 'Sleep Stage'])

# Apply the same scaler
user_scaled = scaler.transform(user_data)

# Make prediction
prediction = model.predict(user_scaled)[0]

# Step 12: Display Result with Analysis
print("\n🩺 Sleep Quality Analysis 🩺")
print(f"Predicted Sleep Quality: {prediction}")

if prediction == "Good":
    print("✅ You seem to have a healthy sleep pattern. Keep maintaining balanced heart rate and temperature!")
elif prediction == "Average":
    print("😴 Your sleep quality is moderate. Try to maintain consistent sleep times and improve room conditions.")
else:
    print("⚠️ Poor sleep detected. Consider improving your room environment or consulting a doctor if persistent.")

print("\nAnalysis based on your inputs:")
print(f"- Heart Rate: {heart_rate} bpm")
print(f"- Body Temperature: {body_temp} °C")
print(f"- Room Temperature: {room_temp} °C")
print(f"- Humidity: {humidity}%")
print(f"- Sleep Stage: {sleep_stage_input}")

print("\n🌙 Thank you for using the Sleep Quality Prediction System 🌙")