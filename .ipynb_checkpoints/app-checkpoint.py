import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# === 1. Load CSV ===
df = pd.read_csv('calories_dataset.csv')

# === 2. Clean text columns ===
df['ActivityType'] = df['ActivityType'].str.strip()
df['SpeedCategory'] = df['SpeedCategory'].str.strip()

# === 3. Map categories to numbers ===
activity_mapping = {'Walk': 0, 'Run': 1, 'Cycle': 2}
speed_mapping = {'Slow': 0, 'Medium': 1, 'Fast': 2}

df['ActivityTypeNumeric'] = df['ActivityType'].map(activity_mapping)
df['SpeedCategoryNumeric'] = df['SpeedCategory'].map(speed_mapping)

# === 4. Drop missing values ===
df = df.dropna()

# === 5. Train model ===
X = df[['Distance', 'Weight', 'SpeedCategoryNumeric', 'ActivityTypeNumeric', 'Time']]
y = df['Calories']

model = LinearRegression()
model.fit(X, y)

# === 6. Streamlit app ===
st.title("Calories Prediction App")
st.write("""
Enter your details below to predict calories burned.
""")

with st.form("prediction_form"):
    distance = st.number_input("Distance (km)", min_value=0.0, step=0.1)
    weight = st.number_input("Weight (kg)", min_value=0.0, step=0.1)
    speed_category = st.selectbox("Speed Category", ['Slow', 'Medium', 'Fast'])
    activity_type = st.selectbox("Activity Type", ['Walk', 'Run', 'Cycle'])
    time = st.number_input("Time (minutes)", min_value=0.0, step=0.1)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Convert user input to numeric
    speed_numeric = speed_mapping[speed_category]
    activity_numeric = activity_mapping[activity_type]

    input_features = np.array([[distance, weight, speed_numeric, activity_numeric, time]])
    prediction = model.predict(input_features)

    st.subheader(f"✅ Predicted Calories Burned: {prediction[0]:.2f}")
