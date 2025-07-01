# training_selected_features.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
df = pd.read_csv("mushrooms.csv")

# Define selected features
selected_features = ['odor', 'gill-color', 'bruises', 'spore-print-color', 'gill-size']

# Encode all columns and save encoders
encoders = {}
for column in selected_features + ['class']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

# Save encoders
joblib.dump(encoders, 'encoders_5.pkl')

# Train model using only selected features
X = df[selected_features]
y = df['class']

model = RandomForestClassifier()
model.fit(X, y)

# Save trained model
joblib.dump(model, 'mushroom_model_5.pkl')


# streamlit_app_5features.py
import streamlit as st
import pandas as pd
import joblib

# Load trained model and encoders
model = joblib.load('mushroom_model_5.pkl')
encoders = joblib.load('encoders_5.pkl')

st.title("🍄 Mushroom Edibility Predictor (5 Features)")

# Define the same 5 features
selected_features = ['odor', 'gill-color', 'bruises', 'spore-print-color', 'gill-size']

# Get user input for all 5 features
user_input = {}
for feature in selected_features:
    options = list(encoders[feature].classes_)
    user_input[feature] = st.selectbox(feature.replace("-", " ").capitalize(), options)

# Encode user input
for feature in selected_features:
    user_input[feature] = encoders[feature].transform([user_input[feature]])[0]

# Create input DataFrame
input_df = pd.DataFrame([user_input])

# Predict
prediction = model.predict(input_df)[0]
prediction_label = encoders['class'].inverse_transform([prediction])[0]

# Output result
st.subheader("Prediction:")
st.success("🟢 Edible" if prediction_label == 'e' else "🔴 Poisonous")
