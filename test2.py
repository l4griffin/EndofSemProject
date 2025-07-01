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


import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load('mushroom_model_5.pkl')
encoders = joblib.load('encoders_5.pkl')

st.title("🍄 Mushroom Edibility Predictor (Descriptive Input)")

# Feature label mappings: Full name → short code
feature_mappings = {
    'odor': {
        "Almond": 'a', "Anise": 'l', "Creosote": 'c', "Fishy": 'y',
        "Foul": 'f', "Musty": 'm', "None": 'n', "Pungent": 'p', "Spicy": 's'
    },
    'gill-color': {
        "Black": 'k', "Brown": 'n', "Buff": 'b', "Chocolate": 'h', "Gray": 'g',
        "Green": 'r', "Orange": 'o', "Pink": 'p', "Purple": 'u', "Red": 'e',
        "White": 'w', "Yellow": 'y'
    },
    'bruises': {
        "Bruises": 't', "No Bruises": 'f'
    },
    'spore-print-color': {
        "Black": 'k', "Brown": 'n', "Buff": 'b', "Chocolate": 'h', "Green": 'r',
        "Orange": 'o', "Purple": 'u', "White": 'w', "Yellow": 'y'
    },
    'gill-size': {
        "Broad": 'b', "Narrow": 'n'
    }
}

# Features to collect
selected_features = list(feature_mappings.keys())
user_input = {}

# Show user-friendly dropdowns
for feature in selected_features:
    full_names = list(feature_mappings[feature].keys())
    user_friendly_value = st.selectbox(feature.replace("-", " ").capitalize(), full_names)
    # Convert full name to short code for model input
    code = feature_mappings[feature][user_friendly_value]
    # Encode using saved label encoder
    user_input[feature] = encoders[feature].transform([code])[0]

# Create input DataFrame
input_df = pd.DataFrame([user_input])

# Predict
prediction = model.predict(input_df)[0]
prediction_label = encoders['class'].inverse_transform([prediction])[0]

# Output
st.subheader("Prediction:")
st.success("🟢 Edible" if prediction_label == 'e' else "🔴 Poisonous")
