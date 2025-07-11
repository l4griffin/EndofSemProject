import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the data
df = pd.read_csv("mushrooms.csv")

# HTML + CSS styling
st.markdown("""
    <style>
    body {
        background-image: url('https://cdnb.artstation.com/p/assets/images/images/039/010/409/large/sergei-smekalov-fairy-forest-ts-serjo.jpg?1624690729');
        background-size: cover;
        background-attachment: fixed;
        font-family: 'Caveat', cursive;
        color: #ffffff;
    }
    h1, h2, h3 {
        color: #fffcf2;
        text-shadow: 1px 1px 4px #444;
    }
    div.stButton > button {
        background-color: #7ec850;
        color: white;
        border-radius: 12px;
        font-weight: bold;
        box-shadow: 2px 2px 6px #345920;
        transition: 0.3s ease-in-out;
    }
    div.stButton > button:hover {
        background-color: #9fe870;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# Background ambient music
st.markdown("""
    <audio autoplay loop>
        <source src="https://cdn.pixabay.com/audio/2022/03/23/audio_3d3e08ff3a.mp3" type="audio/mpeg">
    </audio>
""", unsafe_allow_html=True)

# Encode features
selected_features = ['odor', 'gill-color', 'bruises', 'spore-print-color', 'gill-size']
encoders = {}
for column in selected_features + ['class']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

# Save encoders and model
joblib.dump(encoders, 'encoders_5.pkl')
X = df[selected_features]
y = df['class']
model = RandomForestClassifier()
model.fit(X, y)
joblib.dump(model, 'mushroom_model_5.pkl')

# Load model and encoders
model = joblib.load('mushroom_model_5.pkl')
encoders = joblib.load('encoders_5.pkl')

# App Title and Instructions
st.title("The Justoriafin Faerie Oracle Forest 🌈🍄🧚‍♀")
st.subheader("🌸 Whisper your Mushroom's magical traits to the forest...")
st.write("🔮 Speak now, traveler… and the Oracle shall unveil thy mushroom’s fate: nourishment or peril.")

# Mapping for user-friendly inputs
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

# Get user input
user_input = {}
for feature in selected_features:
    options = list(feature_mappings[feature].keys())
    user_choice = st.selectbox(feature.replace("-", " ").capitalize(), options)
    short_code = feature_mappings[feature][user_choice]
    user_input[feature] = encoders[feature].transform([short_code])[0]

# Predict and display result
input_df = pd.DataFrame([user_input])
prediction_label = model.predict(input_df)[0]
decoded_label = encoders['class'].inverse_transform([prediction_label])[0]

if decoded_label == 'e':
    st.balloons()
    st.image("https://i.imgur.com/RXUkmHt.png", width=120)
    st.success("✨ This mushroom is safe and nourishing. You may feast under the moonlight.")
else:
    st.snow()
    st.image("https://i.imgur.com/UcObIsy.png", width=120)
    st.error("⚠ This mushroom holds poisonous magic. Do not touch, lest you be cursed!")

# Sidebar credits
with st.sidebar:
    st.markdown("## 🧚 Welcome to the Justoriafin Oracle Forest")
    st.markdown("""
    Thanks to three slightly overcaffeinated students —  
    Victoria, Justin, and Griffin —  
    you have stumbled into the whimsical world of mushrooms and mystery.

    Choose your mushroom’s traits wisely,  
    and the Oracle Forest shall whisper its secrets to you:  
    all with a pinch of data and a sprinkle of faerie magic 🍄✨
    """)
