import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import streamlit as st

# Load and encode dataset
df = pd.read_csv("mushrooms.csv")
selected_features = ['odor', 'gill-color', 'bruises', 'spore-print-color', 'gill-size']

encoders = {}
for column in selected_features + ['class']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le
joblib.dump(encoders, 'encoders_5.pkl')

X = df[selected_features]
y = df['class']
model = RandomForestClassifier()
model.fit(X, y)
joblib.dump(model, 'mushroom_model_5.pkl')

# Load model and encoders
model = joblib.load('mushroom_model_5.pkl')
encoders = joblib.load('encoders_5.pkl')

# Inject background and audio with working assets
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Caveat&display=swap');
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1607082349564-1810c368d10a?auto=format&fit=crop&w=1350&q=80');
        background-size: cover;
        background-attachment: fixed;
        font-family: 'Caveat', cursive;
        color: #fff;
    }
    h1, h2, h3 {
        color: #fff9e6;
        text-shadow: 1px 1px 4px #000;
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

st.markdown("""
    <audio autoplay loop controls style="margin-top: 10px;">
        <source src="https://cdn.pixabay.com/download/audio/2023/03/06/audio_99e5f07860.mp3?filename=forest-magic-loop-142994.mp3" type="audio/mpeg">
        Your browser does not support the audio element.
    </audio>
""", unsafe_allow_html=True)

# App title and description
st.title("The Justoriafin Faerie Oracle Forest 🌈🍄🧚‍♀")
st.subheader("🌸 Whisper your Mushroom's magical traits to the forest...")
st.write("🔮 Speak now, traveler… and the Oracle shall unveil thy mushroom’s fate: nourishment or peril.")

# Feature label mappings
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
    choice = st.selectbox(feature.replace("-", " ").capitalize(), options)
    code = feature_mappings[feature][choice]
    user_input[feature] = encoders[feature].transform([code])[0]

input_df = pd.DataFrame([user_input])
prediction = model.predict(input_df)[0]
prediction_label = encoders['class'].inverse_transform([prediction])[0]

# Display results
if prediction_label == 'e':
    st.balloons()
    st.image("https://i.imgur.com/RXUkmHt.png", width=120)
    st.success("✨ This mushroom is safe and nourishing. You may feast under the moonlight.")
else:
    st.snow()
    st.image("https://i.imgur.com/UcObIsy.png", width=120)
    st.error("⚠ This mushroom holds poisonous magic. Do not touch, lest you be cursed!")

# Sidebar
with st.sidebar:
    st.markdown("## 🧚 Welcome to the Justoriafin Oracle Forest")
    st.markdown("""
    Thanks to three slightly overcaffeinated students,  
    Victoria, Justin, and Griffin,  
    you have stumbled into the whimsical world of mushrooms and mystery.

    Choose your mushroom’s traits wisely…  
    and the Oracle Forest shall reveal its secrets 🍄
    """)
