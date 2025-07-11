# Training the data models
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import streamlit as st

# Load and prepare the dataset
df = pd.read_csv("mushrooms.csv")
selected_features = ['odor', 'gill-color', 'bruises', 'spore-print-color', 'gill-size']

# Encode categorical features in the app
encoders = {}
for column in selected_features + ['class']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

# Save encoders
joblib.dump(encoders, 'encoders_5.pkl')

# Train the data model
X = df[selected_features]
y = df['class']
model = RandomForestClassifier()
model.fit(X, y)
joblib.dump(model, 'mushroom_model_5.pkl')

# Load the model and the encoders
model = joblib.load('mushroom_model_5.pkl')
encoders = joblib.load('encoders_5.pkl')

#Input custom background to style and set a theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Caveat&display=swap');

    .stApp {
        background-image: url('https://images.unsplash.com/photo-1536002955755-69c99f7f5244');
        background-size: cover;
        background-attachment: fixed;
        font-family: 'Caveat', cursive;
        color: #fff;
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
    .sidebar .sidebar-content {
        background-color: rgba(0, 0, 0, 0.4);
        padding: 1em;
        border-radius: 12px;
        color: #fceee9;
    }
    </style>
""", unsafe_allow_html=True)

#add an enchanted magial faerie audio to set the mood
st.markdown("""
    <audio autoplay loop controls style="margin-top: 10px;">
        <source src="https://cdn.pixabay.com/audio/2022/03/23/audio_6b71c3b7d3.mp3" type="audio/mpeg">
        Your browser does not support the audio element.
    </audio>
""", unsafe_allow_html=True)

#Give the app a title and brief description
st.title("The Justoriafin Faerie Oracle Forest 🌈🍄🧚‍♀")
st.subheader("🌸 Whisper your Mushroom's magical traits to the forest...")
st.write("🔮 Speak now, traveler… and the Oracle shall unveil thy mushroom’s fate: nourishment or peril.")

#Add feature label mappings
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

#Allow the user to input data
user_input = {}
for feature in selected_features:
    options = list(feature_mappings[feature].keys())
    user_choice = st.selectbox(feature.replace("-", " ").capitalize(), options)
    code = feature_mappings[feature][user_choice]
    user_input[feature] = encoders[feature].transform([code])[0]

input_df = pd.DataFrame([user_input])
prediction = model.predict(input_df)[0]
prediction_label = encoders['class'].inverse_transform([prediction])[0]

#Display prediction of User's input
if prediction_label == 'e':
    st.balloons()
    st.image("https://i.imgur.com/RXUkmHt.png", width=120)
    st.success("✨ This mushroom is safe and nourishing. You may feast under the moonlight.")
else:
    st.snow()
    st.image("https://i.imgur.com/UcObIsy.png", width=120)
    st.error("⚠ This mushroom holds poisonous magic. Do not touch, lest you be cursed!")

#Add an additional sidebar highlighting the group(4)
with st.sidebar:
    st.markdown("## 🧚 Welcome to the Justoriafin Oracle Forest")
    st.markdown("""
    Thanks to three slightly overcaffeinated students,  
    Victoria, Justin, and Griffin,  
    you have stumbled into the whimsical world of mushrooms and mystery.

    This curious oracle is part of their Info-sec semester project—  
    where data science meets enchanted spores.

    Choose your mushroom’s traits wisely…  
    and the Oracle Forest shall reveal its secrets:  
    with the help of a pinch of data and a sprinkle of magic 🍄!
    """)
