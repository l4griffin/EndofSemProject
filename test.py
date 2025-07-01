import pandas as pd

df = pd.read_csv("mushrooms.csv")
print(df.head())
print(df.info())


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

for column in df.columns:
    df[column] = label_encoder.fit_transform(df[column])


from sklearn.model_selection import train_test_split

X = df.drop('class', axis=1)  # features
y = df['class']               # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
df = pd.read_csv("mushrooms.csv")

# Label encode each column separately and save encoders
encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

# Save encoders for later use in the app
joblib.dump(encoders, 'encoders.pkl')

# Train/test split
X = df.drop('class', axis=1)
y = df['class']

model = RandomForestClassifier()
model.fit(X, y)

# Save trained model
joblib.dump(model, 'mushroom_model.pkl')



import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load('mushroom_model.pkl')
encoders = joblib.load('encoders.pkl')

st.title("🍄 Mushroom Edibility Predictor")

# Get user input
cap_shape = st.selectbox("Cap Shape", ['b', 'c', 'x', 'f', 'k', 's'])
cap_color = st.selectbox("Cap Color", ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'])
odor = st.selectbox("Odor", ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'])
gill_color = st.selectbox("Gill Color", ['k', 'n', 'g', 'p', 'w', 'h', 'u', 'e', 'b', 'y'])

# Add more features as needed

user_input = {
    'cap-shape': cap_shape,
    'cap-color': cap_color,
    'odor': odor,
    'gill-color': gill_color,
    # Add all other required features from the original dataset
}

# Encode user input using the saved encoders
for feature in user_input:
    le = encoders[feature]
    user_input[feature] = le.transform([user_input[feature]])[0]

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Predict
prediction = model.predict(input_df)[0]
prediction_label = encoders['class'].inverse_transform([prediction])[0]

st.subheader(f"Prediction: {'🟢 Edible' if prediction_label == 'e' else '🔴 Poisonous'}")
