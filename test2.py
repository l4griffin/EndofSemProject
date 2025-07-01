# training_script.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("mushrooms.csv")

# Encode all categorical columns and save individual encoders
encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

# Save encoders
joblib.dump(encoders, 'encoders.pkl')

# Features & target
X = df.drop('class', axis=1)
y = df['class']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, 'mushroom_model.pkl')
