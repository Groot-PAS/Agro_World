import os
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set a seed for reproducibility
random.seed(42)
np.random.seed(42)

# Create a simple dataset for crop recommendations
# Assume the dataset has features like temperature, humidity, pH, rainfall, etc.
data = {
    "temperature": np.random.uniform(20, 40, 1000),
    "humidity": np.random.uniform(30, 90, 1000),
    "pH": np.random.uniform(5.5, 7.5, 1000),
    "rainfall": np.random.uniform(50, 300, 1000),
    "crop": np.random.choice(["Wheat", "Rice", "Maize", "Sugarcane"], 1000)
}

df = pd.DataFrame(data)

# Map crop names to integers for model training
crop_mapping = {"Wheat": 0, "Rice": 1, "Maize": 2, "Sugarcane": 3}
df["crop_label"] = df["crop"].map(crop_mapping)

# Features and target
X = df[["temperature", "humidity", "pH", "rainfall"]]
y = df["crop_label"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model trained successfully! Accuracy on test set: {accuracy:.2f}")

# Save the model for use in the Flask app
import joblib
model_path = "models/crop_recommendation_model.pkl"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)
print(f"Model saved at: {model_path}")
