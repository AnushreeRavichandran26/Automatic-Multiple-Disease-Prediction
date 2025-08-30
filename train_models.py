import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------
# Train Diabetes Model
# -----------------
print("Training Diabetes Model...")
diabetes_data = pd.read_csv("datasets/diabetes.csv")
X = diabetes_data.drop("Outcome", axis=1)
y = diabetes_data["Outcome"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_diabetes = LogisticRegression(max_iter=5000)
model_diabetes.fit(X_train, y_train)

y_pred = model_diabetes.predict(X_test)
print("Diabetes Model Accuracy:", accuracy_score(y_test, y_pred))

pickle.dump(model_diabetes, open("models/diabetes_model.sav", "wb"))


# -----------------
# Train Heart Disease Model
# -----------------
print("\nTraining Heart Disease Model...")
heart_data = pd.read_csv("datasets/heart.csv")
X = heart_data.drop("target", axis=1)
y = heart_data["target"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_heart = LogisticRegression(max_iter=5000)
model_heart.fit(X_train, y_train)

y_pred = model_heart.predict(X_test)
print("Heart Disease Model Accuracy:", accuracy_score(y_test, y_pred))

pickle.dump(model_heart, open("models/heart_disease_model.sav", "wb"))


# -----------------
# Train Parkinson’s Model
# -----------------
print("\nTraining Parkinson’s Model...")
parkinsons_data = pd.read_csv("datasets/parkinsons.csv")

# Drop non-numeric column 'name'
if "name" in parkinsons_data.columns:
    parkinsons_data = parkinsons_data.drop("name", axis=1)

X = parkinsons_data.drop("status", axis=1)
y = parkinsons_data["status"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_parkinsons = LogisticRegression(max_iter=5000)
model_parkinsons.fit(X_train, y_train)

y_pred = model_parkinsons.predict(X_test)
print("Parkinson’s Model Accuracy:", accuracy_score(y_test, y_pred))

pickle.dump(model_parkinsons, open("models/parkinsons_model.sav", "wb"))

print("\n✅ All models trained and saved successfully!")
