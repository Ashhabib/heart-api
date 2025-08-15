from fastapi import FastAPI
from schemas import HeartDiseaseInput  # Make sure this matches the file name
import joblib
import numpy as np

app = FastAPI(title="Heart Disease Prediction API", version="1.0")

# Load model
model = joblib.load("model.pkl")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: HeartDiseaseInput):
    features = np.array([[data.age, data.sex, data.cp, data.trestbps, data.chol,
                          data.fbs, data.restecg, data.thalach, data.exang,
                          data.oldpeak, data.slope, data.ca, data.thal]])
    prediction = model.predict(features)[0]
    return {"heart_disease": bool(prediction)}
