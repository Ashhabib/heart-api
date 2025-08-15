from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class HeartDiseaseInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.post("/predict")
def predict(data: HeartDiseaseInput):
    # Here you'd load your model and predict
    return {"received_data": data.dict()}


