from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np

app = FastAPI(title="Heart Disease Classifier", version="1.0")
model = joblib.load("model.pkl")

FEATURES = ['age','sex','cp','trestbps','chol','fbs','restecg',
            'thalach','exang','oldpeak','slope','ca','thal']

class PredictRequest(BaseModel):
    age: float; sex: float; cp: float; trestbps: float
    chol: float; fbs: float; restecg: float; thalach: float
    exang: float; oldpeak: float; slope: float; ca: float; thal: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    X = np.array([[getattr(req, f) for f in FEATURES]])
    pred = int(model.predict(X)[0])
    prob = round(float(model.predict_proba(X)[0][1]), 4)
    return {"prediction": pred, "probability": prob,
            "label": "Disease detected" if pred == 1 else "No disease"}