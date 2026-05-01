from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("model.joblib")

class Input(BaseModel):
    study_hours: float
    marks: float

@app.get("/")
def home():
    return {"message": "API Running"}

@app.post("/predict")
def predict(data: Input):
    result = model.predict([[data.study_hours, data.marks]])[0]
    return {
        "prediction": int(result),
        "result": "Pass ✅" if result == 1 else "Fail ❌"
    }
