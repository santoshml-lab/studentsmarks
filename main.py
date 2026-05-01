from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS FIX (MOST IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load("model.joblib")

# Input schema
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
