from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any, Dict
import joblib
import pandas as pd
import os

# Initialize FastAPI app
app = FastAPI(title="Social Media Feed ML API")

# Model loading (adjust as needed for your model)
MODEL_PATH = "models/saved/xgboost.joblib"  # Change to your preferred model
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# Example input schema (adjust fields as per your model's requirements)
class PredictRequest(BaseModel):
    data: List[Dict[str, Any]]  # List of feature dicts

class PredictResponse(BaseModel):
    predictions: List[Any]

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        df = pd.DataFrame(request.data)
        preds = model.predict(df)
        return PredictResponse(predictions=preds.tolist())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Optional: Data ingestion endpoint (for future use)
class IngestRequest(BaseModel):
    data: List[Dict[str, Any]]

@app.post("/ingest")
def ingest(request: IngestRequest):
    # Here you could save data to a database or file
    # For demo, just return the count
    return {"received": len(request.data)}

@app.get("/")
def root():
    return {"message": "Social Media Feed ML API is running."} 