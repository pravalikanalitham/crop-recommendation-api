
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import os
import subprocess
import sys


app = FastAPI(
    title="Crop Recommendation API",
    description="ML-based Crop Recommendation System",
    version="1.0"
)


MODEL_PATH = "crop_model.pkl"

if not os.path.exists(MODEL_PATH):
    subprocess.run([sys.executable, "train_model.py"], check=True)

model = joblib.load(MODEL_PATH)


class CropInput(BaseModel):
    Temperature: float = Field(...)
    Humidity: float = Field(...)
    Rainfall: float = Field(...)
    PH: float = Field(...)

    Nitrogen: float = Field(...)
    Phosphorous: float = Field(...)
    Potassium: float = Field(...)
    Carbon: float = Field(...)

    Soil: str = Field(..., min_length=1)
    Season: str = Field(..., min_length=1)
    Secondary_Season: str = Field(..., min_length=1)


@app.get("/")
def home():
    return {"message": "Crop Recommendation API is running!"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/recommend")
def recommend_crop(data: CropInput):
    try:
        input_data = pd.DataFrame([data.model_dump()])

        probabilities = model.predict_proba(input_data)[0]
        crop_names = model.classes_

        top_indices = np.argsort(probabilities)[::-1][:3]

        recommendations = []
        for index in top_indices:
            recommendations.append({
                "crop": str(crop_names[index]),
                "confidence": round(float(probabilities[index] * 100), 2)
            })

        return {
            "message": "Crop recommendations generated successfully",
            "recommendations": recommendations
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
