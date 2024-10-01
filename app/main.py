# sales_forecasting_api/app/main.py

from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
import numpy as np
from pydantic import BaseModel

# Load the trained model and scaler
model = joblib.load("/Users/vega7unk/Documents/3rd_Sem_DSI/Adv_ML/AT2/Project/adv_mla_at2/models/predictive/xgb_predictive_model.pkl")
# scaler = joblib.load("app/model/scaler.pkl")

# Initialize FastAPI
app = FastAPI()

# Define the input data schema
class SalesPredictionInput(BaseModel):
    item_id: int
    dept_id: int
    cat_id: int
    store_id: int
    state_id: int
    day_of_week: int
    month: int
    year: int

# Home endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sales Prediction API"}

# Prediction endpoint
@app.post("/predict_sales/")
def predict_sales(input_data: SalesPredictionInput):
    # Convert input data to DataFrame
    data = pd.DataFrame([input_data.dict()])

    # Scale the features
    scaled_data = scaler.transform(data)

    # Predict sales
    prediction = model.predict(scaled_data)
    predicted_sales = np.expm1(prediction[0])  # If log-transformed during training, use expm1 to inverse transform

    return {"predicted_sales": predicted_sales}
