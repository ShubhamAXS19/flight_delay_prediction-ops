from fastapi import FastAPI, HTTPException
import pandas as pd
import xgboost as xgb
import yaml
import os
from pydantic import BaseModel
from typing import List

# Load parameters
def load_params(config_path="params.yaml"):
    with open(config_path, "r") as file:
        params = yaml.safe_load(file)
    return params

params = load_params()

# Load the model
model_path = os.path.join("models", "xgb_model.json")
model = xgb.XGBClassifier()
model.load_model(model_path)

app = FastAPI()

# Define the request body schema
class FlightData(BaseModel):
    features: List[float]

@app.get("/")
def read_root():
    return {"message": "Welcome to the Flight Delay Prediction API"}

@app.post("/predict/")
def predict_flight_delay(data: FlightData):
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([data.features])

        # Predict using the model
        prediction = model.predict(df)

        # Return the result
        return {"delay_prediction": int(prediction[0])}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)