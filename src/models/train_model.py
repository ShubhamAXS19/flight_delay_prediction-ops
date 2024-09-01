import os
import pandas as pd
import xgboost as xgb
import mlflow
import yaml
from src.utils.s3_utils import upload_to_s3

def load_params(params_path="params.yaml"):
    with open(params_path, "r") as file:
        params = yaml.safe_load(file)
    return params

def train_model(X_train, y_train, model_params):
    model = xgb.XGBClassifier(**model_params)
    model.fit(X_train, y_train)
    return model

def run_training(config_path="params.yaml"):
    params = load_params(config_path)
    
    # Load training data
    X_train = pd.read_csv(os.path.join(params['data']['processed_data_dir'], "X_train.csv"))
    y_train = pd.read_csv(os.path.join(params['data']['processed_data_dir'], "y_train.csv"))
    
    mlflow.set_experiment("flight_delay_prediction")
    with mlflow.start_run():
        mlflow.log_params(params['model']['params'])
        
        model = train_model(X_train, y_train.values.ravel(), params['model']['params'])
        
        # Log model
        mlflow.xgboost.log_model(model, "model")
        
        # Save and upload model
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "xgb_model.json")
        model.save_model(model_path)
        
        upload_to_s3(params['s3']['bucket_name'], "models/xgb_model.json", model_path)

if __name__ == "__main__":
    run_training()
