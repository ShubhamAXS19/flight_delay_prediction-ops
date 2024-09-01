import os
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def evaluate_model(X_test, y_test, model, metrics):
    y_pred = model.predict(X_test)
    
    results = {}
    if "accuracy" in metrics:
        results["accuracy"] = accuracy_score(y_test, y_pred)
    if "f1" in metrics:
        results["f1"] = f1_score(y_test, y_pred)
    if "precision" in metrics:
        results["precision"] = precision_score(y_test, y_pred)
    if "recall" in metrics:
        results["recall"] = recall_score(y_test, y_pred)
    
    return results

def run_evaluation(config_path="params.yaml"):
    with open(config_path, "r") as file:
        params = yaml.safe_load(file)
    
    # Load test data
    X_test = pd.read_csv(os.path.join(params['data']['processed_data_dir'], "X_test.csv"))
    y_test = pd.read_csv(os.path.join(params['data']['processed_data_dir'], "y_test.csv"))
    
    # Load model
    model_path = os.path.join("models", "xgb_model.json")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    # Evaluate
    results = evaluate_model(X_test, y_test.values.ravel(), model, params['evaluation']['metrics'])
    
    # Log results
    for metric, value in results.items():
        print(f"{metric}: {value}")
    
if __name__ == "__main__":
    run_evaluation()
