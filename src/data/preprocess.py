import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(config_path="params.yaml"):
    with open(config_path, 'r') as file:
        params = yaml.safe_load(file)
    
    data_params = params['data']
    preprocess_params = params['preprocessing']

    raw_data_path = os.path.join(data_params['raw_data_dir'], "flight_data.csv")
    df = pd.read_csv(raw_data_path)
    
    # Drop missing values
    df.dropna(inplace=True)
    
    # Encode categorical variables
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    # Split data
    X = df.drop('delay', axis=1)
    y = df['delay']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=preprocess_params['test_size'], 
        random_state=preprocess_params['random_state']
    )
    
    # Save preprocessed data
    os.makedirs(data_params['processed_data_dir'], exist_ok=True)
    X_train.to_csv(os.path.join(data_params['processed_data_dir'], "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(data_params['processed_data_dir'], "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(data_params['processed_data_dir'], "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(data_params['processed_data_dir'], "y_test.csv"), index=False)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_data()
