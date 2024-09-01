import os
import yaml
from src.utils.s3_utils import download_and_unzip_from_s3

def load_data(config_path="params.yaml"):
    with open(config_path, 'r') as file:
        params = yaml.safe_load(file)
    
    s3_params = params['s3']
    data_params = params['data']
    
    download_and_unzip_from_s3(
        bucket_name=s3_params['bucket_name'], 
        file_key=s3_params['raw_data_key'], 
        extract_to=data_params['raw_data_dir']
    )

if __name__ == "__main__":
    load_data()
