import boto3
import zipfile
import os
import io

def download_and_unzip_from_s3(bucket_name, file_key, extract_to='data/raw'):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    with zipfile.ZipFile(io.BytesIO(obj['Body'].read())) as z:
        z.extractall(extract_to)
    print(f"Data extracted to {extract_to}")

def upload_to_s3(bucket_name, file_key, file_path):
    s3 = boto3.client('s3')
    s3.upload_file(file_path, bucket_name, file_key)
    print(f"File uploaded to s3://{bucket_name}/{file_key}")
