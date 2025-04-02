from fastapi import UploadFile
from io import BytesIO
import boto3
import os
from utils.logger_utils import logger
from decorator.time_decorator import timeit

BUCKET_NAME = "skillup-demo-content"

# Create and return S3 client
@timeit
def get_s3_client():
    logger.info("Creating S3 client.")
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('REGION_NAME')
    )

# Upload to S3
@timeit
def upload_to_s3(file: UploadFile, folder: str, s3_client=None):
    if not s3_client:
        logger.info("No S3 client provided. Creating a new one.")
        s3_client = get_s3_client()

    file_path = f"{folder}/{file.filename}"
    logger.info(f"Uploading {file.filename} to S3 at {file_path}.")
    
    try:
        file_stream = BytesIO()
        file_stream.write(file.file.read())
        file_stream.seek(0)
        
        s3_client.upload_fileobj(file_stream, BUCKET_NAME, file_path)
        logger.info(f"Successfully uploaded {file.filename} to {file_path}.")
        
    except Exception as e:
        logger.exception(f"Failed to upload {file.filename} to S3: {e}")
