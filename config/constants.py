import os

from multiprocessing import cpu_count
from dotenv import load_dotenv

load_dotenv()

# Database config
DB_CONFIG = {
    "host": os.getenv("db_host"),
    "user": os.getenv("db_user"),
    "password": os.getenv("db_password"),
    "database": os.getenv("db_database")
}

# AWS S3 config
AWS_CONFIG = {
    "access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
    "secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "region": os.getenv("REGION_NAME"),
    "bucket_name": "skillup-demo-content"
}

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Max Threading
MAX_THREADS = min(4, cpu_count())

