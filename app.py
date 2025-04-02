from typing import List

from decorator.time_decorator import timeit
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
# from langchain_core.pydantic_v1 import BaseModel
# from pydantic import BaseModel

from src.rag_pipeline import set_input
from utils.s3_utils import upload_to_s3, get_s3_client
from utils.logger_utils import logger

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

s3_client = get_s3_client()

@app.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    folder: str = Form(...)
):
    if not folder:
        logger.error("Folder name is required but not provided.")
        raise HTTPException(status_code=422, detail="Folder name is required.")
    
    try:
        for file in files:
            logger.info(f"Uploading {file.filename} to folder: {folder}")
            upload_to_s3(file, folder, s3_client)
            
        logger.info("All files uploaded successfully.")
        return JSONResponse(content={"message": "All files uploaded successfully"}, status_code=200)
    
    except Exception as e:
        logger.exception("Error uploading files")
        return JSONResponse(content={"message": f"Failed to upload files: {e}"}, status_code=500)

@app.get("/response")
@timeit
def get_response(
    query_param: str = Query(...),
    file_path: str = Query(...)
):
    try:
        logger.info(f"Received request - query_param: {query_param}, file_path: {file_path}")
        response = set_input(query_param, file_path)
        return {"response": response}
    
    except Exception as e:
        logger.exception("Failed to process request")
        return JSONResponse(content={"message": f"Failed to process request: {e}"}, status_code=500)
