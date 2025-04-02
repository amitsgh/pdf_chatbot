import io
import os
import re
import boto3
import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageOps
from dotenv import load_dotenv
from langchain.schema import Document
from pdf2image import convert_from_path
from typing import List
from concurrent.futures import ThreadPoolExecutor

from utils.logger_utils import logger
from decorator.time_decorator import timeit
from config.constants import AWS_CONFIG, MAX_THREADS
from utils.utils import preprocess_content

# Load environment
load_dotenv()

# S3 client using config
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_CONFIG['access_key_id'],
    aws_secret_access_key=AWS_CONFIG['secret_access_key'],
    region_name=AWS_CONFIG['region']
)

@timeit
def fetch_and_process_pdf(directory_key: str, bucket_name: str = AWS_CONFIG["bucket_name"]) -> List[Document]:
    logger.info(f"Fetching PDF files from S3: bucket={bucket_name}, folder={directory_key}")
    
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=directory_key)
    if "Contents" not in response:
        logger.warning(f"No files found at {directory_key}")
        return []

    pdf_files = [obj["Key"] for obj in response["Contents"] if obj["Key"].endswith(".pdf")]
    if not pdf_files:
        logger.warning(f"No PDFs found in {directory_key}")
        return []

    documents = []
    temp_dir = "/tmp"
    os.makedirs(temp_dir, exist_ok=True)

    for file_key in pdf_files:
        logger.info(f"Processing file: {file_key}")
        # temp_path = os.path.join(temp_dir, os.path.basename(file_key))
        
        try:
            # download_file_from_s3(bucket_name, file_key, temp_path)
            response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
            file_obj = io.BytesIO(response["Body"].read())
            
            # breakpoint()
            
            pages = extract_text_from_pdf(file_obj, os.path.basename(file_key))
            
            if not pages:
                logger.warning(f"No meaningful pages extracted from {file_key}")
                continue
            
            for page in pages:
                documents.append(Document(
                    page_content=page["text"],
                    metadata={
                        "source_path": file_key,
                        # "source_dir": directory_key,
                        # "file_name": page["file_name"],
                        # "page_number": page["page_number"],
                        "page_index": page["page_index"]
                    }
                ))
        except Exception as e:
            logger.error(f"Failed processing {file_key}: {e}")
        # finally:
        #     if os.path.exists(temp_path):
        #         os.remove(temp_path)
        #         logger.debug(f"Temporary file deleted: {temp_path}")

    return documents

@timeit
def download_file_from_s3(bucket: str, key: str, local_path: str):
    try:
        s3_client.download_file(bucket, key, local_path)
        logger.info(f"Downloaded: {key} to {local_path}")
    except Exception as e:
        logger.error(f"Error downloading {key}: {e}")
        raise

@timeit
def extract_text_from_pdf(file_obj: io.BytesIO, file_name: str) -> List[dict]:
    logger.info(f"Extracting text from PDF: {file_name}")
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    results = []

    try:
        with fitz.open(stream=file_obj, filetype="pdf") as doc:
            # skip = 0
            with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
                pages = list(executor.map(lambda idx: process_page_text(doc, idx, file_name), range(len(doc))))
                # text, page_num = process_page(doc, idx, file_name)
                # text = doc.load_page(idx).get_text()
                
                # if not page_num:
                #     logger.warning(f"No page number for page {idx + 1}, skipping.")
                #     skip += 1
                # elif page_num == idx + 1:
                #     logger.debug(f"Page match confirmed: Page {page_num}")
                # elif page_num > idx + 1:
                #     logger.warning(f"Skipping future-numbered page: {page_num}")
                #     page_num = None
                #     skip += 1
                # elif idx - skip + 1 != page_num and page_num == 1:
                #     page_num = idx - skip + 1
                #     logger.info(f"Adjusted mismatch to: Page {page_num}")
                    
                # breakpoint()
                    
                # clean_text = preprocess_content(text)
                
                # if clean_text.strip():
                #     results.append({
                #         "text": clean_text,
                #         "page_index": idx + 1,
                #         # "page_number": str(page_num),
                #         # "file_name": file_name,
                #         "file_path": pdf_path
                #     })
                # else:
                #     logger.warning(f"Skipping empty or non-meaningful page: {idx + 1}")
            
            # breakpoint()
            
            results = [p for p in pages if p is not None]
                
        # breakpoint()

    except Exception as e:
        logger.error(f"Extraction failed: {e}")

    return results

# def extract_text_from_pdf(pdf_path: str, file_name: str) -> List[dict]:
#     logger.info(f"Extracting text from PDF: {pdf_path}")
#     # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#     results = []

#     try:
#         with fitz.open(pdf_path, filetype="pdf") as doc:
#             # skip = 0
#             with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
#                 pages = list(executor.map(lambda idx: process_page_text(doc, idx, file_name), range(len(doc))))
#                 # text, page_num = process_page(doc, idx, file_name)
#                 # text = doc.load_page(idx).get_text()
                
#                 # if not page_num:
#                 #     logger.warning(f"No page number for page {idx + 1}, skipping.")
#                 #     skip += 1
#                 # elif page_num == idx + 1:
#                 #     logger.debug(f"Page match confirmed: Page {page_num}")
#                 # elif page_num > idx + 1:
#                 #     logger.warning(f"Skipping future-numbered page: {page_num}")
#                 #     page_num = None
#                 #     skip += 1
#                 # elif idx - skip + 1 != page_num and page_num == 1:
#                 #     page_num = idx - skip + 1
#                 #     logger.info(f"Adjusted mismatch to: Page {page_num}")
                    
#                 # breakpoint()
                    
#                 # clean_text = preprocess_content(text)
                
#                 # if clean_text.strip():
#                 #     results.append({
#                 #         "text": clean_text,
#                 #         "page_index": idx + 1,
#                 #         # "page_number": str(page_num),
#                 #         # "file_name": file_name,
#                 #         "file_path": pdf_path
#                 #     })
#                 # else:
#                 #     logger.warning(f"Skipping empty or non-meaningful page: {idx + 1}")
            
#             # breakpoint()
            
#             results = [p for p in pages if p is not None]
                
#         # breakpoint()

#     except Exception as e:
#         logger.error(f"Extraction failed: {e}")

#     return results

@timeit
def process_page_text(doc, idx, file_name):
    try:
        text = doc.load_page(idx).get_text()
        clean_text = preprocess_content(text)
        if clean_text.strip():
            return {
                "text": clean_text,
                "page_index": idx + 1,
                "file_name": file_name,
                "file_path": doc.name
            }
    except Exception as e:
        logger.warning(f"Failed to process page {idx + 1}: {e}")
        
    return None


# @timeit
# def extract_page_number_from_text(text):
#     try:
#         patterns = [
#             r"^\s*(\d+)\s*(?=\b$)", r"Page[:\s]*(\d+)\b", r"(\d+)\s*[•]?\s*[\w\s]*$",
#             r"(\d+)\s*[A-Za-z\s]*\b$", r"(\d+)[\s.•]*\s*$"
#         ]
#         for pattern in patterns:
#             match = re.search(pattern, text, re.IGNORECASE)
#             if match:
#                 return int(match.group(1))
#     except Exception as e:
#         logger.error(f"Regex failed for text: {text} | Error: {e}")
        
#     return None

# @timeit
# def extract_page_number_with_tesseract(image: Image.Image, fallback: int) -> int:
#     try:
#         gray = ImageOps.grayscale(image)
#         contrast = ImageOps.autocontrast(gray)
#         width, height = contrast.size
#         footer_area = contrast.crop((0, height - int(height * 0.), width, height))

#         ocr_result = pytesseract.image_to_string(footer_area, config='--psm 6')
#         page_number = extract_page_number_from_text(ocr_result)

#         if page_number:
#             return page_number

#         logger.warning(f"OCR page number not found, fallback used: {fallback}")
#     except Exception as e:
#         logger.error(f"Tesseract error: {e}")
        
#     return fallback
