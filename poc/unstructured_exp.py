import os
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Text

from utils.logger_utils import logger
from decorator.time_decorator import timeit

DATA_FOLDER = Path("data")

@timeit
def run_unstructured_pdf_extraction(pdf_path: Path):
    logger.info(f"Processing file: {pdf_path.name}")

    try:
        # breakpoint()
        
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy="hi_res",
            infer_table_structure=False
        )

        pagewise_text = {}

        for element in elements:
            if isinstance(element, Text):
                page = element.metadata.page_number if hasattr(element.metadata, 'page_number') else 'Unknown'
                
                if page == 'Unknown':
                    logger.warning(f"Page number is missing for the text element: {element.text[:100]}...")

                text = element.text.strip().replace("\n", " ")
                if page not in pagewise_text:
                    pagewise_text[page] = []
                pagewise_text[page].append(text)

            logger.debug(f"Element Type: {type(element)}")
            logger.debug(f"Element Metadata: {element.metadata}")
            logger.debug(f"Element Text: {element.text[:150]}")
            
            if hasattr(element, 'other_attribute'):
                logger.debug(f"Other Attribute: {element.other_attribute}")
        
        breakpoint()

        for page_num, texts in pagewise_text.items():
            logger.info(f"Page {page_num} contains {len(texts)} text blocks")
            for snippet in texts[:2]:
                logger.debug(f"[Page {page_num}] {snippet[:150]}...")
        
        breakpoint()

        logger.info(f"Completed file: {pdf_path.name} | Total pages: {len(pagewise_text)}")

    except Exception as e:
        logger.error(f"Error while processing {pdf_path.name}: {e}")

@timeit
def process_all_pdfs_in_folder(folder_path: Path):
    if not folder_path.exists():
        logger.error(f"Input folder does not exist: {folder_path}")
        return

    pdf_files = list(folder_path.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in the folder.")
        return

    logger.info(f"Found {len(pdf_files)} PDF file(s) in {folder_path}")

    for pdf_file in pdf_files:
        run_unstructured_pdf_extraction(pdf_file)

if __name__ == "__main__":
    process_all_pdfs_in_folder(DATA_FOLDER)
