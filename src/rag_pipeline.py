from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import os
import re
from typing import List

import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from config.constants import DB_CONFIG, OPENAI_API_KEY, MAX_THREADS
from src import s3_processor
from utils.logger_utils import logger
from decorator.time_decorator import timeit

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

answer_cache = {}

import io
import os
import re
import boto3
from PIL import Image, ImageOps
from dotenv import load_dotenv
from langchain.schema import Document
from pdf2image import convert_from_path
from typing import List
from concurrent.futures import ThreadPoolExecutor

from utils.logger_utils import logger
from decorator.time_decorator import timeit
from config.constants import AWS_CONFIG, MAX_THREADS

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
def split_documents(bucket_name: str, folder_path: str):
    logger.info(f"Splitting documents in: {folder_path}")
    collection_name = folder_path.replace("/", "_")
    persist_dir = "./vector_store"

    chroma_client = chromadb.PersistentClient(path=persist_dir)
    collection_names = chroma_client.list_collections()
    collection_names = [name if isinstance(name, str) else name.name for name in collection_names]  # backward safe

    # Case 1: Collection already exists
    if collection_name in collection_names:
        logger.info(f"Found existing collection: {collection_name}")
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=persist_dir,
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
            client=chroma_client
        )

        logger.info(f"Checking for changes in S3 bucket: {bucket_name}/{folder_path}")

        current_s3_files = set()
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_path)
        if "Contents" in response:
            current_s3_files = {
                obj["Key"] for obj in response["Contents"] if obj["Key"].endswith(".pdf")
            }

        collection_files = set()
        results = vectorstore.get(include=["metadatas"])
        if results.get("metadatas"):
            for metadata in results["metadatas"]:
                if metadata and "source_path" in metadata:
                    collection_files.add(metadata["source_path"])

        new_files = current_s3_files - collection_files
        deleted_files = collection_files - current_s3_files

        if new_files:
            logger.info(f"Found {len(new_files)} new files to add to collection")
            for file_key in new_files:
                try:
                    logger.info(f"Processing new file: {file_key}")
                    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
                    file_obj = io.BytesIO(response["Body"].read())

                    pages = s3_processor.extract_text_from_pdf(file_obj, os.path.basename(file_key))
                    if not pages:
                        logger.warning(f"No meaningful pages extracted from {file_key}")
                        continue

                    documents = [
                        Document(
                            page_content=page["text"],
                            metadata={
                                "source_path": file_key,
                                "page_index": page["page_index"]
                            }
                        ) for page in pages
                    ]

                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
                    split_docs = []
                    for doc in documents:
                        chunks = text_splitter.split_text(doc.page_content)
                        for chunk in chunks:
                            split_docs.append(Document(page_content=chunk, metadata=doc.metadata))

                    if split_docs:
                        vectorstore.add_documents(split_docs)
                        logger.info(f"Added {len(split_docs)} chunks from {file_key}")
                except Exception as e:
                    logger.error(f"Failed processing new file {file_key}: {e}")

        if deleted_files:
            logger.info(f"Found {len(deleted_files)} files to remove from collection")
            for file_key in deleted_files:
                try:
                    vectorstore.delete(where={"source_path": file_key})
                    logger.info(f"Removed documents for deleted file: {file_key}")
                except Exception as e:
                    logger.error(f"Failed removing documents for {file_key}: {e}")

        if new_files or deleted_files:
            logger.info(f"Collection updated: {len(new_files)} files added, {len(deleted_files)} files removed")
        else:
            logger.info("No changes detected in S3 bucket")

    # Case 2: Create new collection
    else:
        logger.info(f"Collection not found, creating new: {collection_name}")
        documents = s3_processor.fetch_and_process_pdf(folder_path, bucket_name)
        if not documents:
            logger.warning("No documents found for embedding.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)

        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            split_docs_nested = executor.map(
                lambda doc: [
                    Document(page_content=chunk, metadata=doc.metadata)
                    for chunk in text_splitter.split_text(doc.page_content)
                ],
                documents
            )
        split_docs = [doc for sublist in split_docs_nested for doc in sublist]

        if not split_docs:
            logger.warning("No meaningful content to embed.")
            return None

        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
            persist_directory=persist_dir,
            collection_name=collection_name,
        )

    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

@timeit
def create_prompt():
    logger.info("Creating prompt template.")
    system_prompt = (
        "You are an assistant designed to answer questions using information retrieved exclusively from the provided PDF context. "
        "If the answer cannot be found in the PDF, respond with 'I don't know.' "
        "Provide a clear and concise answer based only on the PDF content. "
        # "When answering, always extract the page number from the footer details of the PDF. "
        # "This could include extracting the page number embedded within text like 'Version 1 ME, IIT Kharagpur 14' or directly from the footer if it is present as a standalone number. "
        "Always specify the extracted page number in the format it appears in the footer.\n\n"
        "If the PDF contains any points or steps in a procedure, display them exactly as they appear in the PDF.\n\n"
        "Format your response as follows:\n"
        "- Begin with the answer to the question.\n"
        # "- Include a note at the end specifying the extracted page number as it appears in the footer, e.g., '(Source: Page 14)'.\n\n"
        "If no information is available in the PDF, respond with: 'I don't know.'\n\n"
        "{context}"
    )

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

@timeit
def create_chain(llm, prompt, retriever):
    logger.info("Creating RAG chain.")
    qa_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, qa_chain)
    logger.info("RAG chain successfully created.")
    return chain

@timeit
def generate_related_questions(question: str, llm) -> str:
    logger.info(f"Generating related questions for: {question}")
    prompt = f"Based on the question '{question}', suggest three related questions a user might ask."
    return llm.invoke(prompt)

@timeit
def extract_tags_from_question(question: str) -> List[str]:
    logger.info(f"Extracting tags from question: {question}")
    words = re.findall(r'\b\w+\b', question.lower())
    common_words = {'what', 'is', 'how', 'why', 'a', 'the', 'and', 'in', 'of', 'to', 'on', 'for', 'are', 'with', 'by',
                    'this', 'that', 'which', 'from', 'as', 'at', 'it', 'be', 'not', 'an'}
    tags = [word for word in words if word not in common_words]

    db_config = DB_CONFIG

    if len(tags) <= 5:
        logger.info("Not enough tags to query database.")
        return []

    try:
        import mysql.connector
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        query = f"""
            SELECT DISTINCT tags
            FROM learning_bytes
            WHERE {' OR '.join(['tags LIKE %s'] * len(tags))};
        """
        params = [f"%{tag}%" for tag in tags]
        cursor.execute(query, params)
        rows = cursor.fetchall()

        all_tags = []
        for row in rows:
            if row[0]:
                all_tags.extend(row[0].split(','))

        db_tags = set(map(str.strip, all_tags))
        common_tags = list(db_tags.intersection(tags))
        logger.info(f"Common tags found: {common_tags}")
        return common_tags[:5]
    except mysql.connector.Error as err:
        logger.error(f"Database error: {err}")
        return []
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

@timeit
def get_answer(rag_chain, query_param: str) -> dict:
    logger.info(f"Getting answer for query: {query_param}")
    
    # if query_param in answer_cache:
    #     logger.info("Using cached answer.")
    #     return answer_cache[query_param]
    
    results = rag_chain.invoke({"input": query_param})
    
    if "i don't know" in results['answer'].lower() or not results.get("context"):
        return {"answer": "I don't know.", "context": []}
    # else:
    #     answer_cache[query_param] = results
    return results

@timeit
def set_input(query_param: str, file_path: str) -> dict:
    logger.info(f"Starting RAG process for: {query_param}")    
    bucket_name = "skillup-demo-content"
    retriever = split_documents(bucket_name, file_path)    
    
    if not retriever:
        logger.error("No retriever returned — check if documents were loaded or split.")
        return {
            "Answer": {"answer": "I don't know.", "context": []},
            "Video_url": [],
            "Related_Questions": []
        }

    prompt = create_prompt()
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    rag_chain = create_chain(llm, prompt, retriever)

    results = get_answer(rag_chain, query_param)

    if results["answer"].lower() == "i don't know.":
        return {
            "Answer": results,
            "Video_url": [],
            "Related_Questions": []
        }
    
    related_questions = generate_related_questions(query_param, llm)
    video_links = extract_tags_from_question(query_param)

    final_response = {
        "Answer": results,
        "Video_url": video_links,
        "Related_Questions": related_questions
    }

    logger.info("RAG process completed successfully.")
    return final_response
