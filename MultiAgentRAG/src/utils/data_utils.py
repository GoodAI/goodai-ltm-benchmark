# src/utils/data_utils.py

import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import tiktoken
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger('master')

def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-3.5-turbo").encode(text)
    return len(tokens)

def load_and_process_data(directory_path: str):
    all_docs = []
    if not os.path.exists(directory_path):
        logger.error(f"Directory {directory_path} does not exist.")
        return all_docs

    if not os.listdir(directory_path):
        logger.warning(f"Directory {directory_path} is empty. Using default document.")
        default_content = "This is a default document. Add PDF files to the data/raw directory for processing."
        default_doc = Document(page_content=default_content)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=0,
            length_function=tiktoken_len,
        )
        split_chunks = text_splitter.split_documents([default_doc])
        all_docs.extend(split_chunks)
        return all_docs

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        logger.debug(f"Found file: {file_path}")
        if filename.endswith(".pdf"):
            logger.debug(f"Processing file: {filename}")
            try:
                loader = PyMuPDFLoader(file_path)
                docs = loader.load()
                logger.debug(f"Loaded documents from {filename}")

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=300,
                    chunk_overlap=0,
                    length_function=tiktoken_len,
                )
                split_chunks = text_splitter.split_documents(docs)
                logger.debug(f"Split documents from {filename}")

                all_docs.extend(split_chunks)
            except Exception as e:
                logger.error(f"Error processing file {filename}: {e}", exc_info=True)
        else:
            logger.debug(f"Skipping non-PDF file: {filename}")

    logger.info(f"Total documents processed: {len(all_docs)}")
    return all_docs

def structure_memories(memories: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    """
    Transforms a list of memory tuples into a structured list of dictionaries.

    Args:
        memories (List[Tuple[str, str]]): List of memories where each memory is a tuple (query, result).

    Returns:
        List[Dict[str, str]]: List of structured memories with titles, queries, results, and tags.
    """
    structured_memories = []
    for idx, (query, result) in enumerate(memories):
        memory_data = {
            "title": f"Memory_{idx + 1}",
            "query": query,
            "result": result,
            "tags": "retrieved, processed"
        }
        structured_memories.append(memory_data)
    return structured_memories
