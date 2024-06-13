# src/utils/data_utils.py

import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-3.5-turbo").encode(text)
    return len(tokens)

def load_and_process_data(directory_path: str):
    all_docs = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            print(f"Processing file: {filename}")  # Debugging statement
            loader = PyMuPDFLoader(os.path.join(directory_path, filename))
            docs = loader.load()
            print(f"Loaded documents: {docs}")  # Debugging statement
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=0,
                length_function=tiktoken_len,
            )
            split_chunks = text_splitter.split_documents(docs)
            print(f"Split documents: {split_chunks}")  # Debugging statement
            all_docs.extend(split_chunks)
    print(f"Total documents processed: {len(all_docs)}")  # Debugging statement
    return all_docs
