from dotenv import load_dotenv
import os
import shutil

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
root_dir = os.path.join(script_dir, '..')
dotenv_path = os.path.join(root_dir, '.env')

print(f"Attempting to load .env file from: {dotenv_path}")
load_dotenv(dotenv_path=dotenv_path)

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List

SOURCE_DOCUMENTS_DIR = os.path.join(script_dir, "source_documents")
PROCESSED_DOCUMENTS_DIR = os.path.join(script_dir, "processed_documents")
PERSIST_DIRECTORY = os.path.join(script_dir, "db_chroma")
CATEGORIES_FILE = os.path.join(script_dir, "categories.json")

print("Initializing local embedding model (all-MiniLM-L6-v2)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Embedding model initialized.")

LOADER_MAPPING = {
    ".pdf": PyMuPDFLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
}


def load_and_process_documents(source_dir, processed_dir) -> List:
    """
    Loads all documents from source_dir, WALKS sub-folders,
    and assigns metadata based on the folder name.
    """
    all_documents = []
    print(f"Checking for new documents in: {source_dir}")
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
        print(f"Created directory: {source_dir}")
        return []

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            ext = "." + file.split('.')[-1].lower()

            if ext in LOADER_MAPPING:
                relative_path = os.path.relpath(root, source_dir)
                product_line = "general" if relative_path == "." else relative_path.split(os.sep)[0]
                print(f"  > Loading new file: {file} (Product Line: {product_line})")

                try:
                    loader_class = LOADER_MAPPING[ext]
                    loader = loader_class(file_path)
                    loaded_docs = loader.load()

                    for doc in loaded_docs:
                        doc.metadata["product_line"] = product_line
                        doc.metadata["source"] = file
                    all_documents.extend(loaded_docs)

                    processed_subfolder = os.path.join(processed_dir, relative_path)
                    if not os.path.exists(processed_subfolder):
                        os.makedirs(processed_subfolder, exist_ok=True)

                    print(f"    > Moving to {processed_subfolder}")
                    shutil.move(file_path, os.path.join(processed_subfolder, file))

                except Exception as e:
                    print(f"    Error loading or moving {file}: {e}")
            else:
                print(f"  > Skipping (unsupported file type): {file}")

    print(f"Total new documents loaded: {len(all_documents)}")
    return all_documents


def process_and_store_documents(documents: List):
    """
    The main function to split and store documents in ChromaDB.
    It now OPENS AND CLOSES its own DB connection for writing.
    """
    if not documents:
        print("No new documents to process.")
        return

    print("Splitting new documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    print(f"Total new chunks created: {len(chunks)}")

    if not chunks:
        print("No chunks created from new documents. Exiting.")
        return

    print("Opening DB connection for writing...")
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

    print("  > Adding new documents to the database...")
    db.add_documents(chunks)
    print(f"  > Added {len(chunks)} new chunks to the database.")

    del db
    print("Ingestion complete. DB connection closed.")


if __name__ == "__main__":
    """
    This allows the script to still be run standalone
    """
    if not os.path.exists(SOURCE_DOCUMENTS_DIR):
        os.makedirs(SOURCE_DOCUMENTS_DIR)
    if not os.path.exists(PROCESSED_DOCUMENTS_DIR):
        os.makedirs(PROCESSED_DOCUMENTS_DIR)

    print("--- Running standalone ingestion ---")
    documents = load_and_process_documents(SOURCE_DOCUMENTS_DIR, PROCESSED_DOCUMENTS_DIR)

    if documents:
        process_and_store_documents(documents)
    else:
        print("No new documents found to process.")
    print("--- Standalone ingestion finished ---")
