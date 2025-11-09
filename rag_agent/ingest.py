
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
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.api_core import exceptions as google_exceptions

SOURCE_DOCUMENTS_DIR = os.path.join(script_dir, "source_documents")
PROCESSED_DOCUMENTS_DIR = os.path.join(script_dir, "processed_documents")
PERSIST_DIRECTORY = os.path.join(script_dir, "db_chroma")

if "GOOGLE_API_KEY" not in os.environ:
    raise EnvironmentError("GOOGLE_API_KEY not set in environment variables.")

print("Initializing embedding model...")
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print("Embedding model initialized.")
except google_exceptions.PermissionDenied:
    print("\n--- PERMISSION DENIED ---")
    print("Error: Failed to authenticate with Google AI. ")
    print("Please ensure your GOOGLE_API_KEY is correct and has the right permissions.")
    exit(1)
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
    exit(1)


LOADER_MAPPING = {
    ".pdf": PyMuPDFLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
}


def load_and_process_documents(source_dir, processed_dir):
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

                if relative_path == ".":
                    product_line = "general"
                else:
                    product_line = relative_path.split(os.sep)[0]

                print(f"  > Loading new file: {file} (Category : {product_line})")

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

def process_and_store_documents():
    """
    The main function to load, split, and store documents in ChromaDB.
    This is now "idempotent" - it only adds new files.
    """
    documents = load_and_process_documents(SOURCE_DOCUMENTS_DIR, PROCESSED_DOCUMENTS_DIR)

    if not documents:
        print("No new documents to process. Database is up-to-date.")
        return

    print("Splitting new documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    print(f"Total new chunks created: {len(chunks)}")

    if not chunks:
        print("No chunks created from new documents. Exiting.")
        return


    if os.path.exists(PERSIST_DIRECTORY):
        print(f"  > Existing database found at {PERSIST_DIRECTORY}. Loading...")
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
        print("  > Database loaded. Adding new documents...")

        db.add_documents(chunks)
        print(f"  > Added {len(chunks)} new chunks to the database.")

    else:
        print(f"  > No existing database found. Creating a new one at {PERSIST_DIRECTORY}...")
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        print(f"  > New database created with {len(chunks)} chunks.")

    db.persist()
    print("Ingestion complete. Vector database is ready.")


if __name__ == "__main__":
    if not os.path.exists(SOURCE_DOCUMENTS_DIR):
        os.makedirs(SOURCE_DOCUMENTS_DIR)
        print(f"Created directory: {SOURCE_DOCUMENTS_DIR}")
        print(f"Please add your documents (PDFs, Excel files) to this folder.")

    if not os.path.exists(PROCESSED_DOCUMENTS_DIR):
        os.makedirs(PROCESSED_DOCUMENTS_DIR)
        print(f"Created directory: {PROCESSED_DOCUMENTS_DIR}")

    process_and_store_documents()