import os
import json
from dotenv import load_dotenv
from typing import List, Optional
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("Loading environment variables...")
load_dotenv()

script_dir = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DOCS_DIR = os.path.join(script_dir, "rag_agent", "processed_documents")
OUTPUT_FILE = os.path.join(script_dir, "finetune_dataset.jsonl")

LOADER_MAPPING = {
    ".pdf": PyMuPDFLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
}


class QAPair(BaseModel):
    question: str = Field(description="A specific question that can be answered *only* from the provided text.")
    answer: str = Field(description="The answer to the question, extracted *only* from the provided text.")


class QAPairList(BaseModel):
    qa_pairs: List[QAPair] = Field(description="A list of 3-5 question/answer pairs.")


print("Initializing local LLM (phi3) for dataset generation...")
llm = ChatOllama(model="phi3", temperature=0.1)
parser = JsonOutputParser(pydantic_object=QAPairList)

generation_prompt = ChatPromptTemplate.from_template(
    """
You are an expert at creating high-quality, domain-specific training data.
Your task is to read the following chunk of a company's internal document and generate 3-5 specific question-and-answer pairs based *only* on the provided text.

-   The questions should be realistic queries a customer or engineer might ask.
-   The answers must be concise and taken directly from the text.
-   Do not make up any information.
-   Focus on technical specifications, product names, instructions, and policies.

CONTEXT_CHUNK:
---
{context}
---

**You MUST format your response as a JSON object that strictly follows this Pydantic schema:**
{format_instructions}
""",
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

generation_chain = generation_prompt | llm | parser
print("Generation chain created (JSON mode).")



def load_all_processed_documents():
    """Loads all documents from the processed_documents directory."""
    documents = []
    print(f"Scanning for processed documents in: {PROCESSED_DOCS_DIR}")
    if not os.path.exists(PROCESSED_DOCS_DIR):
        print(f"Error: Processed documents directory not found at {PROCESSED_DOCS_DIR}")
        print("Please run the RAG agent and upload some files via the admin panel first.")
        return []

    for root, dirs, files in os.walk(PROCESSED_DOCS_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            ext = "." + file.split('.')[-1].lower()

            if ext in LOADER_MAPPING:
                print(f"  > Loading: {file}")
                try:
                    loader_class = LOADER_MAPPING[ext]
                    loader = loader_class(file_path)
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"    Error loading {file}: {e}")
            else:
                print(f"  > Skipping (unsupported file type): {file}")

    print(f"Total documents loaded: {len(documents)}")
    return documents


def main():
    """Main function to generate the dataset."""

    documents = load_all_processed_documents()
    if not documents:
        print("No documents found. Exiting.")
        return

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    total_pairs = 0

    with open(OUTPUT_FILE, 'w') as f:
        for i, chunk in enumerate(chunks):
            print(f"  > Processing chunk {i + 1}/{len(chunks)}...")
            try:
                response = generation_chain.invoke({"context": chunk.page_content})

                if response.get("qa_pairs"):
                    for pair in response.get("qa_pairs", []):
                        qa_json = {"question": pair.get("question"), "answer": pair.get("answer")}
                        if qa_json["question"] and qa_json["answer"]:
                            f.write(json.dumps(qa_json) + "\n")
                            total_pairs += 1

                    print(f"    > Generated {len(response.get('qa_pairs', []))} pairs. Total so far: {total_pairs}")

            except Exception as e:
                print(f"    > Error processing chunk (will skip): {e}")
                print("    > This is often due to the LLM returning invalid JSON.")

    print("\n--- Dataset Generation Complete! ---")
    print(f"Total Q&A pairs generated: {total_pairs}")
    print(f"Dataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()