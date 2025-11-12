from dotenv import load_dotenv
import os
import shutil
import json
from pydantic import BaseModel, Field
from typing import Literal, List, Dict

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from .ingest import process_and_store_documents, load_and_process_documents, PROCESSED_DOCUMENTS_DIR, SOURCE_DOCUMENTS_DIR
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
root_dir = os.path.join(script_dir, '..')
dotenv_path = os.path.join(root_dir, '.env')

print(f"Attempting to load .env file from: {dotenv_path}")
load_dotenv(dotenv_path=dotenv_path)

PERSIST_DIRECTORY = os.path.join(script_dir, "db_chroma")
CATEGORIES_FILE = os.path.join(script_dir, "categories.json")
FINETUNED_MODEL_DIR = os.path.join(root_dir, "genilia-qwen2-expert")
TRAINING_RESULTS_DIR = os.path.join(root_dir, "results")

def get_categories_list() -> List[str]:
    try:
        with open(CATEGORIES_FILE, 'r') as f:
            categories = json.load(f)
        return categories
    except FileNotFoundError:
        default_categories = ["general"]
        with open(CATEGORIES_FILE, 'w') as f:
            json.dump(default_categories, f, indent=2)
        return default_categories


print("Initializing local embedding model (all-MiniLM-L6-v2)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Embedding model initialized.")

print(f"Database directory is set to: {PERSIST_DIRECTORY}")
if not os.path.exists(PERSIST_DIRECTORY):
    print("Database directory not found. Creating it.")
    os.makedirs(PERSIST_DIRECTORY)


def get_llm(model_name: str = "phi3", google_fallback: str = "gemini-2.0-flash"):
    """
    Tries to load the local Ollama model.
    If it fails (Ollama not running), it falls back to the Google Gemini model.
    """
    try:
        llm = ChatOllama(model=model_name)
        llm.invoke("Hello")
        print(f"--- Successfully connected to local LLM: {model_name} ---")
        return llm
    except Exception as e:
        print(f"--- WARNING: Local model '{model_name}' failed. Falling back to Google. ---")
        print(f"--- Error: {e} ---")
        if "GOOGLE_API_KEY" not in os.environ:
            print("--- ERROR: GOOGLE_API_KEY not found for fallback. ---")
            return None
        return ChatGoogleGenerativeAI(model=google_fallback)


print("Initializing LLMs...")
llm = get_llm()
print("LLM initialized.")


class MetadataFilter(BaseModel):
    product_line: str = Field(
        description="The specific product_line to filter on. If no specific product is mentioned, use 'general'.")


filter_llm = get_llm()
filter_parser = JsonOutputParser(pydantic_object=MetadataFilter)
categories_list_str = ", ".join(get_categories_list())

filter_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", f"""
You are an expert at extracting product categories.
Your job is to identify a 'product_line' to filter the search.
The available categories are: [{categories_list_str}]
If unsure, default to 'general'.

**You must format your response as a JSON object with a single key "product_line".**
"""),
        ("user", "{question}"),
    ]
)
filter_chain = filter_prompt | filter_llm | filter_parser
print("RAG metadata filter chain created (JSON mode).")


class QueryType(BaseModel):
    query_type: Literal["question", "summarization"]


classifier_llm = get_llm()
query_parser = JsonOutputParser(pydantic_object=QueryType)

router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
Classify the user's request as 'question' or 'summarization'.

**You must format your response as a JSON object with a single key "query_type".**
"""),
        ("user", "{question}"),
    ]
)
query_router = router_prompt | classifier_llm | query_parser
print("RAG query router created (JSON mode).")


def get_dynamic_retriever(metadata_filter: Dict):
    print("--- RAG: Opening DB connection for query ---")
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

    product_line = metadata_filter.get("product_line", "general")

    if product_line == "general":
        print("--- RAG: Using GENERAL retriever (no filter) ---")
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    else:
        print(f"--- RAG: Using FILTERED retriever (product_line = '{product_line}') ---")
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5, "filter": {"product_line": product_line}}
        )

    return retriever


qa_template = """
You are a helpful customer support assistant. Answer the user's question based *only* on the provided context.
---
RULES:
1.  If the context doesn't contain the answer, just say "I'm sorry, I don't have that information."
2.  Do not make up answers.
3.  Always use Markdown (bullet points, bolding) to format your answers.
---
CONTEXT:
{context}
QUESTION:
{question}
ANSWER (in Markdown):
"""
qa_prompt = ChatPromptTemplate.from_template(qa_template)
qa_logic_chain = qa_prompt | llm | StrOutputParser()
print("RAG Q&A logic created.")

summarize_template = """
You are an expert summarization assistant. Provide a concise summary of the retrieved context.
---
RULES:
1.  Do not add any information that is not in the context.
2.  Start with a one-sentence overview, followed by a bulleted list of the 3-5 most important key points.
---
CONTEXT:
{context}
USER REQUEST: "{question}"
CONCISE SUMMARY (in Markdown):
"""
summarize_prompt = ChatPromptTemplate.from_template(summarize_template)
summarize_logic_chain = summarize_prompt | llm | StrOutputParser()
print("RAG Summarization logic created.")


def route_and_invoke(input_dict: Dict):
    query_type = input_dict["query_type"].get("query_type", "question")
    retriever = input_dict["retriever"]
    question = input_dict["question"]
    context = retriever.invoke(question)

    if query_type == "summarization":
        print("--- RAG: Routing to Summarization Chain ---")
        return summarize_logic_chain.invoke({"context": context, "question": question})
    else:
        print("--- RAG: Routing to Q&A Chain ---")
        return qa_logic_chain.invoke({"context": context, "question": question})


rag_chain = (
        {"question": RunnablePassthrough()}
        | RunnablePassthrough.assign(metadata_filter=filter_chain)
        | RunnablePassthrough.assign(retriever=RunnableLambda(lambda x: get_dynamic_retriever(x["metadata_filter"])))
        | RunnablePassthrough.assign(query_type=query_router)
        | RunnableLambda(route_and_invoke)
)
print("RAG Master Metadata-Aware Chain created successfully.")

app = FastAPI(
    title="Genilia RAG Agent (Hybrid)",
    description="Microservice for RAG using local-first models with cloud fallback.",
    version="0.7.0"
)



class QueryRequest(BaseModel):
    question: str


@app.get("/")
def get_status():
    return {"status": "ok", "message": "RAG Agent is running!"}


@app.get("/categories")
def get_categories():
    return {"categories": get_categories_list()}


@app.get("/admin", response_class=FileResponse)
async def get_admin_page():
    admin_page_path = os.path.join(script_dir, "admin.html")
    if not os.path.exists(admin_page_path):
        return {"error": "admin.html file not found"}, 404
    return FileResponse(admin_page_path)


# def clear_directory_contents(directory_path):
#     if not os.path.exists(directory_path):
#         print(f"Directory {directory_path} not found, skipping.")
#         return
#     for item_name in os.listdir(directory_path):
#         item_path = os.path.join(directory_path, item_name)
#         try:
#             if os.path.isfile(item_path) or os.path.islink(item_path):
#                 os.unlink(item_path)
#             elif os.path.isdir(item_path):
#                 shutil.rmtree(item_path)
#         except Exception as e:
#             print(f"Failed to delete {item_path}. Reason: {e}")
#
#
# @app.post("/admin/reset")
# def reset_agent_knowledge():
#     print("--- ADMIN: FACTORY RESET REQUESTED ---")
#     try:
#         global filter_chain, filter_prompt, PROCESSED_DOCUMENTS_DIR, SOURCE_DOCUMENTS_DIR
#
#         if os.path.exists(PERSIST_DIRECTORY):
#             shutil.rmtree(PERSIST_DIRECTORY)
#             print(f"Successfully deleted entire database directory: {PERSIST_DIRECTORY}")
#         else:
#             print("Database directory not found, skipping.")
#         os.makedirs(PERSIST_DIRECTORY)
#         print("Re-created empty database directory.")
#
#         print("Clearing processed and source document folders...")
#         clear_directory_contents(PROCESSED_DOCUMENTS_DIR)
#         clear_directory_contents(SOURCE_DOCUMENTS_DIR)
#
#         print("Resetting categories.json...")
#         default_categories = ["general"]
#         with open(CATEGORIES_FILE, 'w') as f:
#             json.dump(default_categories, f, indent=2)
#
#         print("Re-initializing all agent chains...")
#         categories_list_str = ", ".join(default_categories)
#         filter_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", f"""
# You are an expert at extracting product categories.
# Your job is to identify a 'product_line' to filter the search.
# The available categories are: [{categories_list_str}]
# If unsure, default to 'general'.
#
# **You must format your response as a JSON object with a single key "product_line".**
# """),
#                 ("user", "{question}"),
#             ]
#         )
#         filter_chain = filter_prompt | filter_llm | filter_parser
#
#         print("--- ADMIN: SYSTEM RESET COMPLETE ---")
#         return {"status": "ok", "message": "Agent has been fully reset. All knowledge and metadata deleted."}
#
#     except Exception as e:
#         print(f"--- ERROR DURING RESET: {e} ---")
#         return {"error": str(e)}, 500
#

@app.post("/admin/reset")
def reset_agent_knowledge():
    """
    DANGER: This performs a full factory reset of the RAG agent.
    - Wipes the ENTIRE vector database directory
    - Wipes the ENTIRE processed_documents directory
    - Wipes the ENTIRE source_documents directory
    - Wipes the TRAINED MODEL and RESULTS
    - Resets categories to default
    """
    print("--- ADMIN: FACTORY RESET REQUESTED ---")
    try:
        global filter_chain, filter_prompt, llm

        def nuke_and_recreate(dir_path, name):
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                print(f"Wiped directory: {name}")
            os.makedirs(dir_path)
            print(f"Recreated empty directory: {name}")

        nuke_and_recreate(PERSIST_DIRECTORY, "Database (db_chroma)")

        nuke_and_recreate(PROCESSED_DOCUMENTS_DIR, "Processed Documents")
        nuke_and_recreate(SOURCE_DOCUMENTS_DIR, "Source Documents")

        if os.path.exists(FINETUNED_MODEL_DIR):
            shutil.rmtree(FINETUNED_MODEL_DIR)
            print(f"Wiped trained model: {FINETUNED_MODEL_DIR}")
        if os.path.exists(TRAINING_RESULTS_DIR):
            shutil.rmtree(TRAINING_RESULTS_DIR)
            print(f"Wiped training results: {TRAINING_RESULTS_DIR}")

        print("Resetting categories.json...")
        default_categories = ["general"]
        with open(CATEGORIES_FILE, 'w') as f:
            json.dump(default_categories, f, indent=2)

        print("Re-initializing all agent chains...")
        categories_list_str = ", ".join(default_categories)
        filter_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"""
You are an expert at extracting product categories.
Your job is to identify a 'product_line' to filter the search.
The available categories are: [{categories_list_str}]
If unsure, default to 'general'.

**You must format your response as a JSON object with a single key "product_line".**
"""),
                ("user", "{question}"),
            ]
        )
        filter_chain = filter_prompt | filter_llm | filter_parser

        print("--- ADMIN: SYSTEM RESET COMPLETE ---")
        return {"status": "ok", "message": "Agent has been fully reset. All knowledge, metadata, and trained models deleted."}

    except Exception as e:
        print(f"--- ERROR DURING RESET: {e} ---")
        return {"error": str(e)}, 500

@app.post("/upload")
def upload_document(
        file: UploadFile = File(...),
        category: str = Form(...)
):
    try:
        category = category.lower().strip().replace(" ", "_")
        if not category:
            return {"error": "Category cannot be empty"}, 400

        categories = get_categories_list()
        if category not in categories:
            categories.append(category)
            with open(CATEGORIES_FILE, 'w') as f:
                json.dump(categories, f, indent=2)
            print(f"Added new category: {category}")

            global filter_chain, filter_prompt
            categories_list_str = ", ".join(categories)
            filter_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", f"""
You are an expert at extracting product categories.
Your job is to identify a 'product_line' to filter the search.
The available categories are: [{categories_list_str}]
If unsure, default to 'general'.

**You must format your response as a JSON object with a single key "product_line".**
"""),
                    ("user", "{question}"),
                ]
            )
            filter_chain = filter_prompt | filter_llm | filter_parser
            print("--- RAG: Rebuilt filter chain with new categories. ---")

        category_folder = os.path.join(SOURCE_DOCUMENTS_DIR, category)
        if not os.path.exists(category_folder):
            os.makedirs(category_folder)
            print(f"Created new directory: {category_folder}")

        file_path = os.path.join(category_folder, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"File '{file.filename}' saved to {file_path}")

        print("Triggering ingestion process (load phase)...")
        new_documents = load_and_process_documents(SOURCE_DOCUMENTS_DIR, PROCESSED_DOCUMENTS_DIR)

        if new_documents:
            print("Ingestion process (store phase)...")
            process_and_store_documents(new_documents)
            print("Ingestion process finished.")
        else:
            print("Upload complete, but no new documents were loaded.")

        return {
            "status": "success",
            "filename": file.filename,
            "category": category,
            "message": "File uploaded and ingestion process triggered."
        }
    except Exception as e:
        print(f"Error during file upload or ingestion: {e}")
        return {"error": f"An error occurred: {e}"}, 500
    finally:
        file.file.close()


@app.post("/query")
def query_rag_agent(request: QueryRequest):
    print(f"\nReceived query: {request.question}")
    try:
        print("Running filter chain...")
        metadata_filter = filter_chain.invoke({"question": request.question})

        print("--- RAG: Opening DB connection for query ---")
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
        product_line = metadata_filter.get("product_line", "general")
        if product_line == "general":
            print("--- RAG: Using GENERAL retriever (no filter) ---")
            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        else:
            print(f"--- RAG: Using FILTERED retriever (product_line = '{product_line}') ---")
            retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5, "filter": {"product_line": product_line}}
            )

        print("Retrieving context...")
        context = retriever.invoke(request.question)

        del db
        del retriever
        print("--- RAG: DB connection closed ---")

        print("Running query router...")
        query_type_dict = query_router.invoke({"question": request.question})
        query_type = query_type_dict.get("query_type", "question")

        if query_type == "summarization":
            print("--- RAG: Routing to Summarization Chain ---")
            answer = summarize_logic_chain.invoke({"context": context, "question": request.question})
        else:
            print("--- RAG: Routing to Q&A Chain ---")
            answer = qa_logic_chain.invoke({"context": context, "question": request.question})

        print(f"Generated answer: {answer}")
        return {"question": request.question, "answer": answer}

    except Exception as e:
        print(f"Error during RAG chain invocation: {e}")
        print(f"--- RAG Chain ERROR: {e} ---")
        return {"error": f"An error occurred in the RAG chain: {e}"}, 500