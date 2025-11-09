# # from dotenv import load_dotenv
# # import os
# # import shutil  # --- NEW ADDITION ---
# #
# # script_path = os.path.abspath(__file__)
# # script_dir = os.path.dirname(script_path)
# # root_dir = os.path.join(script_dir, '..')
# # dotenv_path = os.path.join(root_dir, '.env')
# #
# # print(f"Attempting to load .env file from: {dotenv_path}")
# # load_dotenv(dotenv_path=dotenv_path)
# # # --- End of .env loading ---
# # from fastapi import FastAPI, UploadFile, File  # --- NEW ADDITION ---
# # from pydantic import BaseModel  # Used for defining request data types
# #
# # # --- LangChain Imports ---
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# # from langchain_community.vectorstores import Chroma
# # from langchain_core.prompts import PromptTemplate
# # from langchain_core.runnables import RunnablePassthrough
# # from langchain_core.output_parsers import StrOutputParser
# #
# # # --- NEW ADDITION: Import our ingestion function ---
# # # from rag_agent.ingest import process_and_store_documents, SOURCE_DOCUMENTS_DIR
# # from ingest import process_and_store_documents , SOURCE_DOCUMENTS_DIR
# #
# # # --- Check for API Key ---
# # if "GOOGLE_API_KEY" not in os.environ:
# #     raise EnvironmentError("GOOGLE_API_KEY not set in environment variables. Please check your .env file.")
# #
# # # --- Global Variables & Initialization ---
# #
# # # 1. Define the paths
# # PERSIST_DIRECTORY = "db_chroma"
# #
# # # 2. Initialize the Embedding Model (must be the same as in ingest.py)
# # print("Initializing embedding model...")
# # embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
# #
# # # 3. Load the persistent ChromaDB
# # print(f"Loading persistent vector database from: {PERSIST_DIRECTORY}")
# # if not os.path.exists(PERSIST_DIRECTORY):
# #     print(f"Warning: Database directory not found: {PERSIST_DIRECTORY}. Running ingest.py to create it...")
# #     # This is a fallback in case the DB doesn't exist on server start
# #     process_and_store_documents()
# #     if not os.path.exists(PERSIST_DIRECTORY):
# #         raise FileNotFoundError(f"Database directory not found: {PERSIST_DIRECTORY}. Ingestion failed.")
# #
# # db = Chroma(
# #     persist_directory=PERSIST_DIRECTORY,
# #     embedding_function=embeddings
# # )
# #
# # # 4. Create a "Retriever"
# # retriever = db.as_retriever(
# #     search_type="similarity",
# #     search_kwargs={"k": 3}  # Ask for the 3 most relevant chunks
# # )
# # print("Vector database loaded and retriever created.")
# #
# # # 5. Initialize the LLM (Gemini)
# # llm = ChatGoogleGenerativeAI(
# #     model="gemini-2.5-pro",  # Using gemini-pro for stable API
# #     temperature=0.3
# # )
# #
# # # 6. Define our RAG Prompt Template
# # template = """
# # You are a helpful customer support assistant for the company 'Genilia'.
# # Your job is to answer the user's question based *only* on the provided context.
# # If the context doesn't contain the answer, just say "I'm sorry, I don't have that information."
# # Do not make up answers. Be concise and polite.
# #
# # CONTEXT:
# # {context}
# #
# # QUESTION:
# # {question}
# #
# # ANSWER:
# # """
# # prompt = PromptTemplate(template=template, input_variables=["context", "question"])
# #
# # # 7. Create the RAG Chain using LangChain Expression Language (LCEL)
# # rag_chain = (
# #         {"context": retriever, "question": RunnablePassthrough()}
# #         | prompt
# #         | llm
# #         | StrOutputParser()
# # )
# # print("RAG chain created successfully.")
# #
# # # --- FastAPI Application ---
# #
# # app = FastAPI(
# #     title="Genilia RAG Agent",
# #     description="Microservice for handling RAG knowledge and document ingestion.",
# #     version="0.1.0"
# # )
# #
# #
# # # Define the input model for our /query endpoint
# # class QueryRequest(BaseModel):
# #     question: str
# #
# #
# # @app.get("/")
# # def get_status():
# #     """
# #     A simple 'heartbeat' endpoint to check if the
# #     server is running.
# #     """
# #     return {"status": "ok", "message": "RAG Agent is running!"}
# #
# #
# # # --- NEW ADDITION: File Upload Endpoint ---
# # @app.post("/upload")
# # def upload_document(file: UploadFile = File(...)):
# #     """
# #     Allows an admin to upload a document.
# #     The file is saved and then the ingestion process is triggered.
# #     """
# #     # Ensure the source_documents directory exists
# #     if not os.path.exists(SOURCE_DOCUMENTS_DIR):
# #         os.makedirs(SOURCE_DOCUMENTS_DIR)
# #         print(f"Created directory: {SOURCE_DOCUMENTS_DIR}")
# #
# #     file_path = os.path.join(SOURCE_DOCUMENTS_DIR, file.filename)
# #
# #     try:
# #         # Save the uploaded file to the source_documents folder
# #         with open(file_path, "wb") as buffer:
# #             shutil.copyfileobj(file.file, buffer)
# #
# #         print(f"File '{file.filename}' saved to {file_path}")
# #
# #         # Now, trigger the ingestion process
# #         print("Triggering ingestion process...")
# #         process_and_store_documents()
# #         print("Ingestion process finished.")
# #
# #         return {
# #             "status": "success",
# #             "filename": file.filename,
# #             "message": "File uploaded and ingestion process triggered."
# #         }
# #     except Exception as e:
# #         print(f"Error during file upload or ingestion: {e}")
# #         return {"error": f"An error occurred: {e}"}, 500
# #     finally:
# #         # Always close the file
# #         file.file.close()
# #
# #
# # @app.post("/query")
# # def query_rag_agent(request: QueryRequest):
# #     """
# #     The main RAG query endpoint.
# #     Receives a question, processes it through the RAG chain,
# #     and returns the answer.
# #     """
# #     print(f"\nReceived query: {request.question}")
# #
# #     try:
# #         # 1. Invoke the chain
# #         answer = rag_chain.invoke(request.question)
# #
# #         print(f"Generated answer: {answer}")
# #
# #         # 2. Return the answer
# #         return {"question": request.question, "answer": answer}
# #
# #     except Exception as e:
# #         print(f"Error during RAG chain invocation: {e}")
# #         # Return a server error
# #         return {"error": f"An error occurred: {e}"}, 500
# from dotenv import load_dotenv
# import os
# import shutil
#
# # --- Absolute Path Setup ---
# script_path = os.path.abspath(__file__)
# script_dir = os.path.dirname(script_path)
# root_dir = os.path.join(script_dir, '..')
# dotenv_path = os.path.join(root_dir, '.env')
#
# print(f"Attempting to load .env file from: {dotenv_path}")
# load_dotenv(dotenv_path=dotenv_path)
# # --- End of Path Setup ---
#
# from fastapi import FastAPI, UploadFile, File
# from pydantic import BaseModel, Field
# from typing import Literal
#
# # --- LangChain Imports ---
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_community.vectorstores import Chroma
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableLambda
# from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
#
# # --- Import our ingestion function ---
# from .ingest import process_and_store_documents
#
# # --- Check for API Key ---
# if "GOOGLE_API_KEY" not in os.environ:
#     raise EnvironmentError("GOOGLE_API_KEY not set in environment variables. Please check your .env file.")
#
# # --- Global Variables & Initialization ---
#
# # 1. Define the paths (ABSOLUTE PATHS)
# PERSIST_DIRECTORY = os.path.join(script_dir, "db_chroma")
# SOURCE_DOCUMENTS_DIR = os.path.join(script_dir, "source_documents")
#
# # 2. Initialize the Embedding Model
# print("Initializing embedding model...")
# embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
#
# # 3. Load/Create the persistent ChromaDB
# print(f"Loading persistent vector database from: {PERSIST_DIRECTORY}")
# if not os.path.exists(PERSIST_DIRECTORY):
#     print(f"Warning: Database directory not found: {PERSIST_DIRECTORY}. Running ingest.py to create it...")
#
#     if not os.path.exists(SOURCE_DOCUMENTS_DIR):
#         os.makedirs(SOURCE_DOCUMENTS_DIR)
#         print(f"Created missing directory: {SOURCE_DOCUMENTS_DIR}")
#         print("Please add documents to this folder and restart.")
#
#     process_and_store_documents()
#
#     if not os.path.exists(PERSIST_DIRECTORY):
#         print(f"Warning: Ingestion ran, but {PERSIST_DIRECTORY} still not found.")
#         print("This is OK if source_documents is empty. Creating an empty DB.")
#         db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
#         db.persist()
#     else:
#         print("Ingestion successful. DB created.")
#         db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
# else:
#     db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
#
# # 4. Create a "Retriever"
# retriever = db.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 5}  # Retrieve more docs for better summarization
# )
# print("Vector database loaded and retriever created.")
#
# # 5. Initialize the LLM (Gemini)
# llm = ChatGoogleGenerativeAI(
#     # model="gemini-2.5-pro",
#     model="gemini-2.0-flash",
#     temperature=0.3
# )
#
#
# # --- 6. NEW: Define Query Router Logic ---
#
# # Pydantic model for our router's output
# class QueryType(BaseModel):
#     query_type: Literal["question", "summarization"] = Field(
#         description="Classifies the user's request as 'question' or 'summarization'"
#     )
#
#
# # Create an LLM with structured output for classification
# classifier_llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
# structured_llm = classifier_llm.with_structured_output(QueryType)
#
# # Prompt for the router
# router_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", """
# You are an expert query classifier. Your job is to analyze the user's input and determine if they are asking a specific question or requesting a general summarization.
# - 'question': Use this for specific lookups (e.g., "What is X?", "How does Y work?", "Compare A and B").
# - 'summarization': Use this for broad, open-ended requests (e.g., "Summarize this document", "What's this about?", "Tell me the key points").
# """),
#         ("user", "{question}"),
#     ]
# )
#
# # The router chain itself
# query_router = router_prompt | structured_llm
# print("RAG query router created.")
#
# # --- 7. NEW: Define Two Separate Chains ---
#
# # --- Chain 1: The Q&A Chain (Upgraded Prompt) ---
# qa_template = """
# You are a helpful customer support assistant for the company 'Genilia'.
# Your job is to answer the user's question based *only* on the provided context.
#
# ---
# RULES:
# 1.  If the context doesn't contain the answer, just say "I'm sorry, I don't have that information."
# 2.  Do not make up answers.
# 3.  Always use Markdown to format your answers.
# 4.  Use bullet points (e.g., * Item 1) for lists of specifications, features, or steps.
# 5.  Use bolding (e.g., **Feature Name**) to highlight key terms from the context.
# ---
#
# CONTEXT:
# {context}
#
# QUESTION:
# {question}
#
# ANSWER (in Markdown):
# """
# qa_prompt = ChatPromptTemplate.from_template(qa_template)
# qa_chain = (
#         {"context": retriever, "question": RunnablePassthrough()}
#         | qa_prompt
#         | llm
#         | StrOutputParser()
# )
# print("RAG Q&A chain created.")
#
# # --- Chain 2: The Summarization Chain (New) ---
# summarize_template = """
# You are an expert summarization assistant. Your job is to provide a concise summary of the retrieved context.
#
# ---
# RULES:
# 1.  Do not add any information that is not in the context.
# 2.  Start with a one-sentence overview.
# 3.  Follow with a bulleted list of the 3-5 most important key points.
# 4.  The summary should be objective and professional.
# ---
#
# CONTEXT:
# {context}
#
# USER REQUEST: "{question}"
#
# CONCISE SUMMARY (in Markdown):
# """
# summarize_prompt = ChatPromptTemplate.from_template(summarize_template)
# summarize_chain = (
#         {"context": retriever, "question": RunnablePassthrough()}
#         | summarize_prompt
#         | llm
#         | StrOutputParser()
# )
# print("RAG Summarization chain created.")
#
#
# # --- 8. NEW: Build the Main Branching Chain ---
#
# # This function links the router's output (a QueryType object) to the correct chain
# def route(query_type: QueryType, input):
#     if query_type.query_type == "summarization":
#         print("--- RAG: Routing to Summarization Chain ---")
#         return summarize_chain.invoke(input["question"])
#     else:  # Default to "question"
#         print("--- RAG: Routing to Q&A Chain ---")
#         return qa_chain.invoke(input["question"])
#
#
# # This is our NEW master chain for the whole RAG agent
# # It's much more complex and powerful
# rag_chain = (
#         {"question": RunnablePassthrough()}  # 1. Get the user's question
#         | RunnablePassthrough.assign(  # 2. Classify the question
#     query_type=query_router
# )
#         | RunnableLambda(lambda x: route(x["query_type"], x))  # 3. Route to the correct chain
# )
# print("RAG Master Branching Chain created successfully.")
#
# # --- FastAPI Application ---
#
# app = FastAPI(
#     title="Genilia RAG Agent",
#     description="Microservice for RAG, Q&A, and Summarization.",
#     version="0.2.0"  # Upped version
# )
#
#
# # ... (rest of the FastAPI app is the same) ...
# # ... (QueryRequest, get_status, etc.) ...
# # --- CUT --- (I'm omitting the identical code for brevity) ---
# # --- PASTE THE REST OF YOUR `main.py` (from QueryRequest down) HERE ---
#
# # Define the input model for our /query endpoint
# class QueryRequest(BaseModel):
#     question: str
#
#
# @app.get("/")
# def get_status():
#     """
#     A simple 'heartbeat' endpoint to check if the
#     server is running.
#     """
#     return {"status": "ok", "message": "RAG Agent is running!"}
#
#
# # --- File Upload Endpoint (Slightly modified to update new chains) ---
# @app.post("/upload")
# def upload_document(file: UploadFile = File(...)):
#     """
#     Allows an admin to upload a document.
#     The file is saved and then the ingestion process is triggered.
#     """
#     if not os.path.exists(SOURCE_DOCUMENTS_DIR):
#         os.makedirs(SOURCE_DOCUMENTS_DIR)
#         print(f"Created directory: {SOURCE_DOCUMENTS_DIR}")
#
#     file_path = os.path.join(SOURCE_DOCUMENTS_DIR, file.filename)
#
#     try:
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
#
#         print(f"File '{file.filename}' saved to {file_path}")
#
#         print("Triggering ingestion process...")
#         process_and_store_documents()
#         print("Ingestion process finished.")
#
#         # --- IMPORTANT: Reload the retriever and REBUILD all chains ---
#         global retriever, qa_chain, summarize_chain, rag_chain
#
#         db = Chroma(
#             persist_directory=PERSIST_DIRECTORY,
#             embedding_function=embeddings
#         )
#         retriever = db.as_retriever(
#             search_type="similarity",
#             search_kwargs={"k": 5}
#         )
#
#         # Rebuild Q&A Chain
#         qa_chain = (
#                 {"context": retriever, "question": RunnablePassthrough()}
#                 | qa_prompt
#                 | llm
#                 | StrOutputParser()
#         )
#
#         # Rebuild Summarization Chain
#         summarize_chain = (
#                 {"context": retriever, "question": RunnablePassthrough()}
#                 | summarize_prompt
#                 | llm
#                 | StrOutputParser()
#         )
#
#         # Rebuild Master Branch
#         rag_chain = (
#                 {"question": RunnablePassthrough()}
#                 | RunnablePassthrough.assign(query_type=query_router)
#                 | RunnableLambda(lambda x: route(x["query_type"], x))
#         )
#         print("Retriever and all RAG chains have been updated.")
#         # -------------------------------------
#
#         return {
#             "status": "success",
#             "filename": file.filename,
#             "message": "File uploaded and ingestion process triggered."
#         }
#     except Exception as e:
#         print(f"Error during file upload or ingestion: {e}")
#         return {"error": f"An error occurred: {e}"}, 500
#     finally:
#         file.file.close()
#
#
# @app.post("/query")
# def query_rag_agent(request: QueryRequest):
#     """
#     The main RAG query endpoint.
#     Receives a question, processes it through the NEW branching RAG chain,
#     and returns the answer.
#     """
#     print(f"\nReceived query: {request.question}")
#
#     try:
#         # 1. Invoke the new master chain
#         # The chain now handles routing internally
#         answer = rag_chain.invoke(request.question)
#
#         print(f"Generated answer: {answer}")
#
#         # 2. Return the answer
#         # The output is now a simple string, so we build the JSON
#         return {"question": request.question, "answer": answer}
#
#     except Exception as e:
#         print(f"Error during RAG chain invocation: {e}")
#         return {"error": f"An error occurred: {e}"}, 500


from dotenv import load_dotenv
import os
import shutil
import json
from pydantic import BaseModel, Field
from typing import Literal, List, Dict

# --- LangChain Imports ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# --- Import our ingestion function ---
from .ingest import process_and_store_documents, PROCESSED_DOCUMENTS_DIR
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse # <-- ADD THIS IMPORT


# --- Absolute Path Setup ---
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
root_dir = os.path.join(script_dir, '..')
dotenv_path = os.path.join(root_dir, '.env')

print(f"Attempting to load .env file from: {dotenv_path}")
load_dotenv(dotenv_path=dotenv_path)
# --- End of Path Setup ---



# --- Check for API Key ---
if "GOOGLE_API_KEY" not in os.environ:
    raise EnvironmentError("GOOGLE_API_KEY not set in environment variables. Please check your .env file.")

# --- Global Variables & Initialization ---

# 1. Define the paths (ABSOLUTE PATHS)
PERSIST_DIRECTORY = os.path.join(script_dir, "db_chroma")
SOURCE_DOCUMENTS_DIR = os.path.join(script_dir, "source_documents")
CATEGORIES_FILE = os.path.join(script_dir, "categories.json")


# --- NEW: Function to load categories from file ---
def get_categories_list() -> List[str]:
    """Loads the categories from the JSON file."""
    try:
        with open(CATEGORIES_FILE, 'r') as f:
            categories = json.load(f)
        return categories
    except FileNotFoundError:
        return ["general"]  # Fallback


# 2. Initialize the Embedding Model
print("Initializing embedding model...")
# --- THIS IS THE NEW, STABLE LINE ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# 3. Load/Create the persistent ChromaDB
print(f"Loading persistent vector database from: {PERSIST_DIRECTORY}")
if not os.path.exists(PERSIST_DIRECTORY):
    print(f"Warning: Database directory not found. Running ingest.py...")
    # ... (omitting identical startup logic for brevity) ...
    if not os.path.exists(SOURCE_DOCUMENTS_DIR):
        os.makedirs(SOURCE_DOCUMENTS_DIR)
    process_and_store_documents()

# Create the base DB connection
db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
)
print("Vector database loaded.")

# 4. Initialize the LLM (Gemini)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3
)


# --- 5. NEW: Metadata Filtering Logic ---

# Pydantic model for our *new* filter
class MetadataFilter(BaseModel):
    product_line: str = Field(
        description="The specific product_line to filter on, e.g., 'cookies' or 'chocolates'. If no specific product is mentioned, use 'general'."
    )


# Create an LLM with structured output for classification
filter_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
structured_filter_llm = filter_llm.with_structured_output(MetadataFilter)

# Prompt for the filter
# We dynamically insert the category list so the LLM knows its options
categories_list_str = ", ".join(get_categories_list())
filter_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", f"""
You are an expert at extracting product categories from a user's question.
Your job is to identify a single 'product_line' to filter the search.
The available categories are: [{categories_list_str}]

- If the user asks a general question (e.g., "return policy", "about the company"), use 'general'.
- If the user asks about a specific product (e.g., "chocolate chip cookies"), use the matching category (e.g., 'cookies').
- If you are unsure, default to 'general'.
"""),
        ("user", "{question}"),
    ]
)

# The filter-extracting chain
filter_chain = filter_prompt | structured_filter_llm
print("RAG metadata filter chain created.")


# --- 6. Query Router Logic (Same as before) ---
class QueryType(BaseModel):
    query_type: Literal["question", "summarization"]


classifier_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
structured_llm = classifier_llm.with_structured_output(QueryType)
router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Classify the user's request as 'question' or 'summarization'."),
        ("user", "{question}"),
    ]
)
query_router = router_prompt | structured_llm
print("RAG query router created.")


# --- 7. NEW: Dynamic Retriever Function ---
# This is the core of our upgrade.
# This function creates a *new retriever* for *every query*
# based on the filter's output.

def get_dynamic_retriever(metadata_filter: MetadataFilter):
    """
    Creates a new retriever with a metadata filter,
    or a default retriever if the category is 'general'.
    """
    product_line = metadata_filter.product_line

    if product_line == "general":
        print("--- RAG: Using GENERAL retriever (no filter) ---")
        return db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    else:
        print(f"--- RAG: Using FILTERED retriever (product_line = '{product_line}') ---")
        return db.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,
                "filter": {"product_line": product_line}
            }
        )


# --- END NEW LOGIC ---

# --- 8. Define Q&A and Summarization Chains ---
# These are the same as before, but we'll feed the retriever in
# differently when we build the master chain.

qa_template = """
You are a helpful customer support assistant... (full prompt here)
CONTEXT:
{context}
QUESTION:
{question}
ANSWER (in Markdown):
"""
qa_prompt = ChatPromptTemplate.from_template(qa_template)
# We can't build the full chain yet, just the prompt + LLM part
qa_logic_chain = qa_prompt | llm | StrOutputParser()
print("RAG Q&A logic created.")

summarize_template = """
You are an expert summarization assistant... (full prompt here)
CONTEXT:
{context}
USER REQUEST: "{question}"
CONCISE SUMMARY (in Markdown):
"""
summarize_prompt = ChatPromptTemplate.from_template(summarize_template)
# We can't build the full chain yet, just the prompt + LLM part
summarize_logic_chain = summarize_prompt | llm | StrOutputParser()
print("RAG Summarization logic created.")


# --- 9. NEW: Build the Main Branching Chain (Upgraded) ---

# This function links the router's output to the correct logic chain
def route_and_invoke(input_dict: Dict):
    query_type = input_dict["query_type"].query_type
    retriever = input_dict["retriever"]  # The dynamic retriever
    question = input_dict["question"]

    # Retrieve context *using the dynamic retriever*
    context = retriever.invoke(question)

    if query_type == "summarization":
        print("--- RAG: Routing to Summarization Chain ---")
        return summarize_logic_chain.invoke({"context": context, "question": question})
    else:  # Default to "question"
        print("--- RAG: Routing to Q&A Chain ---")
        return qa_logic_chain.invoke({"context": context, "question": question})


# This is our NEW master chain for the whole RAG agent
rag_chain = (
        {"question": RunnablePassthrough()}  # 1. Get the user's question
        | RunnablePassthrough.assign(  # 2. Extract the filter
    metadata_filter=filter_chain
)
        | RunnablePassthrough.assign(  # 3. Create the dynamic retriever
    retriever=RunnableLambda(lambda x: get_dynamic_retriever(x["metadata_filter"]))
)
        | RunnablePassthrough.assign(  # 4. Classify Q&A or Summarize
    query_type=query_router
)
        | RunnableLambda(route_and_invoke)  # 5. Route to the correct chain
)
print("RAG Master Metadata-Aware Chain created successfully.")

# --- FastAPI Application ---

app = FastAPI(
    title="Genilia RAG Agent",
    description="Microservice for RAG, Q&A, and Summarization with Metadata Filtering.",
    version="0.4.0"  # Upped version
)


# ... (omitting /categories, /upload, /query as they are identical) ...
# --- CUT --- (I'm omitting the identical FastAPI endpoints for brevity) ---
# --- PASTE THE REST OF YOUR `main.py` (from QueryRequest down) HERE ---

class QueryRequest(BaseModel):
    question: str


@app.get("/")
def get_status():
    return {"status": "ok", "message": "RAG Agent is running!"}


@app.get("/categories")
def get_categories():
    """
    Reads and returns the list of product categories
    from categories.json for the admin UI dropdown.
    """
    return {"categories": get_categories_list()}

# ... (this is your existing @app.get("/") endpoint)


# --- ADD THIS NEW ENDPOINT ---
# ... (this is your existing @app.get("/") endpoint)


# --- ADD THIS NEW ENDPOINT ---
@app.get("/admin/clear-memory")
def clear_all_chat_history():
    """
    Clears all in-memory chat histories for all sessions.
    Called by the Admin Panel's reset button.
    """
    global chat_histories
    count = len(chat_histories)
    chat_histories.clear() # Empties the dictionary
    print(f"--- ADMIN: Cleared {count} chat session histories. ---")
    return {"status": "ok", "message": f"Cleared {count} session histories."}
# --- END OF NEW ENDPOINT ---


# ... (the rest of your file, like @app.post("/chat"), continues) ...
# --- END OF NEW ENDPOINT ---

# ... (this is your existing @app.get("/admin") endpoint)
@app.get("/admin", response_class=FileResponse)
async def get_admin_page():
    """
    Serves the static admin.html page.
    """
    admin_page_path = os.path.join(script_dir, "admin.html")
    if not os.path.exists(admin_page_path):
        return {"error": "admin.html file not found"}, 404
    return FileResponse(admin_page_path)


# --- ADD THIS ENTIRE NEW ENDPOINT ---
# This helper function is new. Add it inside main.py, above the reset endpoint.
# def clear_directory_contents(directory_path):
#     """
#     Deletes all files and sub-folders inside a given directory,
#     but leaves the directory itself.
#     """
#     if not os.path.exists(directory_path):
#         print(f"Directory {directory_path} not found, skipping.")
#         return
#
#     for item_name in os.listdir(directory_path):
#         item_path = os.path.join(directory_path, item_name)
#         try:
#             if os.path.isfile(item_path) or os.path.islink(item_path):
#                 os.unlink(item_path)
#             elif os.path.isdir(item_path):
#                 shutil.rmtree(item_path)
#         except Exception as e:
#             print(f"Failed to delete {item_path}. Reason: {e}")


# --- THIS IS THE NEW, REPLACED RESET ENDPOINT ---
# --- THIS IS THE NEW, MORE POWERFUL RESET ENDPOINT ---
@app.post("/admin/reset")
def reset_agent_knowledge():
    """
    DANGER: This performs a full factory reset of the RAG agent.
    - Deletes the chroma.sqlite3 file
    - Deletes all contents of processed_documents and source_documents
    - Resets categories to default
    """
    print("--- ADMIN: FACTORY RESET REQUESTED ---")
    try:
        # 1. Get all global objects we need to modify
        global db, retriever, qa_chain, summarize_chain, rag_chain, filter_chain, filter_prompt, llm

        # 2. Stop the current DB connection
        # We need to do this before deleting its files
        # --- THIS IS THE NEW, SAFER BLOCK ---
        # 2. Stop the current DB connection
        # We need to do this before deleting its files
        print("Detaching from vector database...")
        if 'db' in globals():
            del db
        if 'retriever' in globals():
            del retriever
        print("Detached from vector database.")

        # 3. Wipe the Vector Database file (as you requested)
        db_file_path = os.path.join(PERSIST_DIRECTORY, "chroma.sqlite3")
        if os.path.exists(db_file_path):
            os.remove(db_file_path)
            print(f"Deleted database file: {db_file_path}")
        else:
            print("Database file not found, skipping.")

        # --- 4. NEW: Nuke and Recreate Document Folders ---
        print("Clearing processed and source document folders...")

        # Delete the entire folders
        if os.path.exists(PROCESSED_DOCUMENTS_DIR):
            shutil.rmtree(PROCESSED_DOCUMENTS_DIR)
        if os.path.exists(SOURCE_DOCUMENTS_DIR):
            shutil.rmtree(SOURCE_DOCUMENTS_DIR)

        # Re-create them empty
        os.makedirs(PROCESSED_DOCUMENTS_DIR)
        os.makedirs(SOURCE_DOCUMENTS_DIR)
        print("Successfully cleared and recreated document folders.")
        # --- END OF NEW LOGIC ---

        # 5. Reset the categories file
        print("Resetting categories.json...")
        default_categories = ["general"]
        with open(CATEGORIES_FILE, 'w') as f:
            json.dump(default_categories, f, indent=2)

        # 6. Re-initialize all chains with empty/default state
        print("Re-initializing all agent chains...")

        # Re-build filter_chain
        categories_list_str = ", ".join(default_categories)
        filter_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"""
You are an expert at extracting product categories...
The available categories are: [{categories_list_str}]
...
"""),
                ("user", "{question}"),
            ]
        )
        filter_chain = filter_prompt | structured_filter_llm

        # Re-init DB connection (this will create a new, empty chroma.sqlite3)
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )

        # Re-build retriever (will be empty)
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # Re-build all logic chains
        qa_logic_chain = qa_prompt | llm | StrOutputParser()
        summarize_logic_chain = summarize_prompt | llm | StrOutputParser()

        # Re-build master chain
        rag_chain = (
                {"question": RunnablePassthrough()}
                | RunnablePassthrough.assign(metadata_filter=filter_chain)
                | RunnablePassthrough.assign(
            retriever=RunnableLambda(lambda x: get_dynamic_retriever(x["metadata_filter"])))
                | RunnablePassthrough.assign(query_type=query_router)
                | RunnableLambda(route_and_invoke)
        )

        print("--- ADMIN: SYSTEM RESET COMPLETE ---")
        return {"status": "ok", "message": "Agent has been fully reset. All knowledge and metadata deleted."}

    except Exception as e:
        print(f"--- ERROR DURING RESET: {e} ---")
        return {"error": str(e)}, 500

# --- END OF NEW ENDPOINT ---


# ... (the rest of your file, like @app.get("/categories"), continues) ...
# ... (the rest of your file, like @app.get("/categories"), continues) ...

@app.post("/upload")
def upload_document(
        file: UploadFile = File(...),
        category: str = Form(...)
):
    """
    Allows an admin to upload a document to a specific category.
    If the category is new, it's created.
    Then, the ingestion process is triggered.
    """
    try:
        category = category.lower().strip().replace(" ", "_")
        if not category:
            return {"error": "Category cannot be empty"}, 400

        # Logic to update categories.json
        categories = get_categories_list()
        if category not in categories:
            categories.append(category)
            with open(CATEGORIES_FILE, 'w') as f:
                json.dump(categories, f, indent=2)
            print(f"Added new category: {category}")

            # --- IMPORTANT: Re-build the filter_chain ---
            global filter_chain, filter_prompt
            categories_list_str = ", ".join(categories)
            filter_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", f"""
You are an expert at extracting product categories from a user's question.
Your job is to identify a single 'product_line' to filter the search.
The available categories are: [{categories_list_str}]
If unsure, default to 'general'.
"""),
                    ("user", "{question}"),
                ]
            )
            filter_chain = filter_prompt | structured_filter_llm
            print("--- RAG: Rebuilt filter chain with new categories. ---")

        # Logic to save file in the correct sub-folder
        category_folder = os.path.join(SOURCE_DOCUMENTS_DIR, category)
        if not os.path.exists(category_folder):
            os.makedirs(category_folder)
            print(f"Created new directory: {category_folder}")

        file_path = os.path.join(category_folder, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"File '{file.filename}' saved to {file_path}")

        # Trigger the ingestion process
        print("Triggering ingestion process...")
        process_and_store_documents()
        print("Ingestion process finished.")

        # --- IMPORTANT: Reload the base DB object ---
        global db
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
        print("Retriever and all RAG chains have been updated.")
        # -------------------------------------

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
    """
    The main RAG query endpoint.
    """
    print(f"\nReceived query: {request.question}")
    try:
        answer = rag_chain.invoke(request.question)
        print(f"Generated answer: {answer}")
        return {"question": request.question, "answer": answer}
    except Exception as e:
        print(f"Error during RAG chain invocation: {e}")
        return {"error": f"An error occurred: {e}"}, 500