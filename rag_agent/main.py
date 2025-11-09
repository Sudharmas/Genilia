
from dotenv import load_dotenv
import os
import shutil
import json
from pydantic import BaseModel, Field
from typing import Literal, List, Dict

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from .ingest import process_and_store_documents, PROCESSED_DOCUMENTS_DIR
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse


script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
root_dir = os.path.join(script_dir, '..')
dotenv_path = os.path.join(root_dir, '.env')

print(f"Attempting to load .env file from: {dotenv_path}")
load_dotenv(dotenv_path=dotenv_path)

if "GOOGLE_API_KEY" not in os.environ:
    raise EnvironmentError("GOOGLE_API_KEY not set in environment variables. Please check your .env file.")


PERSIST_DIRECTORY = os.path.join(script_dir, "db_chroma")
SOURCE_DOCUMENTS_DIR = os.path.join(script_dir, "source_documents")
CATEGORIES_FILE = os.path.join(script_dir, "categories.json")


def get_categories_list() -> List[str]:
    """Loads the categories from the JSON file."""
    try:
        with open(CATEGORIES_FILE, 'r') as f:
            categories = json.load(f)
        return categories
    except FileNotFoundError:
        return ["general"]


print("Initializing embedding model...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


print(f"Loading persistent vector database from: {PERSIST_DIRECTORY}")
if not os.path.exists(PERSIST_DIRECTORY):
    print(f"Warning: Database directory not found. Running ingest.py...")
    if not os.path.exists(SOURCE_DOCUMENTS_DIR):
        os.makedirs(SOURCE_DOCUMENTS_DIR)
    process_and_store_documents()

db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
)
print("Vector database loaded.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3
)


class MetadataFilter(BaseModel):
    product_line: str = Field(
        description="The specific product_line to filter on, e.g., 'cookies' or 'chocolates'. If no specific product is mentioned, use 'general'."
    )


filter_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
structured_filter_llm = filter_llm.with_structured_output(MetadataFilter)

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

filter_chain = filter_prompt | structured_filter_llm
print("RAG metadata filter chain created.")


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


qa_template = """
You are a helpful customer support assistant... (full prompt here)
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
You are an expert summarization assistant... (full prompt here)
CONTEXT:
{context}
USER REQUEST: "{question}"
CONCISE SUMMARY (in Markdown):
"""
summarize_prompt = ChatPromptTemplate.from_template(summarize_template)
summarize_logic_chain = summarize_prompt | llm | StrOutputParser()
print("RAG Summarization logic created.")

def route_and_invoke(input_dict: Dict):
    query_type = input_dict["query_type"].query_type
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
        | RunnablePassthrough.assign(
    metadata_filter=filter_chain
)
        | RunnablePassthrough.assign(
    retriever=RunnableLambda(lambda x: get_dynamic_retriever(x["metadata_filter"]))
)
        | RunnablePassthrough.assign(
    query_type=query_router
)
        | RunnableLambda(route_and_invoke)
)
print("RAG Master Metadata-Aware Chain created successfully.")


app = FastAPI(
    title="Genilia RAG Agent",
    description="Microservice for RAG, Q&A, and Summarization with Metadata Filtering.",
    version="0.4.0"
)

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

@app.get("/admin/clear-memory")
def clear_all_chat_history():
    """
    Clears all in-memory chat histories for all sessions.
    Called by the Admin Panel's reset button.
    """
    global chat_histories
    count = len(chat_histories)
    chat_histories.clear()
    print(f"--- ADMIN: Cleared {count} chat session histories. ---")
    return {"status": "ok", "message": f"Cleared {count} session histories."}


@app.get("/admin", response_class=FileResponse)
async def get_admin_page():
    """
    Serves the static admin.html page.
    """
    admin_page_path = os.path.join(script_dir, "admin.html")
    if not os.path.exists(admin_page_path):
        return {"error": "admin.html file not found"}, 404
    return FileResponse(admin_page_path)


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
        global db, retriever, qa_chain, summarize_chain, rag_chain, filter_chain, filter_prompt, llm

        print("Detaching from vector database...")
        if 'db' in globals():
            del db
        if 'retriever' in globals():
            del retriever
        print("Detached from vector database.")

        db_file_path = os.path.join(PERSIST_DIRECTORY, "chroma.sqlite3")
        if os.path.exists(db_file_path):
            os.remove(db_file_path)
            print(f"Deleted database file: {db_file_path}")
        else:
            print("Database file not found, skipping.")

        print("Clearing processed and source document folders...")

        if os.path.exists(PROCESSED_DOCUMENTS_DIR):
            shutil.rmtree(PROCESSED_DOCUMENTS_DIR)
        if os.path.exists(SOURCE_DOCUMENTS_DIR):
            shutil.rmtree(SOURCE_DOCUMENTS_DIR)

        os.makedirs(PROCESSED_DOCUMENTS_DIR)
        os.makedirs(SOURCE_DOCUMENTS_DIR)
        print("Successfully cleared and recreated document folders.")

        print("Resetting categories.json...")
        default_categories = ["general"]
        with open(CATEGORIES_FILE, 'w') as f:
            json.dump(default_categories, f, indent=2)

        print("Re-initializing all agent chains...")

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

        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )

        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        qa_logic_chain = qa_prompt | llm | StrOutputParser()
        summarize_logic_chain = summarize_prompt | llm | StrOutputParser()

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

        category_folder = os.path.join(SOURCE_DOCUMENTS_DIR, category)
        if not os.path.exists(category_folder):
            os.makedirs(category_folder)
            print(f"Created new directory: {category_folder}")

        file_path = os.path.join(category_folder, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"File '{file.filename}' saved to {file_path}")

        print("Triggering ingestion process...")
        process_and_store_documents()
        print("Ingestion process finished.")

        global db
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
        print("Retriever and all RAG chains have been updated.")

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