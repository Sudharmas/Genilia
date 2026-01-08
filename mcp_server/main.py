# from dotenv import load_dotenv
# import uvicorn
# import os
# import httpx
# from typing import List, Dict
# from contextlib import asynccontextmanager
#
# script_path = os.path.abspath(__file__)
# script_dir = os.path.dirname(script_path)
# root_dir = os.path.join(script_dir, '..')
# dotenv_path = os.path.join(root_dir, '.env')
#
# print(f"Attempting to load .env file from: {dotenv_path}")
# load_dotenv(dotenv_path=dotenv_path)
#
# from fastapi import FastAPI
# from pydantic import BaseModel, Field
# from fastapi.middleware.cors import CORSMiddleware
#
# from langchain_ollama import ChatOllama
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.messages import HumanMessage, AIMessage
#
# RAG_AGENT_URL = "http://127.0.0.1:8000/query"
# ACTION_AGENT_URL = "http://127.0.0.1:8001/run-agent"
#
#
# # FINETUNE_AGENT_URL = "http://127.0.0.1:8003/query"
#
# def get_llm(model_name: str = "phi3", google_fallback: str = "gemini-2.0-flash"):
#     """
#     Tries to load the local Ollama model.
#     If it fails (Ollama not running), it falls back to the Google Gemini model.
#     """
#     try:
#         llm = ChatOllama(model=model_name)
#         llm.invoke("Hello")
#         print(f"--- Successfully connected to local LLM: {model_name} ---")
#         return llm
#     except Exception as e:
#         print(f"--- WARNING: Local model '{model_name}' failed. Falling back to Google. ---")
#         print(f"--- Error: {e} ---")
#         if "GOOGLE_API_KEY" not in os.environ:
#             print("--- ERROR: GOOGLE_API_KEY not found for fallback. ---")
#             return None
#         return ChatGoogleGenerativeAI(model=google_fallback)
#
# print("Initializing local LLMs (phi3)...")
# llm_router = get_llm(model_name="phi3", google_fallback="gemini-2.0-flash")
# llm_condenser = get_llm(model_name="phi3", google_fallback="gemini-2.0-flash")
# print("MCP LLMs initialized.")
#
# routing_prompt = ChatPromptTemplate.from_template(
#     """
# You are an expert router. Your job is to classify a user's query and decide which agent to send it to.
# You have two choices:
#
# 1.  'rag_agent': **This is the primary agent.** Use this for *all* questions related to products. This includes:
#     - Product suggestions (e.g., "suggest a speaker")
#     - Product specifications (e.g., "how big is the X-1000")
#     - Product comparisons (e.g., "A vs B")
#     - Company policies (e.g., "return policy")
#     - FAQs
#
# 2.  'action_agent': **This is the secondary agent.** Use this *only* for general, non-product questions that require a web search, such as:
#     - "What's new on the company blog?"
#     - "Where is the company headquarters?"
#     - "What's the company's contact info?"
#
# User Query:
# "{input}"
#
# Which agent should handle this? (Return *only* 'rag_agent' or 'action_agent')
# """
# )
# routing_chain = routing_prompt | llm_router | StrOutputParser()
# print("MCP Routing chain created.")
#
# condensing_prompt_template = """
# Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
#
# Chat History:
# {chat_history}
#
# Follow Up Input: {input}
# Standalone Question:"""
# condensing_prompt = ChatPromptTemplate.from_template(condensing_prompt_template)
# condensing_chain = condensing_prompt | llm_condenser | StrOutputParser()
# print("MCP Query Condensing chain created.")
#
# chat_histories: Dict[str, List] = {}
#
#
# def get_chat_history(session_id: str):
#     if session_id not in chat_histories:
#         chat_histories[session_id] = []
#     return chat_histories[session_id]
#
#
# def format_history_for_prompt(history: List):
#     return "\n".join(
#         [
#             f"Human: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}"
#             for msg in history
#         ]
#     )
#
#
# RAG_FAILURE_PHRASES = [
#     "i'm sorry, i don't have that information",
#     "i don't have that information",
#     "context doesn't contain the answer"
# ]
#
#
# def is_rag_failure(answer: str) -> bool:
#     if not answer: return True
#     lower_answer = answer.lower()
#     for phrase in RAG_FAILURE_PHRASES:
#         if phrase in lower_answer:
#             return True
#     return False
#
#
# http_client = httpx.AsyncClient(timeout=30.0)
#
#
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     print("HTTPX AsyncClient started.")
#     yield
#     await http_client.aclose()
#     print("HTTPX AsyncClient closed.")
#
#
# app = FastAPI(
#     title="Genilia MCP (Hybrid)",
#     description="The central router for the Genilia agent system (Local-first).",
#     version="1.5.0",
#     lifespan=lifespan
# )
#
# origins = [
#     "http://localhost",
#     "http://localhost:5173",
#     "http://localhost:3000",
#     "http://127.0.0.1:8000",
#     "http://localhost:8000",
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
#
# class ChatRequest(BaseModel):
#     input: str
#     session_id: str = Field(default="default-session-id", description="Unique ID for the chat session")
#
#
# @app.get("/")
# def get_status():
#     return {"status": "ok", "message": "MCP Server is running!"}
#
#
# @app.get("/admin/clear-memory")
# def clear_all_chat_history():
#     global chat_histories
#     count = len(chat_histories)
#     chat_histories.clear()
#     print(f"--- ADMIN: Cleared {count} chat session histories. ---")
#     return {"status": "ok", "message": f"Cleared {count} session histories."}
#
#
# @app.post("/chat")
# async def chat_endpoint(request: ChatRequest):
#     print(f"\n--- MCP RECEIVED QUERY for Session: '{request.session_id}' ---")
#     print(f"Original Input: '{request.input}'")
#
#     try:
#         chat_history = get_chat_history(request.session_id)
#         if not chat_history:
#             condensed_query = request.input
#             print("No history. Using input as standalone query.")
#         else:
#             print("Condensing query with history...")
#             formatted_history = format_history_for_prompt(chat_history)
#             condensed_query = await condensing_chain.ainvoke({
#                 "chat_history": formatted_history,
#                 "input": request.input
#             })
#             print(f"Condensed Query: '{condensed_query}'")
#     except Exception as e:
#         print(f"Error during query condensing: {e}")
#         return {"error": "Failed to process chat history."}, 500
#
#     try:
#         print(f"Routing condensed query: '{condensed_query}'")
#         agent_to_use = await routing_chain.ainvoke({"input": condensed_query})
#         agent_to_use = agent_to_use.strip().replace("'", "")
#         print(f"Decision: First attempt with '{agent_to_use}'")
#     except Exception as e:
#         print(f"Error during routing: {e}")
#         return {"error": "Failed to route query."}, 500
#
#     final_answer = ""
#     final_json_response = {}
#
#     if agent_to_use == "rag_agent":
#         try:
#             print(f"Calling RAG Agent with: '{condensed_query}'")
#             response = await http_client.post(RAG_AGENT_URL, json={"question": condensed_query})
#             response.raise_for_status()
#             final_json_response = response.json()
#             final_answer = final_json_response.get("answer", "")
#
#             if is_rag_failure(final_answer):
#                 print("RAG Agent failed. FALLING BACK to Action Agent.")
#                 agent_to_use = "action_agent"
#             else:
#                 print("RAG Agent succeeded.")
#
#         except httpx.HTTPStatusError as e:
#             print(f"Error calling RAG agent: {e}. FALLING BACK to Action Agent.")
#             agent_to_use = "action_agent"
#         except Exception as e:
#             print(f"Error processing RAG response: {e}. FALLING BACK to Action Agent.")
#             agent_to_use = "action_agent"
#
#     if agent_to_use == "action_agent":
#         try:
#             print(f"Calling Action Agent with: '{condensed_query}'")
#             response = await http_client.post(ACTION_AGENT_URL, json={"input": condensed_query})
#             response.raise_for_status()
#             final_json_response = response.json()
#             final_answer = final_json_response.get("output", "")
#             print("Action Agent succeeded.")
#         except httpx.HTTPStatusError as e:
#             print(f"Error calling Action agent: {e}")
#             final_json_response = {"error": "Action agent is unavailable or failed."}
#         except Exception as e:
#             print(f"Error processing Action response: {e}")
#             final_json_response = {"error": "Failed to process Action agent response."}
#     else:
#         if final_answer:
#             pass
#         else:
#             print(f"Routing logic failed. Unknown agent: '{agent_to_use}'")
#             final_json_response = {"error": f"Routing failed. Unknown agent '{agent_to_use}'."}
#
#     if final_answer:
#         chat_history.append(HumanMessage(content=request.input))
#         chat_history.append(AIMessage(content=final_answer))
#         print(f"Saved to history for session '{request.session_id}'")
#
#     return final_json_response
#
#
# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8002)


from dotenv import load_dotenv
import uvicorn
import os
import httpx
from typing import List, Dict, Literal
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field

# --- Build a robust path to the .env file ---
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
root_dir = os.path.join(script_dir, '..')
dotenv_path = os.path.join(root_dir, '.env')

print(f"Attempting to load .env file from: {dotenv_path}")
load_dotenv(dotenv_path=dotenv_path)
# --- End of .env loading ---

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- UPDATED IMPORTS ---
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser  # <-- Added JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# --- END UPDATED IMPORTS ---

# --- 1. Define Worker Agent Endpoints ---
RAG_AGENT_URL = "http://127.0.0.1:8000/query"
ACTION_AGENT_URL = "http://127.0.0.1:8001/run-agent"
FINETUNE_AGENT_URL = "http://127.0.0.1:8003/query"


# --- 2. Hybrid LLM Loader ---
def get_llm(model_name: str = "phi3", google_fallback: str = "gemini-2.0-flash"):
    try:
        llm = ChatOllama(model=model_name, temperature=0)  # Keep temp 0 for strictness
        llm.invoke("Hello")
        print(f"--- Successfully connected to local LLM: {model_name} ---")
        return llm
    except Exception as e:
        print(f"--- WARNING: Local model '{model_name}' failed. Falling back to Google. ---")
        if "GOOGLE_API_KEY" not in os.environ:
            print("--- ERROR: GOOGLE_API_KEY not found for fallback. ---")
            return None
        return ChatGoogleGenerativeAI(model=google_fallback, temperature=0)


print("Initializing MCP LLMs...")
llm_router = get_llm()
llm_condenser = get_llm()


# --- 3. NEW: Structured Routing Logic (The Fix) ---
class RoutingDecision(BaseModel):
    # This forces the model to pick EXACTLY one of these three strings
    agent_name: Literal["rag_agent", "action_agent", "finetune_agent"] = Field(
        description="The name of the agent best suited to handle the query."
    )


router_parser = JsonOutputParser(pydantic_object=RoutingDecision)

routing_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
You are an expert router. Classify the user's query and decide which agent to send it to.

AGENTS:
1. 'finetune_agent': (Specialist) For specific, factual, technical Q&A lookups (dimensions, power, specs).
2. 'rag_agent': (Generalist) For broad questions, summaries, comparisons, policies, or product suggestions.
3. 'action_agent': (Web Searcher) ONLY for non-product questions like "company blog", "headquarters", "contact info".

**You must format your response as a JSON object with a single key "agent_name".**
"""),
        ("user", "{input}"),
    ]
)

# The chain now returns a Dictionary, not a string
routing_chain = routing_prompt | llm_router | router_parser
print("MCP Routing chain created (JSON mode).")
# --- END NEW LOGIC ---

# --- 4. Query Condensing Logic ---
condensing_prompt_template = """
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}

Follow Up Input: {input}
Standalone Question:"""
condensing_prompt = ChatPromptTemplate.from_template(condensing_prompt_template)
condensing_chain = condensing_prompt | llm_condenser | StrOutputParser()

# --- 5. Chat History & RAG Logic ---
chat_histories: Dict[str, List] = {}


def get_chat_history(session_id: str):
    if session_id not in chat_histories: chat_histories[session_id] = []
    return chat_histories[session_id]


def format_history_for_prompt(history: List):
    return "\n".join([f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in history])


RAG_FAILURE_PHRASES = ["i'm sorry", "don't have that information", "context doesn't contain"]


def is_rag_failure(answer: str) -> bool:
    return any(phrase in answer.lower() for phrase in RAG_FAILURE_PHRASES) if answer else True


# --- FastAPI Application ---
http_client = httpx.AsyncClient(timeout=30.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("HTTPX AsyncClient started.")
    yield
    await http_client.aclose()
    print("HTTPX AsyncClient closed.")


app = FastAPI(
    title="Genilia MCP (Hybrid)",
    description="The central router for the Genilia agent system (Local-first).",
    version="1.6.0",
    lifespan=lifespan
)

origins = ["http://localhost", "http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:8000",
           "http://localhost:8000"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])


class ChatRequest(BaseModel):
    input: str
    session_id: str = Field(default="default-session-id")


@app.get("/")
def get_status():
    return {"status": "ok", "message": "MCP Server is running!"}


@app.get("/admin/clear-memory")
def clear_all_chat_history():
    count = len(chat_histories)
    chat_histories.clear()
    return {"status": "ok", "message": f"Cleared {count} session histories."}


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    print(f"\n--- MCP RECEIVED QUERY for Session: '{request.session_id}' ---")

    # 1. Condense Query
    try:
        chat_history = get_chat_history(request.session_id)
        if not chat_history:
            condensed_query = request.input
        else:
            formatted_history = format_history_for_prompt(chat_history)
            condensed_query = await condensing_chain.ainvoke(
                {"chat_history": formatted_history, "input": request.input})
            print(f"Condensed Query: '{condensed_query}'")
    except Exception as e:
        print(f"Error condensing: {e}")
        condensed_query = request.input

    # 2. Route Query (Now using JSON parsing)
    try:
        print(f"Routing: '{condensed_query}'")
        # The chain returns a DICT now: {'agent_name': '...'}
        route_result = await routing_chain.ainvoke({"input": condensed_query})
        agent_to_use = route_result.get("agent_name", "rag_agent")  # Default to RAG if key missing
        print(f"Decision: '{agent_to_use}'")
    except Exception as e:
        print(f"Error during routing: {e}")
        # Fallback safely if JSON parsing fails completely
        agent_to_use = "rag_agent"
        print("Routing failed, defaulting to 'rag_agent'.")

    final_answer = ""
    final_json_response = {}

    # 3. Execute Agent
    if agent_to_use == "finetune_agent":
        try:
            response = await http_client.post(FINETUNE_AGENT_URL, json={"question": condensed_query})
            response.raise_for_status()
            final_json_response = response.json()
            final_answer = final_json_response.get("answer", "")
        except Exception as e:
            print(f"Finetune failed: {e}. Fallback to RAG.")
            agent_to_use = "rag_agent"

    if agent_to_use == "rag_agent":
        try:
            response = await http_client.post(RAG_AGENT_URL, json={"question": condensed_query})
            response.raise_for_status()
            final_json_response = response.json()
            final_answer = final_json_response.get("answer", "")

            if is_rag_failure(final_answer):
                print("RAG failed. Fallback to Action.")
                agent_to_use = "action_agent"
        except Exception as e:
            print(f"RAG failed: {e}. Fallback to Action.")
            agent_to_use = "action_agent"

    if agent_to_use == "action_agent":
        try:
            response = await http_client.post(ACTION_AGENT_URL, json={"input": condensed_query})
            response.raise_for_status()
            final_json_response = response.json()
            final_answer = final_json_response.get("output", "")
        except Exception as e:
            print(f"Action failed: {e}")
            final_json_response = {"error": "I couldn't find an answer."}

    if final_answer:
        chat_history.append(HumanMessage(content=request.input))
        chat_history.append(AIMessage(content=final_answer))

    return final_json_response


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)