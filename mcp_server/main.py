# from dotenv import load_dotenv
# import uvicorn
# import os
# import httpx  # The library for making API calls
#
# # --- Build a robust path to the .env file ---
# script_path = os.path.abspath(__file__)
# script_dir = os.path.dirname(script_path)
# root_dir = os.path.join(script_dir, '..')
# dotenv_path = os.path.join(root_dir, '.env')
#
# print(f"Attempting to load .env file from: {dotenv_path}")
# load_dotenv(dotenv_path=dotenv_path)
# # --- End of .env loading ---
#
# from fastapi import FastAPI
# from pydantic import BaseModel, Field
#
# # --- LANGCHAIN IMPORTS ---
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
#
# # --- Check for API Keys ---
# if "GOOGLE_API_KEY" not in os.environ:
#     raise EnvironmentError("GOOGLE_API_KEY not set. Please check your .env file.")
#
# # --- 1. Define Worker Agent Endpoints ---
# RAG_AGENT_URL = "http://127.0.0.1:8000/query"
# ACTION_AGENT_URL = "http://127.0.0.1:8001/run-agent"
#
# # --- 2. Initialize the "Brain" of the Router ---
# llm_router = ChatGoogleGenerativeAI(
#     model="gemini-2.5-pro",
#     temperature=0
# )
# print(f"MCP Router LLM {llm_router.model} initialized.")
#
# # --- 3. Create the Routing Logic ---
# routing_prompt = ChatPromptTemplate.from_template(
#     """
# You are an expert router. Your job is to classify a user's query and decide which agent to send it to.
# You have two choices:
#
# 1.  'rag_agent': Use this for questions about company policies, product specifications, FAQs, or information found in internal documents.
# 2.  'action_agent': Use this for questions that require searching the company website, like finding products, checking for blog posts, or looking up dynamic information.
#
# User Query:
# "{input}"
#
# Which agent should handle this? (Return *only* 'rag_agent' or 'action_agent')
# """
# )
#
# routing_chain = routing_prompt | llm_router | StrOutputParser()
# print("MCP Routing chain created.")
#
# # --- NEW: Define RAG failure conditions ---
# # This is how we know the RAG agent "failed"
# # It's based on the prompt in `rag_agent/main.py`
# RAG_FAILURE_PHRASES = [
#     "i'm sorry, i don't have that information",
#     "i don't have that information",
#     "context doesn't contain the answer"
# ]
#
#
# def is_rag_failure(answer: str) -> bool:
#     """Helper to check if the RAG agent's answer is a failure."""
#     if not answer:
#         return True
#
#     lower_answer = answer.lower()
#     for phrase in RAG_FAILURE_PHRASES:
#         if phrase in lower_answer:
#             return True
#     return False
#
#
# # --- END OF NEW LOGIC ---
#
#
# # --- FastAPI Application ---
#
# app = FastAPI(
#     title="Genilia MCP (Mission Control Plane)",
#     description="The central router for the Genilia agent system.",
#     version="1.1.0"  # Upped version for new logic
# )
#
# # This client will manage all our API calls
# http_client = httpx.AsyncClient(timeout=30.0)
#
#
# class ChatRequest(BaseModel):
#     input: str
#
#
# @app.on_event("startup")
# async def startup_event():
#     print("HTTPX AsyncClient started.")
#
#
# @app.on_event("shutdown")
# async def shutdown_event():
#     await http_client.close()
#     print("HTTPX AsyncClient closed.")
#
#
# @app.get("/")
# def get_status():
#     return {"status": "ok", "message": "MCP Server is running!"}
#
#
# @app.post("/chat")
# async def chat_endpoint(request: ChatRequest):
#     """
#     The main user entry point.
#     It routes the query to the correct microservice,
#     with fallback logic for the RAG agent.
#     """
#     user_query = request.input
#     print(f"\n--- MCP RECEIVED QUERY: '{user_query}' ---")
#
#     # --- 1. Decide which agent to use FIRST ---
#     try:
#         print("Routing query...")
#         agent_to_use = await routing_chain.ainvoke({"input": user_query})
#         agent_to_use = agent_to_use.strip().replace("'", "")
#         print(f"Decision: First attempt with '{agent_to_use}'")
#
#     except Exception as e:
#         print(f"Error during routing: {e}")
#         return {"error": "Failed to route query."}, 500
#
#     # --- 2. Call the chosen agent ---
#     if agent_to_use == "rag_agent":
#         try:
#             print(f"Calling RAG Agent at: {RAG_AGENT_URL}")
#             response = await http_client.post(RAG_AGENT_URL, json={"question": user_query})
#             response.raise_for_status()
#
#             rag_json = response.json()
#             rag_answer = rag_json.get("answer", "")
#
#             # --- THIS IS THE NEW FALLBACK LOGIC ---
#             if is_rag_failure(rag_answer):
#                 print("RAG Agent failed. FALLING BACK to Action Agent.")
#                 agent_to_use = "action_agent"  # Change our plan
#             else:
#                 # RAG Succeeded! Return the answer.
#                 print("RAG Agent succeeded.")
#                 return rag_json
#             # --- END OF NEW LOGIC ---
#
#         except httpx.HTTPStatusError as e:
#             print(f"Error calling RAG agent: {e}. FALLING BACK to Action Agent.")
#             agent_to_use = "action_agent"  # Fallback on network error too
#         except Exception as e:
#             print(f"Error processing RAG response: {e}. FALLING BACK to Action Agent.")
#             agent_to_use = "action_agent"  # Fallback on processing error
#
#     # This block now runs if the first choice was 'action_agent'
#     # OR if the 'rag_agent' failed and set the fallback
#     if agent_to_use == "action_agent":
#         try:
#             print(f"Calling Action Agent at: {ACTION_AGENT_URL}")
#             response = await http_client.post(ACTION_AGENT_URL, json={"input": user_query})
#             response.raise_for_status()
#             return response.json()  # Forward the agent's response directly
#
#         except httpx.HTTPStatusError as e:
#             print(f"Error calling Action agent: {e}")
#             return {"error": "Action agent is unavailable or failed."}, 502
#         except Exception as e:
#             print(f"Error processing Action response: {e}")
#             return {"error": "Failed to process Action agent response."}, 500
#
#     else:
#         # This should rarely happen
#         print(f"Routing logic failed. Unknown agent: '{agent_to_use}'")
#         return {"error": f"Routing failed. Unknown agent '{agent_to_use}'."}
#
#
# # This allows us to run 'python main.py' directly
# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8002)


from dotenv import load_dotenv
import uvicorn
import os
import httpx  # The library for making API calls
from typing import List, Dict

# --- LANGCHAIN IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware  # <-- ADD THIS LINE


# --- Build a robust path to the .env file ---
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
root_dir = os.path.join(script_dir, '..')
dotenv_path = os.path.join(root_dir, '.env')

print(f"Attempting to load .env file from: {dotenv_path}")
load_dotenv(dotenv_path=dotenv_path)
# --- End of .env loading ---


# --- Check for API Keys ---
if "GOOGLE_API_KEY" not in os.environ:
    raise EnvironmentError("GOOGLE_API_KEY not set. Please check your .env file.")

# --- 1. Define Worker Agent Endpoints ---
RAG_AGENT_URL = "http://127.0.0.1:8000/query"
ACTION_AGENT_URL = "http://127.0.0.1:8001/run-agent"

# --- 2. Initialize LLMs ---
# We now have TWO LLM-powered chains
llm_router = ChatGoogleGenerativeAI(
    # model="gemini-2.5-pro",
    model="gemini-2.0-flash",
    temperature=0
)
llm_condenser = ChatGoogleGenerativeAI(
    # model="gemini-2.5-pro",
    model="gemini-2.0-flash",
    temperature=0
)
print(f"MCP LLMs {llm_router.model} initialized.")

# --- 3. Create the Routing Logic (No Change) ---
# --- THIS IS THE NEW, EXPLICIT PROMPT ---
routing_prompt = ChatPromptTemplate.from_template(
    """
You are an expert router. Your job is to classify a user's query and decide which agent to send it to.
You have two choices:

1.  'rag_agent': **This is the primary agent.** Use this for *all* questions related to products. This includes:
    - Product suggestions (e.g., "suggest a speaker")
    - Product specifications (e.g., "how big is the X-1000")
    - Product comparisons (e.g., "A vs B")
    - Company policies (e.g., "return policy")
    - FAQs

2.  'action_agent': **This is the secondary agent.** Use this *only* for general, non-product questions that require a web search, such as:
    - "What's new on the company blog?"
    - "Where is the company headquarters?"
    - "What's the company's contact info?"

User Query:
"{input}"

Which agent should handle this? (Return *only* 'rag_agent' or 'action_agent')
"""
)

routing_chain = routing_prompt | llm_router | StrOutputParser()
print("MCP Routing chain created.")

# --- 4. NEW: Create the Query Condensing Logic ---
condensing_prompt_template = """
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}

Follow Up Input: {input}
Standalone Question:"""

condensing_prompt = ChatPromptTemplate.from_template(condensing_prompt_template)
condensing_chain = condensing_prompt | llm_condenser | StrOutputParser()
print("MCP Query Condensing chain created.")

# --- 5. NEW: In-Memory Chat History ---
# This is our simple, non-production session database
# Key: session_id (str), Value: List[BaseMessage]
chat_histories: Dict[str, List] = {}


def get_chat_history(session_id: str):
    """Retrieves or creates a chat history for a session."""
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    return chat_histories[session_id]


def format_history_for_prompt(history: List):
    """Formats the history list for the prompt."""
    return "\n".join(
        [
            f"Human: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}"
            for msg in history
        ]
    )


# --- END NEW LOGIC ---


# --- 6. RAG Failure Logic (No Change) ---
RAG_FAILURE_PHRASES = [
    "i'm sorry, i don't have that information",
    "i don't have that information",
    "context doesn't contain the answer"
]


def is_rag_failure(answer: str) -> bool:
    if not answer: return True
    lower_answer = answer.lower()
    for phrase in RAG_FAILURE_PHRASES:
        if phrase in lower_answer:
            return True
    return False


# --- FastAPI Application ---

# ... (rest of your file) ...

app = FastAPI(
    title="Genilia MCP (Mission Control Plane)",
    description="The central router for the Genilia agent system.",
    version="1.2.0"
)

# --- NEW: ADD THIS ENTIRE BLOCK FOR CORS ---
# --- THIS IS THE NEW, FIXED LIST ---
origins = [
    "http://localhost",
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:8000",  # <-- ADD THIS LINE
    "http://127.0.0.1:8001",  # <-- ADD THIS LINE
    "http://127.0.0.1:8002",  # <-- ADD THIS LINE
    "http://localhost:8000",
    # <-- ADD THIS LINE (just in case)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # Which origins are allowed to talk to us
    allow_credentials=True,
    allow_methods=["*"],       # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],       # Allow all headers
)
# --- END OF NEW BLOCK ---

http_client = httpx.AsyncClient(timeout=30.0)
# ... (rest of your file) ...

# --- NEW: Updated ChatRequest model ---
class ChatRequest(BaseModel):
    input: str
    session_id: str = Field(default="default-session-id", description="Unique ID for the chat session")


@app.on_event("startup")
async def startup_event():
    print("HTTPX AsyncClient started.")


@app.on_event("shutdown")
async def shutdown_event():
    await http_client.close()
    print("HTTPX AsyncClient closed.")


@app.get("/")
def get_status():
    return {"status": "ok", "message": "MCP Server is running!"}

# --- NEW: Secret endpoint to clear chat memory ---
@app.get("/admin/clear-memory")
def clear_all_chat_history():
    """
    DANGER: Clears all in-memory chat histories for all sessions.
    """
    global chat_histories
    count = len(chat_histories)
    chat_histories.clear() # Empties the dictionary
    print(f"--- ADMIN: Cleared {count} chat session histories. ---")
    return {"status": "ok", "message": f"Cleared {count} session histories."}
# --- END OF NEW ENDPOINT ---

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    The main user entry point.
    It now handles chat history and condenses queries.
    """
    print(f"\n--- MCP RECEIVED QUERY for Session: '{request.session_id}' ---")
    print(f"Original Input: '{request.input}'")

    # --- 1. NEW: Get history and condense query ---
    try:
        chat_history = get_chat_history(request.session_id)

        if not chat_history:
            # If no history, the input is the standalone query
            condensed_query = request.input
            print("No history. Using input as standalone query.")
        else:
            # If there is history, condense it
            print("Condensing query with history...")
            formatted_history = format_history_for_prompt(chat_history)
            condensed_query = await condensing_chain.ainvoke({
                "chat_history": formatted_history,
                "input": request.input
            })
            print(f"Condensed Query: '{condensed_query}'")

    except Exception as e:
        print(f"Error during query condensing: {e}")
        return {"error": "Failed to process chat history."}, 500
    # --- END NEW LOGIC ---

    # --- 2. Decide which agent to use (uses the new condensed_query) ---
    try:
        print(f"Routing condensed query: '{condensed_query}'")
        agent_to_use = await routing_chain.ainvoke({"input": condensed_query})
        agent_to_use = agent_to_use.strip().replace("'", "")
        print(f"Decision: First attempt with '{agent_to_use}'")

    except Exception as e:
        print(f"Error during routing: {e}")
        return {"error": "Failed to route query."}, 500

    final_answer = ""  # We'll store the final answer here
    final_json_response = {}  # Store the final JSON to return

    # --- 3. Call the chosen agent (uses condensed_query) ---
    if agent_to_use == "rag_agent":
        try:
            print(f"Calling RAG Agent with: '{condensed_query}'")
            response = await http_client.post(RAG_AGENT_URL, json={"question": condensed_query})
            response.raise_for_status()

            final_json_response = response.json()
            final_answer = final_json_response.get("answer", "")

            if is_rag_failure(final_answer):
                print("RAG Agent failed. FALLING BACK to Action Agent.")
                agent_to_use = "action_agent"  # Change our plan
            else:
                print("RAG Agent succeeded.")

        except httpx.HTTPStatusError as e:
            print(f"Error calling RAG agent: {e}. FALLING BACK to Action Agent.")
            agent_to_use = "action_agent"  # Fallback on network error too
        except Exception as e:
            print(f"Error processing RAG response: {e}. FALLING BACK to Action Agent.")
            agent_to_use = "action_agent"  # Fallback on processing error

    if agent_to_use == "action_agent":
        try:
            print(f"Calling Action Agent with: '{condensed_query}'")
            response = await http_client.post(ACTION_AGENT_URL, json={"input": condensed_query})
            response.raise_for_status()

            final_json_response = response.json()
            final_answer = final_json_response.get("output", "")  # Action agent uses 'output'
            print("Action Agent succeeded.")

        except httpx.HTTPStatusError as e:
            print(f"Error calling Action agent: {e}")
            final_json_response = {"error": "Action agent is unavailable or failed."}
        except Exception as e:
            print(f"Error processing Action response: {e}")
            final_json_response = {"error": "Failed to process Action agent response."}

    else:
        # This block catches if RAG *succeeded* and we don't need to call the action agent
        if final_answer:  # Check if RAG already gave a good answer
            pass  # We're good, we'll just skip to saving the history
        else:
            print(f"Routing logic failed. Unknown agent: '{agent_to_use}'")
            final_json_response = {"error": f"Routing failed. Unknown agent '{agent_to_use}'."}

    # --- 4. NEW: Save to history and return ---
    if final_answer:
        # Add the user's *original* input and the AI's final answer to history
        chat_history.append(HumanMessage(content=request.input))
        chat_history.append(AIMessage(content=final_answer))
        print(f"Saved to history for session '{request.session_id}'")

    return final_json_response


# This allows us to run 'python main.py' directly
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)