from dotenv import load_dotenv
import uvicorn
import os
import operator
from typing import TypedDict, Annotated, List
from langchain.tools import tool

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
root_dir = os.path.join(script_dir, '..')
dotenv_path = os.path.join(root_dir, '.env')

print(f"Attempting to load .env file from: {dotenv_path}")
load_dotenv(dotenv_path=dotenv_path)

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

if "GOOGLE_API_KEY" not in os.environ:
    raise EnvironmentError("GOOGLE_API_KEY not set. Please check your .env file.")
if "TAVILY_API_KEY" not in os.environ:
    raise EnvironmentError("TAVILY_API_KEY not set. Please check your .env file.")

COMPANY_WEBSITES = [
]

_private_tavily_search = TavilySearch(max_results=5)


@tool
def company_website_search(query: str) -> str:
    print(f"---TOOL: Intercepted query: '{query}'---")

    site_queries = " OR ".join([f"site:{domain}" for domain in COMPANY_WEBSITES])

    final_query = f"{query} ({site_queries})"

    print(f"---TOOL: Executing modified query: '{final_query}'---")

    try:
        return _private_tavily_search.invoke(final_query)
    except Exception as e:
        print(f"Error during Tavily search: {e}")
        return f"Error: Could not perform search. {e}"


tools = [company_website_search]
print(f"Tools initialized: {[t.name for t in tools]}")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0
)

llm_with_tools = llm.bind_tools(tools)
print(f"LLM {llm.model} initialized and tools bound.")


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]


def call_model(state: AgentState):
    print("---NODE: CALLING MODEL---")
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    print(f"Model response: {response.tool_calls}")
    return {"messages": [response]}


def call_tool(state: AgentState):
    print("---NODE: CALLING TOOL---")
    last_message = state['messages'][-1]

    tool_results = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']

        print(f"Executing tool: {tool_name} with args {tool_args}")

        for tool in tools:
            if tool.name == tool_name:
                try:
                    result = tool.invoke(tool_args["query"])
                    tool_results.append(
                        ToolMessage(content=str(result), tool_call_id=tool_call['id'])
                    )
                except Exception as e:
                    print(f"Error executing tool: {e}")
                    tool_results.append(
                        ToolMessage(content=f"Error: {e}", tool_call_id=tool_call['id'])
                    )
                break

    return {"messages": tool_results}


def should_continue(state: AgentState) -> str:
    print("---EDGE: CHECKING FOR TOOL CALLS---")
    last_message = state['messages'][-1]
    if not last_message.tool_calls:
        print("Decision: END")
        return "end"
    else:
        print("Decision: call_tool")
        return "call_tool"


print("Building graph...")
graph_builder = StateGraph(AgentState)

graph_builder.add_node("call_model", call_model)
graph_builder.add_node("call_tool", call_tool)

graph_builder.add_edge(START, "call_model")
graph_builder.add_conditional_edges(
    "call_model",
    should_continue,
    {"call_tool": "call_tool", "end": END}
)
graph_builder.add_edge("call_tool", "call_model")

app_graph = graph_builder.compile()
print("Graph compiled. Ready to serve.")

app = FastAPI(
    title="Genilia Action Agent (Site-Search-Only)",
    description="Microservice using LangGraph for *only* company website search.",
    version="0.3.0"
)


class AgentRequest(BaseModel):
    input: str


@app.get("/")
def get_status():
    return {"status": "ok", "message": "Action Agent (Site-Search-Only) is running!"}


@app.post("/run-agent")
def run_agent(request: AgentRequest):
    print(f"\nReceived request: {request.input}")

    try:
        initial_state = {
            "messages": [
                SystemMessage(
                    content="You are a helpful customer support assistant. Your job is to answer the user's question by searching the websites url provided. "
                            "--- "
                            "RESPONSE FORMATTING RULES: "
                            "1. Always use Markdown to format your answers. "
                            "2. Use bullet points (e.g., * Item 1, * Item 2) for lists of products or specifications. "
                            "3. Use bolding (e.g., **Product Name**) to highlight key items. "
                            "4. If you are providing search results, list the top 3-5 relevant ones. "
                            "5. Be clear, concise, and professional."),
                HumanMessage(content=request.input)
            ]
        }

        response = app_graph.invoke(initial_state)

        final_answer = response['messages'][-1].content
        print(f" {final_answer}")

        return {"input": request.input, "output": final_answer}

    except Exception as e:
        print(f"Error during graph invocation: {e}")
        return {"error": f"An error occurred: {e}"}, 500


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
