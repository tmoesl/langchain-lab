"""
LangGraph Application - SQL Database Assistant

Demonstrates building a ReAct (Reasoning & Acting) agent from scratch using LangGraph
to interact with a SQLite database. This example shows how to build an agent that
dynamically discovers database schema, generates SQL queries, self-corrects errors,
and maintains conversation history across multiple turns.

Key Concepts:
-------------
- StateGraph: Manages the flow between reasoning (Agent) and acting (Tools)
- ToolNode: Prebuilt node for executing tool calls from the LLM
- Runtime Context: Inject database connection into tools via ToolRuntime[Context]
- System Prompt: Guide agent behavior without persisting in state
- Conditional Routing: Route between tool execution and final response
- Memory/Persistence: Checkpointer saves conversation history and tool results
- Thread ID: Maintains separate conversation contexts

Application Architecture:
-------------------------
This SQL agent implements a simple but powerful ReAct workflow:

1. Agent Node (Reasoning)
   ├─ Prepends system prompt to message history
   ├─ Invokes LLM with tools (list_tables, get_schema, execute_sql)
   ├─ LLM decides: call tools or respond to user
   └─ Routes to tools or END based on decision

2. Tools Node (Acting)
   ├─ Executes tool calls in parallel if multiple requested
   ├─ Returns results as ToolMessage objects
   └─ Returns control to Agent for next reasoning step

3. Agent Node (Synthesis)
   └─ Generates final response using tool results and context

Tools Available:
----------------
- list_tables: Discover all tables in the database
- get_schema: Inspect table structure and sample rows
- execute_sql: Run SELECT queries (read-only, safety enforced)

Design Pattern:
---------------
The system prompt is prepended fresh on each agent call and never enters
the persistent state. This follows the create_agent pattern from LangChain,
ensuring the prompt stays current without accumulating in message history.

Reference:
---------
- https://docs.langchain.com/oss/python/langgraph/sql-agent
- https://docs.langchain.com/oss/python/langgraph/overview
- https://docs.langchain.com/oss/python/langgraph/workflows-agents
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypedDict

import requests
from dotenv import load_dotenv
from IPython.display import Image, display
from langchain.tools import ToolRuntime, tool
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()


# ==============================================================
# Database Setup
# ==============================================================

# Download Chinook database if not exists
url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
local_path = Path("Chinook.db")

if not local_path.exists():
    print(f"Downloading {local_path}...")
    response = requests.get(url)
    if response.status_code == 200:
        local_path.write_bytes(response.content)
        print("Download complete.")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

# Initialize database connection
db = SQLDatabase.from_uri("sqlite:///Chinook.db")


# ==============================================================
# Context Schema
# ==============================================================


@dataclass
class DataBase:
    """Context containing database access."""

    db: SQLDatabase


# ==============================================================
# Define Tools
# ==============================================================


@tool
def list_tables(runtime: ToolRuntime[DataBase]) -> str:
    """List all table names in the database."""
    db = runtime.context.db
    return ", ".join(db.get_usable_table_names())


@tool
def get_schema(table_name: str, runtime: ToolRuntime[DataBase]) -> str:
    """Get the schema and sample rows for a specific table.

    Args:
        table_name (str): The name of the table to inspect
    """
    db = runtime.context.db
    return db.get_table_info([table_name])


@tool
def execute_sql(query: str, runtime: ToolRuntime[DataBase]):
    """Execute a SQL SELECT query and return the results.

    Args:
        query (str): The SQL SELECT query to execute
    """
    db = runtime.context.db
    try:
        return db.run(query)  # type: ignore
    except Exception as e:
        return f"Error executing SQL: {e}"


tools = [list_tables, get_schema, execute_sql]


# ==============================================================
# Define State
# ==============================================================


class AgentState(TypedDict):
    """State of the agent with message history."""

    messages: Annotated[list[BaseMessage], add_messages]


# ==============================================================
# Define Nodes
# ==============================================================


def agent_node(state: AgentState) -> dict:
    """
    Invokes the model to generate a response or tool call.
    """
    model = ChatOpenAI(model="gpt-5-mini", temperature=0)
    model_with_tools = model.bind_tools(tools)

    system_msg = SystemMessage(
        content="""You are a helpful SQLite analyst
        
        Rules:
        - Think step-by-step.
        - When you need data, call the tool `execute_sql` with ONE SELECT query.
        - Read-only only; no INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/REPLACE/TRUNCATE.
        - Limit to 5 rows of output unless the user explicitly asks otherwise.
        - If the tool returns 'Error:', revise the SQL and try again.
        - Always get the db schema and table names before querying.
        - Prefer explicit column lists; avoid SELECT *
        - Output the result in clear simple language.
        """
    )

    # Filter out system messages to avoid duplication if persisting state
    messages = [msg for msg in state["messages"] if not isinstance(msg, SystemMessage)]
    messages = [system_msg] + messages
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


# ==============================================================
# Define Routing Functions
# ==============================================================


def should_continue(state: AgentState) -> Literal["tools", END]:  # type: ignore
    """
    Determine whether to execute tools or end the conversation.
    """
    last_message = state["messages"][-1]

    if last_message.tool_calls:  # type: ignore
        return "tools"

    return END


# ==============================================================
# Build Graph
# ==============================================================

# Initialize checkpointer for state persistence
memory = InMemorySaver()

# Initialize the graph
graph = StateGraph(AgentState, context_schema=DataBase)

# Add nodes
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))

# Add static and conditional edges
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
graph.add_edge("tools", "agent")

# Compile graph
app = graph.compile(checkpointer=memory)

# Visualize graph
display(Image(app.get_graph().draw_mermaid_png()))


# ==============================================================
# Run Application
# ==============================================================


def run_app():
    """Run the interactive SQL agent."""

    print("=" * 70)
    print("🤖LangGraph Agent - SQL Database Assistant with Memory")
    print("=" * 70)

    example_questions = [
        "Which table has the largest number of entries?",
        "Which genre on average has the longest tracks?",
        "Please list all of the tables",
        "Find the artist with the most tracks",
    ]

    print(f"\nAsk questions about the Chinook database:\n- {' \n- '.join(example_questions)}\n")
    print("💡 The agent will remember our conversation. Type 'exit' to quit.")
    print("📝 Note: Change thread_id to '2' for a separate conversation.\n")

    # Thread ID tracks conversation
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    while True:
        try:
            user_input = input("Enter: ")
            if user_input.lower() == "exit":
                break

            message = [HumanMessage(content=user_input)]
            events = app.stream(
                {"messages": message},  # type: ignore
                config=config,
                context=DataBase(db=db),
                stream_mode="values",
            )

            for event in events:
                event["messages"][-1].pretty_print()

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    run_app()
