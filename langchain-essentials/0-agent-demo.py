"""
LangChain Agent Quickstart - SQL Database Assistant

ReAct (Reasoning & Acting) agent built on LangGraph. Uses a single SQL tool
to discover database schema dynamically and self-correct queries via error feedback.
This example demonstrates the core capabilities: tools, memory, streaming, and runtime context.

Key Concepts:
-------------
- create_agent: Build agents with model, tools, system prompt
- InMemorySaver: Enable memory persistence across conversations
- context_schema: Inject dependencies (e.g., database) into tools
- agent.stream(): Stream responses step-by-step for real-time updates
- thread_id: Separate conversation threads with independent memory

What the Agent Does:
--------------------
- Discovers database schema dynamically (no pre-loaded schema needed)
- Self-corrects SQL queries based on error messages
- Remembers conversation history within each thread
- Streams progress updates for low-latency responses

Runtime Context:
----------------
Tools receive runtime context via ToolRuntime[Context] parameter.
Pass context during invoke: agent.invoke(..., context=DataBase(db=db))

Reference: https://docs.langchain.com/oss/python/langchain/quickstart
"""

from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv
from IPython.display import Image, display
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# ==============================================================
# Database Setup
# ==============================================================

# Download the Chinook database if needed
url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
local_path = Path("Chinook.db")

if local_path.exists():
    print(f"{local_path} already exists, skipping download.")
else:
    response = requests.get(url)
    if response.status_code == 200:
        local_path.write_bytes(response.content)
        print(f"File downloaded and saved as {local_path}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

# Initialize the database
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# ==============================================================
# Context Schema
# ==============================================================


@dataclass
class DataBase:
    """Context containing database access."""

    db: SQLDatabase


# ==============================================================
# Tools
# ==============================================================


@tool
def execute_sql(query: str, runtime: ToolRuntime[DataBase]):
    """Execute a SQLite command and return results."""
    db = runtime.context.db
    try:
        return db.run(query)
    except Exception as e:
        return f"Error executing SQL query: {e}"


# ==============================================================
# Agent with Memory
# ==============================================================

SYSTEM_PROMPT = """You are a helpful SQLite analyst.

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

# Initialize the agent
agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[execute_sql],
    system_prompt=SYSTEM_PROMPT,
    context_schema=DataBase,
    checkpointer=InMemorySaver(),  # Enables memory/persistence
)

# Visualize the graph
display(Image(agent.get_graph().draw_mermaid_png()))

# ==============================================================
# Interactive Chat with Memory
# ==============================================================

print("=" * 70)
print("🤖LangChain Agent - SQL Database Assistant with Memory")
print("=" * 70)

example_questions = [
    "Which table has the largest number of entries?",
    "Which genre on average has the longest tracks?",
    "Please list all of the tables",
    "Find the artist with the most tracks",
]

print(f"Ask questions about the Chinook database:\n- {' \n- '.join(example_questions)}\n")
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
        response = agent.stream(
            {"messages": message},  # type: ignore
            config=config,
            context=DataBase(db=db),
            stream_mode="values",
        )

        for step in response:
            step["messages"][-1].pretty_print()
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Error: {e}")

# Key Notes:
# - agent.stream() returns a lazy stream, consumed step-by-step in the loop
# - Agent discovers database schema independently (no pre-loaded schema)
# - Error messages enable self-correction of SQL queries
# - Schema knowledge persists within each thread via checkpointer
