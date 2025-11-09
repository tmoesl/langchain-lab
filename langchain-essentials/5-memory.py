"""
Memory and Persistence

Demonstrates agent memory using checkpointers for conversation persistence.
Agents remember context across multiple interactions using thread IDs.

Context injection: context_schema enables automatic runtime context injection
into tools via ToolRuntime[Schema]. Context provided at invoke time.
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

# Download Chinook database if needed
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

db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# ==============================================================
# Context and Tools
# ==============================================================


@dataclass
class DataBase:
    db: SQLDatabase


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
- Prefer explicit column lists; avoid SELECT *
- Output the result in clear simple language.
"""

agent = create_agent(
    model="openai:gpt-4o-mini",
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

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

print("🤖 SQL Agent with Memory - Ask questions about the Chinook database!")
print("💡 The agent will remember our conversation. Type 'exit' to quit.\n")

while True:
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
