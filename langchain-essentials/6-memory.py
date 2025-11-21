"""
LangChain Short-Term Memory - Thread-Level Conversation Persistence

Short-term memory enables agents to remember previous interactions within a single
thread or conversation. Memory is stored in agent state and persisted via checkpointers,
allowing conversations to be paused and resumed at any time.

How It Works:
-------------
1. Memory stored in agent state (state["messages"])
2. Checkpointer persists state to database/memory
3. Thread ID identifies which conversation to resume
4. State updates after each invocation and tool call

Checkpointer Types:
-------------------
1. InMemorySaver - Development (data lost on restart)
2. PostgresSaver - Production (persistent database storage)

Thread Usage:
-------------
- config = {"configurable": {"thread_id": "1"}}
- Same thread_id = same conversation
- Different thread_id = separate conversation

Accessing Memory:
-----------------
- Tools: runtime.state["messages"] to read conversation history
- Middleware: @before_model, @after_model for state processing
- Prompts: @dynamic_prompt for dynamic system prompts
- Custom State: Extend AgentState with state_schema parameter

Key Concepts:
-------------
- Thread organizes multiple interactions in a session
- Long conversations may exceed LLM context window
- Common solutions: trim messages, delete messages, summarize messages
- State persists across invocations within same thread

Reference: https://docs.langchain.com/oss/python/langchain/short-term-memory
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


@tool
def get_token_usage(runtime: ToolRuntime[DataBase]) -> str:
    """Get cumulative token usage for this conversation."""
    messages = runtime.state["messages"]

    total_input_tokens = 0
    total_output_tokens = 0

    for msg in messages:
        # AIMessages contain usage_metadata
        if (
            msg.__class__.__name__ == "AIMessage"
            and hasattr(msg, "usage_metadata")
            and msg.usage_metadata
        ):
            usage = msg.usage_metadata
            total_input_tokens += usage.get("input_tokens", 0)
            total_output_tokens += usage.get("output_tokens", 0)

    if total_input_tokens == 0 and total_output_tokens == 0:
        return "No token usage data available yet."

    total_tokens = total_input_tokens + total_output_tokens
    return f"""Token Usage:
    - Input tokens: {total_input_tokens:,}\n
    - Output tokens: {total_output_tokens:,}\n
    - Total tokens: {total_tokens:,}\n"""


# ==============================================================
# Agent with Memory
# ==============================================================

SYSTEM_PROMPT = """You are a helpful SQLite analyst.

Rules:
- Think step-by-step.
- When you need data, call the tool `execute_sql` with ONE SELECT query.
- When you need to know the token usage, call the tool `get_token_usage`.
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
    tools=[execute_sql, get_token_usage],
    system_prompt=SYSTEM_PROMPT,
    context_schema=DataBase,
    checkpointer=InMemorySaver(),  # Enables memory/persistence
)

# Visualize the graph
display(Image(agent.get_graph().draw_mermaid_png()))

# ==============================================================
# Interactive Chat with Memory
# ==============================================================

# Test the agent
example_questions = [
    "Which table has the largest number of entries?",
    "Which genre on average has the longest tracks?",
    "Please list all of the tables",
    "Find the artist with the most tracks",
]
print("🤖 SQL Agent with Memory")
print(f"Ask questions about the Chinook database: {'\n- ' + '\n- '.join(example_questions)}\n")
print("💡 The agent will remember our conversation. Type 'exit' to quit.")
print("📝 Note: Change thread_id to '2' for a separate conversation.\n")

# Thread ID tracks conversation
config: RunnableConfig = {"configurable": {"thread_id": "1"}}

while True:
    user_input = input("Enter: ")
    if user_input.lower() in ["exit", "quit"]:
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
