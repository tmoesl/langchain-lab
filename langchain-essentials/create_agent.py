from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv
from IPython.display import Image, display
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage

load_dotenv()


# Download the database file if it doesn't exist
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


# Define the data structure for the database
@dataclass
class DataBase:
    db: SQLDatabase


# Define the tool to execute SQL queries
@tool
def execute_sql(query: str, runtime: ToolRuntime[DataBase]):
    """Execute a SQLite command and return results."""
    db = runtime.context.db
    try:
        return db.run(query)
    except Exception as e:
        return f"Error executing SQL query: {e}"


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

# Create the agent
agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[execute_sql],
    system_prompt=SYSTEM_PROMPT,
    context_schema=DataBase,
)

# Visualize the graph
display(Image(agent.get_graph().draw_mermaid_png()))

# Test the agent
messages = [
    "Which table has the largest number of entries?",
    "Which genre on average has the longest tracks?",
    "Please list all of the tables",
    "Find the artist with the most tracks",
]

message = messages[3]
response = agent.stream(
    {"messages": [HumanMessage(content=message)]},
    context=DataBase(db=db),
    stream_mode="values",
)

# Debug the agent
for step in response:
    step["messages"][-1].pretty_print()


# Key Notes:
# - agent.stream() returns a lazy stream, consumed step-by-step in the loop
# - Agent discovers database schema independently (no pre-loaded schema)
# - Error messages enable self-correction of SQL queries
# - Agent doesn't retain schema knowledge between invocations
