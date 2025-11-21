"""
LangChain Middleware - Dynamic Prompting

Middleware provides hooks to control agent execution. It allows you to monitor,
modify, and control the agent's behavior at each step of its reasoning process.

This file demonstrates a key "modify" capability: DYNAMIC PROMPTING.

HOW IT WORKS:
-------------
1. @dynamic_prompt Decorator: A simplified 'before_model' hook that intercepts
   the model request.
2. ModelRequest Object: Contains the agent's current state and runtime context,
   which the middleware can inspect.
3. Context-Aware Prompts: The middleware function returns a new system prompt
   tailored to the specific context of the current invocation.

KEY CONCEPTS:
-------------
- Hooks: Functions that run at specific points in the agent lifecycle (e.g.,
  before_model, after_model, before_tools, after_tools).
- Modification: Middleware can change prompts, tool selections, and outputs.
- Control: Middleware can add retries, fallbacks, and human-in-the-loop checks.
- Context Injection: The agent's behavior is dynamically shaped by the context
  provided at runtime (e.g., user permissions).

This example shows an agent whose SQL access rules change based on whether the
user is an "employee" or a "customer".

Reference: https://docs.langchain.com/oss/python/langchain/middleware
"""

from dataclasses import dataclass

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.tools import ToolRuntime, tool
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage

load_dotenv()

# Initialize the database
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# ==============================================================
# Context Schema
# ==============================================================


@dataclass
class DataBase:
    """Context containing user permissions and database access."""

    is_employee: bool
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
# Dynamic Prompting Middleware
# ==============================================================

SYSTEM_PROMPT_TEMPLATE = """You are a helpful SQLite analyst.

Rules:
- Think step-by-step.
- When you need data, call the tool `execute_sql` with ONE SELECT query.
- Read-only only; no INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/REPLACE/TRUNCATE.
- Limit to 5 rows of output unless the user explicitly asks otherwise.
- {table_access_restrictions}
- If the tool returns 'Error:', revise the SQL and try again.
- Always get the db schema and table names before querying.
- Prefer explicit column lists; avoid SELECT *
- Output the result in clear simple language.
"""


@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    """Dynamically adjust system prompt based on user permissions."""

    if request.runtime.context.is_employee:  # type: ignore
        table_access_restrictions = "You have full access to all database tables."
    else:
        table_access_restrictions = (
            "RESTRICTED: Access only these tables: Album, Artist, Genre, Playlist, Track."
        )

    return SYSTEM_PROMPT_TEMPLATE.format(table_access_restrictions=table_access_restrictions)


# ==============================================================
# Agent with Dynamic Prompting
# ==============================================================

agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[execute_sql],
    middleware=[dynamic_system_prompt],  # type: ignore
    context_schema=DataBase,
)

# Visualize the graph
display(Image(agent.get_graph().draw_mermaid_png()))

# ==============================================================
# Test Dynamic Prompting
# ==============================================================

message = "What is the most costly overall purchase by the customer Frank Harris?"

test_cases = [
    {"user_type": "Employee", "is_employee": True, "label": "Full table access"},
    {"user_type": "Customer", "is_employee": False, "label": "Restricted table access"},
]

for case in test_cases:
    print("\n" + "=" * 60)
    print(f"Testing: {case['user_type']} Access ({case['label']})")
    print("=" * 60 + "\n\n")

    context = DataBase(is_employee=case["is_employee"], db=db)
    response = agent.stream(
        {"messages": [HumanMessage(content=message)]},
        context=context,
        stream_mode="values",
    )

    for step in response:
        step["messages"][-1].pretty_print()
