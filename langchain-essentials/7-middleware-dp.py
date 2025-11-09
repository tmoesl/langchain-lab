"""
Middleware - Dynamic Prompting

Demonstrates dynamic prompt modification based on runtime context.
Middleware intercepts model requests to adjust prompts before model invocation.

Key Concepts:
- @dynamic_prompt decorator: Modifies system prompts at runtime
- ModelRequest: Contains current messages and runtime context
- Context-aware prompting: Different prompts for different user types
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
    print("🔍 Executing SQL query")
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
- Prefer explicit column lists; avoid SELECT *
- Output the result in clear simple language.
"""


@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    """Dynamically adjust system prompt based on user permissions."""

    if request.runtime.context.is_employee:
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


# Test the dynamic prompting
def test_dynamic_prompting():
    """Test the agent with different user contexts to demonstrate dynamic prompting."""

    test_cases = [
        {"user_type": "Employee", "is_employee": True},
        {"user_type": "Customer", "is_employee": False},
    ]

    message = "What is the most costly overall purchase by the customer Frank Harris?"

    for case in test_cases:
        print(f"\nTesting: {case['user_type']} Access\n")

        context = DataBase(is_employee=case["is_employee"], db=db)

        response = agent.stream(
            {"messages": [HumanMessage(content=message)]}, context=context, stream_mode="values"
        )

        # Debug the agent output
        for step in response:
            step["messages"][-1].pretty_print()


if __name__ == "__main__":
    test_dynamic_prompting()
