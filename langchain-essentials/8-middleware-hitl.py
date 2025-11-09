from dataclasses import dataclass

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import ToolRuntime, tool
from langchain_community.utilities import SQLDatabase
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()

# ==============================================================
# Database Setup
# ==============================================================

db = SQLDatabase.from_uri("sqlite:///Chinook.db")


# ==============================================================
# Context Schema
# ==============================================================
@dataclass
class DataBase:
    db: SQLDatabase


# ==============================================================
# Tools
# ==============================================================


# Define the tool to execute SQL queries
@tool
def execute_sql(query: str, runtime: ToolRuntime[DataBase]):
    """Execute a SQLite command and return results."""
    db = runtime.context.db
    try:
        return db.run(query)
    except Exception as e:
        return f"Error executing SQL query: {e}"


# ==============================================================
# Agent with Human-in-the-Loop
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
- If tools rejected, provide a single line of explanation for the rejection.
"""

memory = InMemorySaver()
config = {"configurable": {"thread_id": "1"}}

# Create the agent
agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[execute_sql],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=memory,
    context_schema=DataBase,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"execute_sql": {"allowed_decisions": ["approve", "reject"]}}  # type: ignore
        )
    ],
)

# Visualize the graph
display(Image(agent.get_graph().draw_mermaid_png()))


# ==============================================================
# Test Human-in-the-Loop Scenarios
# ==============================================================


def reject_scenario():
    """Test reject scenario."""

    config = {"configurable": {"thread_id": "1"}}

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "What are the names of all the employees?"}]},
        config=config,  # type: ignore
        context=DataBase(db=db),
    )

    if "__interrupt__" in response:
        action = response["__interrupt__"][-1].value["action_requests"][-1]

        print(f"\033[1;3;31mInterrupt:{action['description']}\033[0m\n")

        response = agent.invoke(
            Command(
                resume={"decisions": [{"type": "reject", "message": "the database is offline."}]}
            ),
            config=config,  # type: ignore
            context=DataBase(db=db),
        )

    print(f"Assistant: {response['messages'][-1].content}")


def approve_scenario():
    """Test approve scenario."""

    config = {"configurable": {"thread_id": "2"}}

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "What are the names of all the employees?"}]},
        config=config,  # type: ignore
        context=DataBase(db=db),
    )

    while "__interrupt__" in response:
        action = response["__interrupt__"][-1].value["action_requests"][-1]

        print(f"\033[1;3;31mInterrupt:{action['description']}\033[0m\n")

        response = agent.invoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config=config,  # type: ignore
            context=DataBase(db=db),
        )
    print(f"Assistant: {response['messages'][-1].content}")


if __name__ == "__main__":
    reject_scenario()
    approve_scenario()
