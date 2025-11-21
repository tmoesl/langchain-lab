"""
LangChain Tools - Extending Agent Capabilities with External Actions

Tools enable agents to interact with external systems (APIs, databases, file systems)
using structured input. They encapsulate a callable function and its input schema,
allowing models to generate requests that conform to specified formats.

Tool Definition Approaches:
----------------------------
1. Basic - @tool decorator with type hints and docstring
2. Custom properties - Override name and description
3. Enhanced schema - parse_docstring=True for Google-style arg docs
4. Pydantic validation - args_schema for complex input validation
5. Context injection - ToolRuntime for accessing state and context

Key Concepts:
-------------
- Type hints are REQUIRED (define input schema)
- Docstrings help models understand when to use tools
- Tool name defaults to function name (can override)
- Description defaults to docstring (can override)
- ToolRuntime parameter is hidden from LLM (automatically injected)

Reference: https://docs.langchain.com/oss/python/langchain/tools
"""

from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

load_dotenv()

# ==============================================================
# Basic Tool Definition
# ==============================================================
# Minimal setup: type hints + docstring


@tool
def calculator(a: float, b: float, operation: str) -> float:
    """Perform basic arithmetic operations on two numbers."""
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    else:
        raise ValueError(f"Unsupported operation: {operation}")


agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[calculator],
    system_prompt="You are a helpful math assistant. Use the tool if applicable.",
)

message = HumanMessage(content="What is 7 times 6?")
response = agent.invoke({"messages": [message]})
print(f"Basic Tool Result: {response['messages'][-1].content}\n")

# ==============================================================
# Custom Tool Name and Description
# ==============================================================


@tool("multiply", description="Multiply two numbers together. Use this for any multiplication.")
def mult(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[mult],
    system_prompt="You are a helpful math assistant. Use the tool if applicable.",
)

message = HumanMessage(content="Calculate 12 times 8")
response = agent.invoke({"messages": [message]})
print(f"Custom Name/Description: {response['messages'][-1].content}\n")

# ==============================================================
# Enhanced Schema with parse_docstring=True
# ==============================================================


@tool(parse_docstring=True)
def advanced_calculator(a: float, b: float, operation: Literal["add", "multiply"]) -> float:
    """Perform arithmetic operations with detailed documentation.

    Args:
        a (float): The first number
        b (float): The second number
        operation (Literal["add", "multiply"]): The arithmetic operation to perform

            - "add": Returns the sum of a and b.
            - "multiply": Returns the product of a and b.

    Returns:
        float: The result of the arithmetic operation

    Raises:
        ValueError: If an unsupported operation is provided
    """
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    else:
        raise ValueError(f"Unsupported operation: {operation}")


agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[advanced_calculator],
    system_prompt="You are a helpful math assistant. Use the tool if applicable.",
)

message = HumanMessage(content="Add 15 and 27")
response = agent.invoke({"messages": [message]})
print(f"Enhanced Schema: {response['messages'][-1].content}\n")

# ==============================================================
# Pydantic Validation with args_schema
# ==============================================================


class CalculatorInput(BaseModel):
    """Input schema for calculator with validation."""

    a: float = Field(description="First number", ge=-1000, le=1000)
    b: float = Field(description="Second number", ge=-1000, le=1000)
    operation: Literal["add", "multiply"] = Field(description="Operation to perform")


@tool(args_schema=CalculatorInput)
def validated_calculator(a: float, b: float, operation: str) -> float:
    """Perform arithmetic with input validation."""
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    else:
        raise ValueError(f"Unsupported operation: {operation}")


agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[validated_calculator],
    system_prompt="You are a helpful math assistant. Use the tool if applicable.",
)

message = HumanMessage(content="What is 25 plus 75?")
response = agent.invoke({"messages": [message]})
print(f"Pydantic Validation: {response['messages'][-1].content}\n")

# ==============================================================
# Context Injection: Accessing State
# ==============================================================
# Access conversation state via runtime.state


@tool
def count_messages(runtime: ToolRuntime) -> str:
    """Count how many messages are in the conversation."""
    messages = runtime.state["messages"]

    human_count = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
    ai_count = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")

    return f"Conversation has {human_count} user messages and {ai_count} AI responses"


agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[count_messages],
    system_prompt="You are a helpful math assistant. Use the tool if applicable.",
)

messages = [
    HumanMessage(content="Hello!"),
    AIMessage(content="Hi! How can I help you today?"),
    HumanMessage(content="I'm looking for a restaurant in New York."),
    AIMessage(content="One of the best restaurants in New York is Le Bernardin in Midtown."),
]

response = agent.invoke({"messages": messages + [HumanMessage(content="Count our messages")]})
print(f"State Access: {response['messages'][-1].content}\n")

# ==============================================================
# Context Injection: Accessing Context
# ==============================================================
# Access immutable configuration via runtime.context

USER_DATABASE = {
    "alice": {"name": "Alice Johnson", "balance": 5000, "tier": "Premium"},
    "bob": {"name": "Bob Smith", "balance": 1200, "tier": "Standard"},
}


@dataclass
class UserContext:
    user_id: str


@tool
def get_account_balance(runtime: ToolRuntime[UserContext]) -> str:
    """Get the current user's account balance."""
    user_id = runtime.context.user_id

    if user_id in USER_DATABASE:
        user = USER_DATABASE[user_id]
        return f"Account: {user['name']}\nTier: {user['tier']}\nBalance: ${user['balance']}"
    return "User not found"


agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[get_account_balance],
    context_schema=UserContext,
    system_prompt="You are a helpful math assistant. Use the tool if applicable.",
)

response = agent.invoke(
    {"messages": [HumanMessage(content="What's my account balance?")]},
    context=UserContext(user_id="alice"),
)
print(f"Context Access: {response['messages'][-1].content}")
