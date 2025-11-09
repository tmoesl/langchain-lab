"""
Tools
Demonstrates three approaches to defining and using tools in LangChain.

1. Basic: Minimal setup using type hints only.
2. Documented: Rich docstring parsing with Google-style formatting for improved model understanding.
3. Validated: Pydantic schema-based input validation for production-grade reliability.

Use Cases
- Basic: Quick prototyping
- Documented: Better LLM understanding
- Validated: Production use
"""

from typing import Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

load_dotenv()

# ==============================================================
# Approach 1: Basic Tool (Minimal)
# ==============================================================


@tool
def basic_calculator(a: float, b: float, operation: Literal["add", "multiply"]) -> float:
    """Basic arithmetic operations."""
    print("🧮 Basic calculator called")
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    else:
        raise ValueError(f"Unsupported operation: {operation}")


# ==============================================================
# Approach 2: Documented Tool (parse_docstring=True)
# ==============================================================


@tool(parse_docstring=True)
def documented_calculator(a: float, b: float, operation: Literal["add", "multiply"]) -> float:
    """Performs arithmetic operations with detailed documentation.

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
    print("📚 Documented calculator called")
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    else:
        raise ValueError(f"Unsupported operation: {operation}")


# ==============================================================
# Approach 3: Validated Tool (args_schema)
# ==============================================================


class ValidatedCalculatorSchema(BaseModel):
    a: float = Field(description="The first number", ge=-1000, le=1000)
    b: float = Field(description="The second number", ge=-1000, le=1000)
    operation: Literal["add", "multiply"] = Field(description="The arithmetic operation to perform")


@tool(args_schema=ValidatedCalculatorSchema)
def validated_calculator(a, b, operation) -> float:
    """Performs arithmetic operations with Pydantic validation.

    This calculator includes input validation to ensure safe operations.
    Numbers are constrained to the range [-1000, 1000].
    """
    print("✅ Validated calculator called")
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    else:
        raise ValueError(f"Unsupported operation: {operation}")


# ==============================================================
# Test All Approaches
# ==============================================================


def test_tool_approaches():
    """Test each tool approach with the same input."""

    tools_to_test = [
        ("Basic Tool", [basic_calculator]),
        ("Documented Tool", [documented_calculator]),
        ("Validated Tool", [validated_calculator]),
    ]

    message = HumanMessage(content="What is 7 times 6?")

    for approach_name, tools in tools_to_test:
        print(f"\n{'=' * 50}")
        print(f"Testing: {approach_name}")
        print(f"{'=' * 50}")

        agent = create_agent(
            model="openai:gpt-4o-mini",
            tools=tools,
            system_prompt="You are a helpful math assistant. Use the available calculator tool.",
        )

        response = agent.invoke({"messages": [message]})
        print(f"Assistant: {response['messages'][-1].content}")


if __name__ == "__main__":
    test_tool_approaches()
