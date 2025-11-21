"""
LangChain Structured Output - Predictable Data Format from Agents

Returns data in a specific format (Pydantic, TypedDict, dataclass, JSON Schema)
instead of natural language. Result stored in state["structured_response"].

Key Concepts:
-------------
- Schema type (recommended): Auto-selects best strategy for your model
- ProviderStrategy: Native structured output (OpenAI, Grok) with automatic validation
- ToolStrategy: Tool calling approach (all models) with custom error handling
- Union types: Model chooses appropriate schema dynamically

Supported Schemas:
------------------
Pydantic models, TypedDict, dataclasses, JSON Schema

Error Handling:
---------------
ONLY available with ToolStrategy (NOT ProviderStrategy).

handle_errors parameter options:
- True: Catch all errors with default template (default)
- str: Custom error message for the model
- type[Exception]: Catch specific exception only
- tuple[type[Exception], ...]: Catch multiple exception types
- Callable[[Exception], str]: Custom error handler function
- False: No retry, propagate exceptions

ProviderStrategy handles validation/retries automatically via provider API.

Reference: https://docs.langchain.com/oss/python/langchain/structured-output
"""

from typing import Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

load_dotenv()

# ==============================================================
# Schema Definitions
# ==============================================================
# Note: This example uses Pydantic models (most common).
# LangChain also supports: Dataclass, TypedDict, JSON Schema.
# All work the same way with response_format parameter.


class ContactInfo(BaseModel):
    """Contact information for a person."""

    name: str = Field(description="Person's name")
    email: str = Field(description="Email address")
    phone: str = Field(description="Phone number")


class EventInfo(BaseModel):
    """Event information."""

    event_name: str = Field(description="Name of the event")
    date: str = Field(description="Event date")
    location: str = Field(description="Event location")


class MeetingAction(BaseModel):
    """Action items from meeting transcript."""

    task: str = Field(description="Task to be completed")
    assignee: str = Field(description="Person responsible")
    priority: Literal["low", "medium", "high"] = Field(description="Priority level")


# ==============================================================
# Test Data
# ==============================================================

contact_message = "John Smith from ABC Corp, email: john@abc.com, phone: 555-0123"
event_message = "Tech Conference on March 15th, 2024 at Convention Center in Sydney"
meeting_message = "Sarah needs to update the project timeline within the next 2 days"

# ==============================================================
# Basic Usage: Auto-select Strategy
# ==============================================================

print("=" * 60)
print("Basic Usage (Auto-select Strategy)")
print("=" * 60)

agent = create_agent(
    model="openai:gpt-5-mini",
    system_prompt="Extract contact information from the text.",
    response_format=ContactInfo,  # Auto-selects best strategy
)

result = agent.invoke({"messages": [HumanMessage(content=contact_message)]})
contact = result["structured_response"]

print(f"Name: {contact.name}")
print(f"Email: {contact.email}")
print(f"Phone: {contact.phone}")
print(f"Type: {type(contact).__name__}\n")


# ==============================================================
# Basic Usage: Auto-select Strategy with Multiple Schemas
# ==============================================================

print("=" * 60)
print("Basic Usage (Auto-select Strategy with Multiple Schemas)")
print("=" * 60)

agent = create_agent(
    model="openai:gpt-5-mini",
    system_prompt="Extract the relevant information from the text.",
    response_format=ContactInfo | EventInfo,  # type: ignore[arg-type]
)

# Test with contact info
result = agent.invoke({"messages": [HumanMessage(content=contact_message)]})
print(f"Contact → {type(result['structured_response']).__name__}")
print(f"Structured response → {result['structured_response']}")

# Test with event info
result = agent.invoke({"messages": [HumanMessage(content=event_message)]})
print(f"\nEvent → {type(result['structured_response']).__name__}")
print(f"Structured response → {result['structured_response']}\n")

# ==============================================================
# ProviderStrategy: Native Structured Output
# ==============================================================
# Note: ProviderStrategy uses the model provider's native structured output API.
# Validation and retries are handled automatically by the provider (e.g., OpenAI).

print("=" * 60)
print("ProviderStrategy (Native Structured Output)")
print("=" * 60)

agent_provider = create_agent(
    model="openai:gpt-5-mini",
    system_prompt="Extract contact information.",
    response_format=ProviderStrategy(ContactInfo),
)
result = agent_provider.invoke({"messages": [HumanMessage(content=contact_message)]})
print(f"Structured response → {result['structured_response']}\n")

print("\nMessage history:")
for msg in result["messages"]:
    msg.pretty_print()


# ==============================================================
# ToolStrategy: Tool Calling Approach
# ==============================================================
# Note: ToolStrategy converts the schema into a tool that the model can call.
# This works with any model that supports tool calling and allows custom error handling.

print("=" * 60)
print("ToolStrategy (Tool Calling)")
print("=" * 60)

agent_tool = create_agent(
    model="openai:gpt-5-mini",
    system_prompt="Extract contact information.",
    response_format=ToolStrategy(
        schema=ContactInfo,
        tool_message_content="✓ Contact information captured and added to database!",
        handle_errors=True,  # default
    ),
)
result = agent_tool.invoke({"messages": [HumanMessage(content=contact_message)]})
print(f"Structured response → {result['structured_response']}")

print("\nMessage history:")
for msg in result["messages"]:
    msg.pretty_print()


# ==============================================================
# ToolStrategy: Error Handling Configuration
# ==============================================================

print("\n" + "=" * 60)
print("ToolStrategy (Error Handling)")
print("=" * 60)

agent_with_error_handling = create_agent(
    model="openai:gpt-5-mini",
    system_prompt="Extract meeting action items.",
    response_format=ToolStrategy(
        schema=MeetingAction,
        handle_errors="Please ensure priority is exactly 'low', 'medium', or 'high'.",
    ),
)

result = agent_with_error_handling.invoke({"messages": [HumanMessage(content=meeting_message)]})
print(f"Structured response → {result['structured_response']}")

print("\nMessage history:")
for msg in result["messages"]:
    msg.pretty_print()
