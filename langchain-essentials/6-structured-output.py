"""
Structured Output

Three strategies for structured output:
1. ProviderStrategy: Native structured output (OpenAI, Grok)
2. ToolStrategy: Tool calling for structured output
3. Simple Schema: Auto-selects best strategy

Supports: Pydantic models, dataclasses, TypedDict, JSON schema
"""

from dataclasses import dataclass
from typing import TypedDict, Union

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

load_dotenv()

contact_message = (
    "John Smith from ABC Corp can be reached via email at john@abc.com or by phone at 555-0123."
)

event_message = (
    "The Tech Conference will take place on March 15th, 2024, at the Convention Center in Sydney."
)

# ==============================================================
# Schema Types
# ==============================================================


# 1. Pydantic Model
class ContactInfo(BaseModel):
    name: str = Field(description="Person's name")
    email: str = Field(description="Email address")
    phone: str = Field(description="Phone number")


# 2. Dataclass
@dataclass
class ContactDataclass:
    name: str
    email: str
    phone: str


# 3. TypedDict
class ContactDict(TypedDict):
    name: str
    email: str
    phone: str


# 4. JSON Schema
contact_json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Person's name"},
        "email": {"type": "string", "description": "Email address"},
        "phone": {"type": "string", "description": "Phone number"},
    },
    "required": ["name", "email", "phone"],
}


# 5. Additional Schema for Union Example
class EventInfo(BaseModel):
    event_name: str = Field(description="Name of the event")
    date: str = Field(description="Event date")
    location: str = Field(description="Event location")


# ==============================================================
# Strategy 1: ProviderStrategy (Explicit)
# ==============================================================

agent_provider = create_agent(
    model="openai:gpt-4o-mini",
    system_prompt="You are a helpful assistant. Extract the required contact information.",
    response_format=ProviderStrategy(ContactInfo),
)

# ==============================================================
# Strategy 2: ToolStrategy (Explicit)
# ==============================================================

agent_tool = create_agent(
    model="openai:gpt-4o-mini",
    system_prompt="You are a helpful assistant. Extract the required contact information.",
    response_format=ToolStrategy(ContactInfo),
)

# ==============================================================
# Strategy 3: Simple Schema (Auto-select)
# ==============================================================

agent_auto = create_agent(
    model="openai:gpt-4o-mini",
    system_prompt="You are a helpful assistant. Extract the required contact information.",
    response_format=ContactInfo,  # Auto-selects best strategy
)

# ==============================================================
# Test All Approaches
# ==============================================================


def test_structured_output():
    """Test each strategy with the same input."""

    strategies = [
        ("ProviderStrategy", agent_provider),
        ("ToolStrategy", agent_tool),
        ("Auto-select", agent_auto),
    ]

    for name, agent in strategies:
        print(f"\n{'=' * 50}")
        print(f"Testing: {name}")
        print(f"{'=' * 50}")

        response = agent.invoke({"messages": [HumanMessage(content=contact_message)]})
        contact = response["structured_response"]

        print("Structured Response:")
        print(f"  Name: {contact.name}")
        print(f"  Email: {contact.email}")
        print(f"  Phone: {contact.phone}")
        print(f"  Type: {type(contact).__name__}")

        print("\nRaw AI Response (JSON string):")
        print(response["messages"][-1].content)


def test_schema_types():
    """Test different schema types with ToolStrategy."""

    schemas = [
        ("Pydantic Model", ContactInfo, contact_message),
        ("Dataclass", ContactDataclass, contact_message),
        ("TypedDict", ContactDict, contact_message),
        ("JSON Schema", contact_json_schema, contact_message),
        ("Union Types", Union[ContactInfo, EventInfo], event_message),
    ]

    print(f"\n{'=' * 60}")
    print("Testing Different Schema Types")
    print(f"{'=' * 60}")

    for name, schema, test_message in schemas:
        print(f"\n{name}:")

        agent = create_agent(
            model="openai:gpt-4o-mini",
            system_prompt="Extract information from the given text.",
            response_format=ToolStrategy(schema),
        )

        response = agent.invoke({"messages": [HumanMessage(content=test_message)]})
        result = response["structured_response"]

        print(f"  Result: {result}")
        print(f"  Type: {type(result).__name__}")

        # Show which schema was chosen for Union
        if name == "Union Types":
            if isinstance(result, ContactInfo):
                print("  → Agent chose ContactInfo schema")
            elif isinstance(result, EventInfo):
                print("  → Agent chose EventInfo schema")


if __name__ == "__main__":
    # Test the three strategies
    test_structured_output()

    # Test different schema types (including Union)
    test_schema_types()
