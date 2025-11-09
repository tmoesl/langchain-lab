"""
Streaming

Streamin modes:
- values: return state after every step
- messages: token by token as the LLM produces them

"""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.config import get_stream_writer

load_dotenv()

agent = create_agent(
    model="openai:gpt-5-nano",
    system_prompt="You are an hilarious comedian.",
)

message = HumanMessage(content="Tell me a Dad joke.")


# ==============================================================
# No Streaming (Invocation)
# ==============================================================

message = HumanMessage(content="Tell me a Dad joke.")
response = agent.invoke({"messages": [message]})
print(f"Assistant: {response['messages'][-1].content}\n")

# ==============================================================
# Streaming Mode (Values)
# ==============================================================

message = HumanMessage(content="Tell me a Dad joke.")
response = agent.stream({"messages": [message]}, stream_mode="values")

for step in response:
    step["messages"][-1].pretty_print()

# ==============================================================
# Streaming Mode (Messages)
# ==============================================================

message = HumanMessage(content="Tell me a poem.")
response = agent.stream({"messages": [message]}, stream_mode="messages")

for token, _ in response:
    print(f"{token.content}", end="", flush=True)

# ==============================================================
# Streaming Mode (Custom)
# ==============================================================
# Allows to stream messages from tools as they get executed


@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    writer = get_stream_writer()

    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"


agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
)

message = HumanMessage(content="What is the weather in SF?")
response = agent.stream({"messages": [message]}, stream_mode="custom")

for chunk in response:
    print(chunk)


# ==============================================================
# Streaming Mode (Values and Custom)
# ==============================================================

message = HumanMessage(content="What is the weather in SF?")
response = agent.stream({"messages": [message]}, stream_mode=["values", "custom"])

# Tuple of (mode, data)
for step in response:
    print(step)
