"""
LangChain Streaming - Real-time Updates for Agent Execution

Streaming reduces latency in interactive applications by providing progressive output
before complete responses are ready, significantly improving user experience.

Streaming Modes:
----------------
1. invoke() - No streaming, waits for complete response
2. stream_mode="values" - Full state after each agent step
3. stream_mode="updates" - Only incremental changes per node (model/tools)
4. stream_mode="messages" - Token-by-token as LLM generates
5. stream_mode="custom" - User-defined updates from tools via get_stream_writer()
6. stream_mode=["values", "custom"] - Multiple modes combined

Reference: https://docs.langchain.com/oss/python/langchain/streaming
"""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.config import get_stream_writer

load_dotenv()

# Simple agent for basic examples
agent = create_agent(
    model="openai:gpt-5-mini",
    system_prompt="You are a hilarious comedian.",
)


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
    print(f"{token.content}", end="", flush=True)  # type: ignore

# ==============================================================
# Tool Setup for Remaining Examples
# ==============================================================


@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    writer = get_stream_writer()

    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")

    city_map = {
        "Paris": "It's always rainy in Paris!",
        "London": "It's always rainy in London!",
        "Tokyo": "It's always cloudy in Tokyo!",
        "New York": "It's always sunny in New York!",
        "San Francisco": "It's always windy in San Francisco!",
    }

    return city_map[city]


agent_with_tools = create_agent(
    model="openai:gpt-5-mini",
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
)

# ==============================================================
# Streaming Mode (Updates)
# ==============================================================

message = HumanMessage(content="What is the weather in Paris?")
response = agent_with_tools.stream({"messages": [message]}, stream_mode="updates")

for chunk in response:
    for node_name, update_data in chunk.items():
        print(f"\nNode: {node_name} | Message: {update_data['messages'][-1].content}")

# ==============================================================
# Streaming Mode (Custom)
# ==============================================================

message = HumanMessage(content="What is the weather in SF?")
response = agent_with_tools.stream({"messages": [message]}, stream_mode="custom")

for chunk in response:
    print(chunk)

# ==============================================================
# Streaming Mode (Multiple: Values and Custom)
# ==============================================================

message = HumanMessage(content="What is the weather in Tokyo?")
response = agent_with_tools.stream({"messages": [message]}, stream_mode=["values", "custom"])

for step in response:
    print(step)
