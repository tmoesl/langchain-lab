"""
LangChain Messages - The Fundamental Unit of Context for LLMs

Messages represent inputs and outputs of models, carrying content and metadata needed
to represent conversation state. They are standardized across providers, enabling
model independence without code changes.

Message Types:
--------------
1. SystemMessage - Instructions that prime the model's behavior
2. HumanMessage - User input (text, images, audio, files)
3. AIMessage - Model responses (text, tool calls, metadata)
4. ToolMessage - Tool call outputs

Input Formats:
--------------
1. Message objects: HumanMessage("Hello")
2. Strings: "Hello" (auto-converted to HumanMessage)
3. Dictionaries: {"role": "user", "content": "Hello"}

Key Concepts:
-------------
- Messages are stored in a persistent scratch pad shared by all nodes
- System prompt is the most important message (defines agent behavior)
- Messages contain: content, usage_metadata, response_metadata, tool_calls

Reference: https://docs.langchain.com/oss/python/langchain/messages
"""

import json

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()

# ==============================================================
# Basic Message Usage
# ==============================================================

agent = create_agent(
    model="openai:gpt-5-mini",
    system_prompt="You are a helpful assistant.",
)

message = HumanMessage(content="What is the capital of France?")
response = agent.invoke({"messages": [message]})
print(f"Assistant: {response['messages'][-1].content}\n")

# ==============================================================
# Input Format: String (auto-converted to HumanMessage)
# ==============================================================

response = agent.invoke({"messages": "What is the capital of France?"})  # type: ignore
print(f"Assistant: {response['messages'][-1].content}\n")

# ==============================================================
# Input Format: Dictionary (OpenAI chat completions format)
# ==============================================================

messages = [
    {"role": "system", "content": "You are a geography expert"},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris"},
    {"role": "user", "content": "What is its population?"},
]
response = agent.invoke({"messages": messages})  # type: ignore
print(f"Assistant: {response['messages'][-1].content}\n")

# ==============================================================
# Message Types: HumanMessage, AIMessage, ToolMessage
# ==============================================================


@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


agent_with_tools = create_agent(
    model="openai:gpt-5-mini",
    tools=[add],
    system_prompt="You are a helpful assistant. Use tools when relevant to answer the question.",
)

message = HumanMessage(content="Add 2 and 3.")
response = agent_with_tools.invoke({"messages": [message]})

print(f"Total messages: {len(response['messages'])}\n")

# Visualize all message types in the conversation
for msg in response["messages"]:
    msg.pretty_print()

# ==============================================================
# AIMessage Attributes and Metadata
# ==============================================================

last_message = response["messages"][-1]

print(f"\nContent: {last_message.content}")
print(f"Tool calls: {last_message.tool_calls}")
print(f"\nUsage Metadata:\n{json.dumps(last_message.usage_metadata, indent=2)}")
print(f"\nResponse Metadata:\n{json.dumps(last_message.response_metadata, indent=2)}")
