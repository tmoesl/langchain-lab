"""Models and Messages"""

import json

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()

agent = create_agent(
    model="openai:gpt-5-nano",
    system_prompt="You are a helpful assistant.",
)

message = HumanMessage(content="What is the capital of France?")
response = agent.invoke({"messages": [message]})
print(f"Assistant: {response['messages'][-1].content}\n")


# ==============================================================
# Alternative Formats
# ==============================================================

# Strings
# -> Automatically converted to HumanMessage
response = agent.invoke({"messages": "What is the capital of France?"})  # type: ignore
print(f"Assistant: {response['messages'][-1].content}\n")


# Dictionaries
# -> Automatically converted to list of messages (HumanMessage, AIMessage)
messages = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris"},
    {"role": "user", "content": "What is its city population?"},
]
response = agent.invoke({"messages": messages})  # type: ignore
print(f"Assistant: {response['messages'][-1].content}\n")


# ==============================================================
# Output Format
# ==============================================================
# Output List: HumanMessage, AIMessage, ToolMessage


@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


agent = create_agent(
    model="openai:gpt-5-nano",
    tools=[add],
    system_prompt="You are a helpful assistant. Use tools when relevant to answer the question.",
)

message = HumanMessage(content="Add 2 and 3.")
response = agent.invoke({"messages": [message]})

print(f"Total number of messages: {len(response['messages'])}\n")
print(f"Assistant: {response['messages'][-1].content}\n")


# Visualize the conversation
for msg in response["messages"]:
    msg.pretty_print()

# ==============================================================
# Information
# ==============================================================

last_message = response["messages"][-1]
print(f"Content: {last_message.content}\n")
print(f"Usage Metadata: {json.dumps(last_message.usage_metadata, indent=2)}\n")
print(f"Metadata: {json.dumps(last_message.response_metadata, indent=2)}\n")
