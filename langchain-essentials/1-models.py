"""
LangChain Models - The Reasoning Engine of Agents

Models provide the reasoning component in ReAct agents. LangChain supports
100+ model providers with a unified interface, making model switching seamless.

Key Concepts:
-------------
- init_chat_model: Unified initialization for all providers
- Provider-specific classes: Direct provider imports (ChatOpenAI, etc.)
- Configuration: Control behavior via parameters (temperature, max_tokens, etc.)
- Invocation: invoke(), stream(), batch() - standard methods across provider
- Streaming: Covered in detail in 3-streaming.py

Supported Providers:
--------------------
OpenAI, Anthropic, Google Gemini, AWS Bedrock, Azure, and 100+ others.

References:
https://docs.langchain.com/oss/python/langchain/models
https://docs.langchain.com/oss/python/integrations/providers/overview
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

# ==============================================================
# Basic Model Initialization (init_chat_model)
# ==============================================================

print("=" * 60)
print("Basic Model Initialization (init_chat_model)")
print("=" * 60)

# Unified initialization syntax: provider:model_name
model = init_chat_model("openai:gpt-5-mini")

# Simple invocation
response = model.invoke("What is 2+2?")
print(f"Response: {response.content}\n")

# ==============================================================
# Provider-Specific Initialization
# ==============================================================

print("=" * 60)
print("Provider-Specific Initialization")
print("=" * 60)

# Direct provider class imports (alternative approach)
model_openai = ChatOpenAI(model="openai:gpt-5-mini")
model_anthropic = ChatAnthropic(model="claude-haiku-4-5-20251001")  # type: ignore

print(f"OpenAI: {model_openai.invoke('Hello!').content}")
print(f"Anthropic: {model_anthropic.invoke('Hello!').content}\n")

# ==============================================================
# Model Configuration Parameters
# ==============================================================

print("=" * 60)
print("Model Configuration Parameters")
print("=" * 60)

# Configure model behavior with parameters
configured_model = init_chat_model(
    "openai:openai:gpt-5-mini",
    temperature=0.7,  # Controls randomness (0=deterministic, 1=creative)
    max_tokens=100,  # Limit response length
    max_retries=3,  # Retry attempts on failure
)

response = configured_model.invoke("Tell me a fun fact.")
print(f"Response: {response.content}\n")

# ==============================================================
# 4. Simple Conversation Example
# ==============================================================
# Messages enable multi-turn conversations with roles (system, human, AI)

print("=" * 60)
print("4. Simple Conversation Example")
print("=" * 60)

# Message-based invocation with conversation history
messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="Translate: I love programming."),
]

model = init_chat_model("openai:openai:gpt-5-mini")
response = model.invoke(messages)
print(f"Translation: {response.content}")
