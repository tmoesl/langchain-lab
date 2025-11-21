# LangChain Essentials

![Status](https://img.shields.io/badge/-Ongoing-F8B84F?style=flat&label=Project&labelColor=23555555)
![Languages](https://img.shields.io/github/languages/count/tmoesl/langchain-lab?label=Languages)
![Top Language](https://img.shields.io/github/languages/top/tmoesl/langchain-lab?color=white)

A hands-on guide to building production-ready AI agents with **LangChain v1.0**. This collection demonstrates the unified agent framework, featuring `create_agent` as the standard entry point for building agents with any model provider. Learn the core elements: models, messages, tools, streaming, memory, structured outputs, and middleware patterns.

## 📋 Quick Start

### SQL Database Agent Demo

- **[0-agent-demo.py](0-agent-demo.py)** - SQL Database Assistant  

<table><tr>
<td width="60%" valign="top">ReAct agent (Reason → Act → Observe → Repeat) built on LangGraph that interacts with a SQLite database through a single tool, discovers schema dynamically, and self-corrects queries using error feedback.<br><br>
<strong>Key Features:</strong><br>
- ReAct agent loop powered by LangGraph<br>
- Automatic reasoning, tool calls, and retries<br>
- System prompt handling without state pollution<br>
- Dynamic schema discovery via SQL queries<br>
- Short-term memory and thread management<br>
- Streaming responses for real-time updates<br>
- Runtime dependency injection to access database</td>
<td width="40%" align="center" valign="middle"><a href="https://blog.langchain.com/langchain-langgraph-1dot0/" target="_blank"><img src="https://blog.langchain.com/content/images/2025/10/Screenshot-2025-10-08-at-5.15.25---PM--1-.png" alt="ReAct Agent Graph" width="60%"/></a></td>
</tr></table>



## 🧱 Core Building Blocks

### Models & Messages
- **[1_model_basics.py](1-models.py)** - Model initialization, configuration, and invocation across 100+ providers
- **[2-messages.py](2-messages.py)** - Message types, formats, and the persistent scratch pad pattern

*Docs: [Models](https://docs.langchain.com/oss/python/langchain/models) | [Messages](https://docs.langchain.com/oss/python/langchain/messages)*

### Execution & Interaction
- **[3-streaming.py](3-streaming.py)** - Real-time updates with multiple streaming modes (values, updates, messages, custom)
- **[4-tools.py](4-tools.py)** - Tool definition approaches from basic decorators to Pydantic validation
- **[5-tools-with-mcp.py](5-tools-with-mcp.py)** - Model Context Protocol integration for standardized external services

*Docs: [Streaming](https://docs.langchain.com/oss/python/langchain/streaming) | [Tools](https://docs.langchain.com/oss/python/langchain/tools) | [MCP Integration](https://docs.langchain.com/oss/python/langchain/mcp)*

### State & Output
- **[6-memory.py](6-memory.py)** - Short-term memory with checkpointers and thread management inclusive runtime memory access
- **[7-structured-output.py](7-structured-output.py)** - Predictable data formats with ProviderStrategy and ToolStrategy

*Docs: [Short-Term Memory](https://docs.langchain.com/oss/python/langchain/short-term-memory) | [Structured Output](https://docs.langchain.com/oss/python/langchain/structured-output)*

### Middleware
- **[8-middleware-dynamic-prompt.py](8-middleware-dynamic-prompt.py)** - Context-aware system prompts that adapt at runtime
- **[9-middleware-human-in-the-loop.py](9-middleware-human-in-the-loop.py)** - Human approval workflows for high-stakes operations

*Docs: [Middleware](https://docs.langchain.com/oss/python/langchain/middleware)*

## 🏗️ File Structure

```
langchain-essentials/
├── 0-agent-demo.py                # Quickstart: Full agent example
├── 1-models.py                    # Models: Reasoning engine
├── 2-messages.py                  # Messages: Context units
├── 3-streaming.py                 # Streaming: Real-time updates
├── 4-tools.py                     # Tools: External actions
├── 5-tools-with-mcp.py            # MCP: Standardized protocols
├── 6-memory.py                    # Memory: State persistence
├── 7-structured-output.py         # Structured Output: Predictable formats
├── 8-middleware-dynamic-prompt.py # Middleware: Dynamic prompts
└── 9-middleware-human-in-the-loop.py # Middleware: HITL patterns
```

## 🎯 Key Highlights

- **Unified Agent Framework**: `create_agent` is the fastest way to build agents with any model provider, built on LangGraph runtime
- **Durable State & Persistence**: Automatic state persistence, save/resume workflows, production-ready reliability
- **Middleware System**: Fine-grained control with hooks at every step (before_model, after_model, before_tools, etc.)
- **Standard Content Blocks**: Provider-agnostic spec for consistent content types across 100+ providers
- **Streamlined Package**: Focused on core abstractions with legacy functionality moved to `langchain-classic`

## 📚 References
- LangChain Academy: [LangChain Essentials - Python](https://academy.langchain.com/courses/langchain-essentials-python)
- LangChain Documentation: [LangChain Documentation](https://docs.langchain.com/oss/python/langchain)
- LangGraph Documentation: [LangGraph Documentation](https://docs.langchain.com/langgraph)
- LangChain Blog: [LangChain/LangGraph v1.0 Release Blog](https://blog.langchain.com/langchain-langgraph-1dot0)

## Disclaimer

This repository is for educational purposes only and is not affiliated with or endorsed by the LangChain team. Its content and structure are inspired by the official LangChain documentation and the content available in the LangChain Academy.

---

*Built in Python with LangChain v1.0 and LangGraph v1.0*

