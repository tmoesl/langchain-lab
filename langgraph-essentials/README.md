# LangGraph Essentials

![Status](https://img.shields.io/badge/-Ongoing-F8B84F?style=flat&label=Project&labelColor=23555555)
![Languages](https://img.shields.io/github/languages/count/tmoesl/langchain-lab?label=Languages)
![Top Language](https://img.shields.io/github/languages/top/tmoesl/langchain-lab?color=white)

A hands-on guide to building stateful, multi-step AI workflows with **LangGraph v1.0**. This collection demonstrates how to build production-ready agent systems from scratch using graphs, nodes, and state management. Learn the core concepts: state flow, parallel execution, conditional routing, persistence, human-in-the-loop patterns, and complete application architectures.

## 📋 Quick Start

### SQL Database Agent Application

- **[8-application-sql-agent.py](8-application-sql-agent.py)** - SQL Database Assistant  

<table><tr>
<td width="60%" valign="top">A ReAct agent built from scratch using LangGraph that interacts with a SQLite database. Discovers schema dynamically, generates SQL queries, self-corrects errors, and maintains conversation history across multiple turns.<br><br>
<strong>Key Features:</strong><br>
- ReAct agent loop built from scratch with LangGraph<br>
- Automatic reasoning, tool calls, and retries<br>
- System prompt handling without state pollution<br>
- Dynamic schema discovery via SQL queries<br>
- Short-term memory and thread management<br>
- Streaming responses for real-time updates<br>
- Runtime dependency injection to access database</td>
<td width="40%" align="center" valign="middle"><a href="https://docs.langchain.com/oss/python/langgraph/overview" target="_blank"><img src="https://blog.langchain.com/content/images/2025/10/Screenshot-2025-10-08-at-5.15.25---PM--1-.png" alt="LangGraph Agent" width="60%"/></a></td>
</tr></table>


## 🧱 Core Building Blocks

### Graph Fundamentals
- **[1-nodes.py](1-nodes.py)** - State management, nodes, and basic graph construction
- **[2-static-edges.py](2-static-edges.py)** - Parallel execution with static edges and state reducers
- **[3-conditional-edges.py](3-conditional-edges.py)** - Dynamic routing with conditional edges and Command objects

*Docs: [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api) | [State](https://docs.langchain.com/oss/python/langgraph/graph-api#state) | [Nodes](https://docs.langchain.com/oss/python/langgraph/graph-api#nodes) | [Edges](https://docs.langchain.com/oss/python/langgraph/graph-api#edges)*

### Persistence & Control Flow
- **[4-memory.py](4-memory.py)** - Checkpointers for state persistence and conversation memory
- **[5-human-in-the-loop.py](5-human-in-the-loop.py)** - Interrupt patterns for human approval workflows
- **[6-tools.py](6-tools.py)** - Tool integration with prebuilt ToolNode and custom implementations

*Docs: [Memory](https://docs.langchain.com/oss/python/langgraph/add-memory) | [Human-in-the-Loop](https://docs.langchain.com/oss/python/langgraph/human-in-the-loop) | [Tools](https://docs.langchain.com/oss/python/langgraph/quickstart)*




### Complete Applications
- **[7-application-email-agent.py](7-application-email-agent.py)** - Production email triage system with classification, parallel processing, and human review
- **[8-application-sql-agent.py](8-application-sql-agent.py)** - SQL database agent with dynamic schema discovery and error self-correction

*Docs: [Building Applications](https://docs.langchain.com/oss/python/langgraph/workflows-agents) | [Agent Architectures](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph)*

## 🏗️ File Structure

```
langgraph-essentials/
├── 1-nodes.py                     # State & Nodes: Building blocks
├── 2-static-edges.py              # Static Edges: Parallel execution
├── 3-conditional-edges.py         # Conditional Edges: Dynamic routing
├── 4-memory.py                    # Memory: State persistence
├── 5-human-in-the-loop.py         # HITL: Interrupt patterns
├── 6-tools.py                     # Tools: ToolNode integration
├── 7-application-email-agent.py   # Application: Email triage system
└── 8-application-sql-agent.py     # Application: SQL database agent
```

## 🎯 Key Highlights

- **Graph-Based Architecture**: Build complex workflows by composing nodes, edges, and state transformations
- **Parallel Execution**: Execute multiple nodes simultaneously with super steps and state reducers
- **Conditional Routing**: Dynamic control flow with `add_conditional_edges` or `Command` objects
- **State Persistence**: Checkpointers enable pause/resume, time travel, and graceful failure recovery
- **Human-in-the-Loop**: Interrupt execution for human approval on high-stakes operations
- **Production-Ready Patterns**: Complete application examples demonstrating real-world agent architectures

## 📊 Application Examples

### Email Agent (7-application-email-agent.py)
Complex workflow demonstrating:
- Structured output for email classification
- Parallel execution (fan-out/fan-in pattern)
- Conditional routing based on priority
- Human-in-the-loop for critical emails
- RAG search integration
- Batch processing with memory

### SQL Agent (8-application-sql-agent.py)
ReAct agent demonstrating:
- Runtime context injection (database access)
- Dynamic schema discovery
- Error self-correction through tool feedback
- System prompt handling without state pollution
- Conversation memory across turns
- Prebuilt ToolNode for tool execution

## 📚 References
- LangGraph Documentation: [LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph)
- LangGraph Quickstart: [Building Your First Agent](https://docs.langchain.com/oss/python/langgraph/quickstart)
- LangGraph Academy: [LangGraph Essentials - Python](https://academy.langchain.com/courses/langgraph-essentials-python)
- LangChain Blog: [LangChain/LangGraph v1.0 Release Blog](https://blog.langchain.com/langchain-langgraph-1dot0)

## Disclaimer

This repository is for educational purposes only and is not affiliated with or endorsed by the LangChain team. Its content and structure are inspired by the official LangGraph documentation and the content available in the LangChain Academy.

---

*Built in Python with LangGraph v1.0*

