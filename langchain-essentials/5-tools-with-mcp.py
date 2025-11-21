"""
LangChain MCP - Model Context Protocol Integration

MCP is an open protocol that standardizes how applications provide tools and context
to LLMs. The langchain-mcp-adapters library converts MCP tools into LangChain tools,
enabling agents to use external services via standardized protocol.

How It Works:
-------------
1. MCP server provides tool schemas (name, description, input schema)
2. MultiServerMCPClient connects to MCP servers via transport protocol
3. Adapter converts MCP tools to LangChain StructuredTools
4. LangChain agent calls tools → adapter wraps call → MCP server executes

Loading Tools:
--------------
1. get_tools() - Stateless (default, recommended)
   - await client.get_tools() → All tools from all servers
   - await client.get_tools("math") → Tools from specific server
   - Each tool call: creates session → executes → cleans up
   - Best for: Most use cases (simple, automatic cleanup)

2. load_mcp_tools() - Stateful session
   - async with client.session("time") as session:
   -     tools = await load_mcp_tools(session)
   - Session persists across tool calls (server can maintain context)
   - Best for: Servers needing state between calls

Transport Types:
----------------
- stdio: Local subprocess (command + args)
- streamable_http: Remote HTTP server (url)
- SSE: Server-Sent Events for real-time streaming

Reference: https://docs.langchain.com/oss/python/langchain/mcp
"""

import asyncio
import os

import nest_asyncio
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

nest_asyncio.apply()
load_dotenv()


# ==============================================================
# MCP Client Setup
# ==============================================================


async def setup_mcp_client():
    """Setup MCP client with multiple servers."""
    brave_api_key = os.getenv("BRAVE_API_KEY")
    if not brave_api_key:
        raise ValueError("BRAVE_API_KEY not found in environment variables")

    mcp_client = MultiServerMCPClient(
        {
            "time": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@theo.foobar/mcp-time"],
            },
            "brave-search": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@brave/brave-search-mcp-server"],
                "env": {
                    "BRAVE_API_KEY": brave_api_key,
                },
            },
        }
    )
    return mcp_client


# ==============================================================
# Loading MCP Tools
# ==============================================================


async def loading_mcp_tools():
    """Demonstrate different ways to load MCP tools."""

    client = await setup_mcp_client()

    # Method 1: Get all tools (stateless)
    all_tools = await client.get_tools()
    print(f"All tools: {[t.name for t in all_tools]}\n")

    # Method 2: Get tools from specific server (stateless)
    time_tools = await client.get_tools(server_name="time")
    print(f"Time server: {[t.name for t in time_tools]}\n")

    brave_tools = await client.get_tools(server_name="brave-search")
    print(f"Brave server: {[t.name for t in brave_tools]}\n")

    # Method 3: Stateful session with load_mcp_tools()
    async with client.session("time") as session:
        session_tools = await load_mcp_tools(session)
        print(f"Time server (stateful): {[t.name for t in session_tools]}\n")


# ==============================================================
# Create Agent and Test (using all tools)
# ==============================================================


async def agent_with_mcp_tools(message: str):
    """Create and test agent using MCP tools."""

    client = await setup_mcp_client()

    # Get all tools
    tools = await client.get_tools()

    # Create agent
    agent = create_agent(
        model="openai:gpt-5-mini",
        tools=tools,
        system_prompt="""
        You are a helpful assistant.
        Use format: YYYY-MM-DD HH:MM:SS for time.
        Use the tools to answer the question if applicable.
        """,
    )

    # Test agent
    response = await agent.ainvoke({"messages": [HumanMessage(content=message)]})
    print(f"Response: {response['messages'][-1].content}\n")

    # Show message history
    for msg in response["messages"]:
        msg.pretty_print()


async def main():
    """Run the MCP tools examples."""
    # Show available tools
    await loading_mcp_tools()

    print("=" * 60)
    print("Testing Agent with MCP Tools")
    print("=" * 60)

    # Scenario 1: Time server (get current time)
    print("\n[Scenario 1: Time Server]")
    await agent_with_mcp_tools("What is the current time?")

    # Scenario 2: Brave Search server (web search)
    print("\n[Scenario 2: Brave Search Server]")
    await agent_with_mcp_tools("What is the latest news on OpenAI and Microsoft partnership?")


if __name__ == "__main__":
    asyncio.run(main())
