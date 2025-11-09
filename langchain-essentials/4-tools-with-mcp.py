"""
MCP Tools Integration

Demonstrates using Model Context Protocol (MCP) tools with LangChain agents.
MCP enables external tools/services integration via standardized protocol.

Requires async: use await agent.ainvoke() instead of agent.invoke().
"""

import asyncio

import nest_asyncio
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

nest_asyncio.apply()
load_dotenv()


async def main():
    """Main async function to handle MCP tool integration."""

    # ==============================================================
    # MCP Client Setup
    # ==============================================================

    mcp_client = MultiServerMCPClient(
        {
            "time": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@theo.foobar/mcp-time"],
            }
        }
    )

    # ==============================================================
    # Load MCP Tools and Create Agent
    # ==============================================================

    mcp_tools = await mcp_client.get_tools()
    print(f"Loaded {len(mcp_tools)} MCP tools: {[t.name for t in mcp_tools]}")

    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=mcp_tools,
        system_prompt="You are a helpful assistant. Use format: YYYY-MM-DD HH:MM:SS for time.",
    )

    # ==============================================================
    # Test MCP Tools
    # ==============================================================

    message = HumanMessage(
        content="What is the current time? Convert it to the timezone of Sydney?"
    )
    response = await agent.ainvoke({"messages": [message]})

    print(f"Assistant: {response['messages'][-1].content}")

    # Debug: Show all messages
    for msg in response["messages"]:
        msg.pretty_print()

    print(response)


if __name__ == "__main__":
    asyncio.run(main())
