"""
Memory and Persistence

Demonstrates how to add memory to LangGraph using checkpointers for state persistence.
State is preserved across graph runs using thread IDs for conversation continuity.

Key Concepts:
- Super Steps: Multiple nodes can execute in parallel within a single step
- Checkpointers: Store state snapshots at the end of each super step
- Thread ID: Identifies conversation threads for state persistence
- State Accumulation: Messages accumulate across invocations using add_messages reducer

Benefits of Checkpointing:
- Graceful recovery from failures
- Time travel: restore state from previous points
- Persistent state: preserved when graph isn't running
- Resume execution: pick up exactly where left off

Checkpointer Options:
- InMemorySaver: Simple in-memory storage (development)
- PostgresSaver: Production database storage
- SQLiteSaver: File-based database storage
"""

import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

# ==============================================================
# State Definition
# ==============================================================


class AgentState(TypedDict):
    """State schema with message accumulation across invocations."""

    messages: Annotated[list[BaseMessage], add_messages]


# ==============================================================
# Node Definition
# ==============================================================


def call_llm(state: AgentState) -> dict:
    """Call LLM and return response."""
    model = ChatOpenAI(model="gpt-4o-mini")
    system_msg = SystemMessage(content="You are a helpful assistant.")

    messages = [system_msg] + state["messages"]
    response = model.invoke(messages)

    return {"messages": [response]}


# ==============================================================
# Graph Setup with Memory
# ==============================================================


def create_memory_graph():
    """Create a graph with persistent memory using checkpointer."""
    # Build graph structure
    builder = StateGraph(AgentState)
    builder.add_node("llm", call_llm)
    builder.add_edge(START, "llm")
    builder.add_edge("llm", END)

    # Set up checkpointer for memory
    memory = InMemorySaver()

    # Compile graph with checkpointer
    return builder.compile(checkpointer=memory)


# ==============================================================
# Interactive Chat with Memory
# ==============================================================


def test_memory_persistence():
    """Test memory persistence across multiple invocations."""
    agent = create_memory_graph()

    # Display graph structure
    if hasattr(sys, "ps1"):
        display(Image(agent.get_graph().draw_mermaid_png()))

    # Configure thread for persistent conversation
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    print("\n🧠 Testing Memory Persistence")
    print("=" * 50)
    print("💡 Tip: Messages accumulate across invocations using the same thread_id")
    print("💡 Type 'exit' to quit\n")

    while True:
        user_input = input("Enter: ")

        if user_input.lower() == "exit":
            break

        # Invoke with persistent state
        response = agent.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

        # Show user msg if in interactive mode (only)
        if hasattr(sys, "ps1"):
            print(f"User: {user_input}")
        print(f"Assistant: {response['messages'][-1].content}")
        print(
            f"📝 Thread: {config['configurable']['thread_id']} | "
            f"Messages in state: {len(response['messages'])}\n"
        )


if __name__ == "__main__":
    test_memory_persistence()
