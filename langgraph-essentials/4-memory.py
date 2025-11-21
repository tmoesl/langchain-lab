"""
LangGraph - Memory and Persistence with Checkpointers

Demonstrates how to add memory to LangGraph using checkpointers for state persistence.
This enables multi-turn conversations where the graph remembers context across invocations.

Key Concepts:
-------------
- Checkpointers: Store state snapshots at the end of each super step
- Thread ID: Identifies conversation threads for state persistence
- InMemorySaver: Simple in-memory storage (for development/testing)
- add_messages: Built-in reducer that intelligently merges message lists

Benefits of Checkpointing:
--------------------------
- Graceful Recovery: Restore state and restart after node failures
- Time Travel: Restore state from any previous checkpoint
- Persistent State: State preserved even when graph isn't running
- Resume Execution: Pick up exactly where execution left off

Production Checkpointers:
-------------------------
- InMemorySaver: Development/testing (used in this example)
- PostgresSaver: Production-grade database storage
- SQLiteSaver: File-based database storage

Execution Flow:
---------------
1. Graph is compiled with a checkpointer
2. Each invocation uses a thread_id in the config
3. State is automatically saved at the end of each super step
4. Subsequent invocations with the same thread_id restore previous state
5. Messages accumulate across multiple invocations

References:
https://docs.langchain.com/oss/python/langgraph/add-memory
https://docs.langchain.com/oss/python/langgraph/persistence
"""

from typing import Annotated, TypedDict

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()


# ==============================================================
# Define State
# ==============================================================
# Note: This can be replaced with MessagesState directly.


class State(TypedDict):
    """State with accumulating list of messages."""

    messages: Annotated[list[BaseMessage], add_messages]


# ==============================================================
# Initialize Model and System Message
# ==============================================================

model = init_chat_model("openai:gpt-5-mini")

# System message to guide the chatbot's behavior


# ==============================================================
# Define Nodes
# ==============================================================


def agent_node(state: State) -> dict:
    """
    Calls the LLM with full conversation history and returns the AI response.
    The checkpointer automatically saves the updated state after this node completes.
    """

    print("--- Agent Node ---")
    print(f"  - Processing {len(state['messages'])} message(s) in conversation history")

    system_msg = SystemMessage(
        content="You are a helpful assistant. Be concise and friendly in your responses."
    )
    messages = [system_msg] + state["messages"]
    response = model.invoke(messages)  # Invoke the model with full conversation history

    return {"messages": [response]}


# ==============================================================
# Build Graph
# ==============================================================
# Add checkpointer to the graph
# State is saved after each super step, enabling recovery and memory


# Initiate the graph
graph = StateGraph(State)

# Add nodes and edges
graph.add_node("agent", agent_node)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)


# Compile graph
memory = InMemorySaver()
graph = graph.compile(checkpointer=memory)


# ==============================================================
# Helper Function for Running Conversations
# ==============================================================


def run_turn(thread_name: str, turn_number: int, user_msg: str, config: RunnableConfig) -> None:
    """
    Helper function to run a single conversation turn and display results.

    Args:
        thread_name: Display name for the thread (e.g., "Thread 1")
        turn_number: The turn number in this conversation
        user_msg: The user's message
        config: RunnableConfig with thread_id for state persistence
    """
    print(f"\n[{thread_name} - Turn {turn_number}]")
    result = graph.invoke({"messages": [HumanMessage(content=user_msg)]}, config)  # type: ignore
    print(f"\nUser: {user_msg}")
    print(f"Assistant: {result['messages'][-1].content}")
    print(f"Total messages in thread: {len(result['messages'])}")


# ==============================================================
# Run Graph - Demonstrating Memory Across Invocations
# ==============================================================
# Key Takeaway: Each thread_id maintains independent conversation history!

print("=" * 70)
print("Demonstrating Memory: Multi-Turn Conversations with Thread IDs")
print("=" * 70)

# Configuration for thread 1
config_1: RunnableConfig = {"configurable": {"thread_id": "conversation-1"}}

run_turn("Thread 1", 1, "Hi! My name is Alice. I'm 25 years old.", config_1)
run_turn("Thread 1", 2, "What's my name?", config_1)  # Alice
run_turn("Thread 1", 3, "What is my age?", config_1)  # 25 years

# Start a new conversation (new thread)
config_2: RunnableConfig = {"configurable": {"thread_id": "conversation-2"}}
run_turn("Thread 2", 1, "Hi! My name is Bob.", config_2)
run_turn("Thread 2", 2, "What's my name?", config_2)  # Bob

# Return to thread 1 (memory persists)
run_turn("Thread 1", 4, "What's my name again?", config_1)  # Alice

# ==============================================================
# Visualize Graph
# ==============================================================
print("\n" + "=" * 60)
print("Graph Visualization:")
print("=" * 60)

display(Image(graph.get_graph().draw_mermaid_png()))
