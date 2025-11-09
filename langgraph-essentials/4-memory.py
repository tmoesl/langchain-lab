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

import operator
import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from IPython.display import Image, display
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

load_dotenv()

from typing import Literal

# ==============================================================
# Approach 1: Conditional Edges with Separate Router
# ==============================================================


class AgentState(TypedDict):
    """State with accumulating list of visited nodes."""

    nlist: Annotated[list[str], operator.add]


def node_a_basic(state: AgentState) -> None:
    """Entry node - no state updates."""
    print("🚀 Starting at node a")


def node_b_basic(state: AgentState) -> AgentState:
    """Process B path."""
    print("📍 Executing node b")
    return AgentState(nlist=["b"])


def node_c_basic(state: AgentState) -> AgentState:
    """Process C path."""
    print("📍 Executing node c")
    return AgentState(nlist=["c"])


def route_decision(state: AgentState) -> Literal["b", "c", END]:
    """Route based on last input in state."""
    last_input = state["nlist"][-1].lower()

    route_map = {"b": "b", "c": "c", "q": END}
    next_node = route_map.get(last_input, END)
    print(f"🎯 Routing to node {next_node}")
    return next_node


def create_conditional_edge_graph():
    """Create graph using add_conditional_edges approach."""

    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("a", node_a_basic)
    builder.add_node("b", node_b_basic)
    builder.add_node("c", node_c_basic)

    # Add edges
    builder.add_edge(START, "a")
    builder.add_conditional_edges("a", route_decision)
    builder.add_edge("b", END)
    builder.add_edge("c", END)

    memory = InMemorySaver()

    return builder.compile(checkpointer=memory)


def test_conditional_edges():
    """Test the conditional edges approach."""

    print(f"\n{'=' * 50}")
    print("Testing: Conditional Edges Approach")
    print(f"{'=' * 50}")

    graph = create_conditional_edge_graph()
    config = {"configurable": {"thread_id": "1"}}

    # Visualize the graph
    if hasattr(sys, "ps1"):
        display(Image(graph.get_graph().draw_mermaid_png()))

    # Test cases

    while True:
        user = input("Enter: b, c, q or exit to quit: ")
        if user == "exit":
            break

        initial_state = AgentState(nlist=[user])
        response = graph.invoke(initial_state, config=config)
        print(f"Result: {response['nlist']}\n")


if __name__ == "__main__":
    test_conditional_edges()
