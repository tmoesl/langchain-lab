"""
Conditional Edges - Dynamic Routing

Demonstrates two ways to implement conditional routing in LangGraph:
1. add_conditional_edges(): Separate routing function
2. Command with goto: Built-in node routing

Key Concepts:
- Conditional edges: Route based on state values (dashed lines in graph)
- Static edges: Always taken (solid lines in graph)
- Runtime decisions: Next node determined by current state
- Command pattern: Update state + control flow in one return

Both approaches achieve the same result - choose based on preference and complexity.
"""

import operator
from typing import Annotated, Literal, TypedDict

from IPython.display import Image, display
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

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

    return builder.compile()


# ==============================================================
# Approach 2: Command-Based Routing
# ==============================================================


class AgentState(TypedDict):
    """State with accumulating list of visited nodes."""

    nlist: Annotated[list[str], operator.add]


def node_a_command(state: AgentState) -> Command[Literal["b", "c", END]]:
    """Entry node with built-in routing logic."""
    print("🚀 Starting at node A (Command approach)")

    last_input = state["nlist"][-1].lower()

    if last_input == "b":
        next_node = "b"
    elif last_input == "c":
        next_node = "c"
    elif last_input == "q":
        next_node = END
    else:
        next_node = END

    print(f"🎯 Routing to node{next_node}")
    return Command(goto=[next_node])


def node_b_command(state: AgentState) -> AgentState:
    """Process B path."""
    print("📍 Executing node b")
    return AgentState(nlist=["b"])


def node_c_command(state: AgentState) -> AgentState:
    """Process C path."""
    print("📍 Executing node c")
    return AgentState(nlist=["c"])


def create_command_graph():
    """Create graph using Command approach."""

    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("a", node_a_command)
    builder.add_node("b", node_b_command)
    builder.add_node("c", node_c_command)

    # Add edges (no conditional edges needed - handled by Command)
    builder.add_edge(START, "a")
    builder.add_edge("b", END)
    builder.add_edge("c", END)

    return builder.compile()


# ==============================================================
# Test Both Approaches
# ==============================================================


def test_conditional_edges():
    """Test the conditional edges approach."""

    print(f"\n{'=' * 50}")
    print("Testing: Conditional Edges Approach")
    print(f"{'=' * 50}")

    graph = create_conditional_edge_graph()

    # Visualize the graph
    display(Image(graph.get_graph().draw_mermaid_png()))

    # Test cases
    test_cases = ["b", "c", "q", "invalid"]

    for test_input in test_cases:
        print(f"\nInput: '{test_input}'")
        initial_state = AgentState(nlist=[test_input])

        try:
            response = graph.invoke(initial_state)
            print(f"Result: {response['nlist']}")
        except Exception as e:
            print(f"Error: {e}")


def test_command_routing():
    """Test the Command-based routing approach."""

    print(f"\n{'=' * 50}")
    print("Testing: Command Routing Approach")
    print(f"{'=' * 50}")

    graph = create_command_graph()

    # Visualize the graph
    display(Image(graph.get_graph().draw_mermaid_png()))

    # Test cases
    test_cases = ["b", "c", "q", "invalid"]

    for test_input in test_cases:
        print(f"\nInput: '{test_input}'")
        initial_state = AgentState(nlist=[test_input])

        try:
            response = graph.invoke(initial_state)
            print(f"Result: {response['nlist']}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # Test both approaches separately
    test_conditional_edges()
    test_command_routing()
