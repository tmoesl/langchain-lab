"""
LangGraph - Conditional Edges for Dynamic Routing

Demonstrates the two primary ways to implement conditional routing in LangGraph,
allowing the graph to make decisions based on the current state.

Key Concepts:
-------------
- Conditional Edges: Enable dynamic, looping, and agentic behaviors by routing
  control flow based on the current state.

- Method 1: Conditional Edges (`add_conditional_edges`)
  - A separate router function inspects the state and returns the next node(s).
  - When to Use: Best for separating routing logic from a node's main task.
    Use this to route between nodes without updating the state in the router.

- Method 2: Command (`update` and `goto`)
  - A node returns a `Command` object to update the state and control routing directly.
  - When to Use: Best for combining state updates and routing logic in a
    single, self-contained function.
  - NOTE: Return annotations declare node routing paths for rendering.

Execution Flow:
---------------
Both approaches build the same graph: an entry node A decides which path
to take, routing to either node B or C. Both paths then converge at a final node D.

References:
https://docs.langchain.com/oss/python/langgraph/graph-api#conditional-edges
https://docs.langchain.com/oss/python/langgraph/graph-api#command
"""

import operator
import random
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from IPython.display import Image, display
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

load_dotenv()


# ==============================================================
# Define State
# ==============================================================


class State(TypedDict):
    """State with accumulating list of messages."""

    messages: Annotated[list[str], operator.add]


# ==============================================================
# Define Shared Nodes for both approaches
# ==============================================================


def node_b(state: State) -> State:
    print(f"--- Node B --- \n  - Received: {state['messages']}\n")
    return {"messages": ["B"]}


def node_c(state: State) -> State:
    print(f"--- Node C --- \n  - Received: {state['messages']}\n")
    return {"messages": ["C"]}


def node_d(state: State) -> State:
    print(f"--- Node D --- \n  - Received merged state: {state['messages']}\n")
    return {"messages": ["D"]}


# ==============================================================
# Approach 1: Conditional Edges with Separate Router
# ==============================================================
print("=" * 60)
print("Approach 1: Conditional Edges with Separate Router")
print("=" * 60)


def node_a(state: State) -> State:  # type: ignore
    """
    Updates the state with a randomly chosen path. The actual routing
    decision is handled by the separate `conditional_router` function,
    demonstrating a separation of concerns.
    """
    choice = random.choice(["B", "C"])
    print(f"--- Node A --- \n  - Input: {state['messages']}\n  - Path: '{choice}'\n")
    return {"messages": [f"Path to {choice}"]}


def conditional_router(state: State) -> Literal["B", "C"]:
    """Inspects the state to decide the next path."""
    last_message = state["messages"][-1]
    return "B" if "B" in last_message else "C"


# Initiate the graph
graph = StateGraph(State)

# Add nodes
graph.add_node("A", node_a)
graph.add_node("B", node_b)
graph.add_node("C", node_c)
graph.add_node("D", node_d)

# Add edges
graph.add_edge(START, "A")
graph.add_conditional_edges("A", conditional_router, {"B": "B", "C": "C"})
graph.add_edge("B", "D")
graph.add_edge("C", "D")
graph.add_edge("D", END)

# Compile the graph (runnable object)
graph = graph.compile()

# Invoke the graph
initial_state = State(messages=["Initial String"])
response = graph.invoke(initial_state)
print(f"Result: {response['messages']}\n")

# Display the graph
display(Image(graph.get_graph().draw_mermaid_png()))


# ==============================================================
# Approach 2: Using a Command to Update State and Route
# ==============================================================
print("\n" + "=" * 60)
print("Approach 2: Using a Command to Update State and Route")
print("=" * 60)


def node_a(state: State) -> Command[Literal["B", "C"]]:
    """
    Combines state updates and routing logic in a single function.
    This node randomly chooses a path, updates the state with its choice,
    and returns a Command to route to that path all in one step.
    NOTE: Return annotations declare node routing paths for rendering.
    """
    choice = random.choice(["B", "C"])
    print(f"--- Node A --- \n  - Input: {state['messages']}\n  - Path: '{choice}'\n")

    state_update = {"messages": [f"Routing to {choice} node"]}
    return Command(update=state_update, goto=choice)  # type: ignore


# Initiate the graph
graph = StateGraph(State)

# Add nodes
graph.add_node("A", node_a)
graph.add_node("B", node_b)
graph.add_node("C", node_c)
graph.add_node("D", node_d)

# Add edges
graph.add_edge(START, "A")
graph.add_edge("B", "D")
graph.add_edge("C", "D")
graph.add_edge("D", END)

# Compile the graph (runnable object)
graph = graph.compile()

# Invoke the graph
initial_state = State(messages=["Initial String"])
response = graph.invoke(initial_state)
print(f"Result: {response['messages']}\n")

# Display the graph
display(Image(graph.get_graph().draw_mermaid_png()))
