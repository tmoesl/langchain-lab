"""
LangGraph - Parallel Execution with Static Edges

Demonstrates how to build graphs with branching paths that execute in parallel
and then converge. This example uses the `operator.add` reducer to accumulate
results from the parallel branches into a single state key.

Key Concepts:
-------------
- Static Edges: Define a direct, non-conditional path from one node to another.
- Parallel Execution: A node with multiple outgoing edges triggers parallel
  execution of the downstream nodes.
- Super Steps: LangGraph waits for all parallel nodes to complete before
  moving to the next step, ensuring the state is fully updated.
- State Reducers: Define how state updates are merged.
    - Default Behavior: Overwrite (last write wins), as seen in 1-nodes.py.
    - `operator.add`: Appends values to a list, accumulating results.

Execution Flow:
---------------
1. Node A executes first.
2. It branches out to nodes B, C, and D, which run in parallel.
3. The `operator.add` reducer merges the outputs from B, C, and D into the state['messages'].
4. Node E executes last, receiving the combined state.

References:
https://docs.langchain.com/oss/python/langgraph/graph-api
https://docs.langchain.com/oss/python/langgraph/graph-api#edges
https://docs.langchain.com/oss/python/langgraph/graph-api#reducers
"""

import operator
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from IPython.display import Image, display
from langgraph.graph import END, START, StateGraph

load_dotenv()


# ==============================================================
# Define State
# ==============================================================
# `Annotated` attaches a reducer to a state key. `operator.add` concatenates
# list updates from parallel nodes instead of the default overwrite behavior.


class State(TypedDict):
    """State with accumulating list of messages."""

    messages: Annotated[list[str], operator.add]


# ==============================================================
# Define Nodes
# ==============================================================


def node_a(state: State) -> State:
    print(f"--- Node A --- \n  - Received: {state['messages']}\n")
    return {"messages": ["A"]}


def node_b(state: State) -> State:
    print(f"--- Node B --- \n  - Received: {state['messages']}\n")
    return {"messages": ["B"]}


def node_c(state: State) -> State:
    print(f"--- Node C --- \n  - Received: {state['messages']}\n")
    return {"messages": ["C"]}


def node_d(state: State) -> State:
    print(f"--- Node D --- \n  - Received: {state['messages']}\n")
    return {"messages": ["D"]}


def node_e(state: State) -> State:
    print(f"--- Node E --- \n  - Received merged state: {state['messages']}")
    return {"messages": ["E"]}


# ==============================================================
# Build Graph
# ==============================================================

# Initiate the graph
builder = StateGraph(State)

# Add nodes
builder.add_node("A", node_a)
builder.add_node("B", node_b)
builder.add_node("C", node_c)
builder.add_node("D", node_d)
builder.add_node("E", node_e)

# Add edges
builder.add_edge(START, "A")
builder.add_edge("A", "B")
builder.add_edge("A", "C")
builder.add_edge("A", "D")
builder.add_edge("B", "E")
builder.add_edge("C", "E")
builder.add_edge("D", "E")
builder.add_edge("E", END)

# Compile the graph (runnable object)
graph = builder.compile()


# ==============================================================
# Run Graph
# ==============================================================
print("=" * 60)
print("Running the graph...")
print("=" * 60)

# Invoke the graph
initial_state = State(messages=["Initial String"])
response = graph.invoke(initial_state)

print(f"\nResult: {response}")


# ==============================================================
# Visualize Graph
# ==============================================================
print("\n" + "=" * 60)
print("Visualizing the graph:")
print("=" * 60)

# Display the graph
display(Image(graph.get_graph().draw_mermaid_png()))
