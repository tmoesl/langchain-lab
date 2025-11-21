"""
LangGraph Nodes and State: The Building Blocks of Graph-Based Workflows

LangGraph functions like a programming language for building AI workflows, where data (“state”)
flows through a network of functional units (“nodes”) connected by control paths (“edges”).

Key Concepts:
-------------
- State: A shared data structure (e.g., TypedDict) that is passed between nodes and updated by them.
- Nodes: Functions that take the current state as input and return updates to be merged back into the state.
- Edges: Define the control flow, determining which node to execute next.
- Graphs: Are stateless; all application data is managed in the state.
- Checkpointing: Persists state for resilience, allowing the graph to recover from failures and be paused.
- Human-in-the-Loop: Allows for pausing the graph to wait for human input before continuing.

Execution Flow:
---------------
1. Initialize Graph: A StateGraph is instantiated with a state definition.
2. Add Nodes: Nodes (functions) are added to the graph with unique identifiers.
3. Define Edges: Edges connect nodes to define the sequence of operations. START and END are special nodes.
4. Compile: The graph definition is compiled into a runnable object.
5. Invoke: The graph is run by passing in an initial state. The runtime executes nodes, updates the state, and returns the final state.

References:
https://docs.langchain.com/oss/python/langgraph/graph-api
https://docs.langchain.com/oss/python/langgraph/graph-api#state
https://docs.langchain.com/oss/python/langgraph/graph-api#nodes
https://docs.langchain.com/oss/python/langgraph/graph-api#edges
"""

from typing import TypedDict

from dotenv import load_dotenv
from IPython.display import Image, display
from langgraph.graph import END, START, StateGraph

load_dotenv()


# ==============================================================
# Define State
# ==============================================================


class State(TypedDict):
    """A list of messages. This state is passed between nodes."""

    messages: list[str]


# ==============================================================
# Define Nodes
# ==============================================================


def greeting_node(state: State) -> State:
    """A node that overwrites the 'messages' list in the state with a greeting."""
    print("--- Executing Node: greeter ---")
    print(f"Received state: {state}")
    new_message = "Hello from the greeting node!"
    return {"messages": [new_message]}


# ==============================================================
# Build Graph
# ==============================================================

# Initiate the graph
builder = StateGraph(State)

# Add nodes and edges
builder.add_node("greeter", greeting_node)
builder.add_edge(START, "greeter")
builder.add_edge("greeter", END)

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

# Display the graph
print(graph.get_graph().draw_mermaid())
