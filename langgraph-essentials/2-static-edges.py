"""
Static Edges - Control Flow and State Reducers

Demonstrates LangGraph's edge types and state management across all node executions.

Edge Types:
- Static edges: Always taken (solid lines)
- Conditional edges: Decision-based routing (dashed lines)
- MapReduce edges: Variable number of downstream nodes

Execution Model:
- Super Steps: All active nodes complete before continuing
- Sequential & Parallel: Reducers work for both execution patterns
- State accumulation: Values merge across ALL node updates, not just parallel

State Management:
- Default: Last write overwrites previous state
- Reducers: Define how ALL writes to same key merge (sequential + parallel)
- operator.add: Accumulates values from every node that updates the key
- add_messages: Smart merging for BaseMessage objects across all nodes

"""

import operator
from typing import Annotated, TypedDict

from IPython.display import Image, display
from langgraph.graph import END, START, StateGraph


class AgentState(TypedDict):
    nlist: Annotated[list[str], operator.add]


def node_a(state: AgentState) -> AgentState:
    print(f"Adding 'A' to {state['nlist']}")
    return {"nlist": ["A"]}


def node_b(state: AgentState) -> AgentState:
    print(f"Adding 'B' to {state['nlist']}")
    return {"nlist": ["B"]}


def node_c(state: AgentState) -> AgentState:
    print(f"Adding 'C' to {state['nlist']}")
    return {"nlist": ["C"]}


def node_d(state: AgentState) -> AgentState:
    print(f"Adding 'D' to {state['nlist']}")
    return {"nlist": ["D"]}


def node_e(state: AgentState) -> AgentState:
    print(f"Adding 'E' to {state['nlist']}")
    return {"nlist": ["E"]}


builder = StateGraph(AgentState)

# Add nodes
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_node("c", node_c)
builder.add_node("d", node_d)
builder.add_node("e", node_e)
# Add edges
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", "e")
builder.add_edge("a", "c")
builder.add_edge("c", "e")
builder.add_edge("a", "d")
builder.add_edge("d", "e")
builder.add_edge("e", END)

# Compile and display the graph
graph = builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))

# Invoke the graph
initial_state = AgentState(nlist=["Initial String"])
response = graph.invoke(initial_state)
print(f"Assistant: {response['nlist']}")
