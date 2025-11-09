"""
Nodes and State

LangGraph functions like a programming language for building AI workflows, where data (“state”)
flows through a network of functional units (“nodes”) connected by control paths (“edges”).

Key points about nodes and state:
- State: shared data passed and updated across nodes, usually a Python structure.
- Nodes: functions that take state as input and return updates to it.
- Edges: control execution flow between nodes (static or conditional) and (serial, parallel).
- Checkpointing: persists state for recovery and resilience after failures.
- Graphs: are stateless; all the data lives in the shared state object passed between nodes.
- Human in the Loop: Pause the graph execution and wait for human input.

Execution:
- Runtime: On `invoke`, LangGraph initializes state and selects nodes to run.
- State Flow: Each node processes the current state and returns updates.
- Graph Return: After all nodes complete, the graph outputs the final state.
"""

from typing import Annotated, TypedDict

from IPython.display import Image, display
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def agent_node(state: AgentState) -> AgentState:
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = model.invoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)

graph = builder.compile()

response = graph.invoke({"messages": [HumanMessage(content="What is the capital of France?")]})
print(response["messages"][-1].content)

# Display the graph
display(Image(graph.get_graph().draw_mermaid_png()))
# print(graph.get_graph().draw_mermaid())
