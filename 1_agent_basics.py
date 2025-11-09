from typing import Annotated, TypedDict

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import trim_messages
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState, add_messages

load_dotenv()


# ==============================================================
# Example 1: Simple LLM Call (No Graph)
# ==============================================================
# - Direct LLM invocation without graph structure
# - Single message, no state management

llm = ChatOpenAI(model="gpt-4o-mini")

message = "What is the capital of France?"
response = llm.invoke([HumanMessage(content=message)])
print(f"AI: {response.content}\n")


# ==============================================================
# Example 2: Simple LLM Call with StateGraph
# ==============================================================
# - Basic graph: State → Node → Edges → Compile → Invoke
# - Single invocation, no conversation history

llm = ChatOpenAI(model="gpt-4o-mini")


class AgentState(TypedDict):
    """State schema: list of messages"""

    messages: list[BaseMessage]


def call_llm(state: AgentState) -> dict:
    """Node: calls LLM and returns response"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# Build graph
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_edge(START, "llm")
graph.add_edge("llm", END)

# Compile and invoke
agent = graph.compile()
response = agent.invoke({"messages": [HumanMessage(content=message)]})
print(f"AI: {response['messages'][-1].content}\n")

# Visualize
display(Image(agent.get_graph().draw_mermaid_png()))


# ==============================================================
# Example 3: Chat Agent with Short-Term Memory (Manual)
# ==============================================================
# - Uses add_messages reducer
# - Manually maintains conversation history in memory
# - No checkpointer, state lost when loop exits

llm = ChatOpenAI(model="gpt-4o-mini")


class AgentState(TypedDict):
    """State schema: list of messages with add_messages reducer"""

    messages: Annotated[list[BaseMessage], add_messages]


def call_llm(state: AgentState) -> dict:
    """Node: calls LLM and returns response"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# Build graph
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_edge(START, "llm")
graph.add_edge("llm", END)

# Compile
agent = graph.compile()

# Chat loop: manually maintain conversation history
history = {"messages": [SystemMessage(content="You are a helpful assistant.")]}

while True:
    user_input = input("Enter: ")
    if user_input.lower() == "exit":
        break

    # Add user message to history
    history["messages"].append(HumanMessage(content=user_input))  # type: ignore

    # Invoke with full history
    response = agent.invoke({"messages": history["messages"]})  # type: ignore

    # Update history with full conversation
    history["messages"] = response["messages"]

    print(f"User: {user_input}")
    print(f"AI: {response['messages'][-1].content}\n")


# ==============================================================
# Example 4: Chat Agent with Short-Term Memory (InMemorySaver)
# ==============================================================
# - Uses InMemorySaver to persist state per thread_id
# - Only pass new messages, checkpointer handles history
# - Custom state definition with add_messages reducer

llm = ChatOpenAI(model="gpt-4o-mini")


class AgentState(TypedDict):
    """State schema: list of messages with add_messages reducer"""

    messages: Annotated[list[BaseMessage], add_messages]


def call_llm(state: AgentState) -> dict:
    """Node: calls LLM and returns response"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# Build graph
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_edge(START, "llm")
graph.add_edge("llm", END)

# Compile with checkpointer
memory = InMemorySaver()
agent = graph.compile(checkpointer=memory)

# Thread ID tracks conversation
config: RunnableConfig = {"configurable": {"thread_id": "conversation-1"}}

# Initialize conversation
agent.invoke({"messages": [SystemMessage(content="You are a helpful assistant.")]}, config)

# Chat loop: only pass new messages, checkpointer handles history
while True:
    user_input = input("Enter: ")
    if user_input.lower() == "exit":
        break

    response = agent.invoke({"messages": [HumanMessage(content=user_input)]}, config)

    print(f"User: {user_input}")
    print(f"AI: {response['messages'][-1].content}\n")


# ==============================================================
# Example 5: Chat Agent with Short-Term Memory (MessagesState)
# ==============================================================
# - Same as Example 4 but uses built-in MessagesState
# - No custom state definition needed
# - InMemorySaver persists state per thread_id

llm = ChatOpenAI(model="gpt-4o-mini")


def call_llm(state: MessagesState) -> dict:
    """Node: calls LLM and returns response"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# Build graph
graph = StateGraph(MessagesState)
graph.add_node("llm", call_llm)
graph.add_edge(START, "llm")
graph.add_edge("llm", END)

# Compile with checkpointer
memory = InMemorySaver()
agent = graph.compile(checkpointer=memory)

# Thread ID tracks conversation
config: RunnableConfig = {"configurable": {"thread_id": "conversation-1"}}

# Initialize conversation
agent.invoke({"messages": [SystemMessage(content="You are a helpful assistant.")]}, config)

# Chat loop: only pass new messages, checkpointer handles history
while True:
    user_input = input("Enter: ")
    if user_input.lower() == "exit":
        break

    response = agent.invoke({"messages": [HumanMessage(content=user_input)]}, config)

    print(f"User: {user_input}")
    print(f"AI: {response['messages'][-1].content}\n")

# ==============================================================
# Example 6: Chat Agent with Short-Term Memory (Trim Messages)
# ==============================================================
# - Trims message history to last 5 messages before LLM call
# - Always keeps system message (include_system=True)
# - InMemorySaver persists full history, trim only for LLM

llm = ChatOpenAI(model="gpt-4o-mini")


class AgentState(TypedDict):
    """State schema: list of messages with add_messages reducer"""

    messages: Annotated[list[BaseMessage], add_messages]


def call_llm(state: AgentState) -> dict:
    """Node: calls LLM and returns response, trims messages to 5"""

    full_history = state["messages"]

    trimmed_history = trim_messages(
        full_history,
        max_tokens=5,  # Keep last 5 messages
        strategy="last",  # Keep most recent
        token_counter=len,  # Count messages (not tokens)
        include_system=True,  # Always keep system message
        allow_partial=False,
    )

    response = llm.invoke(trimmed_history)
    return {"messages": [response]}


# Build graph
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_edge(START, "llm")
graph.add_edge("llm", END)

# Compile with checkpointer
memory = InMemorySaver()
agent = graph.compile(checkpointer=memory)

# Thread ID tracks conversation
config: RunnableConfig = {"configurable": {"thread_id": "conversation-1"}}

# Initialize conversation
agent.invoke({"messages": [SystemMessage(content="You are a helpful assistant.")]}, config)

# Chat loop: only pass new messages, checkpointer handles history
while True:
    user_input = input("Enter: ")
    if user_input.lower() == "exit":
        break

    response = agent.invoke({"messages": [HumanMessage(content=user_input)]}, config)

    print(f"User: {user_input}")
    print(f"AI: {response['messages'][-1].content}\n")
