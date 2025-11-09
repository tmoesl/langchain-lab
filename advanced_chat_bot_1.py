import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()


# ==============================================================
# Example 7: Chat Agent with Short-Term Memory (Manual Summarization)
# ==============================================================
# - Router triggers summarization when messages exceed threshold
# - Summarizes all messages, keeps last 2, removes older ones
# - InMemorySaver persists state across invocations

MAX_SUMMARY_TOKENS = 256
MESSAGE_THRESHOLD = 10

# Initialize models
model = ChatOpenAI(model="gpt-4o-mini")
summarization_model = model.bind(max_tokens=MAX_SUMMARY_TOKENS)


class AgentState(TypedDict):
    """State schema: list of messages with add_messages reducer"""

    messages: Annotated[list[BaseMessage], add_messages]
    summary: str


def call_model_node(state: AgentState) -> dict:
    """Node: calls LLM and returns response"""

    full_history = state["messages"]
    print(f"Full history length: {len(full_history)}")

    system_prompt = "You are a helpful assistant."

    # Build system message with summary if available
    if summary := state.get("summary", ""):
        system_content = (
            f"{system_prompt}\n\nContext from earlier conversation:\n{summary}"
            "Continue the conversation naturally using this context."
        )
    else:
        system_content = system_prompt

    # Inject system message dynamically | Not saved to messages
    llm_messages = [SystemMessage(content=system_content)] + full_history

    print(f"Sending {len(llm_messages)} messages to LLM")

    response = model.invoke(llm_messages)
    return {"messages": [response]}


def summarize_history(state: AgentState) -> dict:
    """Node: summarizes old messages and removes them from history"""

    messages = state["messages"]

    messages_to_summarize = [msg for msg in messages if not isinstance(msg, SystemMessage)]

    print(f"Summarizing {len(messages_to_summarize)} messages")

    # Build summary prompt
    if existing_summary := state.get("summary", ""):
        prompt_msg = (
            f"This is a summary of the conversation to date: {existing_summary}\n\n"
            "Extend the summary by taking into account the new messages above."
        )
    else:
        prompt_msg = "Create a concise summary of the conversation above:"

    # Add prompt to ALL messages for summarization (LLM sees full context)
    summary_input = messages_to_summarize + [HumanMessage(content=prompt_msg)]

    # Generate summary
    response = summarization_model.invoke(summary_input)
    new_summary = response.content

    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=msg.id) for msg in messages_to_summarize[:-2]]  # type: ignore
    print(f"Removing {len(delete_messages)} messages from history")

    return {"summary": new_summary, "messages": delete_messages}


def should_continue(state: AgentState) -> str:
    """Router: decide whether to summarize"""
    if len(state["messages"]) > MESSAGE_THRESHOLD:
        return "summarize_conversation"
    return "generate_response"


# Build graph
graph = StateGraph(AgentState)
graph.add_node("call_model", call_model_node)
graph.add_node("summarize", summarize_history)
graph.add_conditional_edges(
    START,
    should_continue,
    {"summarize_conversation": "summarize", "generate_response": "call_model"},
)
graph.add_edge("summarize", "call_model")
graph.add_edge("call_model", END)

# Compile with checkpointer
memory = InMemorySaver()
agent = graph.compile(checkpointer=memory)

# Thread ID tracks conversation
config: RunnableConfig = {"configurable": {"thread_id": "1"}}

# Visualize graph if in interactive mode
if hasattr(sys, "ps1"):
    display(Image(agent.get_graph().draw_mermaid_png()))


# Chat loop: only pass new messages, checkpointer handles history
# @traceable(name="chat_session", run_type="chain")
def run_chatbot():
    while True:
        user_input = input("Enter: ")
        if user_input.lower() == "exit":
            break

        response = agent.invoke({"messages": [HumanMessage(content=user_input)]}, config)  # type: ignore

        # Show user msg if in interactive mode (only)
        if hasattr(sys, "ps1"):
            print(f"User: {user_input}")

        last_message = response["messages"][-1]
        print(f"Assistant: {last_message.content}\n")


if __name__ == "__main__":
    run_chatbot()
