import sys
from typing import TypedDict

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langmem.short_term import RunningSummary, SummarizationNode

load_dotenv()


# ==============================================================
# Example 8: Chat Agent with Short-Term Memory (Auto-Summarization)
# ==============================================================
# - SummarizationNode auto-summarizes when exceeding token threshold
# - Keeps recent messages + running summary within budget
# - InMemorySaver persists state across invocations


MAX_SUMMARY_TOKENS = 256
MAX_TOKENS = 2000
MAX_TOKENS_BEFORE_SUMMARY = 600

# Initialize models
model = ChatOpenAI(model="gpt-4o-mini")
summarization_model = model.bind(max_tokens=MAX_SUMMARY_TOKENS)


class AgentState(MessagesState):
    context: dict[str, RunningSummary]


class LLMInputState(TypedDict):
    summarized_messages: list[AnyMessage]
    context: dict[str, RunningSummary]


# Optimized summarization node configuration
summarization_node = SummarizationNode(
    token_counter=model.get_num_tokens_from_messages,  # type: ignore
    model=summarization_model,  # LLM used for summarization
    max_tokens=MAX_TOKENS,  # total output token budget (summary + tail)
    max_tokens_before_summary=MAX_TOKENS_BEFORE_SUMMARY,  # trigger summarization when exceeding
    max_summary_tokens=MAX_SUMMARY_TOKENS,  # reserve tokens for summary in budget calculation
)


def call_model_node(state: LLMInputState) -> dict:
    response = model.invoke(state["summarized_messages"])
    return {"messages": [response]}


# Build graph
builder = StateGraph(AgentState)
builder.add_node("call_model", call_model_node)
builder.add_node("summarize", summarization_node)
builder.add_edge(START, "summarize")
builder.add_edge("summarize", "call_model")

# Compile with checkpointer
memory = InMemorySaver()
agent = builder.compile(checkpointer=memory)

# Thread ID tracks conversation
config: RunnableConfig = {"configurable": {"thread_id": "1"}}

# Visualize graph if in interactive mode
if hasattr(sys, "ps1"):
    display(Image(agent.get_graph().draw_mermaid_png()))


# Chat loop: only pass new messages, checkpointer handles history
# @traceable(name="chat_session", run_type="chain")
def run_chatbot() -> dict | None:
    response = None
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
    return response


def show_conversation_context(response: dict):
    """Show conversation context in a readable format."""
    # Display all messages
    if messages := response.get("messages"):
        print("\n\nMESSAGE HISTORY")
        for msg in messages:
            msg.pretty_print()

    if summary := response.get("context", {}).get("running_summary"):
        print("\n\nCONVERSATION SUMMARY")
        print(f"{summary.summary}\n")


if __name__ == "__main__":
    response = run_chatbot()

    if response:
        user_input = input("Should we display the conversation context? (y/n): ")
        if user_input.lower() == "y":
            show_conversation_context(response)
