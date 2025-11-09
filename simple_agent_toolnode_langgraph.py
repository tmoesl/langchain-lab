from typing import Annotated, TypedDict

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")
summarization_model = model.bind(max_tokens=256)


class AgentState(TypedDict):
    """State schema: list of messages and LLM call count"""

    messages: Annotated[list[AnyMessage], add_messages]
    llm_call_count: int


@tool
def add(num1: int, num2: int) -> int:
    """Add two numbers"""
    return num1 + num2


@tool
def multiply(num1: int, num2: int) -> int:
    """Multiply two numbers"""
    return num1 * num2


# Augment the LLM with tools
# tools: list of instances of StructuredTool (via @tool decorator)
tools = [add, multiply]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)


def agent_node(state: AgentState) -> dict:
    """Calls LLM with tools, returns response and updates the LLM call count"""
    system_msg = SystemMessage(
        content="""You are a helpful assistant that performs arithmetic operations.
        For tasks with dependencies, call tools sequentially. Wait for each result 
        before using it in the next tool call. Plan your actions and call the tools accordingly."""
    )
    messages = [system_msg] + state["messages"]
    response = model_with_tools.invoke(messages)
    counter = state.get("llm_call_count", 0) + 1

    return {"messages": [response], "llm_call_count": counter}


def should_continue(state: AgentState) -> str:
    """Routes to tools if the last message has tool calls, otherwise ends."""
    last_message = state["messages"][-1]

    if last_message.tool_calls:  # type: ignore
        return "tools"
    else:
        return END


graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, ["tools", END])
graph.add_edge("tools", "agent")

agent = graph.compile()

# Visualize the graph
display(Image(agent.get_graph().draw_mermaid_png()))

# Test the agent
message = [HumanMessage(content="Multiply 2 by 5 and then add 10 to the result.")]
response = agent.invoke({"messages": message})  # type: ignore
print(f"Assistant: {response['messages'][-1].content}")

# Debug the agent (observe all messages)
for msg in response["messages"]:
    msg.pretty_print()
