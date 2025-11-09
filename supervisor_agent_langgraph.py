"""
Personal Assistant Supervisor - Pure LangGraph Implementation

This example demonstrates the subgraph pattern for multi-agent systems.
A supervisor agent coordinates specialized sub-agents (calendar and email)
that are implemented as full subgraphs.

Key differences from supervisor_agent.py:
- Uses explicit LangGraph subgraphs instead of create_agent
- No HumanInTheLoopMiddleware (removed for simplicity)
- Same functionality: calendar scheduling and email management
"""

from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()

# ============================================================================
# Step 1: Define low-level API tools (stubbed)
# ============================================================================


@tool
def create_calendar_event(
    title: str,
    start_time: str,  # ISO format: "2024-01-15T14:00:00"
    end_time: str,  # ISO format: "2024-01-15T15:00:00"
    attendees: list[str],  # email addresses
    location: str = "",
) -> str:
    """Create a calendar event. Requires exact ISO datetime format."""
    return f"Event created: {title} from {start_time} to {end_time} with {len(attendees)} attendees"


@tool
def send_email(
    to: list[str],  # email addresses
    subject: str,
    body: str,
    cc: list[str] = [],
) -> str:
    """Send an email via email API. Requires properly formatted addresses."""
    return f"Email sent to {', '.join(to)} - Subject: {subject}"


@tool
def get_available_time_slots(
    attendees: list[str],
    date: str,  # ISO format: "2024-01-15"
    duration_minutes: int,
) -> list[str]:
    """Check calendar availability for given attendees on a specific date."""
    return ["09:00", "14:00", "16:00"]


# ============================================================================
# Step 2: Create specialized sub-agent subgraphs
# ============================================================================

model = init_chat_model("openai:gpt-4o-mini")


def create_calendar_subgraph():
    """
    Create a calendar agent subgraph.

    This subgraph handles calendar-related tasks:
    - Parsing natural language into ISO datetime
    - Checking availability
    - Creating events
    """
    llm_with_tools = model.bind_tools([create_calendar_event, get_available_time_slots])

    workflow = StateGraph(MessagesState)

    def calendar_agent(state: MessagesState) -> dict:
        """Calendar agent node with system prompt"""
        messages = state["messages"]
        system_msg = SystemMessage(
            content=(
                "You are a calendar scheduling assistant. "
                "Parse natural language scheduling requests (e.g., 'next Tuesday at 2pm') "
                "into proper ISO datetime formats. "
                "Use get_available_time_slots to check availability when needed. "
                "Use create_calendar_event to schedule events. "
                "Always confirm what was scheduled in your final response."
            )
        )
        response = llm_with_tools.invoke([system_msg] + messages)
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> Literal["tools", "end"]:
        """Route to tools or end based on tool calls"""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"

    # Build the graph
    workflow.add_node("agent", calendar_agent)
    workflow.add_node("tools", ToolNode([create_calendar_event, get_available_time_slots]))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    workflow.add_edge("tools", "agent")

    return workflow.compile()


def create_email_subgraph():
    """
    Create an email agent subgraph.

    This subgraph handles email-related tasks:
    - Composing professional emails
    - Extracting recipient information
    - Sending emails
    """
    llm_with_tools = model.bind_tools([send_email])

    workflow = StateGraph(MessagesState)

    def email_agent(state: MessagesState) -> dict:
        """Email agent node with system prompt"""
        messages = state["messages"]
        system_msg = SystemMessage(
            content=(
                "You are an email assistant. "
                "Compose professional emails based on natural language requests. "
                "Extract recipient information and craft appropriate subject lines and body text. "
                "Use send_email to send the message. "
                "Always confirm what was sent in your final response."
            )
        )
        response = llm_with_tools.invoke([system_msg] + messages)
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> Literal["tools", "end"]:
        """Route to tools or end based on tool calls"""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"

    # Build the graph
    workflow.add_node("agent", email_agent)
    workflow.add_node("tools", ToolNode([send_email]))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# Create the subgraphs
calendar_graph = create_calendar_subgraph()
email_graph = create_email_subgraph()


# ============================================================================
# Step 3: Wrap sub-agent subgraphs as tools for the supervisor
# ============================================================================


@tool
def schedule_event(request: str) -> str:
    """Schedule calendar events using natural language.

    Use this when the user wants to create, modify, or check calendar appointments.
    Handles date/time parsing, availability checking, and event creation.

    Input: Natural language scheduling request (e.g., 'meeting with design team
    next Tuesday at 2pm')
    """
    result = calendar_graph.invoke({"messages": [HumanMessage(content=request)]})
    return result["messages"][-1].content


@tool
def manage_email(request: str) -> str:
    """Send emails using natural language.

    Use this when the user wants to send notifications, reminders, or any email
    communication. Handles recipient extraction, subject generation, and email
    composition.

    Input: Natural language email request (e.g., 'send them a reminder about
    the meeting')
    """
    result = email_graph.invoke({"messages": [HumanMessage(content=request)]})
    return result["messages"][-1].content


# ============================================================================
# Step 4: Create the supervisor agent subgraph
# ============================================================================


def create_supervisor_graph():
    """
    Create the supervisor agent subgraph.

    The supervisor coordinates between calendar and email sub-agents,
    breaking down complex requests into appropriate tool calls.
    """
    llm_with_tools = model.bind_tools([schedule_event, manage_email])

    workflow = StateGraph(MessagesState)

    def supervisor_agent(state: MessagesState) -> dict:
        """Supervisor agent node with system prompt"""
        messages = state["messages"]
        system_msg = SystemMessage(
            content=(
                "You are a helpful personal assistant. "
                "You can schedule calendar events and send emails. "
                "Break down user requests into appropriate tool calls and coordinate the results. "
                "When a request involves multiple actions, use multiple tools in sequence."
            )
        )
        response = llm_with_tools.invoke([system_msg] + messages)
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> Literal["tools", "end"]:
        """Route to tools or end based on tool calls"""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"

    # Build the graph
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("tools", ToolNode([schedule_event, manage_email]))

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges("supervisor", should_continue, {"tools": "tools", "end": END})
    workflow.add_edge("tools", "supervisor")

    return workflow.compile()


# ============================================================================
# Step 5: Initialize and use the supervisor
# ============================================================================

supervisor_agent = create_supervisor_graph()


if __name__ == "__main__":
    user_request = (
        "Schedule a meeting with the design team on Tuesday at 2pm for 1 hour, "
        "and send them an email reminder about reviewing the new mockups."
    )

    # Invoke the supervisor agent
    result = supervisor_agent.invoke({"messages": [HumanMessage(content=user_request)]})

    for message in result["messages"]:
        message.pretty_print()
