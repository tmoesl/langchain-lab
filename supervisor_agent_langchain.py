"""
Personal Assistant Supervisor - Pure LangChain Implementation

This example demonstrates the tool calling pattern for multi-agent systems.
A supervisor agent coordinates specialized sub-agents (calendar and email)
that are wrapped as tools.
"""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

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
# Step 2: Create specialized sub-agents
# ============================================================================

model = init_chat_model("openai:gpt-4o-mini")  # for example

calendar_agent = create_agent(
    model,
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=(
        "You are a calendar scheduling assistant. "
        "Parse natural language scheduling requests (e.g., 'next Tuesday at 2pm') "
        "into proper ISO datetime formats. "
        "Use get_available_time_slots to check availability when needed. "
        "Use create_calendar_event to schedule events. "
        "Always confirm what was scheduled in your final response."
    ),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"create_calendar_event": False},
            description_prefix="Calendar event pending approval",
        )
    ],
)

email_agent = create_agent(
    model,
    tools=[send_email],
    system_prompt=(
        "You are an email assistant. "
        "Compose professional emails based on natural language requests. "
        "Extract recipient information and craft appropriate subject lines and body text. "
        "Use send_email to send the message. "
        "Always confirm what was sent in your final response."
    ),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"send_email": True},
            description_prefix="Outbound email pending approval",
        )
    ],
)

# ============================================================================
# Step 3: Wrap sub-agents as tools for the supervisor
# ============================================================================


@tool
def schedule_event(request: str) -> str:
    """Schedule calendar events using natural language.

    Use this when the user wants to create, modify, or check calendar appointments.
    Handles date/time parsing, availability checking, and event creation.

    Input: Natural language scheduling request (e.g., 'meeting with design team
    next Tuesday at 2pm')
    """
    result = calendar_agent.invoke({"messages": [{"role": "user", "content": request}]})
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
    result = email_agent.invoke({"messages": [{"role": "user", "content": request}]})
    return result["messages"][-1].content


# ============================================================================
# Step 4: Create the supervisor agent
# ============================================================================

memory = InMemorySaver()

supervisor_agent = create_agent(
    model,
    tools=[schedule_event, manage_email],
    system_prompt=(
        "You are a helpful personal assistant. "
        "You can schedule calendar events and send emails. "
        "Break down user requests into appropriate tool calls and coordinate the results. "
        "When a request involves multiple actions, use multiple tools in sequence."
        "In case tool calls are rejected, provide the reason for the rejection in the response"
    ),
    checkpointer=memory,
)

# ============================================================================
# Step 5: Use the supervisor
# ============================================================================

if __name__ == "__main__":
    user_request = (
        "Schedule a meeting with the design team on Tuesday at 2pm for 1 hour, "
        "and send them an email reminder about reviewing the new mockups."
    )

    config = {"configurable": {"thread_id": "1"}}

    # Step 1: Initial invoke
    result = supervisor_agent.invoke(
        {"messages": [HumanMessage(content=user_request)]}, config=config
    )

    # Step 2: Handle interrupts (approve or reject once, no retry)
    if "__interrupt__" in result:
        interrupts = result["__interrupt__"]

        resume_dict = {}
        for interrupt_ in interrupts:
            print(f"\n\033[1;31mINTERRUPTED: {interrupt_.id}\033[0m")
            for request in interrupt_.value["action_requests"]:
                print(f"\033[1;33m{request['description']}\033[0m")

            decision_type = "reject"  # Change to "approve" to allow execution
            message = {
                "approve": f"Action {decision_type}ed. Tool was executed successfully.",
                "reject": f"Action {decision_type}ed. Tool was not executed (security policy).",
            }

            resume_dict[interrupt_.id] = {
                "decisions": [
                    {
                        "type": decision_type,
                        "message": message[decision_type],
                    }
                ]
            }
            print(f"\033[1;32m✓ Decision: {decision_type.upper()}\033[0m\n")

        # Step 3: Resume once (sub-agents will respond to rejection)
        result = supervisor_agent.invoke(Command(resume=resume_dict), config)

    # Step 4: Print final outcome
    if "messages" in result:
        for message in result["messages"]:
            message.pretty_print()
