"""
LangChain Middleware - Human-in-the-Loop (HITL)

Middleware provides hooks to control agent execution. Human-in-the-Loop (HITL) pauses the agent
at a critical step (like a tool call) and waits for human input before proceeding
This is essential for high-stakes operations where human oversight is required.

How It Works:
-------------
1. HumanInTheLoopMiddleware: Configured to interrupt on specific tool calls.
2. Checkpointer: Required to save the agent's state during the interruption.
3. Interrupt Signal: agent.invoke() returns `__interrupt__` with the pending action.
4. User Decision: The user can 'approve', 'reject', or 'edit' the action.
5. Resume Execution: A `Command(resume=...)` is passed back to agent.invoke()
   to continue with the user's decision.

Key Concepts:
-------------
- Control Flow: HITL is a powerful way to control the agent's execution path.
- Statefulness: Checkpointers are essential for maintaining state across interruptions.
- Safety and Compliance: Critical for sensitive actions like sending emails,
  making purchases, or modifying databases.

This example simulates an agent responding to a budget proposal email. The agent
will draft a response, but a human must approve, edit, or reject it before sending.

Note on 'Reject' vs. 'Edit':
- Reject: Creates a feedback loop. The tool is NOT run, and the rejection
  reason is sent to the model so it can try again.
- Edit (in this example): Directly executes the tool with the modified
  arguments, bypassing a second model call for simplicity.

Reference: https://docs.langchain.com/oss/python/langchain/middleware
"""

import json

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()

# ==============================================================
# Agent Components
# ==============================================================

# A checkpointer is required for HITL to save the agent's state.
memory = InMemorySaver()

# Each conversation needs a unique thread_id.
config = {"configurable": {"thread_id": "hitl-thread-1"}}

# ==============================================================
# Tools
# ==============================================================


@tool(parse_docstring=True)
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email to a recipient.

    Args:
        recipient: Email address of the recipient.
        subject: Subject line of the email.
        body: Body content of the email.

    Returns:
        Confirmation message.
    """
    print(f"\nSending email to {recipient}...")
    return f"Email sent successfully to {recipient}"


# ==============================================================
# Agent with Human-in-the-Loop Middleware
# ==============================================================

agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[send_email],
    system_prompt="You are a helpful assistant. Your task is to draft email responses.",
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"send_email": True})],
    checkpointer=memory,
)

# Visualize the graph
display(Image(agent.get_graph().draw_mermaid_png()))

# ==============================================================
# Interactive Chat with Human-in-the-Loop
# ==============================================================

# Incoming email (in production from API)
incoming_email = """
From: partner@startup.com
Subject: Budget proposal for Q1 2026
Body: Hey Sydney, we need your sign-off on the $1M engineering budget for Q1.
Can you approve and reply by EOD? This is critical for our timeline.
"""

print("🤖 Email Agent with Human-in-the-Loop")
print("Agent will now process the incoming email and draft a response.")

# 1. Initial invocation with incoming email
response = agent.invoke({"messages": [HumanMessage(content=incoming_email)]}, config=config)  # type: ignore

# 2. Loop to handle interruptions
while "__interrupt__" in response:
    action = response["__interrupt__"][-1].value["action_requests"][-1]

    print("\n\033[1;3;31mTool execution requires approval.\033[0m\n")
    print(f"\033[1;3;31mTool: {action['name']}\033[0m")
    print(f"\033[1;3;31mArgs: {json.dumps(action['args'], indent=2)}\033[0m\n")

    decision = ""
    while decision.lower() not in ["approve", "edit", "reject", "exit"]:
        decision = input("Your decision (approve / edit / reject) or stop (exit): ")

    if decision.lower() == "exit":
        break

    resume_decision = None
    if decision.lower() == "approve":
        resume_decision = {"decisions": [{"type": "approve"}]}

    elif decision.lower() == "reject":
        rejection_reason = input("Reason for rejection: ")
        resume_decision = {"decisions": [{"type": "reject", "message": rejection_reason}]}

    elif decision.lower() == "edit":
        # NOTE FOR PRODUCTION: For maximum safety, you could instead use a 'reject'
        # decision with feedback, instructing the model to reissue the tool call
        # with the edits. This would force a second 'approve' step, ensuring the
        # user gives final sign-off on the exact arguments being executed.
        print("Applying a pre-defined edit: Changing budget approval to $500k.")
        edited_args = {
            "recipient": "partner@startup.com",
            "subject": "Re: Budget proposal for Q1 2026",
            "body": "I can only approve up to $500k at this time. Please send over a revised proposal.",
        }
        resume_decision = {
            "decisions": [
                {
                    "type": "edit",
                    "edited_action": {
                        "name": "send_email",
                        "args": edited_args,
                    },
                }
            ]
        }

    # 3. Resume execution with the user's decision
    response = agent.invoke(Command(resume=resume_decision), config=config)  # type: ignore


# 4. Print the final response from the agent if the loop wasn't exited
if "messages" in response and decision.lower() != "exit":
    print(f"\nAssistant: {response['messages'][-1].content}")
