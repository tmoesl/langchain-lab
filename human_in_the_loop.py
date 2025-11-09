"""
Human-in-the-Loop workflow with simple LangChain agents.

Demonstrates pausing agent execution for human approval before executing
sensitive operations. Uses HumanInTheLoopMiddleware to interrupt tool calls
and wait for user decisions (approve/edit/reject).

Key components:
- HumanInTheLoopMiddleware: Pauses execution for specific tools
- InMemorySaver: Maintains conversation state during interrupts
- Command: Resumes execution with decisions

Configuration:
- interrupt_on={tool_name: True} - Requires approval (all decisions allowed)
- interrupt_on={tool_name: False} - Auto-approve (no interrupt)
- interrupt_on={tool_name: {"allowed_decisions": [...]}} - Restrict decisions

How it works:
The HumanInTheLoopMiddleware.after_model is the gatekeeper that:
- Intercepts every tool call before execution
- Routes based on decision:
  * approve/edit → tools (execute immediately)
  * reject → model (try again, no tool execution)
- Repeats until model has no more tool calls

Example flows:
1. APPROVE: model → interrupt → [human approves] → tools → model → end
2. EDIT: model → interrupt → [human edits args] → tools → model → end
3. REJECT: model → interrupt → [human rejects] → model (new attempt) →
           interrupt again → [human approves] → tools → model → end

Note: Reject creates a loop because it sends feedback to the model without
executing the tool, causing the model to generate a new tool call.
"""

import json
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()

# Initialize agent components
model = ChatOpenAI(model="gpt-4o-mini")
memory = InMemorySaver()  # Required for state persistence during interrupts
config = {"configurable": {"thread_id": "human-in-the-loop1"}}  # Required for conversation tracking


# Define tool
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
    return f"Email sent successfully to {recipient}"


# Create agent
agent = create_agent(
    model=model,
    tools=[send_email],
    system_prompt="You are a helpful assistant that can send emails.",
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"send_email": True})],
    checkpointer=memory,
)

# Incoming email (example)
incoming_email = """
Respond to the following email:
From: partner@startup.com
Subject: Budget proposal for Q1 2026
Body: Hey Sydney, we need your sign-off on the $1M engineering budget for Q1.
Can you approve and reply by EOD? This is critical for our timeline.
"""

# Invoke agent with incoming email
result = agent.invoke({"messages": [HumanMessage(content=incoming_email)]}, config=config)


# Handle interrupts
while "__interrupt__" in result:
    action = result["__interrupt__"][0].value["action_requests"][0]

    print(f"\n📧 Approval needed: {action['name']}")
    print(json.dumps(action["args"], indent=2))

    decision = input("\nAction (approve/edit/reject): ")

    match decision:
        case "approve":
            resume_decision = {"decisions": [{"type": "approve", "message": "Email approved."}]}

        case "edit":
            resume_decision = {
                "decisions": [
                    {
                        "type": "edit",
                        "edited_action": {
                            "name": "send_email",
                            "args": {
                                "recipient": "partner@startup.com",
                                "subject": "Budget proposal for Q1 2026",
                                "body": "I can only approve up to 500k, please send over details.",
                            },
                        },
                    }
                ]
            }

        case "reject":
            resume_decision = {
                "decisions": [{"type": "reject", "message": "Ask for more budget details first."}]
            }

        case _:
            print("Invalid action")
            sys.exit(1)

    # Resume execution with decision
    result = agent.invoke(Command(resume=resume_decision), config=config)

    for message in result["messages"]:
        message.pretty_print()

print(f"Assistant: {result['messages'][-1].content}")
