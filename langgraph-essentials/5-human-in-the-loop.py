"""
Interrupts - Human in the Loop

Demonstrates pausing graph execution for external input using interrupt().
Unlike conditional branching that collects input during runtime, interrupts
suspend the graph and wait indefinitely for human response before resuming.

Key Concepts:
- interrupt(): Pauses execution and returns control to Python runtime
- Command(resume=...): Provides human input back to the interrupted node
- Checkpointer: Enables save/restore operation for suspended state
- Thread ID: Maintains conversation context across pause/resume cycles

Command Options:
- resume: Provides input to resume an interrupted node (after interrupt())
- update: Modifies graph state before continuing execution
- goto: Routes execution to specific nodes dynamically

How Interrupts Work:
1. Graph executes until interrupt() is called
2. Execution suspends, state is saved via checkpointer
3. Control returns to caller with __interrupt__ data
4. Human provides input via Command(resume=...)
5. Node restarts from beginning with resume value
6. Execution continues with human input

Node Restart Behavior:
- When resumed, the entire node re-executes from the start
- Code before interrupt() runs again (must be idempotent)
- LangGraph automatically supplies cached responses to seen interrupts
- This avoids keeping intermediate state alive during suspension

Multiple Interrupts:
- Multiple nodes can raise interrupts simultaneously (parallel execution)
- __interrupt__ field contains a list of all active interrupts
- LangGraph tracks which interrupt you're responding to

Use Cases:
- Human approval before critical operations (database writes, API calls)
- Content review and editing workflows
- Quality control checkpoints
- External system integration points

This pattern enables true human-in-the-loop workflows where operations can
pause for minutes, hours, or days while waiting for human input.
"""

from typing import Literal, TypedDict

from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

# ==============================================================
# State Definition
# ==============================================================


class State(TypedDict):
    action_details: str
    status: Literal["pending", "approved", "rejected"]


# ==============================================================
# Node Definitions
# ==============================================================


def review_node(state: State) -> Command[Literal["approve", "reject", "edit"]]:
    """Review action and route based on decision."""

    decision = interrupt(
        {
            "question": "What would you like to do?",
            "details": state["action_details"],
            "options": ["approve", "reject", "edit"],
        }
    )

    # Use Command for dynamic routing (no static edges needed)
    if decision == "approve":
        return Command(goto="approve")
    elif decision == "reject":
        return Command(goto="reject")
    else:
        return Command(goto="edit")


def approve_node(state: State) -> dict:
    """Approve the action."""
    return {"status": "approved"}


def reject_node(state: State) -> dict:
    """Reject the action."""
    return {"status": "rejected"}


def edit_node(state: State) -> dict:
    """Edit action details and return to review."""

    edited_text = interrupt(
        {
            "question": "Edit the action details",
            "details": state["action_details"],
        }
    )

    print(f"📝 Updated to: {edited_text}")

    return {"action_details": edited_text}


# ==============================================================
# Graph Setup
# ==============================================================

builder = StateGraph(State)
builder.add_node("review", review_node)
builder.add_node("approve", approve_node)
builder.add_node("reject", reject_node)
builder.add_node("edit", edit_node)

# Only add static edges to END (Command handles all other routing)
builder.add_edge(START, "review")
builder.add_edge("approve", END)
builder.add_edge("reject", END)
builder.add_edge("edit", "review")

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
display(Image(graph.get_graph().draw_mermaid_png()))

# ==============================================================
# Interactive Testing
# ==============================================================


def handle_interrupts(initial_state: State):
    """Handle the complete interrupt workflow with consistent thread ID."""
    config = {"configurable": {"thread_id": "1"}}  # Consistent thread ID

    # Start the workflow
    response = graph.invoke(initial_state, config=config)

    # Handle interrupts in sequence
    while "__interrupt__" in response:
        interrupt_data = response["__interrupt__"][0].value
        print(f"\n⏸️ Interrupted: {interrupt_data['question']}")
        print(f"📄 Details: {interrupt_data['details']}")

        if "options" in interrupt_data:
            print(f"🔧 Options: {interrupt_data['options']}")

        # Get user input
        decision = input("Enter your decision: ")

        if decision.lower() == "exit":
            return None

        # Resume with decision
        response = graph.invoke(Command(resume=decision), config=config)

    return response


print("🧠 Testing Approve/Reject/Edit Loop Workflow")
print("=" * 50)
print("💡 Try: approve → done, reject → done, edit → loops back to review")

# Test the workflow
action_details = input("\n📝 Enter action details: ")
initial_state = State(action_details=action_details, status="pending")

final_result = handle_interrupts(initial_state)

print(f"\n{final_result}") if final_result else print("\nWorkflow exited by user")
