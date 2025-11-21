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

⚠️ CRITICAL: Interrupt Placement
---------------------------------
Place interrupts at the START of nodes.
Code before interrupt() re-executes on resume!
"""

from typing import Literal, TypedDict

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

load_dotenv()

# ==============================================================
# State Definition
# ==============================================================


class State(TypedDict):
    proposal_text: str
    status: Literal["pending", "approved", "rejected"]


# ==============================================================
# Node Definitions
# ==============================================================


def review_node(state: State) -> Command[Literal["approve", "reject", "edit"]]:
    """Review action and route based on decision."""

    print("🔄 Entered review_node (runs again after resume!)")

    decision = interrupt(
        {
            "question": "What would you like to do?",
            "details": state["proposal_text"],
            "options": ["approve", "reject", "edit"],
        }
    )

    print(f"✅ Decision received: {decision}")

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

    print("🔄 Entered edit_node (runs again after resume!)")

    edited_text = interrupt(
        {
            "question": "Edit the proposal text",
            "details": state["proposal_text"],
        }
    )

    print(f"📝 Updated to: {edited_text}")

    return {"proposal_text": edited_text}


# ==============================================================
# Build Graph
# ==============================================================

# Initialize graph
graph = StateGraph(State)

# Add nodes
graph.add_node("review", review_node)
graph.add_node("approve", approve_node)
graph.add_node("reject", reject_node)
graph.add_node("edit", edit_node)

# Only add static edges to END (Command handles all other routing)
graph.add_edge(START, "review")
graph.add_edge("approve", END)
graph.add_edge("reject", END)
graph.add_edge("edit", "review")

# Compile graph
checkpointer = MemorySaver()
graph = graph.compile(checkpointer=checkpointer)


# ==============================================================
# Interactive Testing
# ==============================================================


def handle_interrupts(initial_state: State):
    """Handle the complete interrupt workflow with consistent thread ID."""
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}  # Consistent thread ID

    # Invoke graph
    response = graph.invoke(initial_state, config=config)

    # Handle interrupts in sequence
    while "__interrupt__" in response:
        # Show interrupt structure
        print("\n" + "=" * 50)
        print("📋 INTERRUPT STRUCTURE:")
        print(f"   Type: {type(response['__interrupt__'])}")
        print(f"   Count: {len(response['__interrupt__'])}")
        print(f"   Interrupt ID: {response['__interrupt__'][0].id}")
        print("=" * 50)

        interrupt_data = response["__interrupt__"][0].value
        print(f"\n⏸️ Interrupted: {interrupt_data['question']}")
        print(f"📄 Proposal Text: {interrupt_data['details']}")

        if "options" in interrupt_data:
            print(f"🔧 Options: {interrupt_data['options']}")

        # Get user input (use the question from interrupt as prompt)
        user_input = input(f"\n{interrupt_data['question']}: ")

        if user_input.lower() == "exit":
            return None

        # Resume with user input
        response = graph.invoke(Command(resume=user_input), config=config)

    return response


print("🧠 Testing Approve/Reject/Edit Loop Workflow")
print("=" * 50)
print("💡 Try: approve → done, reject → done, edit → loops back to review")

# Test the workflow
proposal_text = input("\n📝 Enter proposal text: ")
initial_state = State(proposal_text=proposal_text, status="pending")

final_result = handle_interrupts(initial_state)

print(f"\n{final_result}") if final_result else print("\nWorkflow exited by user")

# ==============================================================
# Visualize Graph
# ==============================================================
print("\n" + "=" * 60)
print("Graph Visualization:")
print("=" * 60)

display(Image(graph.get_graph().draw_mermaid_png()))
