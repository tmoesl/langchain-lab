"""
LangGraph Application - Building a Complete Email Agent Workflow

Demonstrates building a production-ready application that combines all LangGraph
concepts: complex state management, parallel execution, conditional routing,
human-in-the-loop, and memory. This example shows how to build an email triage
and response system that classifies emails, searches documentation, creates bug
tickets, and drafts responses with human review for critical cases.

Key Concepts:
-------------
- Complex State Schema: Multiple TypedDicts for structured, typed data management
- Structured Output: LLM returns typed classification objects (EmailClassification)
- Parallel Execution: Multiple processing paths run simultaneously (fan-out pattern)
- Conditional Routing: Dynamic path selection based on email classification
- Human-in-the-Loop: Interrupt workflow for approval on high-priority emails
- Tool Integration: RAG search for knowledge base documentation lookup
- Batch Processing: Handle multiple emails with different priorities efficiently
- Memory/Persistence: Checkpointer saves state for pause/resume workflows

Application Architecture:
-------------------------
This email agent implements a sophisticated multi-stage workflow:

1. Email Classification (LLM Node)
   ├─ Classifies intent: question, bug, feature, other
   ├─ Assigns priority: low, medium, high, critical
   ├─ Extracts topic and creates summary
   └─ Routes to appropriate processing path

2. Parallel Processing (Conditional Fan-out)
   ├─ Bug reports → Create ticket + Search docs (parallel execution)
   └─ Other intents → Search docs only

3. Response Generation (LLM Node - Fan-in)
   ├─ Synthesizes context from search results and tickets
   ├─ Drafts professional, contextualized response
   └─ Routes based on priority

4. Human Review Gate (Conditional)
   ├─ High/Critical priority → Requires approval (interrupt)
   └─ Low/Medium priority → Auto-send

5. Send Reply (Action Node)
   └─ Delivers the final response

Workflow Patterns Demonstrated:
--------------------------------
- Fan-out/Fan-in: Parallel branches converge before next step
- Priority-based routing: Different paths for different urgency levels
- Review gates: Human checkpoints for critical decisions
- Batch processing: Handle multiple items with consistent logic
- State accumulation: Gather data from multiple sources

Real-World Applications:
------------------------
This pattern can be adapted for:
- Customer support automation and ticket triage
- Content moderation workflows with escalation
- Document processing pipelines with approval gates
- Multi-step approval processes in enterprise systems
- Incident response and escalation systems
- Lead qualification and routing

Example Email Scenarios:
------------------------
- Simple question: "How do I reset my password?" → Auto-response
- Bug report: "Export crashes with PDF format" → Creates ticket + response
- Urgent billing: "I was charged twice!" → Requires human review
- Feature request: "Can you add dark mode?" → Auto-response
- Critical issue: "API fails with 504 errors" → Requires human review

References:
-----------
- https://docs.langchain.com/oss/python/langgraph/overview
- https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph
- https://docs.langchain.com/oss/python/langgraph/quickstart
- https://docs.langchain.com/oss/python/langgraph/workflows-agents
"""

import uuid
from typing import Literal, TypedDict

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

load_dotenv()


# ==============================================================
# State Definitions
# ==============================================================
# NOTE: For production, use Pydantic BaseModel with Field descriptions instead of TypedDict.
# Pydantic provides better LLM guidance, runtime validation, and error handling.


# Define the schema for the email classification (structured output)
class EmailClassification(TypedDict):
    """
    Structured output from the LLM's email classification task.
    This TypedDict ensures the LLM returns data in a consistent format.
    """

    intent: Literal["question", "bug", "feature", "other"]
    priority: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str


# Define the schema for the agent state
class AgentState(TypedDict):
    """
    Complete state schema for the email agent workflow.
    State flows through nodes and accumulates information at each stage.
    Fields are grouped by workflow stage for clarity.
    """

    # Input data
    email: str
    subject: str
    body: str

    # Classification results
    classification: EmailClassification | None

    # Bug tracking (conditional)
    ticket_id: str | None
    ticket_description: str | None

    # Knowledge base search
    search_results: list[str] | None
    customer_history: dict[str, str] | None

    # Generated response
    draft_response: str | None

    # Human review decision
    human_decision: Literal["approve", "reject"]


# ==============================================================
# Tool Definitions
# ==============================================================


@tool()
def rag_search(query: str) -> str:
    """
    Search the knowledge base for relevant documentation.
    In production, this would query a vector database or search API.
    """
    documents = [
        "Reset password via Settings > Security > Change Password",
        "Password must be at least 12 characters",
        "Include uppercase, lowercase, numbers, and symbols",
        "Dark mode is currently in development, potential release time in November 2025",
    ]
    search_results = "\n".join(doc for doc in documents)
    return search_results


# ==============================================================
# Node Definitions
# ==============================================================


def read_email(state: AgentState) -> None:
    """
    Entry point: Read and validate email data.
    In production, this would parse email headers, sanitize input, etc.
    """
    pass


def classify_intent(state: AgentState) -> dict:
    """
    LLM Node: Classify email intent and priority using structured output.
    Returns an EmailClassification dict that determines downstream routing.
    """
    email = state.get("email", "")
    subject = state.get("subject", "")
    body = state.get("body", "")

    system_prompt = f"""
    You are an email assistant. Your task is to read and analyse the email.

    Email: {email}
    Subject: {subject}
    Body: {body}

    Tasks:
    - Classify the email into one of the following categories: question, bug, feature, other.
    - Label the priority of the email into one of the following categories: low, medium, high, critical.
    - Specify the topic of the email.
    - Create a concise summary of the email.
    """

    model = ChatOpenAI(model="gpt-5-mini")

    # Use structured output to ensure consistent EmailClassification format
    structured_model = model.with_structured_output(EmailClassification)
    response = structured_model.invoke([SystemMessage(content=system_prompt)])

    return {"classification": response}  # type: ignore


def search_documentation(state: AgentState) -> dict:
    """
    LLM + Tool Node: Search knowledge base using RAG.
    LLM decides what to search for, then executes the rag_search tool.
    """
    classification = state.get("classification") or {}
    intent = classification.get("intent", "")
    topic = classification.get("topic", "")

    system_prompt = f"""
    You are an email assistant. Your task is to create a query to search the knowledge base
    for relevant information. The query should be specific to the intent and topic of the email.

    Intent: {intent}
    Topic: {topic}
    """

    model = ChatOpenAI(model="gpt-5-mini")
    model_with_tools = model.bind_tools([rag_search])

    response = model_with_tools.invoke([SystemMessage(content=system_prompt)])

    # Execute the tool calls
    search_results = []
    for tool_call in response.tool_calls:
        if tool_call["name"] == "rag_search":
            result = rag_search.invoke(tool_call["args"])
            search_results.append(result)

    return {"search_results": search_results}


def bug_tracking(state: AgentState) -> dict:
    """
    Action Node: Create a bug ticket.
    In production, this would call an external API (Jira, Linear, etc.).
    """
    classification = state.get("classification") or {}
    intent = classification.get("intent", "")
    priority = classification.get("priority", "")
    topic = classification.get("topic", "")
    summary = classification.get("summary", "")

    ticket_id = f"BUG-{uuid.uuid4()}"
    ticket_description = f"intent: {intent} - priority: {priority} - topic: {topic} \n\n{summary}"

    return {"ticket_id": ticket_id, "ticket_description": ticket_description}


def write_response(state: AgentState) -> dict:
    """
    LLM Node: Generate response using accumulated context.
    This is the fan-in point where parallel branches (bug tracking + search) converge.
    """
    email = state["email"]
    ticket_id = state.get("ticket_id", "")
    search_results = state.get("search_results", [])
    customer_history = state.get("customer_history", "")
    classification = state.get("classification") or {}

    # Build context sections conditionally
    context_sections = []

    if ticket_id:
        context_sections.append(f"Ticket ID: {ticket_id}")

    if search_results:
        docs = "\n".join(str(doc) for doc in search_results)
        context_sections.append(f"Search results:\n{docs}")

    if customer_history:
        context_sections.append(f"Customer history: {customer_history}")

    context = chr(10).join(context_sections) if context_sections else "No context available"

    system_prompt = f"""
    You are an email assistant. Write a professional response to the customer's inquiry.
    
    CUSTOMER EMAIL: {email}

    ORIGINAL INQUIRY:
    Intent: {classification.get("intent", "Unknown")}
    Priority: {classification.get("priority", "medium")}
    Content: {state["body"]}

    CONTEXT:
    {context}

    INSTRUCTIONS:
    - Use the context to provide accurate, helpful information if it relates to the original enquiry
    - Be professional and empathetic
    - Only include the ticket ID if available in the context
    - If you cannot help, explain what steps the customer should take

    Always end with:
    Best regards,
    Customer Support Team
    """

    model = ChatOpenAI(model="gpt-5-mini")
    response = model.invoke([SystemMessage(content=system_prompt)])
    return {"draft_response": response.content}


def human_review(state: AgentState) -> dict:
    """
    Interrupt Node: Pause workflow for human approval.
    Only triggered for high/critical priority or unclear intents.
    """
    classification = state.get("classification") or {}

    human_decision = interrupt(
        {
            "enquiry": state["body"],
            "draft_response": state.get("draft_response", ""),
            "priority": classification.get("priority"),
            "intent": classification.get("intent"),
            "ticket_id": state.get("ticket_id", ""),
            "action": "Please review and approve/edit this response",
            "options": ["approve", "reject"],
        }
    )

    return {"human_decision": human_decision}


def send_reply(state: AgentState) -> None:
    """
    Action Node: Send the email response.
    In production, this would call an email API or SMTP server.
    """
    email = state.get("email", "")
    print(f"Sending reply to {email}")
    pass


# ==============================================================
# Routing Functions
# ==============================================================


def route_from_classification(state: AgentState) -> list[str]:
    """
    Conditional router: Determine parallel processing paths.
    Bugs trigger both bug_tracking AND search_documentation (fan-out).
    Other intents only trigger search_documentation.
    """
    classification = state.get("classification") or {}
    intent = classification.get("intent", "")

    if intent == "bug":
        return ["bug_tracking", "search_documentation"]
    else:
        return ["search_documentation"]


def decide_review_needed(state: AgentState) -> str:
    """
    Conditional router: Determine if human review is needed.
    High/critical priority or unclear intents require approval.
    """
    classification = state.get("classification") or {}
    needs_review = (
        classification.get("priority") in ["high", "critical"]
        or classification.get("intent") == "other"
    )
    return "human_review" if needs_review else "send_reply"


def route_after_review(state: AgentState) -> str:
    """
    Conditional router: Route based on human decision.
    Approved emails proceed to send, rejected emails end workflow.
    """
    human_decision = state.get("human_decision", "")
    return "send_reply" if human_decision == "approve" else END


# ==============================================================
# Build Graph
# ==============================================================

# Initialize checkpointer for state persistence
memory = InMemorySaver()

# Initialize the graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("read_email", read_email)
graph.add_node("classify_intent", classify_intent)
graph.add_node("bug_tracking", bug_tracking)
graph.add_node("search_documentation", search_documentation)
graph.add_node("write_response", write_response)
graph.add_node("human_review", human_review)
graph.add_node("send_reply", send_reply)

# Add static and conditional edges
graph.add_edge(START, "read_email")
graph.add_edge("read_email", "classify_intent")
graph.add_conditional_edges(
    "classify_intent",
    route_from_classification,
    {
        "bug_tracking": "bug_tracking",
        "search_documentation": "search_documentation",
    },
)
graph.add_edge("bug_tracking", "write_response")
graph.add_edge("search_documentation", "write_response")
graph.add_conditional_edges(
    "write_response",
    decide_review_needed,
    {"human_review": "human_review", "send_reply": "send_reply"},
)
graph.add_conditional_edges(
    "human_review", route_after_review, {"send_reply": "send_reply", END: END}
)
graph.add_edge("send_reply", END)

# Compile graph
app = graph.compile(checkpointer=memory)

# Visualize graph
display(Image(app.get_graph().draw_mermaid_png()))


# ==============================================================
# Email Agent Workflow
# ==============================================================
# Note: For real-time approval workflows (e.g., financial transactions),
# use instant interruption by processing one item at a time and blocking
# until approval. For email triage, batch processing is more efficient.


def process_email_batch(app, test_emails: list[dict]) -> list[dict]:
    """
    Process a batch of emails through the agent.
    Returns a list of results that need human approval.
    """
    needs_approval = []

    print("\n" + "=" * 70)
    print("PROCESSING EMAIL BATCH")
    print("=" * 70 + "\n")

    for i, email_data in enumerate(test_emails):
        email_id = f"email_{i + 1:03d}"
        thread_id = str(uuid.uuid4())

        initial_state = {
            "email": email_data["email"],
            "subject": email_data["subject"],
            "body": email_data["body"],
        }

        config = {"configurable": {"thread_id": thread_id}}

        # Display email being processed
        print(f"\033[1;33m{'─' * 70}\033[0m")
        print(f"\033[1;36m[{email_id}]\033[0m {email_data['subject'][:50]:<50} ", end="")

        # Process email
        result = app.invoke(initial_state, config)  # type: ignore

        # Display ticket id, intent, priority, and draft response
        print(f"\nTicket ID: {result.get('ticket_id', 'N/A')}")
        print(f"\nIntent: {result.get('classification', {}).get('intent', 'N/A')}")
        print(f"\nPriority: {result.get('classification', {}).get('priority', 'N/A')}")
        print(f"\nDraft Response: {result.get('draft_response', 'N/A')}")

        # Check if needs approval
        if "__interrupt__" in result:
            interrupt_data = result["__interrupt__"][-1].value
            priority = interrupt_data.get("priority", "unknown")

            # Store for later approval
            result["thread_id"] = thread_id
            result["email_id"] = email_id
            needs_approval.append(result)

            print(f"\033[1;31m⚠️  REVIEW [{priority.upper()}]\033[0m")
        else:
            print("\033[1;32m✓ SENT\033[0m")

    print("\n" + "=" * 70)
    print(f"BATCH PROCESSING COMPLETE: {len(test_emails)} emails processed")
    print(f"  • Sent automatically: {len(test_emails) - len(needs_approval)}")
    print(f"  • Awaiting approval: {len(needs_approval)}")
    print("=" * 70)

    return needs_approval


def process_review_queue(app, needs_approval: list[dict]):
    """
    Process the queue of emails requiring human review.
    Simulates human decisions and resumes the agent workflow.
    """
    if not needs_approval:
        return

    print("\n" + "=" * 70)
    print("HUMAN REVIEW QUEUE")
    print("=" * 70 + "\n")

    for item in needs_approval:
        interrupt_data = item["__interrupt__"][-1].value

        print(f"\033[1;33m{'─' * 70}\033[0m")
        print(f"\033[1;36mEmail ID:\033[0m {item['email_id']}")
        print(f"\033[1;36mPriority:\033[0m {interrupt_data.get('priority', 'N/A')}")
        print(f"\033[1;36mIntent:\033[0m {interrupt_data.get('intent', 'N/A')}")
        print(f"\033[1;36mTicket:\033[0m {interrupt_data.get('ticket_id', 'None')}")
        print(f"\033[1;36mEnquiry:\033[0m {interrupt_data.get('enquiry', 'N/A')[:80]}...")
        print(f"\033[1;36mDraft:\033[0m {interrupt_data.get('draft_response', 'N/A')}...")

        # Simulate human decision (in production, this would be actual human input)
        priority = interrupt_data.get("priority", "medium")
        intent = interrupt_data.get("intent", "other")

        if priority == "critical":
            decision = "reject"
        elif priority == "high" or intent == "other":
            decision = "approve"
        else:
            decision = "approve"

        print(f"\n\033[1;32m✓ Decision: {decision}\033[0m")

        # Resume workflow with Command(resume=decision)
        config = {"configurable": {"thread_id": item["thread_id"]}}
        app.invoke(Command(resume=decision), config)  # type: ignore

        if decision == "approve":
            print("\033[1;32m✓ Email approved and sent\033[0m\n")
        else:
            print("\033[1;31m✗ Email rejected\033[0m\n")

    print("=" * 70)
    print("ALL APPROVALS PROCESSED")
    print("=" * 70)


# ==============================================================
# Test Scenarios
# ==============================================================


def get_test_emails() -> list[dict]:
    """
    Returns the list of test email scenarios.
    """
    return [
        # Case 1: Question → Auto-send
        {
            "subject": "Password reset help",
            "body": "How do I reset my password?",
            "email": "user@example.com",
        },
        # Case 2: Bug report → Creates ticket + Requires review
        {
            "subject": "Export feature issue",
            "body": "The export feature crashes when I select PDF format",
            "email": "developer@company.com",
        },
        # Case 3: Billing issue → Requires review
        {
            "subject": "Urgent billing issue",
            "body": "I was charged twice for my subscription!",
            "email": "billing@example.com",
        },
        # Case 4: Feature request → Auto-send
        {
            "subject": "Feature request",
            "body": "Can you add dark mode to the mobile app?",
            "email": "features@example.com",
        },
        # Case 5: Bug report → Requires review
        {
            "subject": "CRITICAL: Production down",
            "body": "Our API integration fails intermittently with 504 errors",
            "email": "tech@enterprise.com",
        },
        # Case 6: Question → Auto-send
        {
            "subject": "General inquiry",
            "body": "I have concerns about recent changes to your terms and privacy policy",
            "email": "legal@company.com",
        },
        # Case 7: Other intent → Requires review (doesn't fit question/bug/feature)
        {
            "subject": "Data Deletion Request",
            "body": "Per GDPR Article 17, I request deletion of all my personal data from your systems.",
            "email": "privacy@user.com",
        },
    ]


# ==============================================================
# Application Entry Point
# ==============================================================


def main():
    """
    Main function to execute the email batch processing workflow.
    Wraps the script logic to allow it to be called when run directly.
    """
    # 1. Get scenarios
    test_emails = get_test_emails()

    # 2. Execute Batch Processing
    needs_approval = process_email_batch(app, test_emails)

    # 3. Process Human Review Queue
    process_review_queue(app, needs_approval)


if __name__ == "__main__":
    main()
