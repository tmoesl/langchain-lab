"""
Simple product question: “How do I reset my password?”
Bug report: “The export feature crashes when I select PDF format”
Urgent billing issue: “I was charged twice for my subscription!”
Feature request: “Can you add dark mode to the mobile app?”
Complex technical issue: “Our API integration fails intermittently with 504 errors”

"""

import uuid

from dotenv import load_dotenv

load_dotenv()
from typing import Literal, TypedDict

from IPython.display import Image, display
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

model = ChatOpenAI(model="gpt-4o-mini")


class EmailClassification(TypedDict):
    intent: Literal["question", "bug", "feature", "other"]
    priority: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str


class AgentState(TypedDict):
    # Raw email data
    email: str
    subject: str
    body: str

    # Email classification
    classification: EmailClassification | None

    # Bug tracking
    ticket_id: str | None
    ticket_description: str | None

    # Raw search results
    search_results: list[str] | None
    customer_history: dict[str, str] | None

    # Draft response
    draft_response: str | None

    # Human review
    human_decision: Literal["approve", "reject"]


@tool()
def rag_search(query: str) -> str:
    """Search the RAG database for the most relevant information."""
    documents = [
        "Reset password via Settings > Security > Change Password",
        "Password must be at least 12 characters",
        "Include uppercase, lowercase, numbers, and symbols",
        "Dark mode is currently in development, potential release time in November 2025",
    ]
    search_results = "\n".join(doc for doc in documents)
    return search_results


# data
def read_email(state: AgentState) -> None:
    """Read the email and extract the relevant information."""
    pass


# LLM
def classify_intent(state: AgentState) -> dict:
    """Use LLM to classify the intent and priority of the email."""
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

    model = ChatOpenAI(model="gpt-4o-mini")

    # Structured output (model obtains the EmailClassification schema)
    structured_model = model.with_structured_output(EmailClassification)
    response = structured_model.invoke([SystemMessage(content=system_prompt)])

    return {"classification": response}


# data
def search_documentation(state: AgentState) -> dict:
    """Search the knowledge base for relevant information"""
    classification = state.get("classification", {})
    intent = classification.get("intent", "")
    topic = classification.get("topic", "")

    system_prompt = f"""
    You are an email assistant. Your task is to create a query to search the knowledge base
    for relevant information. The query should be specific to the intent and topic of the email.

    Intent: {intent}
    Topic: {topic}
    """

    model = ChatOpenAI(model="gpt-4o-mini")
    model_with_tools = model.bind_tools([rag_search])

    response = model_with_tools.invoke([SystemMessage(content=system_prompt)])

    # Execute the tool calls
    search_results = []
    for tool_call in response.tool_calls:
        if tool_call["name"] == "rag_search":
            result = rag_search.invoke(tool_call["args"])
            search_results.append(result)

    return {"search_results": search_results}


# action (In production, API would handle the ticket creation)
def bug_tracking(state: AgentState) -> dict:
    """Create a ticket in the bug tracking system"""
    classification = state.get("classification", {})
    intent = classification.get("intent", "")
    priority = classification.get("priority", "")
    topic = classification.get("topic", "")
    summary = classification.get("summary", "")

    ticket_id = f"BUG-{uuid.uuid4()}"
    ticket_description = f"intent: {intent} - priority: {priority} - topic: {topic} \n\n{summary}"

    return {"ticket_id": ticket_id, "ticket_description": ticket_description}


# LLM
def write_response(state: AgentState) -> dict:
    """Generate a response to the original enquire using context."""
    # Extract data safely
    email = state["email"]
    ticket_id = state.get("ticket_id", "")
    search_results = state.get("search_results", [])
    customer_history = state.get("customer_history", "")
    classification = state.get("classification", {})

    # Build context sections conditionally
    context_sections = []

    if ticket_id:
        context_sections.append(f"Ticket ID: {ticket_id}")

    if search_results:
        # Fixed: Proper handling of search results
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

    model = ChatOpenAI(model="gpt-4o-mini")
    response = model.invoke([SystemMessage(content=system_prompt)])
    return {"draft_response": response.content}


def decide_review_needed(state: AgentState) -> str:
    """Decide if the draft response needs human review."""
    classification = state.get("classification", {})
    needs_review = (
        classification.get("priority") in ["high", "critical"]
        or classification.get("intent") == "other"
    )
    return "human_review" if needs_review else "send_reply"


def route_after_review(state: AgentState) -> str:
    """Route after human review - decide next action based on human decision."""
    human_decision = state.get("human_decision", "")
    return "send_reply" if human_decision == "approve" else END


def route_from_classification(state: AgentState) -> list[str]:
    """Route after email classification - decide which processing paths to take."""
    classification = state.get("classification", {})
    intent = classification.get("intent", "")

    if intent == "bug":
        # For bugs: do both bug tracking AND search documentation in parallel
        return ["bug_tracking", "search_documentation"]
    else:
        # For other intents: only search documentation
        return ["search_documentation"]


# user input
def human_review(state: AgentState) -> dict:
    """Review the draft response and provide feedback."""
    classification = state.get("classification", {})

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


# action
def send_reply(state: AgentState) -> None:
    """Send the email reply."""
    email = state.get("email", "")
    print(f"Sending reply to {email}")
    pass


memory = InMemorySaver()
config = {"configurable": {"thread_id": "1"}}

graph = StateGraph(AgentState)
graph.add_node("read_email", read_email)
graph.add_node("classify_intent", classify_intent)
graph.add_node("bug_tracking", bug_tracking)
graph.add_node("search_documentation", search_documentation)
graph.add_node("write_response", write_response)
graph.add_node("human_review", human_review)
graph.add_node("send_reply", send_reply)


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


agent = graph.compile(checkpointer=memory)

display(Image(agent.get_graph().draw_mermaid_png()))


# ============================================================================
# TEST SCENARIOS
# ============================================================================

# Test emails covering different scenarios
# Review triggers: priority in ["high", "critical"] OR intent == "other"
test_emails = [
    # Case 1: Simple question (low priority, no review)
    {
        "subject": "Password reset help",
        "body": "How do I reset my password?",
        "email": "user@example.com",
    },
    # Case 2: Bug report (HIGH priority, REQUIRES review, creates ticket)
    {
        "subject": "Export feature issue",
        "body": "The export feature crashes when I select PDF format",
        "email": "developer@company.com",
    },
    # Case 3: Urgent billing issue (HIGH priority, REQUIRES review)
    {
        "subject": "Urgent billing issue",
        "body": "I was charged twice for my subscription!",
        "email": "billing@example.com",
    },
    # Case 4: Feature request (low priority, no review)
    {
        "subject": "Feature request",
        "body": "Can you add dark mode to the mobile app?",
        "email": "features@example.com",
    },
    # Case 5: Complex technical issue (CRITICAL priority, REQUIRES review)
    {
        "subject": "CRITICAL: Production down",
        "body": "Our API integration fails intermittently with 504 errors",
        "email": "tech@enterprise.com",
    },
    # Case 6: Unclear/Other intent (intent = "other", REQUIRES review)
    {
        "subject": "General inquiry",
        "body": "I have some concerns about the recent changes to your terms of service and privacy policy",
        "email": "legal@company.com",
    },
]

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
    print(f"\033[1;36m[{email_id}]\033[0m {email_data['subject'][:50]:<50} ", end="")

    # Process email
    result = agent.invoke(initial_state, config)
    # show ticket id, draft_response
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

# Display summary
print("\n" + "=" * 70)
print(f"BATCH PROCESSING COMPLETE: {len(test_emails)} emails processed")
print(f"  • Sent automatically: {len(test_emails) - len(needs_approval)}")
print(f"  • Awaiting approval: {len(needs_approval)}")
print("=" * 70)

# Process approvals
if needs_approval:
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

        # Auto-decision based on priority and intent (in production, this would be human input)
        priority = interrupt_data.get("priority", "medium")
        intent = interrupt_data.get("intent", "other")

        # Approve high priority, reject critical (needs escalation), approve "other" for review
        if priority == "critical":
            decision = "reject"  # Critical issues need escalation
        elif priority == "high" or intent == "other":
            decision = "approve"  # High priority and unclear intents get approved after review
        else:
            decision = "approve"  # Default approve

        print(f"\n\033[1;32m✓ Decision: {decision}\033[0m")

        # Resume workflow
        config = {"configurable": {"thread_id": item["thread_id"]}}
        final_result = agent.invoke(Command(resume=decision), config)

        if decision == "approve":
            print("\033[1;32m✓ Email approved and sent\033[0m\n")
        else:
            print("\033[1;31m✗ Email rejected\033[0m\n")

    print("=" * 70)
    print("ALL APPROVALS PROCESSED")
    print("=" * 70)
