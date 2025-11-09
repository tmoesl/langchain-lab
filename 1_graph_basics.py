import random
from typing import Literal

from dotenv import load_dotenv
from IPython.display import Image, display
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()


# --------------------------------------------------------------
# Simple Graph (Single Input)
# --------------------------------------------------------------


class AgentState(TypedDict):
    user_name: str
    result: str | None


def greeting_node(state: AgentState) -> AgentState:
    """Simple node that adds a greeting message to the state"""

    state["result"] = f"Hello {state['user_name']}! How is your day going?!"

    return state


graph = StateGraph(AgentState)

graph.add_node("greeting", greeting_node)
graph.add_edge(START, "greeting")
graph.add_edge("greeting", END)

graph = graph.compile()

message = AgentState(user_name="John")  # type: ignore
response = graph.invoke(message)
print(response["result"])

# Display the graph
display(Image(graph.get_graph().draw_mermaid_png()))


# --------------------------------------------------------------
# Simple Graph (Multiple Inputs)
# --------------------------------------------------------------
class AgentState(TypedDict):
    values: list[int]
    name: str
    result: str | None


def process_values_node(state: AgentState) -> AgentState:
    """Simple node that processes the values in the list"""

    result = sum(state["values"])
    state["result"] = f"Hello {state['name']}! The sum of the values is {result}"

    return state


graph = StateGraph(AgentState)

graph.add_node("process_values", process_values_node)
graph.set_entry_point("process_values")
graph.set_finish_point("process_values")

graph = graph.compile()

message = AgentState(values=[1, 2, 3], name="John")  # type: ignore
response = graph.invoke(message)
print(response["result"])

# Display the graph
display(Image(graph.get_graph().draw_mermaid_png()))


# --------------------------------------------------------------
# Simple Graph (Multiple Inputs with Operation)
# --------------------------------------------------------------


class AgentState(BaseModel):
    values: list[int] = Field(description="The list of values to be processed")
    operation: Literal["add", "multiply"] = Field(description="The operation to be performed")
    name: str = Field(description="The name of the user")
    result: str | None = Field(default=None, description="The result of the calculation")


def process_values_node(state: AgentState) -> AgentState:
    """Simple node that processes the values in the list"""

    match state.operation:
        case "add":
            result = sum(state.values)
        case "multiply":
            result = 1
            for value in state.values:
                result *= value

    state.result = f"Hello {state.name}! The result is {result}"
    return state


graph = StateGraph(AgentState)

graph.add_node("process_values", process_values_node)
graph.set_entry_point("process_values")
graph.set_finish_point("process_values")

graph = graph.compile()

message = AgentState(values=[1, 2, 3, 8], name="John", operation="add")
response = graph.invoke(message)
print(response["result"])

# Display the graph
display(Image(graph.get_graph().draw_mermaid_png()))


# --------------------------------------------------------------
# Simple Graph (Sequential Nodes)
# --------------------------------------------------------------


class AgentState(TypedDict):
    name: str
    age: int
    skills: list[str]
    result: str | None


def first_node(state: AgentState) -> AgentState:
    """Simple node that adds a greeting message to the state"""

    state["result"] = f"Hello {state['name']}, welcome to the session!"

    return state


def second_node(state: AgentState) -> AgentState:
    """Simple node that adds a greeting message to the state"""

    state["result"] = f"{state['result']} You're {state['age']} years old."

    return state


def third_node(state: AgentState) -> AgentState:
    """Simple node that adds a greeting message to the state"""

    state["result"] = (
        f"{state['result']} You have the following skills: {', '.join(state['skills'])}"
    )

    return state


graph = StateGraph(AgentState)

graph.add_node("first_node", first_node)
graph.add_node("second_node", second_node)
graph.add_node("third_node", third_node)

graph.add_edge(START, "first_node")
graph.add_edge("first_node", "second_node")
graph.add_edge("second_node", "third_node")
graph.add_edge("third_node", END)

graph = graph.compile()

message = AgentState(name="John", age=30, skills=["Python", "SQL", "Machine Learning"])  # type: ignore
response = graph.invoke(message)
print(response["result"])

# Display the graph
display(Image(graph.get_graph().draw_mermaid_png()))


# --------------------------------------------------------------
# Simple Graph (Conditional Nodes)
# --------------------------------------------------------------


class AgentState(TypedDict):
    number1: int
    number2: int
    operation: str
    result: int | None


def add_node(state: AgentState) -> AgentState:
    """Simple node that adds two numbers"""

    state["result"] = state["number1"] + state["number2"]

    return state


def multiply_node(state: AgentState) -> AgentState:
    """Simple node that multiplies two numbers"""

    state["result"] = state["number1"] * state["number2"]

    return state


def router_node(state: AgentState) -> AgentState:
    """Router node that passes through the state"""

    return state


def route_decision(state: AgentState) -> str:  # No modification of the state
    """Simple node that routes the operation to the appropriate node"""

    if state["operation"] == "add":
        return "add_operation"
    elif state["operation"] == "multiply":
        return "multiply_operation"
    else:
        raise ValueError(f"Invalid operation: {state['operation']}")


graph = StateGraph(AgentState)

graph.add_node("router_node", router_node)  # or lambda state: state
graph.add_node("add_node", add_node)
graph.add_node("multiply_node", multiply_node)


graph.add_edge(START, "router_node")
graph.add_conditional_edges(
    "router_node",
    route_decision,
    {
        # Edge : Node mapping
        "add_operation": "add_node",
        "multiply_operation": "multiply_node",
    },
)
graph.add_edge("add_node", END)
graph.add_edge("multiply_node", END)

graph = graph.compile()

message = AgentState(number1=10, number2=5, operation="add")
response = graph.invoke(message)
print(response["result"])

# Display the graph
display(Image(graph.get_graph().draw_mermaid_png()))

# --------------------------------------------------------------
# Simple Graph (Looping Logic)
# --------------------------------------------------------------


class AgentState(TypedDict):
    name: str
    numbers: list[int]
    counter: int
    result: str


def greeting_node(state: AgentState) -> AgentState:
    """Simple node that adds a greeting message to the state"""

    state["result"] = f"Hello {state['name']}! Welcome to the session!"
    state["counter"] = 0

    return state


def random_node(state: AgentState) -> AgentState:
    """Simple node that generates a random number"""

    number = random.randint(1, 10)
    state["numbers"].append(number)
    state["counter"] += 1

    return state


def route_decision(state: AgentState) -> str:  # No modification of the state
    """Simple node that routes the operation to the appropriate node"""

    if state["counter"] < 5:
        print(f"Entering loop, {state['counter']}")
        return "loop"
    else:
        print(f"Exiting loop, {state['counter']}")
        return "exit"


graph = StateGraph(AgentState)

graph.add_node("greeting_node", greeting_node)
graph.add_node("random_node", random_node)

graph.add_edge(START, "greeting_node")
graph.add_edge("greeting_node", "random_node")
graph.add_conditional_edges(
    "random_node",
    route_decision,
    {
        "loop": "random_node",
        "exit": END,
    },
)

graph = graph.compile()

message = AgentState(name="John", numbers=[], counter=0)
response = graph.invoke(message)
print(response["result"])
print(f"Your generated numbers: {', '.join(str(v) for v in response['numbers'])}")

# Display the graph
display(Image(graph.get_graph().draw_mermaid_png()))


# --------------------------------------------------------------
# Simple Graph (Looping Logic)
# --------------------------------------------------------------


class AgentState(TypedDict):
    name: str
    attempts: int
    lower_bound: int
    upper_bound: int
    target_number: int
    guesses: list[int]


def setup_node(state: AgentState) -> AgentState:
    """Simple node that sets up the game"""

    state["attempts"] = 0
    state["lower_bound"] = 1
    state["upper_bound"] = 20
    state["target_number"] = random.randint(state["lower_bound"], state["upper_bound"])
    state["guesses"] = []

    print(f"Hello {state['name']}! Welcome to the session!")

    return state


def guess_node(state: AgentState) -> AgentState:
    """Simple node to guess the target number"""

    upper_bound = state["upper_bound"]
    lower_bound = state["lower_bound"]

    guess = random.randint(lower_bound, upper_bound)
    state["guesses"].append(guess)
    state["attempts"] += 1

    print(f"Attempt {state['attempts']}: Guessing {guess} (Range: {lower_bound}-{upper_bound})")

    return state


def hint_node(state: AgentState) -> AgentState:
    """Simple node to hint if the guess is correct"""

    target = state["target_number"]
    latest_guess = state["guesses"][-1]

    if target > latest_guess:
        state["lower_bound"] = latest_guess + 1
    elif target < latest_guess:
        state["upper_bound"] = latest_guess - 1

    return state


def route_decision(state: AgentState) -> str:  # No modification of the state
    """Simple node that routes the operation to the appropriate node"""
    target = state["target_number"]
    latest_guess = state["guesses"][-1]

    if target == latest_guess:
        print(f"Number found in {state['attempts']} attempts! The number was {target}.")
        return "exit"
    elif state["attempts"] >= 7:
        print(f"Maximum attempts reached. The number was {target}.")
        return "exit"
    else:
        return "continue"


graph = StateGraph(AgentState)

graph.add_node("setup", setup_node)
graph.add_node("guess", guess_node)
graph.add_node("hint", hint_node)

graph.add_edge(START, "setup")
graph.add_edge("setup", "guess")
graph.add_edge("guess", "hint")
graph.add_conditional_edges("hint", route_decision, {"continue": "guess", "exit": END})

graph = graph.compile()


message = AgentState(name="John")
response = graph.invoke(message)

# Display the graph
display(Image(graph.get_graph().draw_mermaid_png()))
