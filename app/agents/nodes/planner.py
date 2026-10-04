from app.agents.state import AgentState
import logfire
from langchain_groq import ChatGroq

from app.config import settings


llm = ChatGroq(
    model=settings.GROQ_MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=0,
    max_tokens=1024
)


def planner_node(state: AgentState) -> dict:
    """
    Planner Node

    Decides whether the query is:
    - CONVERSATIONAL
    - RETRIEVAL

    Uses conversation history + latest user message.
    """

    messages = state.get("messages", [])
    history_lines = []
    for msg in messages[:-1]:
        if isinstance(msg, dict):
            role, content = msg.get("role", "user"), msg.get("content", "")
        else:
            role, content = getattr(msg, "type", "user"), getattr(msg, "content", "")
        history_lines.append(f"{role}: {content}")
    history = "\n".join(history_lines)

    last_message = messages[-1] if messages else None
    if isinstance(last_message, dict):
        user_message = last_message.get("content", "")
    elif last_message is not None:
        user_message = getattr(last_message, "content", "")
    else:
        user_message = state.get("current_query", "")

    prompt = f"""
You are an intent classifier.

Conversation History:
{history}

Latest User Message:
{user_message}

Classify the user's intent.

Reply with ONLY ONE WORD.

CONVERSATIONAL

OR

RETRIEVAL

No explanation.
"""

    with logfire.span("Generating plan with Groq LLM"):
        decision = str(llm.invoke(prompt).content).strip().upper()

        logfire.info(f"INTENT IDENTIFIED: {decision}")

    if decision == "CONVERSATIONAL":

        return {
            "route": "CONVERSATIONAL",
            "current_query": user_message,
            "status": "Handling conversational request",
            "plan": [
                "Intent: Conversational",
                "Retrieval Skipped"
            ]
        }

    return {
        "route": "RETRIEVAL",
        "current_query": user_message,
        "status": "Technical query detected",
        "plan": [
            "Intent: Technical",
            f"Search: {user_message}"
        ]
    }
