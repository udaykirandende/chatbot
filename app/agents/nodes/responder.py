import logfire
from typing import Any

from app.agents.state import AgentState
from app.gateway import get_langchain_llm


def generate_node(state: AgentState) -> dict[str, Any]:
    """
    Final response generation node.

    - Uses conversation memory for conversational queries.
    - Uses retrieved documents for technical/RAG queries.
    - Uses Portkey Gateway.
    """

    route = state.get("route", "RETRIEVAL")
    query = state.get("current_query", "")

    history_str = ""

    if state.get("messages"):
        for msg in state["messages"][:-1]:
            if isinstance(msg, dict):
                role, content = msg.get("role", "user"), msg.get("content", "")
            else:
                role, content = getattr(msg, "type", "user"), getattr(msg, "content", "")
            history_str += f"{'User' if role in ('human', 'user') else 'Assistant'}: {content}\n"

        last_message = state["messages"][-1]
        user_msg = (
            last_message.get("content", "")
            if isinstance(last_message, dict)
            else getattr(last_message, "content", "")
        )
    else:
        user_msg = query

    # ----------------------------
    # Conversational Response
    # ----------------------------

    if route == "CONVERSATIONAL":

        logfire.info("Generating conversational response using memory.")

        prompt = f"""
You are a friendly Enterprise AI Assistant.

Conversation History:
{history_str}

Latest User Message:
{user_msg}

Answer naturally.
"""

    # ----------------------------
    # Technical RAG Response
    # ----------------------------

    else:

        logfire.info("Generating technical RAG response.")

        documents = state.get("documents", [])

        context = ""

        for doc in documents:
            context += f"{doc}\n\n"

        prompt = f"""
You are a Senior Technical AI Assistant.

Answer ONLY from the provided context.

Technical Context:
{context}

Conversation History:
{history_str}

User Question:
{user_msg}

If the answer is not in the context, clearly say you don't know.
"""

    # ----------------------------
    # LLM Call
    # ----------------------------

    with logfire.span("LLM Response Generation"):

        try:

            response = get_langchain_llm(feature="rag").invoke(prompt)
            answer = response.content
            if not isinstance(answer, str):
                answer = str(answer)

            plan = state.get("plan", [])

            logfire.info("LLM Response Generated")
            status = "Completed"

            return {

                "final_answer": answer,

                "messages": [
                    {"role": "assistant", "content": answer}
                ],

                "status": status,

                "plan": plan

            }

        except Exception as exc:
            logfire.error("LLM response generation failed", error=str(exc))
            raise
