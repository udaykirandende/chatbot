from langchain_openai import ChatOpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

from app.config import settings


def _gateway_config() -> dict:
    if not settings.GROQ_SLUG or not settings.GROQ_SLUG_2:
        raise RuntimeError("Set GROQ_SLUG and GROQ_SLUG_2 for Portkey routing.")

    return {
        "strategy": {"mode": "fallback"},
        "cache": {"mode": "simple"},
        "retry": {"attempts": 2, "on_status_codes": [429, 503]},
        "targets": [
            {"override_params": {"model": f"@{settings.GROQ_SLUG}/openai/gpt-oss-120b"}},
            {"override_params": {"model": f"@{settings.GROQ_SLUG_2}/openai/gpt-oss-20b"}},
        ],
    }


def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    """Build an OpenAI-compatible LangChain client routed through Portkey."""
    if not settings.PORTKEY_API_KEY:
        raise RuntimeError("Set PORTKEY_API_KEY to use the Portkey model gateway.")

    config = _gateway_config()
    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=f"@{settings.GROQ_SLUG}/openai/gpt-oss-120b",
        temperature=0,
        max_tokens=1024,
        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            config=config,
            metadata={
                "feature": feature,
                "_user": "rag-system",
                "environment": "production",
            },
        ),
    )
