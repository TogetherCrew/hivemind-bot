from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

try:
    # Prefer the official OpenAI client if available in the environment
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - fallback for environments without openai
    OpenAI = None  # type: ignore


DEFAULT_REPHRASE_MODEL = "gpt-5-nano-2025-08-07"


def _build_messages(question: str, context_hint: Optional[str] = None) -> list[dict[str, str]]:
    system = (
        "You rephrase user questions to improve retrieval. "
        "Return a single, concise rephrasing that preserves meaning and entities. "
        "Do not answer the question. Output only the rephrased question."
    )
    if context_hint:
        user = (
            f"Original question: {question}\n"
            f"Context hint: {context_hint}\n"
            "Rephrase the original question accordingly."
        )
    else:
        user = f"Original question: {question}\nRephrase the original question."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=6))
def rephrase_question(question: str, context_hint: Optional[str] = None, model: str = DEFAULT_REPHRASE_MODEL) -> str:
    """
    Rephrase a question using a small, deterministic LLM to potentially improve retrieval.

    Falls back to the original question if the client is not available or an error occurs.
    """
    if OpenAI is None:
        return question

    try:
        client = OpenAI()
        messages = _build_messages(question, context_hint)
        response = client.chat.completions.create(model=model, messages=messages, temperature=0.0)
        content = (response.choices[0].message.content or question).strip()
        # Normalize simple wrapping quotes if present
        if (content.startswith("\"") and content.endswith("\"")) or (
            content.startswith("'") and content.endswith("'")
        ):
            content = content[1:-1].strip()
        # Avoid returning empty string
        return content or question
    except Exception:
        return question


