"""LLM answer generation with strict context-only guardrails."""

from __future__ import annotations

from openai import OpenAI

from src.config import settings

GUARDRAIL_PROMPT = """You are a strict retrieval QA assistant.
Answer the user ONLY using the retrieved context below.
Rules:
1) Do NOT use external knowledge.
2) If the answer is not explicitly in the context, reply exactly:
   The information for your question is not present in the provided data.
3) Be concise and factual.

Retrieved context:
{context}
"""


class AnswerGenerator:
    """Generate context-grounded responses via OpenAI Chat Completions."""

    def __init__(self) -> None:
        self.client = None
        if settings.openai_api_key:
            self.client = OpenAI(api_key=settings.openai_api_key)

    def answer_from_context(self, question: str, context_chunks: list[str]) -> str:
        context = "\n\n".join(f"- {chunk}" for chunk in context_chunks)
        if not context.strip():
            return "The information for your question is not present in the provided data."

        if not self.client:
            # Deterministic fallback if API key is not configured.
            return (
                "OpenAI API key is missing. Unable to generate an LLM answer. "
                "Please set OPENAI_API_KEY to enable context-grounded responses."
            )

        response = self.client.chat.completions.create(
            model=settings.llm_model,
            temperature=0,
            messages=[
                {"role": "system", "content": GUARDRAIL_PROMPT.format(context=context)},
                {"role": "user", "content": question},
            ],
        )
        return response.choices[0].message.content or ""
