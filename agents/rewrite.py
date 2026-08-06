"""Agent 0 — Contextualization: fold conversation history into a standalone query.

Multi-turn fix. A follow-up like "MI250, June 5" is meaningless alone: the
clarification agent sees no subject and loops forever. This agent rewrites the
follow-up into a self-contained question using prior turns, so every downstream
agent (clarification, retrieval, generation, judge) operates on one clear query
and needs no history of its own.

No history -> return the query untouched (zero extra LLM cost on turn 1).
"""
from typing import Dict, List

from llm import call_gemini

REWRITE_PROMPT = """You rewrite a passenger's latest message into a single, self-contained flight question, using the conversation so far to fill in any references.

Rules:
1. Resolve pronouns and fragments ("this", "it", "MI250, June 5") using the history.
2. Carry forward entities the user gave earlier (flight number, dates, item, service).
3. If the latest message is already self-contained, return it unchanged.
4. Output ONLY the rewritten question. No preamble, no quotes.

Conversation so far:
{history}

Latest message: {query}

Standalone question:"""

# Only the last few turns matter for reference resolution; keep the prompt small.
MAX_HISTORY_TURNS = 6


def _format_history(history: List[Dict]) -> str:
    turns = history[-MAX_HISTORY_TURNS:]
    return "\n".join(f'{m["role"]}: {m["content"]}' for m in turns)


def rewrite_agent(query: str, history: List[Dict] | None = None) -> str:
    """Return a standalone query. With no history, the query is returned as-is."""
    if not history:
        return query
    prompt = (
        REWRITE_PROMPT
        .replace("{history}", _format_history(history))
        .replace("{query}", query)
    )
    return call_gemini(prompt, temperature=0.0).strip()
