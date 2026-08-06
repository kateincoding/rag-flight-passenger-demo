"""Agent 3 — Generation: source-grounded answering with citations."""
from typing import Dict, List

from llm import call_gemini

GENERATION_PROMPT = """You are a flight assistant. Answer the passenger's question using ONLY the information in the provided sources.

Rules:
1. Base your answer strictly on the sources. Do not add outside knowledge.
2. Cite the source IDs (e.g. [BAG-003]) after any specific claim.
3. If the sources do not fully answer the question, say so explicitly.
4. Keep the answer concise and passenger-friendly.

Passenger question: {query}

Sources:
{sources}

Answer:"""


def _format_sources(chunks: List[Dict]) -> str:
    return "\n".join(f'[{c["chunk_id"]}] {c["text"]}' for c in chunks)


def generation_agent(query: str, retrieved_chunks: List[Dict]) -> str:
    """Generate an answer grounded in the retrieved chunks."""
    prompt = (
        GENERATION_PROMPT
        .replace("{query}", query)
        .replace("{sources}", _format_sources(retrieved_chunks))
    )
    return call_gemini(prompt, temperature=0.2)
