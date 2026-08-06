"""Agent 1 — Clarification: an input guardrail before retrieval.

Scores query answerability, names missing entities, and generates the
questions to ask back. Runs before any retrieval/generation budget is spent.
"""
from dataclasses import dataclass, field
from typing import List

from llm import call_gemini
from utils import parse_json_object

CLARIFICATION_PROMPT = """You are a clarification agent for a flight assistant. Your job is to decide if a passenger query has enough information to be answered from a knowledge base of flight policies, baggage rules, schedules, and services.

Analyze the query and return a JSON object with:
- "clarity_score": float from 0.0 (completely ambiguous) to 1.0 (perfectly clear)
- "missing_entities": list of what's missing (e.g. ["item_name", "flight_number"])
- "clarifying_questions": list of specific questions to ask the user (empty if clarity_score >= 0.7)
- "reasoning": one sentence explaining the score

Examples:

Query: "can I bring this?"
Output: {"clarity_score": 0.2, "missing_entities": ["item_name"], "clarifying_questions": ["What item are you asking about?", "Do you mean carry-on or checked baggage?"], "reasoning": "The item being referenced is not specified."}

Query: "when does AF1234 depart?"
Output: {"clarity_score": 0.95, "missing_entities": [], "clarifying_questions": [], "reasoning": "Flight number and question are both specified."}

Now analyze this query:
Query: "{query}"

Return only the JSON object, no other text."""


@dataclass
class ClarificationResult:
    clarity_score: float
    missing_entities: List[str] = field(default_factory=list)
    clarifying_questions: List[str] = field(default_factory=list)
    reasoning: str = ""


def clarification_agent(query: str) -> ClarificationResult:
    """Analyze query and return the clarification decision."""
    prompt = CLARIFICATION_PROMPT.replace("{query}", query)
    data = parse_json_object(call_gemini(prompt, temperature=0.0, json_mode=True))
    return ClarificationResult(
        clarity_score=float(data["clarity_score"]),
        missing_entities=data.get("missing_entities", []),
        clarifying_questions=data.get("clarifying_questions", []),
        reasoning=data.get("reasoning", ""),
    )
