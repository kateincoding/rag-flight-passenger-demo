"""Agent 4 — LLM-as-Judge: the evaluation layer.

Hardened vs. the notebook:
- The PASS/FAIL verdict is computed in *code* from the scores, never taken
  from the LLM (which was observed to emit a numeric verdict).
- Groundedness gets a deterministic cross-check: are the cited ids a subset
  of the retrieved ids? A citation the model invented is caught without
  trusting the model to catch itself.
"""
import re
from dataclasses import dataclass
from typing import Dict, List

import config
from llm import call_gemini
from utils import clamp01, parse_json_object

JUDGE_PROMPT = """You are evaluating the quality of a flight assistant's answer. Rate each dimension from 0.0 to 1.0.

Query: {query}

Sources provided:
{sources}

Answer generated:
{answer}

Evaluate:
1. faithfulness (0-1): does the answer stay within what the sources say? Penalize claims not in the sources.
2. relevance (0-1): does the answer address the query directly?
3. groundedness (0-1): are citations present and correct?

Return a JSON object with keys: faithfulness, relevance, groundedness, reasoning (one sentence). Do NOT include a verdict."""


@dataclass
class JudgeResult:
    faithfulness: float
    relevance: float
    groundedness: float
    verdict: str          # computed in code, not from the LLM
    reasoning: str
    citation_check: bool  # deterministic: all cited ids were retrieved


def check_citations(answer: str, retrieved_chunks: List[Dict]) -> bool:
    """Deterministic: every [ID] cited in the answer must be a retrieved chunk."""
    cited = set(re.findall(r"\[([A-Z]+-\d+)\]", answer))
    retrieved_ids = {c["chunk_id"] for c in retrieved_chunks}
    return cited.issubset(retrieved_ids)   # True also when nothing is cited


def llm_judge(query: str, retrieved_chunks: List[Dict], answer: str) -> JudgeResult:
    """Score the answer, then decide PASS/FAIL in code."""
    sources = "\n".join(f'[{c["chunk_id"]}] {c["text"]}' for c in retrieved_chunks)
    prompt = (
        JUDGE_PROMPT
        .replace("{query}", query)
        .replace("{sources}", sources)
        .replace("{answer}", answer)
    )
    data = parse_json_object(call_gemini(prompt, temperature=0.0, json_mode=True))

    faith = clamp01(data["faithfulness"])
    rel = clamp01(data["relevance"])
    ground = clamp01(data["groundedness"])
    citations_ok = check_citations(answer, retrieved_chunks)

    # Verdict decided here, not by the model. Bad citations force a FAIL.
    passed = (
        min(faith, rel, ground) >= config.JUDGE_PASS_THRESHOLD
        and citations_ok
    )

    return JudgeResult(
        faithfulness=faith,
        relevance=rel,
        groundedness=ground,
        verdict="PASS" if passed else "FAIL",
        reasoning=data.get("reasoning", ""),
        citation_check=citations_ok,
    )
