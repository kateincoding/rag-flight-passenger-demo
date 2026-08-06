"""Orchestrator — composes the four agents into one pipeline.

Early-exit routing: if the query is below the clarity threshold, stop at
stage 1 and ask for clarification — never spend retrieval/generation/judge
budget on an unanswerable query. Otherwise run retrieve -> generate -> judge.

Every run accumulates a `stages` trace (the observability seam).
"""
from typing import Dict, List, Optional

import config
from agents.clarification import clarification_agent
from agents.generation import generation_agent
from agents.judge import llm_judge
from agents.retrieval import VectorStore, retrieval_agent
from agents.rewrite import rewrite_agent


def answer_query(
    query: str,
    store: VectorStore,
    history: Optional[List[Dict]] = None,
    top_k: int = config.DEFAULT_TOP_K,
) -> Dict:
    """Full agentic RAG pipeline. Returns a structured result with a stage trace.

    `history` is the prior chat turns ([{role, content}, ...]); it lets the
    rewrite agent resolve follow-ups into standalone queries. The rest of the
    pipeline runs on that rewritten query.
    """
    result: Dict = {"query": query, "stages": []}

    # ── STAGE 0: Rewrite (multi-turn contextualization) ──
    standalone = rewrite_agent(query, history)
    result["standalone_query"] = standalone
    if standalone != query:
        result["stages"].append({
            "stage": "rewrite",
            "original": query,
            "standalone": standalone,
        })
    query = standalone

    # ── STAGE 1: Clarification ──
    clarification = clarification_agent(query)
    result["stages"].append({
        "stage": "clarification",
        "clarity_score": clarification.clarity_score,
        "missing_entities": clarification.missing_entities,
        "reasoning": clarification.reasoning,
    })

    if clarification.clarity_score < config.CLARITY_THRESHOLD:
        result["action"] = "ask_clarification"
        result["clarifying_questions"] = clarification.clarifying_questions
        result["final_response"] = (
            "I need a bit more information to help you:\n"
            + "\n".join(f"  - {q}" for q in clarification.clarifying_questions)
        )
        return result

    # ── STAGE 2: Retrieval ──
    retrieved = retrieval_agent(store, query, top_k=top_k)
    result["stages"].append({
        "stage": "retrieval",
        "chunks": [{"id": c["chunk_id"], "score": c["score"]} for c in retrieved],
    })

    # ── STAGE 3: Generation ──
    answer = generation_agent(query, retrieved)
    result["stages"].append({"stage": "generation", "answer": answer})

    # ── STAGE 4: LLM-as-Judge ──
    judge = llm_judge(query, retrieved, answer)
    result["stages"].append({
        "stage": "judge",
        "faithfulness": judge.faithfulness,
        "relevance": judge.relevance,
        "groundedness": judge.groundedness,
        "citation_check": judge.citation_check,
        "verdict": judge.verdict,
        "reasoning": judge.reasoning,
    })

    result["action"] = "answer"
    result["final_response"] = answer
    result["retrieved"] = retrieved
    result["judge"] = judge
    if judge.verdict == "FAIL":
        result["warning"] = "Judge flagged this answer — human review recommended."

    return result
