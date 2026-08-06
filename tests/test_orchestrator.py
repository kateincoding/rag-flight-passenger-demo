"""Routing tests for the orchestrator — agents are stubbed, so no network is used.

These prove the *control flow* (early-exit, full pipeline, rewrite trace),
which is the part that makes this 'agentic' rather than a fixed chain.
"""
import config
from agents.clarification import ClarificationResult
from agents.judge import JudgeResult
import orchestrator


def _stage_names(result):
    return [s["stage"] for s in result["stages"]]


def test_low_clarity_early_exits_before_retrieval(monkeypatch):
    monkeypatch.setattr(orchestrator, "rewrite_agent", lambda q, h: q)
    monkeypatch.setattr(orchestrator, "clarification_agent", lambda q: ClarificationResult(
        clarity_score=0.2,
        missing_entities=["item_name"],
        clarifying_questions=["What item?"],
        reasoning="ambiguous",
    ))
    # If retrieval were reached, this would blow up — proving early-exit.
    monkeypatch.setattr(orchestrator, "retrieval_agent",
                        lambda *a, **k: pytest_fail("retrieval should not run"))

    result = orchestrator.answer_query("can I bring this?", store=None)

    assert result["action"] == "ask_clarification"
    assert _stage_names(result) == ["clarification"]
    assert "What item?" in result["final_response"]


def test_clear_query_runs_full_pipeline(monkeypatch):
    chunks = [{"chunk_id": "SVC-006", "category": "service", "text": "docs", "score": 0.9}]
    monkeypatch.setattr(orchestrator, "rewrite_agent", lambda q, h: q)
    monkeypatch.setattr(orchestrator, "clarification_agent", lambda q: ClarificationResult(
        clarity_score=0.95, reasoning="clear"))
    monkeypatch.setattr(orchestrator, "retrieval_agent", lambda *a, **k: chunks)
    monkeypatch.setattr(orchestrator, "generation_agent",
                        lambda q, c: "You need a health certificate [SVC-006].")
    monkeypatch.setattr(orchestrator, "llm_judge", lambda q, c, a: JudgeResult(
        faithfulness=0.9, relevance=0.9, groundedness=0.9,
        verdict="PASS", reasoning="grounded", citation_check=True))

    result = orchestrator.answer_query("service dog docs for MI250?", store=None)

    assert result["action"] == "answer"
    assert _stage_names(result) == ["clarification", "retrieval", "generation", "judge"]
    assert "[SVC-006]" in result["final_response"]
    assert "warning" not in result


def test_failed_judge_attaches_warning(monkeypatch):
    chunks = [{"chunk_id": "SVC-006", "category": "service", "text": "docs", "score": 0.9}]
    monkeypatch.setattr(orchestrator, "rewrite_agent", lambda q, h: q)
    monkeypatch.setattr(orchestrator, "clarification_agent", lambda q: ClarificationResult(
        clarity_score=0.95, reasoning="clear"))
    monkeypatch.setattr(orchestrator, "retrieval_agent", lambda *a, **k: chunks)
    monkeypatch.setattr(orchestrator, "generation_agent", lambda q, c: "made up [XX-999]")
    monkeypatch.setattr(orchestrator, "llm_judge", lambda q, c, a: JudgeResult(
        faithfulness=0.4, relevance=0.9, groundedness=0.3,
        verdict="FAIL", reasoning="ungrounded", citation_check=False))

    result = orchestrator.answer_query("anything", store=None)

    assert result["action"] == "answer"
    assert "warning" in result


def test_rewrite_recorded_only_when_query_changes(monkeypatch):
    monkeypatch.setattr(orchestrator, "clarification_agent", lambda q: ClarificationResult(
        clarity_score=0.1, clarifying_questions=["?"], reasoning="x"))

    # Rewrite changes the query → a 'rewrite' stage is recorded.
    monkeypatch.setattr(orchestrator, "rewrite_agent", lambda q, h: "standalone version")
    changed = orchestrator.answer_query("MI250", store=None, history=[{"role": "user", "content": "prev"}])
    assert "rewrite" in _stage_names(changed)
    assert changed["standalone_query"] == "standalone version"

    # Rewrite is a no-op → no 'rewrite' stage clutters the trace.
    monkeypatch.setattr(orchestrator, "rewrite_agent", lambda q, h: q)
    same = orchestrator.answer_query("clear query", store=None)
    assert "rewrite" not in _stage_names(same)


def pytest_fail(msg):
    raise AssertionError(msg)
