"""Tests for the rewrite agent's non-network behaviour."""
from agents.rewrite import _format_history, rewrite_agent


def test_no_history_returns_query_unchanged():
    # Turn 1: no history → passthrough, no LLM call.
    assert rewrite_agent("can I bring a laptop?", history=None) == "can I bring a laptop?"


def test_empty_history_returns_query_unchanged():
    assert rewrite_agent("hello", history=[]) == "hello"


def test_format_history_renders_role_and_content():
    history = [
        {"role": "user", "content": "docs for my service dog?"},
        {"role": "assistant", "content": "Which flight and date?"},
    ]
    out = _format_history(history)
    assert "user: docs for my service dog?" in out
    assert "assistant: Which flight and date?" in out


def test_format_history_truncates_to_last_turns():
    from agents.rewrite import MAX_HISTORY_TURNS
    history = [{"role": "user", "content": str(i)} for i in range(MAX_HISTORY_TURNS + 3)]
    lines = _format_history(history).splitlines()
    assert len(lines) == MAX_HISTORY_TURNS
