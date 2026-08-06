"""Unit tests for utils — pure, no network."""
import json

import pytest

from utils import clamp01, parse_json_object


class TestParseJsonObject:
    def test_plain_object(self):
        assert parse_json_object('{"a": 1}') == {"a": 1}

    def test_strips_markdown_fence(self):
        raw = '```json\n{"clarity_score": 0.9}\n```'
        assert parse_json_object(raw) == {"clarity_score": 0.9}

    def test_ignores_leading_and_trailing_prose(self):
        raw = 'Here you go:\n{"ok": true}\nHope that helps.'
        assert parse_json_object(raw) == {"ok": True}

    def test_raises_on_garbage(self):
        with pytest.raises(json.JSONDecodeError):
            parse_json_object("not json at all")


class TestClamp01:
    @pytest.mark.parametrize("value,expected", [
        (0.5, 0.5),
        (-1.0, 0.0),
        (2.0, 1.0),
        (0, 0.0),
        (1, 1.0),
        ("0.3", 0.3),   # string coercion
    ])
    def test_bounds(self, value, expected):
        assert clamp01(value) == pytest.approx(expected)
