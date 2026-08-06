"""Shared helpers used across agents.

Kept deliberately dependency-free so the pieces that don't touch the network
(JSON parsing, math) can be unit-tested in isolation.
"""
import json
from typing import Dict


def parse_json_object(text: str) -> Dict:
    """Parse a JSON object from model output, tolerating ```json fences or stray prose.

    Models frequently wrap JSON in Markdown fences or add a sentence before/after
    it. This strips fences and slices to the outermost ``{ ... }`` before parsing.
    Raises ``json.JSONDecodeError`` if no valid object is found.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return json.loads(text)


def clamp01(x) -> float:
    """Clamp a value into the [0.0, 1.0] range."""
    return max(0.0, min(1.0, float(x)))
