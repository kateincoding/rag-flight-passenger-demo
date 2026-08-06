"""Shared Gemini client + the single LLM call seam.

Centralising every model call here gives one place to own the realities of a
shared, rate-limited serving layer: retry-on-429 with the server's own
suggested delay, plus a clean place to add logging/latency/token metrics later.
"""
import time

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

import config

_configured = False


def _ensure_configured() -> None:
    global _configured
    if not _configured:
        genai.configure(api_key=config.require_api_key())
        _configured = True


def call_gemini(prompt: str, temperature: float = 0.0, json_mode: bool = False,
                max_retries: int = 3) -> str:
    """Call the generation model. Retry only on 429, waiting the API's suggested delay."""
    _ensure_configured()
    model = genai.GenerativeModel(
        model_name=config.LLM_MODEL,
        generation_config=genai.GenerationConfig(
            temperature=temperature,
            response_mime_type="application/json" if json_mode else "text/plain",
        ),
    )
    for attempt in range(max_retries):
        try:
            return model.generate_content(prompt).text
        except ResourceExhausted as e:
            delay = getattr(e, "retry_delay", None)
            wait = delay.seconds if delay else 2 ** attempt * 5
            if attempt == max_retries - 1:
                raise
            print(f"429 rate-limited. Waiting {wait}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)


def embed(text: str, task_type: str) -> "list[float]":
    """Embed text with the configured embedding model for a given task type."""
    _ensure_configured()
    result = genai.embed_content(
        model=config.EMBEDDING_MODEL,
        content=text,
        task_type=task_type,
    )
    return result["embedding"]
