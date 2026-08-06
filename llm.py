"""Shared Gemini client + the single LLM call seam.

Centralising every model call here gives one place to own the realities of a
shared, rate-limited serving layer: retry-on-429 with the server's own
suggested delay, plus a clean place to add logging/latency/token metrics later.

Uses the current `google-genai` SDK (the older `google-generativeai` package is
deprecated). Because both generation and embedding go through this module,
swapping providers is a one-file change.
"""
import time
from typing import List, Optional

from google import genai
from google.genai import types
from google.genai.errors import ClientError

import config

_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    """Lazily build one client; the API key is read the first time it's needed."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.require_api_key())
    return _client


def _retry_delay_seconds(err: ClientError) -> Optional[int]:
    """Pull the server's suggested retry delay (e.g. '5s') out of a 429, if present."""
    try:
        for item in err.details.get("error", {}).get("details", []):
            delay = item.get("retryDelay")
            if isinstance(delay, str) and delay.endswith("s"):
                return int(float(delay[:-1]))
    except (AttributeError, TypeError, ValueError):
        pass
    return None


def call_gemini(prompt: str, temperature: float = 0.0, json_mode: bool = False,
                max_retries: int = 3) -> str:
    """Call the generation model. Retry only on 429, waiting the API's suggested delay."""
    client = _get_client()
    generation_config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json" if json_mode else "text/plain",
    )
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=config.LLM_MODEL,
                contents=prompt,
                config=generation_config,
            )
            return response.text
        except ClientError as e:
            if e.code != 429:        # only rate limits are retryable here
                raise
            wait = _retry_delay_seconds(e) or 2 ** attempt * 5
            if attempt == max_retries - 1:
                raise
            print(f"429 rate-limited. Waiting {wait}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)
    # Unreachable: the loop returns on success and re-raises on the last attempt.
    raise RuntimeError("call_gemini exhausted retries without returning")


def embed(text: str, task_type: str) -> List[float]:
    """Embed text with the configured embedding model for a given task type."""
    client = _get_client()
    response = client.models.embed_content(
        model=config.EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return response.embeddings[0].values
