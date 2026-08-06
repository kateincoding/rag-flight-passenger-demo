"""Central configuration: models, thresholds, API key from environment.

No secrets live in code. Set the key before running:
    export GOOGLE_API_KEY='...'
"""
import os

# ── Models ──
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 3072                     # gemini-embedding-001 output dim
LLM_MODEL = "gemini-flash-lite-latest"   # alias with free-tier quota. flash-latest -> gemini-3.6-flash caps at 20 req/day


# ── Thresholds ──
CLARITY_THRESHOLD = 0.7                  # below this, the clarification agent asks for more info
JUDGE_PASS_THRESHOLD = 0.7               # min of faith/relevance/groundedness to PASS

# ── Retrieval ──
DEFAULT_TOP_K = 3

# ── Paths ──
_HERE = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_PATH = os.path.join(_HERE, "knowledge", "chunks.json")

# ── API key (from env — never hardcode) ──
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")


def require_api_key() -> str:
    """Return the key or raise a clear error if it's missing."""
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Run:  export GOOGLE_API_KEY='your-key'  "
            "before launching the app."
        )
    return GOOGLE_API_KEY
