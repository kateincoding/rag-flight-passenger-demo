# ✈️ Multi-Agentic RAG — Flight Passenger Assistant

A production-minded **agentic Retrieval-Augmented Generation** system that answers flight-passenger questions (baggage rules, schedules, in-flight services) from a grounded knowledge base — and, crucially, **knows when it doesn't have enough information to answer**.

This is *agentic* RAG, not a fixed pipeline: an orchestrator with **early-exit routing** decides, per query, whether to clarify, retrieve, generate, and whether the generated answer is trustworthy enough to return. Four specialised agents make those decisions — the control flow is driven by their verdicts, not hardcoded.

Built with Google Gemini, FAISS, and Streamlit. The architecture is deliberately provider-agnostic so it ports to a managed cloud stack (Azure or AWS) with no structural change.

> I built this to demonstrate how I approach LLM systems beyond a naive "embed → retrieve → generate" loop: with **input guardrails**, an **evaluation layer**, and a design that treats a hosted inference API as the unreliable, rate-limited dependency it actually is.

---

## Why this is agentic RAG, not "just RAG"

Standard RAG is a fixed chain: embed → retrieve → generate, every query, no decisions. This system instead lets agents *route* the query. Standard RAG also assumes every query is well-formed and every generated answer is trustworthy — neither holds in production. Two controls wrap the classic retrieval loop, and each can change the control flow:

- **A clarification agent** runs *before* retrieval. If a query is ambiguous (`"can I bring this?"`) or out-of-domain (`"what is the meaning of life?"`), the system asks a follow-up instead of confidently retrieving garbage. Guardrails belong at the cheapest point in the pipeline.
- **An LLM-as-Judge** runs *after* generation. Every answer is scored on faithfulness, relevance, and groundedness, with a **deterministic citation check** on top — a hallucinated citation fails the answer without asking the model to grade its own homework.

---

## Architecture

```
                        ┌─────────────────────────────────────┐
   user query  ─────────▶            Orchestrator             │
   + chat history       │        (early-exit routing)          │
                        └───────────────┬─────────────────────┘
                                        │
              ┌─────────────────────────▼──────────────────────────┐
              │  0. Rewrite Agent  (multi-turn contextualization)   │
              │     history + follow-up ──▶ standalone query        │
              └─────────────────────────┬──────────────────────────┘
                                        │
              ┌─────────────────────────▼──────────────────────────┐
              │  1. Clarification Agent   (input guardrail)         │
              │     score < 0.7  ──▶  ask a follow-up, stop here    │
              └─────────────────────────┬──────────────────────────┘
                                        │ score ≥ 0.7
              ┌─────────────────────────▼──────────────────────────┐
              │  2. Retrieval Agent   (FAISS + Gemini embeddings)   │
              │     top-k semantic search over the knowledge base   │
              └─────────────────────────┬──────────────────────────┘
                                        │
              ┌─────────────────────────▼──────────────────────────┐
              │  3. Generation Agent  (grounded answer + citations) │
              └─────────────────────────┬──────────────────────────┘
                                        │
              ┌─────────────────────────▼──────────────────────────┐
              │  4. LLM-as-Judge  (faithfulness / relevance /       │
              │     groundedness)  +  deterministic citation check  │
              └─────────────────────────────────────────────────────┘
```

Each query returns a full **stage trace** — the seam I'd wire into observability (latency, judge scores, drift) in production.

---

## Project structure

```
.
├── app.py              # Streamlit frontend — chat UI, 4 stages expanding live
├── orchestrator.py     # answer_query(): early-exit routing + stage trace
├── llm.py              # shared Gemini client + retry/backoff on 429
├── config.py           # models, thresholds, API key from env (no secrets in code)
├── agents/
│   ├── rewrite.py        # Agent 0 — multi-turn contextualization
│   ├── clarification.py  # Agent 1 — answerability guardrail
│   ├── retrieval.py      # Agent 2 — VectorStore + retrieval_agent()
│   ├── generation.py     # Agent 3 — grounded, cited answering
│   └── judge.py          # Agent 4 — LLM-as-Judge + citation check
├── knowledge/
│   └── chunks.json       # the knowledge base (synthetic flight policies)
├── requirements.txt
└── README.md
```

---

## Tech stack

| Layer          | Choice                                   | Why |
|----------------|------------------------------------------|-----|
| Generation     | Gemini (`gemini-flash-lite-latest`)      | Fast, cheap, generous free tier |
| Embeddings     | `gemini-embedding-001` (3072-dim)        | Asymmetric doc/query task types |
| Vector store   | FAISS `IndexFlatIP`                      | Exact cosine search, zero infra for the demo |
| Frontend       | Streamlit                                | Fastest path to a professional chat UI |
| Language       | Python 3.12                              | — |

---

## Getting started

### Prerequisites
- **Python 3.12** (managed with [`pyenv`](https://github.com/pyenv/pyenv) below)
- A **Google Gemini API key** — free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### 1. Clone
```bash
git clone <your-repo-url>
cd <repo-name>
```

### 2. Set up Python with pyenv
```bash
# Install pyenv if you don't have it (macOS)
brew install pyenv

# Install and pin the Python version for this project
pyenv install 3.12.13
pyenv local 3.12.13          # writes .python-version
```

### 3. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

### 4. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Provide your API key (never hardcode it)
```bash
export GOOGLE_API_KEY='your-key-here'
```
Or copy the template and fill it in:
```bash
cp .env.example .env
# edit .env, then:  export $(grep -v '^#' .env | xargs)
```

### 6. Run
```bash
streamlit run app.py
```
The app opens at `http://localhost:8501`. Ask a question and watch all four agent stages expand in real time.

---

## Try these queries

| Query | Expected behaviour |
|-------|--------------------|
| `can I bring a laptop in my carry-on?` | Full pipeline → cited answer, judge **PASS** |
| `when does flight AF1234 depart from Barcelona?` | Schedule lookup → cited answer |
| `can I bring this?` | **Ambiguous** → asks which item / carry-on vs checked |
| `what time does my flight leave?` | **Missing entity** → asks for flight number |
| `what is the meaning of life?` | **Out of domain** → declines, asks to stay on-topic |

**Multi-turn (contextualization):** ask `what documents do I need for my service dog?` → the assistant asks for the flight and date → reply `MI250, June 5`. The rewrite agent folds that fragment back into a standalone question and the full pipeline runs — no re-asking, no loop.

---

## Design decisions worth calling out

- **The judge verdict is computed in code, not by the LLM.** I observed the model occasionally emit a numeric verdict instead of `PASS`/`FAIL`, which would silently let a bad answer through. The LLM only supplies the *scores*; the pass/fail threshold lives in `judge.py`.
- **Groundedness has a deterministic cross-check.** `check_citations()` verifies every cited `[ID]` in the answer is actually one of the retrieved chunks. An invented citation forces a FAIL — no need to trust the model to catch itself.
- **Retry-on-429 with the server's own delay.** A hosted, multi-tenant inference API *will* throttle you. `llm.py` treats a `429` as an expected signal: it reads the API's suggested `retry_delay` and backs off, rather than blindly sleeping or crashing.
- **Asymmetric embeddings.** Documents are embedded with `RETRIEVAL_DOCUMENT`, queries with `RETRIEVAL_QUERY` — a small change that measurably tightens retrieval.
- **The model id is configuration.** After hitting a hard `quota: 0` on a deprecated pinned model, I treat the model name as a swappable config value, not a constant baked into the code.

---

## Known limitations & next steps

This is a focused demo, and I'd harden it in this order before production:

1. **Stronger / different judge model** than the generator, to break correlated blind spots in self-evaluation.
2. **Adversarial judge tests** — feed known-hallucinated answers and confirm the judge fails them, so the evaluation layer is proven, not asserted.
3. **Hybrid retrieval** (vector + BM25) for exact policy numbers, plus a cross-encoder reranker.
4. **Retrieval score floor** as a second out-of-domain guard for queries that slip past clarification.
5. ~~**Conversation memory** for multi-turn clarification flows.~~ **Done** — a rewrite agent (Stage 0) folds prior turns into a standalone query, so follow-ups like `MI250, June 5` resolve instead of looping. Next step here: a retrieval score floor tied to the *rewritten* query, and trimming very long histories by token budget.
6. **Observability** — wrap `llm.py` with latency, token counting, and structured logging; monitor judge-score drift.

---

## Production mapping (Azure & AWS)

The architecture was kept provider-agnostic on purpose. Porting to a managed cloud stack is a swap of implementations, not a redesign — the same four agents, orchestrator, and stage trace stay put:

| This demo | Azure production | AWS production |
|-----------|------------------|----------------|
| Gemini `gemini-flash-lite-latest` | Azure OpenAI GPT-4o | Amazon Bedrock (Claude / Nova) |
| `gemini-embedding-001` | Azure OpenAI `text-embedding-3` | Bedrock Titan Text Embeddings |
| FAISS `IndexFlatIP` | Azure AI Search (hybrid + semantic ranking) | Amazon OpenSearch Serverless (k-NN + hybrid) |
| API key in env | Azure Key Vault | AWS Secrets Manager |
| Streamlit app | Azure Functions + Azure OpenAI + Azure AI Search | AWS Lambda + API Gateway (or App Runner) + Bedrock |
| `stages` trace | Application Insights / Azure Monitor | Amazon CloudWatch + X-Ray |

---

## Author

**Katherine Soto** — built as a multi-agent RAG demonstration.
