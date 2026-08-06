"""Streamlit frontend for the multi-agent RAG demo.

Run:
    export GOOGLE_API_KEY='your-key'
    streamlit run app.py
"""
import streamlit as st

import config
from agents.retrieval import build_vector_store
from orchestrator import answer_query

st.set_page_config(page_title="Multi-Agent RAG — Flight Assistant", page_icon="✈️")


@st.cache_resource(show_spinner="Building vector store...")
def get_store():
    """Build the FAISS store once and reuse across reruns."""
    return build_vector_store()


st.title("✈️ Multi-Agent RAG — Flight Assistant")
st.caption(
    f"Clarification → Retrieval → Generation → LLM-as-Judge  ·  "
    f"model: `{config.LLM_MODEL}`"
)

# ── API key guard ──
if not config.GOOGLE_API_KEY:
    st.error(
        "`GOOGLE_API_KEY` is not set. Stop the app, run "
        "`export GOOGLE_API_KEY='your-key'`, then `streamlit run app.py`."
    )
    st.stop()

store = get_store()

with st.sidebar:
    st.header("Try these")
    st.markdown(
        "- can I bring this?\n"
        "- can I bring a laptop in my carry-on?\n"
        "- when does flight AF1234 depart from Barcelona?\n"
        "- what time does my flight leave?\n"
        "- what is the meaning of life?"
    )
    st.divider()
    st.caption(
        f"clarity threshold: {config.CLARITY_THRESHOLD}  ·  "
        f"judge pass: {config.JUDGE_PASS_THRESHOLD}  ·  "
        f"top-k: {config.DEFAULT_TOP_K}"
    )

# ── chat history ──
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def render_stages(result: dict) -> None:
    """Show each pipeline stage in an expander."""
    stages = {s["stage"]: s for s in result["stages"]}

    # Stage 0 — rewrite (only present when the follow-up was contextualized)
    if "rewrite" in stages:
        r = stages["rewrite"]
        with st.expander("0 · Rewrite — follow-up made standalone", expanded=True):
            st.write("**You asked:**", r["original"])
            st.write("**Interpreted as:**", r["standalone"])

    # Stage 1 — clarification
    if "clarification" in stages:
        c = stages["clarification"]
        clear = c["clarity_score"] >= config.CLARITY_THRESHOLD
        label = f"1 · Clarification — score {c['clarity_score']:.2f} " \
                f"[{'CLEAR' if clear else 'AMBIGUOUS'}]"
        with st.expander(label, expanded=not clear):
            st.write(c["reasoning"])
            if c.get("missing_entities"):
                st.write("**Missing:**", ", ".join(c["missing_entities"]))

    # Stage 2 — retrieval
    if "retrieval" in stages:
        with st.expander("2 · Retrieval — top-k chunks"):
            for ch in stages["retrieval"]["chunks"]:
                st.write(f"`{ch['id']}`  ·  score {ch['score']:.3f}")

    # Stage 3 — generation
    if "generation" in stages:
        with st.expander("3 · Generation — grounded answer", expanded=True):
            st.markdown(stages["generation"]["answer"])

    # Stage 4 — judge
    if "judge" in stages:
        j = stages["judge"]
        passed = j["verdict"] == "PASS"
        with st.expander(f"4 · LLM-as-Judge — {j['verdict']}", expanded=not passed):
            col1, col2, col3 = st.columns(3)
            col1.metric("Faithfulness", f"{j['faithfulness']:.2f}")
            col2.metric("Relevance", f"{j['relevance']:.2f}")
            col3.metric("Groundedness", f"{j['groundedness']:.2f}")
            st.write("Citation check (cited ids ⊆ retrieved):",
                     "✅ pass" if j["citation_check"] else "❌ fail")
            st.caption(j["reasoning"])


if prompt := st.chat_input("Ask a flight question..."):
    # Snapshot history BEFORE appending the current turn — the rewrite agent
    # needs prior turns to resolve follow-ups, not the message being asked now.
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Running agent pipeline..."):
            result = answer_query(prompt, store, history=history)
        render_stages(result)
        final = result["final_response"]
        if result.get("warning"):
            st.warning(result["warning"])
        st.markdown("---")
        st.markdown(final)

    st.session_state.messages.append({"role": "assistant", "content": result["final_response"]})
