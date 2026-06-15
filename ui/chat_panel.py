"""
ARIA Chat Panel — Industrial Skeuomorphic
==========================================
ARIA = Anomaly Response & Intelligence Assistant.

FIXES vs previous version
──────────────────────────
FIX-1  st.rerun() was called at the end of the chat handler BEFORE the
       response was safely committed to st.session_state.  Streamlit aborts
       the script on st.rerun(), so the assistant turn was appended to a
       *local* `history` list that never got written back.  The user saw the
       streaming answer flash, then "ARIA encountered an error" on the next
       call because the conversation state was inconsistent.

       Resolution: write history to st.session_state FIRST, clear
       chat_processing, THEN call st.rerun().

FIX-2  pending_question was popped with st.session_state.pop() inside the
       same script run that set it (via st.button → st.rerun).  On fast
       re-runs the pop happened before the question was consumed.

       Resolution: read with .get(), only delete after we have actually
       used the value.

FIX-3  conv_state was shared across all anomalies via a single key when
       aid was empty string.  Pinned to aid so different anomalies get
       independent conversation contexts.

FIX-4  Error handler appended a placeholder message then fell through to
       st.rerun() — the next run would see the placeholder and mark ARIA
       as having answered, hiding the real error.  Now we write state before
       rerunning unconditionally.
"""
from __future__ import annotations
import json, logging, re
from typing import Optional
import streamlit as st

logger = logging.getLogger(__name__)
SENTINEL_RE = re.compile(r'<!--VERIFY:(.*?)-->', re.DOTALL)

QUESTION_TYPE_LABELS = {
    "slice_drilldown":    ("Slice drilldown",    "#378add", "rgba(55,138,221,.12)"),
    "time_comparison":    ("Time comparison",    "#f59e0b", "rgba(245,158,11,.12)"),
    "causal_hypothesis":  ("Causal hypothesis",  "#7c3aed", "rgba(124,58,237,.12)"),
    "action_request":     ("Action request",     "#ff4757", "rgba(255,71,87,.12)"),
    "metric_detail":      ("Metric detail",      "#22c55e", "rgba(34,197,94,.12)"),
    "financial_impact":   ("Financial impact",   "#f59e0b", "rgba(245,158,11,.12)"),
    "scope_check":        ("Scope check",        "#636e72", "rgba(99,110,114,.12)"),
    "out_of_scope":       ("Out of scope",       "#ff4757", "rgba(255,71,87,.12)"),
    "initial_diagnostic": ("Initial diagnostic", "#378add", "rgba(55,138,221,.12)"),
}

SUGGESTED_QUESTIONS = {
    "3ds_acs_failure":      ["Which countries were affected?", "Was non-3DS impacted?",
                             "What does the RC 65 spike tell us?", "What SCA exemptions can we apply?"],
    "processor_outage":     ["Which card ranges are affected?", "Was there a retry storm?",
                             "How does RC 96 compare to baseline?", "Who should I escalate to?"],
    "fraud_attack":         ["Which merchant categories are concentrated?", "Is the avg ticket unusual?",
                             "Was RC 59 elevated?", "What rules should we apply now?"],
    "acquirer_routing":     ["How long has this drift continued?", "Which corridor is affected?",
                             "Is RC 91 trending upward?", "What should I tell the acquirer?"],
    "network_rule_change":  ["Is this only on weekends?", "Which countries are affected?",
                             "What should we tell cardholders?", "How does RC 61 compare to last month?"],
    "issuer_rules_misfire": ["Which rule is triggering incorrectly?", "Is fraud rate actually elevated?",
                             "What changed in fraud rules recently?", "How do we revert the rule?"],
    "undetermined":         ["What does the RC distribution show?", "Which channels are affected?",
                             "Is fraud rate elevated?", "What should I investigate first?"],
}
DEFAULT_QUESTIONS = [
    "What does the evidence show?", "Which dimensions are most affected?",
    "What are the recommended next steps?", "What is the financial impact?",
]


def _coerce_meta(obj) -> dict:
    """
    Ensure meta is always a plain dict with 'risk' and 'warnings' keys.

    The LLM client may return a typed VerificationResult object (or similar)
    instead of a plain dict.  Accessing .is_clean or other attributes that
    don't exist on a dict raises AttributeError.  We normalise here so the
    rest of the panel only ever sees a plain dict.
    """
    if isinstance(obj, dict):
        return obj
    # Typed object — pull out the fields we care about defensively
    risk = getattr(obj, "risk", None) or ("none" if getattr(obj, "is_clean", True) else "high")
    warnings = list(getattr(obj, "warnings", []) or [])
    question_type = getattr(obj, "question_type", None)
    result: dict = {"risk": risk, "warnings": warnings}
    if question_type is not None:
        result["question_type"] = question_type
    return result


def _stream_to_placeholder(placeholder, generator) -> tuple[str, dict]:
    """
    Consume the streaming generator, update the placeholder live,
    then parse the optional VERIFY sentinel from the completed text.
    Returns (clean_text, meta_dict).
    """
    chunks: list[str] = []
    for chunk in generator:
        chunks.append(chunk)
        display = SENTINEL_RE.sub("", "".join(chunks))
        placeholder.markdown(display + "▌")

    full = "".join(chunks)
    match = SENTINEL_RE.search(full)
    if match:
        try:
            raw_meta = json.loads(match.group(1))
        except Exception:
            raw_meta = {"risk": "unknown", "warnings": []}
        clean = SENTINEL_RE.sub("", full).strip()
    else:
        raw_meta  = {"risk": "none", "warnings": []}
        clean = full.strip()

    meta = _coerce_meta(raw_meta)
    placeholder.markdown(clean)
    return clean, meta


def _aria_header(anomaly: dict) -> None:
    st.markdown(
        "<div style='background:#1e2530;border:1px solid #2d3a4a;border-radius:8px;"
        "padding:10px 14px;margin-bottom:14px;"
        "box-shadow:0 2px 8px rgba(0,0,0,.4);display:flex;align-items:center;gap:10px'>"
        "<div style='width:9px;height:9px;border-radius:50%;background:#ff4757;"
        "box-shadow:0 0 6px 2px rgba(255,71,87,.5);flex-shrink:0'></div>"
        "<div>"
        "<div style='font-family:\"JetBrains Mono\",monospace;font-size:11px;font-weight:700;"
        "letter-spacing:.1em;text-transform:uppercase;color:#e0e5ec'>ARIA</div>"
        "<div style='font-family:\"JetBrains Mono\",monospace;font-size:9px;color:#5a6a7a;"
        "letter-spacing:.04em'>Anomaly Response &amp; Intelligence Assistant</div>"
        "</div>"
        "<div style='margin-left:auto;font-family:\"JetBrains Mono\",monospace;font-size:9px;"
        "font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:#22c55e;"
        "background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.2);"
        "padding:2px 8px;border-radius:4px'>ONLINE</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_type_badge(q_type: str) -> None:
    if q_type and q_type in QUESTION_TYPE_LABELS:
        lbl, color, bg = QUESTION_TYPE_LABELS[q_type]
        st.markdown(
            f"<span style='font-family:\"JetBrains Mono\",monospace;font-size:9px;"
            f"font-weight:700;letter-spacing:.06em;text-transform:uppercase;"
            f"color:{color};background:{bg};padding:2px 7px;border-radius:3px'>"
            f"{lbl}</span>",
            unsafe_allow_html=True,
        )


def _render_grounding(h_risk: str, warnings: list) -> None:
    risk_colors = {"none": "#22c55e", "low": "#f59e0b", "medium": "#f59e0b", "high": "#ff4757"}
    risk_labels = {"none": "✓ Verified", "low": "~ Low risk", "medium": "⚠ Review", "high": "✗ Check values"}
    risk_col = risk_colors.get(h_risk, "#636e72")
    risk_lbl = risk_labels.get(h_risk, "—")
    st.markdown(
        f"<span style='font-family:\"JetBrains Mono\",monospace;font-size:9px;"
        f"color:{risk_col};margin-left:6px'>{risk_lbl}</span>",
        unsafe_allow_html=True,
    )
    if h_risk in ("medium", "high"):
        for w in warnings[:2]:
            st.caption(f"  ↳ {w}")


def render_chat_panel(anomaly: dict, llm_client=None) -> None:
    aid = anomaly.get("anomaly_id", "") or "default"
    fc  = anomaly.get("failure_class", "undetermined")

    # ── Session-state keys scoped to this anomaly ──────────────────────────
    hist_key = f"chat_history_{aid}"
    proc_key = f"chat_processing_{aid}"
    conv_key = f"conv_state_{aid}"           # FIX-3: scoped per anomaly
    pend_key = f"pending_question_{aid}"

    if hist_key not in st.session_state:
        st.session_state[hist_key] = []
    if proc_key not in st.session_state:
        st.session_state[proc_key] = False

    history = st.session_state[hist_key]

    # ── No-API-key state ────────────────────────────────────────────────────
    if llm_client is None:
        import os
        st.markdown(
            "<div style='background:#1e2530;border:1px solid #2d3a4a;border-radius:8px;"
            "padding:10px 14px;display:flex;align-items:center;gap:10px;margin-bottom:14px;"
            "box-shadow:0 2px 8px rgba(0,0,0,.4)'>"
            "<div style='width:8px;height:8px;border-radius:50%;background:#f59e0b;"
            "box-shadow:0 0 6px rgba(245,158,11,.7);flex-shrink:0'></div>"
            "<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
            "font-weight:700;letter-spacing:.06em;text-transform:uppercase;"
            "color:#e0e5ec;white-space:nowrap'>ARIA offline</span>"
            "<span style='font-family:JetBrains Mono,monospace;font-size:10px;"
            "color:#5a6a7a;margin-left:4px'>· add Groq API key in sidebar to activate</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        key_val = st.text_input(
            "aria_key",
            type="password",
            placeholder="Paste Groq API key (free at console.groq.com)…",
            label_visibility="collapsed",
            key=f"aria_tab_api_key_{aid}",
        )
        if key_val:
            os.environ["GROQ_API_KEY"] = key_val
            st.rerun()

        questions = SUGGESTED_QUESTIONS.get(fc, DEFAULT_QUESTIONS)
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:9px;font-weight:700;"
            "letter-spacing:.08em;text-transform:uppercase;color:#5a6a7a;margin:14px 0 8px'>"
            "Questions you can ask once connected</div>",
            unsafe_allow_html=True,
        )
        for q in questions:
            st.markdown(
                f"<div style='font-family:JetBrains Mono,monospace;font-size:11px;"
                f"color:#8892a4;padding:7px 12px;background:#1a2030;"
                f"border:1px solid #2d3a4a;border-radius:6px;margin-bottom:5px'>"
                f"→ {q}</div>",
                unsafe_allow_html=True,
            )
        return

    # ── Active ARIA header with inline Clear button ─────────────────────────
    # Only split into columns when there is history to show the Clear button.
    # Creating a column pair when the button is absent leaves a ghost transparent
    # block below the header — avoided by rendering the header alone when empty.
    if history:
        col_hdr, col_clr = st.columns([6, 1])
        with col_hdr:
            _aria_header(anomaly)
        with col_clr:
            st.markdown("<div style='padding-top:6px'>", unsafe_allow_html=True)
            if st.button(
                "✕ Clear",
                key=f"reset_{aid}",
                help="Clear conversation history",
            ):
                st.session_state[hist_key] = []
                st.session_state.pop(conv_key, None)
                st.session_state[proc_key] = False
                st.session_state.pop(pend_key, None)
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        _aria_header(anomaly)

    # ── Render existing history ─────────────────────────────────────────────
    for msg in history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
                _render_type_badge(msg.get("question_type", ""))
                _render_grounding(msg.get("h_risk", "none"), msg.get("h_warnings", []))

    # ── Suggested questions (only when history is empty) ────────────────────
    if not history:
        questions = SUGGESTED_QUESTIONS.get(fc, DEFAULT_QUESTIONS)
        st.markdown(
            "<div style='font-family:\"JetBrains Mono\",monospace;font-size:9px;font-weight:700;"
            "letter-spacing:.08em;text-transform:uppercase;color:#5a6a7a;margin-bottom:8px'>"
            "Suggested questions for this incident</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        for i, q in enumerate(questions[:4]):
            with cols[i % 2]:
                if st.button(q, key=f"sugg_{fc}_{i}_{aid}", use_container_width=True):
                    # FIX-2: set pending key, rerun; consume it at top of next run
                    st.session_state[pend_key] = q
                    st.rerun()

    # ── FIX-2: consume pending question safely ──────────────────────────────
    pending_q = st.session_state.get(pend_key)

    user_input = st.chat_input(
        placeholder="Ask ARIA about this incident...",
        key=f"chat_input_{aid}",
        disabled=st.session_state.get(proc_key, False),
    )

    question = user_input or pending_q
    if not question:
        return

    # Consume pending so it doesn't repeat
    if pending_q:
        st.session_state.pop(pend_key, None)

    # ── Initialise conversation state if needed ─────────────────────────────
    if conv_key not in st.session_state:
        try:
            state = llm_client.start_conversation(anomaly)
            st.session_state[conv_key] = state
        except Exception as exc:
            st.error(f"ARIA failed to initialise: {exc}")
            return
    else:
        state = st.session_state[conv_key]

    # ── Stream the answer ────────────────────────────────────────────────────
    st.session_state[proc_key] = True

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # NOTE: we do NOT call st.empty() here before we need it.
        # st.empty() used to create a ghost transparent block in Streamlit because
        # the placeholder was allocated immediately but left unfilled until the
        # generator yielded its first chunk. Instead we create it inside the try block.
        placeholder = None
        try:
            placeholder = st.empty()
            clean_text, meta = _stream_to_placeholder(
                placeholder,
                llm_client.ask_followup_stream(state, question),
            )
            meta     = _coerce_meta(meta)
            q_type   = meta.get("question_type", "metric_detail")
            h_risk   = meta.get("risk", "none")
            warnings = meta.get("warnings", [])

            _render_type_badge(q_type)
            _render_grounding(h_risk, warnings)

            # FIX-1: write to session_state BEFORE st.rerun()
            history.append({"role": "user", "content": question})
            history.append({
                "role":          "assistant",
                "content":       clean_text,
                "question_type": q_type,
                "h_risk":        h_risk,
                "h_warnings":    warnings,
                "is_oos":        (q_type == "out_of_scope"),
            })

        except Exception as exc:
            logger.error("ARIA response failed: %s", exc)
            err = str(exc)
            if "rate_limit" in err.lower() or "429" in err:
                answer = "⏱ Rate limit reached. Please wait a moment and retry."
            else:
                answer = f"ARIA error: `{err[:120]}`"

            if placeholder is not None:
                placeholder.markdown(answer)
            else:
                st.markdown(answer)

            # FIX-4: still commit both turns so history is consistent
            history.append({"role": "user", "content": question})
            history.append({
                "role":          "assistant",
                "content":       answer,
                "question_type": None,
                "h_risk":        "none",
                "h_warnings":    [],
                "is_oos":        False,
            })

    # FIX-1: commit THEN rerun
    st.session_state[hist_key]  = history
    st.session_state[proc_key]  = False
    st.rerun()