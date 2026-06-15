"""
ARIA Chat Panel — Industrial Skeuomorphic
==========================================
ARIA = Anomaly Response & Intelligence Assistant.
Full industrial styling, proper empty state, grounding badges.
"""
from __future__ import annotations
import json, logging, re
from typing import Optional
import streamlit as st

logger = logging.getLogger(__name__)
SENTINEL_RE = re.compile(r'<!--VERIFY:(.*?)-->', re.DOTALL)

QUESTION_TYPE_LABELS = {
    "slice_drilldown":   ("Slice drilldown",   "#378add","rgba(55,138,221,.12)"),
    "time_comparison":   ("Time comparison",   "#f59e0b","rgba(245,158,11,.12)"),
    "causal_hypothesis": ("Causal hypothesis", "#7c3aed","rgba(124,58,237,.12)"),
    "action_request":    ("Action request",    "#ff4757","rgba(255,71,87,.12)"),
    "metric_detail":     ("Metric detail",     "#22c55e","rgba(34,197,94,.12)"),
    "financial_impact":  ("Financial impact",  "#f59e0b","rgba(245,158,11,.12)"),
    "scope_check":       ("Scope check",       "#636e72","rgba(99,110,114,.12)"),
    "out_of_scope":      ("Out of scope",      "#ff4757","rgba(255,71,87,.12)"),
    "initial_diagnostic":("Initial diagnostic","#378add","rgba(55,138,221,.12)"),
}

SUGGESTED_QUESTIONS = {
    "3ds_acs_failure":     ["Which countries were affected?","Was non-3DS impacted?",
                            "What does the RC 65 spike tell us?","What SCA exemptions can we apply?"],
    "processor_outage":    ["Which card ranges are affected?","Was there a retry storm?",
                            "How does RC 96 compare to baseline?","Who should I escalate to?"],
    "fraud_attack":        ["Which merchant categories are concentrated?","Is the avg ticket unusual?",
                            "Was RC 59 elevated?","What rules should we apply now?"],
    "acquirer_routing":    ["How long has this drift continued?","Which corridor is affected?",
                            "Is RC 91 trending upward?","What should I tell the acquirer?"],
    "network_rule_change": ["Is this only on weekends?","Which countries are affected?",
                            "What should we tell cardholders?","How does RC 61 compare to last month?"],
    "issuer_rules_misfire":["Which rule is triggering incorrectly?","Is fraud rate actually elevated?",
                            "What changed in fraud rules recently?","How do we revert the rule?"],
    "undetermined":        ["What does the RC distribution show?","Which channels are affected?",
                            "Is fraud rate elevated?","What should I investigate first?"],
}
DEFAULT_QUESTIONS = ["What does the evidence show?","Which dimensions are most affected?",
                     "What are the recommended next steps?","What is the financial impact?"]


def _stream_to_placeholder(placeholder, generator) -> tuple[str, dict]:
    chunks = []
    for chunk in generator:
        chunks.append(chunk)
        display = SENTINEL_RE.sub("","".join(chunks))
        placeholder.markdown(display + "▌")
    full = "".join(chunks)
    match = SENTINEL_RE.search(full)
    if match:
        try:
            meta = json.loads(match.group(1))
        except Exception:
            meta = {"risk":"unknown","warnings":[]}
        clean = SENTINEL_RE.sub("",full).strip()
    else:
        meta  = {"risk":"none","warnings":[]}
        clean = full.strip()
    placeholder.markdown(clean)
    return clean, meta


def _aria_header(anomaly: dict) -> None:
    fc = anomaly.get("failure_class","undetermined")
    st.markdown(
        f"<div style='background:#2d3436;border-radius:8px;padding:11px 14px;margin-bottom:12px;"
        f"box-shadow:8px 8px 16px #babecc,-8px -8px 16px #ffffff;display:flex;align-items:center;gap:10px'>"
        f"<div style='width:10px;height:10px;border-radius:50%;background:#ff4757;"
        f"box-shadow:0 0 8px 2px rgba(255,71,87,.7);flex-shrink:0'></div>"
        f"<div>"
        f"<div style='font-family:\"JetBrains Mono\",monospace;font-size:11px;font-weight:700;"
        f"letter-spacing:.1em;text-transform:uppercase;color:#e0e5ec'>ARIA</div>"
        f"<div style='font-family:\"JetBrains Mono\",monospace;font-size:9px;color:#636e72;"
        f"letter-spacing:.06em'>Anomaly Response &amp; Intelligence Assistant</div>"
        f"</div>"
        f"<div style='margin-left:auto;font-family:\"JetBrains Mono\",monospace;font-size:9px;"
        f"font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:#22c55e;"
        f"background:rgba(34,197,94,.1);padding:3px 8px;border-radius:3px'>ONLINE</div>"
        f"</div>",
        unsafe_allow_html=True
    )


def render_chat_panel(anomaly: dict, llm_client=None) -> None:
    aid = anomaly.get("anomaly_id","")
    fc  = anomaly.get("failure_class","undetermined")

    col_title, col_reset = st.columns([4,1])
    col_reset.button("Clear", key=f"reset_{aid}", use_container_width=True)

    if st.session_state.get(f"_reset_trigger_{aid}"):
        st.session_state["chat_history"] = []
        st.session_state.pop(f"conv_state_{aid}", None)
        st.session_state.pop("chat_processing", None)
        st.session_state.pop(f"_reset_trigger_{aid}", None)
        st.rerun()

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "chat_processing" not in st.session_state:
        st.session_state["chat_processing"] = False

    history = st.session_state["chat_history"]

    # No API key state
    if llm_client is None:
        import os
        # ── Sleek API key entry strip ─────────────────────────────────────
        st.markdown(
            "<div style='background:#2d3436;border-radius:8px;padding:10px 14px;"
            "display:flex;align-items:center;gap:10px;margin-bottom:12px;"
            "box-shadow:8px 8px 16px #babecc,-8px -8px 16px #fff'>"
            "<div style='width:8px;height:8px;border-radius:50%;background:#f59e0b;"
            "box-shadow:0 0 8px rgba(245,158,11,.8);flex-shrink:0'></div>"
            "<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
            "font-weight:700;letter-spacing:.06em;text-transform:uppercase;"
            "color:#e0e5ec;white-space:nowrap'>ARIA offline</span>"
            "<span style='font-family:JetBrains Mono,monospace;font-size:10px;"
            "color:#a8b2d1;margin-left:4px'>· add Groq API key to activate</span>"
            "</div>",
            unsafe_allow_html=True
        )
        key_val = st.text_input(
            "aria_key",
            type="password",
            placeholder="Paste Groq API key (free at console.groq.com)…",
            label_visibility="collapsed",
            key="aria_tab_api_key",
        )
        if key_val:
            os.environ["GROQ_API_KEY"] = key_val
            st.rerun()

        questions = SUGGESTED_QUESTIONS.get(fc, DEFAULT_QUESTIONS)
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:9px;font-weight:500;"
            "letter-spacing:.06em;text-transform:uppercase;color:#9ca3af;margin:12px 0 6px'>"
            "Questions you can ask once connected</div>",
            unsafe_allow_html=True
        )
        for q in questions:
            st.markdown(
                f"<div style='font-family:JetBrains Mono,monospace;font-size:11px;"
                f"color:#4a5568;padding:6px 10px;background:#e0e5ec;border-radius:6px;"
                f"margin-bottom:4px;box-shadow:4px 4px 8px #babecc,-4px -4px 8px #fff'>"
                f"— {q}</div>",
                unsafe_allow_html=True
            )
        return

    # Active ARIA header
    _aria_header(anomaly)

    # Chat history
    for msg in history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
                q_type = msg.get("question_type","")
                h_risk = msg.get("h_risk","none")
                if q_type and q_type in QUESTION_TYPE_LABELS:
                    lbl, color, bg = QUESTION_TYPE_LABELS[q_type]
                    st.markdown(
                        f"<span style='font-family:\"JetBrains Mono\",monospace;font-size:9px;"
                        f"font-weight:700;letter-spacing:.06em;text-transform:uppercase;"
                        f"color:{color};background:{bg};padding:2px 7px;border-radius:3px'>"
                        f"{lbl}</span>",
                        unsafe_allow_html=True
                    )
                # Grounding indicator
                risk_colors = {"none":"#22c55e","low":"#f59e0b","medium":"#f59e0b","high":"#ff4757"}
                risk_labels = {"none":"✓ Verified","low":"~ Low risk","medium":"⚠ Review","high":"✗ Check values"}
                risk_col = risk_colors.get(h_risk,"#636e72")
                risk_lbl = risk_labels.get(h_risk,"—")
                st.markdown(
                    f"<span style='font-family:\"JetBrains Mono\",monospace;font-size:9px;"
                    f"color:{risk_col};margin-left:6px'>{risk_lbl}</span>",
                    unsafe_allow_html=True
                )
                if h_risk in ("medium","high"):
                    for w in msg.get("h_warnings",[])[:2]:
                        st.caption(f"  ↳ {w}")

    # Suggested questions
    if not history:
        questions = SUGGESTED_QUESTIONS.get(fc, DEFAULT_QUESTIONS)
        st.markdown(
            "<div style='font-family:\"JetBrains Mono\",monospace;font-size:9px;font-weight:700;"
            "letter-spacing:.1em;text-transform:uppercase;color:#636e72;margin-bottom:8px'>"
            "Suggested questions for this incident</div>",
            unsafe_allow_html=True
        )
        cols = st.columns(2)
        for i, q in enumerate(questions[:4]):
            with cols[i%2]:
                if st.button(q, key=f"sugg_{fc}_{i}_{aid}", use_container_width=True):
                    st.session_state["pending_question"] = q
                    st.rerun()

    pending_q  = st.session_state.pop("pending_question", None)
    user_input = st.chat_input(
        placeholder="Ask ARIA about this incident...",
        key=f"chat_input_{aid}",
        disabled=st.session_state.get("chat_processing", False),
    )
    question = user_input or pending_q
    if not question:
        return

    st.session_state["chat_processing"] = True
    history.append({"role":"user","content":question})
    with st.chat_message("user"):
        st.markdown(question)

    conv_key = f"conv_state_{aid}"
    if conv_key not in st.session_state:
        try:
            state = llm_client.start_conversation(anomaly)
            st.session_state[conv_key] = state
        except Exception as exc:
            st.error(f"ARIA failed to initialise: {exc}")
            st.session_state["chat_processing"] = False
            return
    else:
        state = st.session_state[conv_key]

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            clean_text, meta = _stream_to_placeholder(
                placeholder,
                llm_client.ask_followup_stream(state, question),
            )
            q_type   = meta.get("question_type","metric_detail")
            h_risk   = meta.get("risk","none")
            warnings = meta.get("warnings",[])

            if q_type in QUESTION_TYPE_LABELS:
                lbl, color, bg = QUESTION_TYPE_LABELS[q_type]
                st.markdown(
                    f"<span style='font-family:\"JetBrains Mono\",monospace;font-size:9px;"
                    f"font-weight:700;letter-spacing:.06em;text-transform:uppercase;"
                    f"color:{color};background:{bg};padding:2px 7px;border-radius:3px'>"
                    f"{lbl}</span>",
                    unsafe_allow_html=True
                )

            risk_colors = {"none":"#22c55e","low":"#f59e0b","medium":"#f59e0b","high":"#ff4757"}
            risk_labels = {"none":"✓ Verified","low":"~ Low risk","medium":"⚠ Review","high":"✗ Check values"}
            risk_col = risk_colors.get(h_risk,"#636e72")
            risk_lbl = risk_labels.get(h_risk,"—")
            st.markdown(
                f"<span style='font-family:\"JetBrains Mono\",monospace;font-size:9px;"
                f"color:{risk_col};margin-left:6px'>{risk_lbl}</span>",
                unsafe_allow_html=True
            )

            if h_risk in ("medium","high"):
                for w in warnings[:2]:
                    st.caption(f"  ↳ {w}")

            history.append({
                "role":"assistant","content":clean_text,
                "question_type":q_type,"h_risk":h_risk,
                "h_warnings":warnings,"is_oos":(q_type=="out_of_scope"),
            })
        except Exception as exc:
            logger.error("ARIA response failed: %s", exc)
            err = str(exc)
            if "rate_limit" in err.lower() or "429" in err:
                placeholder.markdown("⏱ Rate limit reached. Please wait a moment and retry.")
            else:
                placeholder.markdown(f"ARIA error: `{err[:100]}`")
            history.append({"role":"assistant","content":"ARIA encountered an error — please retry.",
                            "question_type":None,"h_risk":"none","h_warnings":[],"is_oos":False})

    st.session_state["chat_history"]    = history
    st.session_state["chat_processing"] = False
    st.rerun()
