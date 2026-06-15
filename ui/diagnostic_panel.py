"""
Diagnostic Panel — Industrial Skeuomorphic
============================================
Neumorphic tiles, dark escalation panel, screw corners.
"""
from __future__ import annotations
import json, logging, re
from typing import Optional
import streamlit as st

logger = logging.getLogger(__name__)
SENTINEL_RE = re.compile(r'<!--VERIFY:(.*?)-->', re.DOTALL)

SEV_CFG = {
    "critical": {"color":"#ff4757","bg":"rgba(255,71,87,.08)"},
    "high":     {"color":"#f59e0b","bg":"rgba(245,158,11,.08)"},
    "medium":   {"color":"#378add","bg":"rgba(55,138,221,.08)"},
    "low":      {"color":"#22c55e","bg":"rgba(34,197,94,.08)"},
}

FC_LABELS = {
    "3ds_acs_failure":      "3DS authentication failure",
    "processor_outage":     "Processor outage",
    "fraud_attack":         "Fraud attack",
    "acquirer_routing":     "Routing issue",
    "network_rule_change":  "Network rule change",
    "issuer_rules_misfire": "Incorrect declines",
    "undetermined":         "Under investigation",
}

CONF_LABELS = {"high":"High confidence","medium":"Medium confidence","low":"Low confidence"}

RISK_BADGES = {
    "none":   ("✓ All values verified",              "#22c55e","rgba(34,197,94,.12)"),
    "low":    ("~ Minor uncited numbers",            "#f59e0b","rgba(245,158,11,.12)"),
    "medium": ("⚠ Review cited values",              "#f59e0b","rgba(245,158,11,.12)"),
    "high":   ("✗ Value mismatch detected",          "#ff4757","rgba(255,71,87,.12)"),
}


def _dh_tile(label: str, value: str, sub: str, accent: bool = False) -> str:
    color = "#ff4757" if accent else "#636e72"
    val_color = "#ff4757" if accent else "#2d3436"
    return (
        f"<div style='background:#e0e5ec;border-radius:8px;padding:10px 12px;"
        f"box-shadow:8px 8px 16px #babecc,-8px -8px 16px #ffffff;position:relative;overflow:hidden'>"
        f"<div style='position:absolute;top:0;left:0;right:0;height:2px;background:{'#ff4757' if accent else '#babecc'}'></div>"
        # screw
        f"<div style='position:absolute;top:8px;left:8px;width:7px;height:7px;border-radius:50%;"
        f"background:radial-gradient(circle at 3px 3px,rgba(255,255,255,.5) 1.5px,transparent 2px),#babecc;"
        f"box-shadow:1px 1px 2px rgba(0,0,0,.18),-1px -1px 1px rgba(255,255,255,.6)'></div>"
        f"<div style='font-family:\"JetBrains Mono\",monospace;font-size:9px;font-weight:700;"
        f"letter-spacing:.08em;text-transform:uppercase;color:#4a5568;margin-bottom:4px'>{label}</div>"
        f"<div style='font-family:\"JetBrains Mono\",monospace;font-size:15px;font-weight:700;"
        f"color:{val_color};letter-spacing:-.01em;line-height:1'>{value}</div>"
        f"<div style='font-family:\"JetBrains Mono\",monospace;font-size:9px;color:{color};margin-top:2px'>{sub}</div>"
        f"</div>"
    )


def _strip_sentinel(text: str) -> tuple[str, dict]:
    match = SENTINEL_RE.search(text)
    if match:
        try:
            meta = json.loads(match.group(1))
        except Exception:
            meta = {"risk":"unknown","warnings":[]}
        return SENTINEL_RE.sub("",text).strip(), meta
    return text.strip(), {"risk":"none","warnings":[]}


def _render_streaming(placeholder, generator) -> tuple[str, dict]:
    chunks = []
    for chunk in generator:
        chunks.append(chunk)
        display = SENTINEL_RE.sub("","".join(chunks))
        placeholder.markdown(display + "▌")
    full  = "".join(chunks)
    clean, meta = _strip_sentinel(full)
    placeholder.markdown(clean)
    return clean, meta


def _static_narrative(anomaly: dict) -> str:
    fc    = anomaly.get("failure_class","undetermined")
    fc_lbl = FC_LABELS.get(fc,fc)
    sev   = anomaly.get("severity","medium").upper()
    ts    = anomaly.get("first_seen_ts","")[:10]
    obs   = anomaly.get("observed_value",0)
    base  = anomaly.get("baseline_value",0)
    sigma = anomaly.get("deviation_sigma",0)
    dur   = anomaly.get("duration_hours",0)
    conf  = anomaly.get("failure_class_confidence","low")
    sl    = anomaly.get("affected_slice",{})
    slice_str = " / ".join(
        str(v) for v in sl.values()
        if str(v).lower() not in ("all","","none")
    )[:60]
    evidence = anomaly.get("evidence",[])
    ev_lines = "\n".join(f"- {ev}" for ev in evidence[:4])

    p1 = (f"**WHAT HAPPENED**\n\n"
          f"On **{ts}**, a **{sev}** anomaly was detected on **{slice_str}**. "
          f"The approval rate registered at **{obs:.1%}** against a 7-day baseline of **{base:.1%}** — "
          f"a deviation of **{sigma:.1f}σ** persisting for **{dur} hours**.")

    p2 = (f"\n\n**WHY IT HAPPENED**\n\n"
          f"The detection engine attributes this to **{fc_lbl}** ({conf} confidence). "
          f"Supporting evidence:\n\n{ev_lines}")

    esc = anomaly.get("recommended_escalation","")
    p3  = (f"\n\n**WHAT TO DO NOW**\n\n{esc}" if esc
           else "\n\n**WHAT TO DO NOW**\n\nSee escalation path below.")
    return p1 + p2 + p3


def render_diagnostic_panel(anomaly: dict, llm_client=None) -> None:
    aid   = anomaly.get("anomaly_id","")
    sev   = anomaly.get("severity","medium")
    cfg   = SEV_CFG.get(sev, SEV_CFG["medium"])
    fc    = anomaly.get("failure_class","undetermined")
    fc_lbl = FC_LABELS.get(fc,fc)
    conf  = CONF_LABELS.get(anomaly.get("failure_class_confidence","low"),"")
    sigma = anomaly.get("deviation_sigma",0)
    dur   = anomaly.get("duration_hours",0)
    txns  = anomaly.get("volume_evidence",{}).get("txn_count_observed",0)
    ts    = anomaly.get("first_seen_ts","")[:10]

    # Batch rank
    batch_rank = "—"
    for ev in anomaly.get("evidence",[]):
        if "Batch rank #" in ev:
            batch_rank = ev.split("Batch rank ")[1]
            break

    # 4-tile header
    tiles_html = (
        _dh_tile("Incident type", fc_lbl, conf, accent=True)
        + _dh_tile("Severity score", f"{sigma:+.1f}σ", sev.upper(), accent=True)
        + _dh_tile("Duration", f"{dur}h" if dur<48 else f"{dur//24}d", f"From {ts}")
        + _dh_tile("Txns in window", f"{int(txns):,}" if txns else "—", batch_rank)
    )

    st.markdown(
        f"<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:13px'>"
        f"{tiles_html}</div>",
        unsafe_allow_html=True
    )

    # Anomaly ID line
    st.markdown(
        f"<div style='font-family:\"JetBrains Mono\",monospace;font-size:9px;color:#636e72;"
        f"letter-spacing:.06em;text-transform:uppercase;padding:5px 0;margin-bottom:8px;"
        f"border-bottom:1px solid #babecc'>ID: {aid} · Detected: {ts}</div>",
        unsafe_allow_html=True
    )

    # Narrative
    narrative_key    = f"narrative_{aid}"
    verification_key = f"verify_{aid}"
    conv_key         = f"conv_state_{aid}"

    if narrative_key in st.session_state:
        # Wrap narrative in neumorphic panel
        _narrative_panel(st.session_state[narrative_key])
        _grounding_badge(st.session_state.get(verification_key,{"risk":"none","warnings":[]}))
    elif llm_client is None:
        narrative = _static_narrative(anomaly)
        st.session_state[narrative_key]    = narrative
        st.session_state[verification_key] = {"risk":"none","warnings":[]}
        _narrative_panel(narrative)
        st.caption("Static narrative — add GROQ_API_KEY in sidebar for AI diagnostics")
    else:
        try:
            if conv_key not in st.session_state:
                with st.spinner("Initialising ARIA diagnostic context..."):
                    state = llm_client.start_conversation(anomaly)
                    st.session_state[conv_key] = state
            else:
                state = st.session_state[conv_key]
            placeholder = st.empty()
            with st.spinner("ARIA generating diagnostic..."):
                narrative, meta = _render_streaming(
                    placeholder,
                    llm_client.get_initial_diagnostic_stream(state)
                )
            st.session_state[narrative_key]    = narrative
            st.session_state[verification_key] = meta
            _narrative_panel(narrative)
            _grounding_badge(meta)
        except Exception as exc:
            logger.error("LLM diagnostic failed: %s", exc)
            err = str(exc)
            if "GROQ_API_KEY" in err or "api_key" in err.lower():
                st.warning("Groq API key not configured — add it in the sidebar.")
            else:
                st.error(f"ARIA error: {err[:120]}")
            narrative = _static_narrative(anomaly)
            st.session_state[narrative_key]    = narrative
            st.session_state[verification_key] = {"risk":"none","warnings":[]}
            _narrative_panel(narrative)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # Escalation — dark panel
    esc = anomaly.get("recommended_escalation","")
    if esc:
        st.markdown(
            f"<div style='background:#2d3436;border-radius:8px;padding:12px 14px;"
            f"margin-bottom:11px;box-shadow:8px 8px 16px #babecc,-8px -8px 16px #ffffff'>"
            f"<div style='font-family:\"JetBrains Mono\",monospace;font-size:9px;font-weight:700;"
            f"letter-spacing:.1em;text-transform:uppercase;color:#ff4757;margin-bottom:5px;"
            f"display:flex;align-items:center;gap:5px'>"
            f"<i class='ti ti-alert-triangle' style='font-size:11px' aria-hidden='true'></i>"
            f"What to do next</div>"
            f"<div style='font-size:12px;color:#a8b2d1;line-height:1.55'>{esc}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    # Evidence sections
    _evidence_section(anomaly)


def _narrative_panel(text: str) -> None:
    st.markdown(
        f"<div style='background:#e0e5ec;border-radius:0 10px 10px 0;border-left:3px solid #ff4757;"
        f"padding:13px 15px;box-shadow:8px 8px 16px #babecc,-8px -8px 16px #ffffff;"
        f"margin-bottom:10px;position:relative'>",
        unsafe_allow_html=True
    )
    # Screw corners
    st.markdown(
        "<div style='position:absolute;top:8px;right:8px;width:7px;height:7px;border-radius:50%;"
        "background:radial-gradient(circle at 3px 3px,rgba(255,255,255,.5) 1.5px,transparent 2px),#babecc;"
        "box-shadow:1px 1px 2px rgba(0,0,0,.18),-1px -1px 1px rgba(255,255,255,.6)'></div>",
        unsafe_allow_html=True
    )
    st.markdown(text)
    st.markdown("</div>", unsafe_allow_html=True)


def _grounding_badge(meta: dict) -> None:
    risk   = meta.get("risk","none")
    label, color, bg = RISK_BADGES.get(risk, RISK_BADGES["none"])
    st.markdown(
        f"<div style='display:inline-flex;align-items:center;gap:5px;font-family:\"JetBrains Mono\",monospace;"
        f"font-size:9px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;"
        f"color:{color};background:{bg};padding:3px 10px;border-radius:3px;margin-bottom:8px'>"
        f"ARIA verification: {label}</div>",
        unsafe_allow_html=True
    )
    if risk in ("medium","high"):
        for w in meta.get("warnings",[])[:2]:
            st.caption(f"  ↳ {w}")


def _evidence_section(anomaly: dict) -> None:
    evidence     = anomaly.get("evidence",[])
    ruled_out    = anomaly.get("ruled_out",[])
    not_affected = anomaly.get("not_affected",[])
    rc_ev        = anomaly.get("reason_code_evidence",{})

    with st.expander(f"Supporting evidence ({len(evidence)} items)", expanded=False):
        for ev in evidence:
            st.markdown(f"- {ev}")
        if rc_ev:
            st.markdown("**Decline reason code shifts:**")
            for code, d in sorted(rc_ev.items(),
                                  key=lambda x: abs(x[1].get("delta_pp",0)),
                                  reverse=True)[:5]:
                curr  = d.get("current_share",0)
                base_ = d.get("baseline_share",0)
                delta = d.get("delta_pp",0)
                lbl   = d.get("label",f"Code {code}")
                arrow = "▲" if delta>0 else "▼"
                st.markdown(f"- RC {code} ({lbl}): `{curr:.1%}` vs `{base_:.1%}` {arrow} **{abs(delta):.1f}pp**")

    if ruled_out:
        with st.expander(f"Why other causes were ruled out ({len(ruled_out)})", expanded=False):
            for item in ruled_out:
                st.markdown(f"- {item}")

    if not_affected:
        with st.expander(f"What was not affected ({len(not_affected)})", expanded=False):
            for item in not_affected:
                st.markdown(f"- {item}")
