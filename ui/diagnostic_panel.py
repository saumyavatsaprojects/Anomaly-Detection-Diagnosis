"""
Diagnostic Panel — Industrial Skeuomorphic
============================================
Neumorphic tiles, dark escalation panel, screw corners.

FIXES vs previous version
──────────────────────────
FIX-5  _narrative_panel() opened a <div> via st.markdown(unsafe_allow_html=True),
       then rendered markdown content with a plain st.markdown() (Streamlit native),
       then tried to close the div with st.markdown("</div>").

       Streamlit strips orphaned/unmatched closing tags from HTML fragments.
       The opening <div> was emitted as a raw HTML node into the DOM with no
       matching close tag, which the browser auto-closes at end-of-body.
       Result: a full-height invisible block occupying the entire tab below
       the ARIA header.

       Resolution: render the narrative text in Python, inject it as a single
       self-contained HTML block.  We also need to handle markdown-in-HTML:
       Streamlit's markdown renderer does NOT process markdown inside HTML
       strings, so we run a minimal inline conversion (bold **…**, headers,
       bullet lists) before injecting.

FIX-6  The same orphaned-div pattern appears in the escalation section.
       Converted to a single self-contained st.markdown() call.
"""
from __future__ import annotations
import json, logging, re
from typing import Optional
import streamlit as st

logger = logging.getLogger(__name__)
SENTINEL_RE = re.compile(r'<!--VERIFY:(.*?)-->', re.DOTALL)

SEV_CFG = {
    "critical": {"color": "#ff4757", "bg": "rgba(255,71,87,.08)"},
    "high":     {"color": "#f59e0b", "bg": "rgba(245,158,11,.08)"},
    "medium":   {"color": "#378add", "bg": "rgba(55,138,221,.08)"},
    "low":      {"color": "#22c55e", "bg": "rgba(34,197,94,.08)"},
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

CONF_LABELS = {"high": "High confidence", "medium": "Medium confidence", "low": "Low confidence"}

RISK_BADGES = {
    "none":   ("✓ All values verified",     "#22c55e", "rgba(34,197,94,.12)"),
    "low":    ("~ Minor uncited numbers",   "#f59e0b", "rgba(245,158,11,.12)"),
    "medium": ("⚠ Review cited values",     "#f59e0b", "rgba(245,158,11,.12)"),
    "high":   ("✗ Value mismatch detected", "#ff4757", "rgba(255,71,87,.12)"),
}


# ── Minimal markdown → HTML (bold, headers, bullets, newlines) ─────────────
def _md_to_html(text: str) -> str:
    """
    Convert a small subset of markdown to HTML so it renders correctly
    inside an HTML string passed to st.markdown(unsafe_allow_html=True).
    Streamlit's markdown processor does NOT run inside raw HTML blocks.
    """
    # Escape any existing < > that aren't part of our own tags
    # (we skip full escaping to keep it simple — these are trusted LLM strings)
    lines = text.split("\n")
    html_lines = []
    in_ul = False

    for raw in lines:
        line = raw.rstrip()

        # ATX headings
        if line.startswith("### "):
            if in_ul:
                html_lines.append("</ul>"); in_ul = False
            content = _inline_md(line[4:])
            html_lines.append(
                f"<div style='font-size:12px;font-weight:700;color:#2d3436;"
                f"margin:12px 0 4px;letter-spacing:.02em'>{content}</div>"
            )
            continue
        if line.startswith("## "):
            if in_ul:
                html_lines.append("</ul>"); in_ul = False
            content = _inline_md(line[3:])
            html_lines.append(
                f"<div style='font-size:13px;font-weight:700;color:#2d3436;"
                f"margin:12px 0 4px'>{content}</div>"
            )
            continue

        # Bullet list items
        if line.startswith("- ") or line.startswith("* "):
            if not in_ul:
                html_lines.append(
                    "<ul style='margin:4px 0 4px 16px;padding:0;"
                    "list-style:disc;color:#4a5568'>"
                )
                in_ul = True
            content = _inline_md(line[2:])
            html_lines.append(f"<li style='font-size:12px;line-height:1.6;margin-bottom:2px'>{content}</li>")
            continue

        # Blank line — close list if open
        if not line.strip():
            if in_ul:
                html_lines.append("</ul>"); in_ul = False
            html_lines.append("<div style='height:6px'></div>")
            continue

        # Regular paragraph
        if in_ul:
            html_lines.append("</ul>"); in_ul = False
        content = _inline_md(line)
        html_lines.append(
            f"<div style='font-size:13px;color:#2d3436;line-height:1.65;"
            f"margin-bottom:2px'>{content}</div>"
        )

    if in_ul:
        html_lines.append("</ul>")

    return "\n".join(html_lines)


def _inline_md(text: str) -> str:
    """Convert **bold** and *italic* inline markdown to HTML."""
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Italic
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    return text


def _dh_tile(label: str, value: str, sub: str, accent: bool = False) -> str:
    color     = "#ff4757" if accent else "#636e72"
    val_color = "#ff4757" if accent else "#2d3436"
    return (
        f"<div style='background:#e0e5ec;border-radius:8px;padding:10px 12px;"
        f"box-shadow:8px 8px 16px #babecc,-8px -8px 16px #ffffff;position:relative;overflow:hidden'>"
        f"<div style='position:absolute;top:0;left:0;right:0;height:2px;"
        f"background:{'#ff4757' if accent else '#babecc'}'></div>"
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
            meta = {"risk": "unknown", "warnings": []}
        return SENTINEL_RE.sub("", text).strip(), meta
    return text.strip(), {"risk": "none", "warnings": []}


def _render_streaming(placeholder, generator) -> tuple[str, dict]:
    chunks: list[str] = []
    for chunk in generator:
        chunks.append(chunk)
        display = SENTINEL_RE.sub("", "".join(chunks))
        placeholder.markdown(display + "▌")
    full  = "".join(chunks)
    clean, meta = _strip_sentinel(full)
    placeholder.markdown(clean)
    return clean, meta


def _static_narrative(anomaly: dict) -> str:
    fc      = anomaly.get("failure_class", "undetermined")
    fc_lbl  = FC_LABELS.get(fc, fc)
    sev     = anomaly.get("severity", "medium").upper()
    ts      = anomaly.get("first_seen_ts", "")[:10]
    obs     = anomaly.get("observed_value", 0)
    base    = anomaly.get("baseline_value", 0)
    sigma   = anomaly.get("deviation_sigma", 0)
    dur     = anomaly.get("duration_hours", 0)
    conf    = anomaly.get("failure_class_confidence", "low")
    sl      = anomaly.get("affected_slice", {})
    slice_str = " / ".join(
        str(v) for v in sl.values()
        if str(v).lower() not in ("all", "", "none")
    )[:60]
    evidence  = anomaly.get("evidence", [])
    ev_lines  = "\n".join(f"- {ev}" for ev in evidence[:4])

    p1 = (
        f"**WHAT HAPPENED**\n\n"
        f"On **{ts}**, a **{sev}** anomaly was detected on **{slice_str}**. "
        f"The approval rate registered at **{obs:.1%}** against a 7-day baseline of **{base:.1%}** — "
        f"a deviation of **{sigma:.1f}σ** persisting for **{dur} hours**."
    )
    p2 = (
        f"\n\n**WHY IT HAPPENED**\n\n"
        f"The detection engine attributes this to **{fc_lbl}** ({conf} confidence). "
        f"Supporting evidence:\n\n{ev_lines}"
    )
    esc = anomaly.get("recommended_escalation", "")
    p3  = (
        f"\n\n**WHAT TO DO NOW**\n\n{esc}" if esc
        else "\n\n**WHAT TO DO NOW**\n\nSee escalation path below."
    )
    return p1 + p2 + p3


# ── FIX-5: self-contained narrative block (no split open/close divs) ────────
def _narrative_panel(text: str) -> None:
    """
    Render the diagnostic narrative inside a neumorphic card.

    Previous approach (broken):
        st.markdown("<div ...>", unsafe_allow_html=True)   # opens div
        st.markdown(text)                                   # content — BUT this
                                                            # renders OUTSIDE the div
                                                            # in Streamlit's virtual DOM
        st.markdown("</div>", unsafe_allow_html=True)      # Streamlit strips lone </div>

    Fixed approach: convert markdown to HTML inline, emit ONE self-contained block.
    """
    inner_html = _md_to_html(text)
    screw = (
        "<div style='position:absolute;top:8px;right:8px;width:7px;height:7px;"
        "border-radius:50%;background:radial-gradient(circle at 3px 3px,"
        "rgba(255,255,255,.5) 1.5px,transparent 2px),#babecc;"
        "box-shadow:1px 1px 2px rgba(0,0,0,.18),-1px -1px 1px rgba(255,255,255,.6)'></div>"
    )
    st.markdown(
        f"<div style='background:#e0e5ec;border-radius:0 10px 10px 0;"
        f"border-left:3px solid #ff4757;padding:13px 15px;"
        f"box-shadow:8px 8px 16px #babecc,-8px -8px 16px #ffffff;"
        f"margin-bottom:10px;position:relative'>"
        f"{screw}"
        f"{inner_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _grounding_badge(meta: dict) -> None:
    risk          = meta.get("risk", "none")
    label, color, bg = RISK_BADGES.get(risk, RISK_BADGES["none"])
    st.markdown(
        f"<div style='display:inline-flex;align-items:center;gap:5px;"
        f"font-family:\"JetBrains Mono\",monospace;font-size:9px;font-weight:700;"
        f"letter-spacing:.06em;text-transform:uppercase;color:{color};"
        f"background:{bg};padding:3px 10px;border-radius:3px;margin-bottom:8px'>"
        f"ARIA verification: {label}</div>",
        unsafe_allow_html=True,
    )
    if risk in ("medium", "high"):
        for w in meta.get("warnings", [])[:2]:
            st.caption(f"  ↳ {w}")


def render_diagnostic_panel(anomaly: dict, llm_client=None) -> None:
    aid    = anomaly.get("anomaly_id", "")
    sev    = anomaly.get("severity", "medium")
    fc     = anomaly.get("failure_class", "undetermined")
    fc_lbl = FC_LABELS.get(fc, fc)
    conf   = CONF_LABELS.get(anomaly.get("failure_class_confidence", "low"), "")
    sigma  = anomaly.get("deviation_sigma", 0)
    dur    = anomaly.get("duration_hours", 0)
    txns   = anomaly.get("volume_evidence", {}).get("txn_count_observed", 0)
    ts     = anomaly.get("first_seen_ts", "")[:10]

    batch_rank = "—"
    for ev in anomaly.get("evidence", []):
        if "Batch rank #" in ev:
            batch_rank = ev.split("Batch rank ")[1]
            break

    # 4-tile header
    tiles_html = (
        _dh_tile("Incident type",   fc_lbl, conf, accent=True)
        + _dh_tile("Severity score", f"{sigma:+.1f}σ", sev.upper(), accent=True)
        + _dh_tile("Duration",       f"{dur}h" if dur < 48 else f"{dur // 24}d", f"From {ts}")
        + _dh_tile("Txns in window", f"{int(txns):,}" if txns else "—", batch_rank)
    )
    st.markdown(
        f"<div style='display:grid;grid-template-columns:repeat(4,1fr);"
        f"gap:10px;margin-bottom:13px'>{tiles_html}</div>",
        unsafe_allow_html=True,
    )

    # Anomaly ID line
    st.markdown(
        f"<div style='font-family:\"JetBrains Mono\",monospace;font-size:9px;color:#636e72;"
        f"letter-spacing:.06em;text-transform:uppercase;padding:5px 0;margin-bottom:8px;"
        f"border-bottom:1px solid #babecc'>ID: {aid} · Detected: {ts}</div>",
        unsafe_allow_html=True,
    )

    # Narrative + grounding badge
    narrative_key    = f"narrative_{aid}"
    verification_key = f"verify_{aid}"
    conv_key         = f"conv_state_{aid}"

    if narrative_key in st.session_state:
        _narrative_panel(st.session_state[narrative_key])
        _grounding_badge(st.session_state.get(verification_key, {"risk": "none", "warnings": []}))

    elif llm_client is None:
        narrative = _static_narrative(anomaly)
        st.session_state[narrative_key]    = narrative
        st.session_state[verification_key] = {"risk": "none", "warnings": []}
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
                    llm_client.get_initial_diagnostic_stream(state),
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
            st.session_state[verification_key] = {"risk": "none", "warnings": []}
            _narrative_panel(narrative)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── FIX-6: escalation panel — single self-contained block ───────────────
    esc = anomaly.get("recommended_escalation", "")
    if esc:
        st.markdown(
            f"<div style='background:#2d3436;border-radius:8px;padding:12px 14px;"
            f"margin-bottom:11px;box-shadow:8px 8px 16px #babecc,-8px -8px 16px #ffffff'>"
            f"<div style='font-family:\"JetBrains Mono\",monospace;font-size:9px;font-weight:700;"
            f"letter-spacing:.1em;text-transform:uppercase;color:#ff4757;margin-bottom:5px'>"
            f"What to do next</div>"
            f"<div style='font-size:12px;color:#a8b2d1;line-height:1.55'>{esc}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    _evidence_section(anomaly)


def _evidence_section(anomaly: dict) -> None:
    evidence     = anomaly.get("evidence", [])
    ruled_out    = anomaly.get("ruled_out", [])
    not_affected = anomaly.get("not_affected", [])
    rc_ev        = anomaly.get("reason_code_evidence", {})

    with st.expander(f"Supporting evidence ({len(evidence)} items)", expanded=False):
        for ev in evidence:
            st.markdown(f"- {ev}")
        if rc_ev:
            st.markdown("**Decline reason code shifts:**")
            for code, d in sorted(
                rc_ev.items(), key=lambda x: abs(x[1].get("delta_pp", 0)), reverse=True
            )[:5]:
                curr  = d.get("current_share", 0)
                base_ = d.get("baseline_share", 0)
                delta = d.get("delta_pp", 0)
                lbl   = d.get("label", f"Code {code}")
                arrow = "▲" if delta > 0 else "▼"
                st.markdown(
                    f"- RC {code} ({lbl}): `{curr:.1%}` vs `{base_:.1%}` {arrow} **{abs(delta):.1f}pp**"
                )

    if ruled_out:
        with st.expander(f"Why other causes were ruled out ({len(ruled_out)})", expanded=False):
            for item in ruled_out:
                st.markdown(f"- {item}")

    if not_affected:
        with st.expander(f"What was not affected ({len(not_affected)})", expanded=False):
            for item in not_affected:
                st.markdown(f"- {item}")
