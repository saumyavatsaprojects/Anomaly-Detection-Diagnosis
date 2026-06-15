"""
Anomaly Feed — Industrial Skeuomorphic
Left column: scrollable card list with dropdown selector.
Filter text is small and muted (not red).
"""
from __future__ import annotations
import streamlit as st
from datetime import datetime
from typing import Optional

SEVERITY_CONFIG = {
    "critical": {"color":"#ff4757","border":"#ff4757","label":"CRITICAL"},
    "high":     {"color":"#f59e0b","border":"#f59e0b","label":"HIGH"},
    "medium":   {"color":"#378add","border":"#378add","label":"MEDIUM"},
    "low":      {"color":"#22c55e","border":"#22c55e","label":"LOW"},
}

FAILURE_CLASS_LABELS = {
    "3ds_acs_failure":      "Authentication failure",
    "processor_outage":     "Processor outage",
    "fraud_attack":         "Fraud attack",
    "acquirer_routing":     "Routing issue",
    "network_rule_change":  "Network rule change",
    "issuer_rules_misfire": "Incorrect declines",
    "undetermined":         "Under investigation",
}

SEVERITY_ORDER = {"critical":0,"high":1,"medium":2,"low":3}


def _format_slice(sl: dict) -> str:
    parts = []
    for key in ["country","mcc_group","channel","auth_type","corridor","bin_bucket"]:
        val = sl.get(key)
        if val and str(val).lower() not in ("all","","none"):
            parts.append(str(val))
    return " · ".join(parts[:4]) if parts else "all dimensions"


def _format_ts(ts_str: str) -> str:
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z","+00:00"))
        return ts.strftime("%b %d")
    except Exception:
        return str(ts_str)[:10]


def _format_metric(metric: str, value: float) -> str:
    if "rate" in metric or "share" in metric:
        return f"{value:.1%}"
    if metric == "txn_count":
        return f"{int(value):,}"
    return f"{value:.4f}"


def _delta_str(obs: float, base: float, metric: str) -> str:
    if base == 0:
        return ""
    if "rate" in metric or "share" in metric:
        delta_pp = (obs - base) * 100
        return f"{delta_pp:+.1f}pp"
    pct = (obs - base) / abs(base) * 100
    return f"{pct:+.1f}%"


def _sort_anomalies(anomalies: list) -> list:
    return sorted(
        anomalies,
        key=lambda a: (
            SEVERITY_ORDER.get(a.get("severity","low"),9),
            -abs(a.get("deviation_sigma",0)),
        )
    )


def render_anomaly_feed(anomalies: list) -> Optional[dict]:
    """Renders the left-column incident feed."""

    total    = len(anomalies)
    critical = sum(1 for a in anomalies if a.get("severity")=="critical")
    high     = sum(1 for a in anomalies if a.get("severity")=="high")

    # ── Feed header ───────────────────────────────────────────────────────
    st.markdown(
        f"<div style='padding:0 4px 0 6px'>"
        f"<div style='font-family:JetBrains Mono,monospace;font-size:9px;font-weight:700;"
        f"letter-spacing:.1em;text-transform:uppercase;color:#636e72;margin-bottom:4px;"
        f"display:flex;align-items:center;justify-content:space-between'>"
        f"Active alerts"
        f"<span style='font-size:9px;font-weight:700;color:#ff4757;"
        f"background:rgba(255,71,87,.12);padding:1px 6px;border-radius:2px'>{total}</span>"
        f"</div>"
        f"<div style='font-family:JetBrains Mono,monospace;font-size:9px;color:#636e72;"
        f"margin-bottom:10px'>🔴 {critical} critical &nbsp;🟠 {high} high</div>",
        unsafe_allow_html=True
    )

    # ── Filters — muted, compact ──────────────────────────────────────────
    all_sevs  = sorted(set(a.get("severity","low") for a in anomalies),
                       key=lambda s: SEVERITY_ORDER.get(s,9))
    all_fcs   = sorted(set(a.get("failure_class","undetermined") for a in anomalies))
    fc_labels = [FAILURE_CLASS_LABELS.get(fc,fc) for fc in all_fcs]

    if "feed_sev_filter" not in st.session_state:
        st.session_state["feed_sev_filter"] = all_sevs
    if "feed_fc_filter" not in st.session_state:
        st.session_state["feed_fc_filter"] = fc_labels

    # Filter — compact toggle button, no expander (avoids blank box)
    if "show_filters" not in st.session_state:
        st.session_state["show_filters"] = False

    if st.button(
        "⚙ Filter incidents",
        key="filter_toggle_btn",
        help="Filter by severity and incident type",
    ):
        st.session_state["show_filters"] = not st.session_state["show_filters"]

    if st.session_state.get("show_filters"):
        st.multiselect(
            "Severity",
            options=all_sevs,
            default=all_sevs,
            format_func=lambda s: f"{'🔴🟠🟡🟢'[SEVERITY_ORDER.get(s,3)]} {s.upper()}",
            key="feed_sev_filter",
            label_visibility="collapsed",
        )
        st.multiselect(
            "Incident type",
            options=fc_labels,
            default=fc_labels,
            key="feed_fc_filter",
            label_visibility="collapsed",
        )

    sel_sevs      = st.session_state.get("feed_sev_filter", all_sevs)
    sel_fc_labels = st.session_state.get("feed_fc_filter", fc_labels)
    label_to_fc   = {v:k for k,v in FAILURE_CLASS_LABELS.items()}
    sel_fcs       = [label_to_fc.get(l,l) for l in sel_fc_labels]
    filtered      = [
        a for a in anomalies
        if a.get("severity","low") in sel_sevs
        and a.get("failure_class","undetermined") in sel_fcs
    ]
    sorted_a = _sort_anomalies(filtered if filtered else anomalies)

    if "selected_anomaly_id" not in st.session_state:
        st.session_state["selected_anomaly_id"] = (
            sorted_a[0]["anomaly_id"] if sorted_a else None
        )

    # ── Dropdown selector ─────────────────────────────────────────────────
    sev_icons = {"critical":"🔴","high":"🟠","medium":"🟡","low":"🟢"}

    def _opt(a: dict) -> str:
        icon  = sev_icons.get(a.get("severity","low"),"⚪")
        fc    = FAILURE_CLASS_LABELS.get(a.get("failure_class","undetermined"),"—")
        sl    = _format_slice(a.get("affected_slice",{}))[:28]
        ts    = _format_ts(a.get("first_seen_ts",""))
        obs   = a.get("observed_value",0)
        base  = a.get("baseline_value",0)
        dlt   = _delta_str(obs, base, a.get("metric","approval_rate_computed"))
        return f"{icon} {fc} · {sl} · {ts}" + (f"  {dlt}" if dlt else "")

    options = [_opt(a) for a in sorted_a]
    ids     = [a["anomaly_id"] for a in sorted_a]
    sel_id  = st.session_state.get("selected_anomaly_id")
    cur_idx = ids.index(sel_id) if sel_id in ids else 0

    st.markdown(
        "<div style='font-family:JetBrains Mono,monospace;font-size:9px;font-weight:500;"
        "letter-spacing:.06em;text-transform:uppercase;color:#9ca3af;margin:8px 0 3px'>",
        unsafe_allow_html=True
    )
    chosen = st.selectbox(
        "incident_selector",
        options=options,
        index=cur_idx,
        key="incident_dropdown",
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    chosen_idx = options.index(chosen) if chosen in options else 0
    chosen_id  = ids[chosen_idx]

    if chosen_id != st.session_state.get("selected_anomaly_id"):
        st.session_state["selected_anomaly_id"] = chosen_id
        st.session_state["conversation_state"]  = None
        st.session_state["conv_anomaly_id"]     = chosen_id
        st.session_state["chat_history"]        = []
        st.session_state["aria_expanded"]       = False
        st.rerun()

    selected = next((a for a in sorted_a if a["anomaly_id"]==chosen_id), sorted_a[0])

    # ── Expanded detail card for selected incident ────────────────────────
    sev       = selected.get("severity","low")
    cfg       = SEVERITY_CONFIG.get(sev, SEVERITY_CONFIG["low"])
    fc        = selected.get("failure_class","undetermined")
    fc_lbl    = FAILURE_CLASS_LABELS.get(fc, fc)
    sigma     = selected.get("deviation_sigma",0)
    dur       = selected.get("duration_hours",0)
    dur_str   = f"{dur}h" if dur < 48 else f"{dur//24}d"
    metric    = selected.get("metric","approval_rate_computed")
    obs       = selected.get("observed_value",0)
    base      = selected.get("baseline_value",0)
    obs_str   = _format_metric(metric, obs)
    base_str  = _format_metric(metric, base)
    dlt       = _delta_str(obs, base, metric)
    slice_str = _format_slice(selected.get("affected_slice",{}))
    ts        = _format_ts(selected.get("first_seen_ts",""))
    sev_color = cfg["color"]
    sev_label = cfg["label"]

    vent = ("<div style='width:3px;height:14px;border-radius:2px;background:#d1d9e6;"
            "box-shadow:inset 1px 1px 2px rgba(0,0,0,.12),"
            "inset -1px -1px 2px rgba(255,255,255,.8)'></div>")
    vents = f"<div style='display:flex;gap:3px'>{vent}{vent}{vent}</div>"

    dlt_badge = (
        f"<span style='font-family:JetBrains Mono,monospace;font-size:10px;"
        f"font-weight:700;padding:1px 5px;border-radius:2px;"
        f"background:{sev_color}22;color:{sev_color}'>{dlt}</span>"
        if dlt else ""
    )

    st.markdown(
        f"<div style='background:#e8dfe0;border-radius:0 8px 8px 0;margin-top:8px;"
        f"border-left:3px solid {sev_color};padding:11px 12px;position:relative;"
        f"box-shadow:4px 4px 10px #babecc,-4px -4px 10px #fff,"
        f"inset 0 0 0 1px rgba(255,71,87,.12)'>"
        # screw TL
        f"<div style='position:absolute;top:8px;left:8px;width:7px;height:7px;"
        f"border-radius:50%;background:radial-gradient(circle at 3px 3px,"
        f"rgba(255,255,255,.5) 1.5px,transparent 2px),#babecc;"
        f"box-shadow:1px 1px 2px rgba(0,0,0,.2),-1px -1px 1px rgba(255,255,255,.6)'></div>"
        # screw TR
        f"<div style='position:absolute;top:8px;right:8px;width:7px;height:7px;"
        f"border-radius:50%;background:radial-gradient(circle at 3px 3px,"
        f"rgba(255,255,255,.5) 1.5px,transparent 2px),#babecc;"
        f"box-shadow:1px 1px 2px rgba(0,0,0,.2),-1px -1px 1px rgba(255,255,255,.6)'></div>"
        # title + delta
        f"<div style='display:flex;justify-content:space-between;align-items:center;"
        f"margin-bottom:4px'>"
        f"<span style='font-size:13px;font-weight:700;color:#2d3436'>{fc_lbl}</span>"
        f"{dlt_badge}</div>"
        # big metric
        f"<div style='display:flex;align-items:baseline;gap:8px;margin-bottom:3px'>"
        f"<span style='font-family:JetBrains Mono,monospace;font-size:22px;"
        f"font-weight:700;color:{sev_color}'>{obs_str}</span>"
        f"<span style='font-family:JetBrains Mono,monospace;font-size:10px;"
        f"color:#636e72'>vs {base_str} baseline</span>"
        f"<span style='font-family:JetBrains Mono,monospace;font-size:10px;"
        f"font-weight:700;padding:1px 6px;border-radius:3px;background:#d1d9e6;"
        f"color:#4a5568;margin-left:auto;box-shadow:inset 1px 1px 2px #babecc,"
        f"inset -1px -1px 2px #fff'>{sigma:+.1f}σ</span></div>"
        # slice + meta + vents
        f"<div style='font-family:JetBrains Mono,monospace;font-size:9px;"
        f"color:#636e72;margin-bottom:4px;white-space:nowrap;overflow:hidden;"
        f"text-overflow:ellipsis'>{slice_str}</div>"
        f"<div style='display:flex;justify-content:space-between;align-items:center'>"
        f"<span style='font-family:JetBrains Mono,monospace;font-size:10px;"
        f"font-weight:700;color:{sev_color}'>{sev_label} · {dur_str} · {ts}</span>"
        f"{vents}</div></div>",
        unsafe_allow_html=True
    )

    st.button("▶ Investigating", key="investigating_btn",
              use_container_width=True, type="primary")

    st.markdown("</div>", unsafe_allow_html=True)
    return selected
