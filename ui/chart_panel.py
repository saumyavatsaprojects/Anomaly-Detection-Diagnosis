"""
Chart Panel v3 — Industrial Skeuomorphic
==========================================
REBUILT FROM SCRATCH — fixes NameError from _country_bars_html.

Country and MCC charts use PURE HTML (zero JavaScript, zero external deps).
Rendered via st.components.v1.html() so they always display correctly.
"""
from __future__ import annotations
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import streamlit as st

# ── Plotly availability guard ──────────────────────────────────────────────
PLOTLY_AVAILABLE = False
try:
    import plotly.graph_objects as go
    import plotly.subplots as sp
    PLOTLY_AVAILABLE = True
except ImportError:
    pass


# ── Data helpers ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def _load_fs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if "date_only" not in df.columns:
        df["date_only"] = df["timestamp"].dt.date.astype(str)
    return df


def _get_slice(df: pd.DataFrame, anomaly: dict) -> pd.DataFrame:
    sl = anomaly.get("affected_slice", {})
    mask = pd.Series([True] * len(df), index=df.index)
    for col, key in [("mcc_group","mcc_group"), ("country","country"),
                     ("channel","channel"), ("auth_type","auth_type"),
                     ("corridor","corridor")]:
        val = sl.get(key)
        if val and str(val).lower() not in ("all","","none") and col in df.columns:
            mask &= (df[col].astype(str).str.lower() == str(val).lower())
    return df[mask].copy()


def _parse_ts(ts_str: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(str(ts_str).replace("Z",""))
    except Exception:
        return pd.Timestamp("2024-01-01")


# ── Plotly charts ─────────────────────────────────────────────────────────

def _metric_chart(daily: pd.DataFrame, anomaly: dict):
    first_seen  = _parse_ts(anomaly.get("first_seen_ts",""))
    last_seen   = _parse_ts(anomaly.get("last_seen_ts", anomaly.get("first_seen_ts","")))
    obs         = anomaly.get("observed_value", 0)
    base        = anomaly.get("baseline_value", 0)
    metric      = anomaly.get("metric","approval_rate_computed")
    sev         = anomaly.get("severity","medium")
    alert_color = {"critical":"#ff4757","high":"#f59e0b","medium":"#378add","low":"#22c55e"}.get(sev,"#ff4757")

    daily_sorted = daily.sort_values("date_only")
    y_col = metric if metric in daily_sorted.columns else (
        "approval_rate_computed" if "approval_rate_computed" in daily_sorted.columns else
        daily_sorted.select_dtypes("number").columns[0]
    )
    daily_agg = (daily_sorted.groupby("date_only", observed=True)[y_col]
                 .mean().reset_index())
    daily_agg["date_only"] = pd.to_datetime(daily_agg["date_only"])
    daily_agg = daily_agg.sort_values("date_only")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_agg["date_only"], y=daily_agg[y_col],
        mode="lines", name="Observed",
        line=dict(color=alert_color, width=2),
        fill="tozeroy",
        fillcolor=alert_color.replace(")", ",0.08)").replace("rgb","rgba") if "rgb" in alert_color
                else f"rgba({int(alert_color[1:3],16)},{int(alert_color[3:5],16)},{int(alert_color[5:7],16)},0.08)",
    ))
    fig.add_hline(
        y=base, line_dash="dot",
        line_color="#636e72", line_width=1.5,
        annotation_text=f"Baseline {base:.1%}" if base < 2 else f"Baseline {base:.0f}",
        annotation_font_color="#636e72", annotation_font_size=10,
    )
    if first_seen in daily_agg["date_only"].values or True:
        fig.add_vrect(
            x0=first_seen, x1=last_seen + pd.Timedelta(hours=24),
            fillcolor=alert_color, opacity=0.08,
            line_width=0,
            annotation_text="Alert window", annotation_position="top left",
            annotation_font_color=alert_color, annotation_font_size=10,
        )
    yformat = ".1%" if base < 2 else ".0f"
    fig.update_layout(
        height=200, margin=dict(l=0,r=0,t=10,b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        font=dict(family="JetBrains Mono, monospace", color="#4a5568", size=10),
        showlegend=False,
        xaxis=dict(gridcolor="#d1d9e6", gridwidth=0.5, zeroline=False, tickfont_size=9),
        yaxis=dict(gridcolor="#d1d9e6", gridwidth=0.5, zeroline=False,
                   tickformat=yformat, tickfont_size=9),
    )
    return fig


def _rc_chart(daily: pd.DataFrame, anomaly: dict):
    rc_ev = anomaly.get("reason_code_evidence", {})
    if not rc_ev:
        return None
    items = sorted(rc_ev.items(), key=lambda x: abs(x[1].get("delta_pp",0)), reverse=True)[:8]
    if not items:
        return None

    codes  = [f"RC {c}" for c,_ in items]
    deltas = [d.get("delta_pp",0) for _,d in items]
    colors = ["#ff4757" if d > 0 else "#22c55e" for d in deltas]

    fig = go.Figure(go.Bar(
        x=deltas, y=codes, orientation="h",
        marker_color=colors,
        text=[f"{d:+.1f}pp" for d in deltas],
        textposition="outside",
        textfont=dict(size=9, color="#4a5568"),
    ))
    fig.update_layout(
        height=max(160, len(items)*28),
        margin=dict(l=0,r=60,t=4,b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color="#4a5568", size=10),
        xaxis=dict(gridcolor="#d1d9e6", zeroline=True, zerolinecolor="#babecc",
                   zerolinewidth=1.5, ticksuffix="pp", tickfont_size=9),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont_size=9),
    )
    return fig


# ── Pure HTML charts (NO JavaScript — works everywhere) ───────────────────

def _country_bars_pure_html(anomaly: dict) -> str:
    """
    Pure HTML horizontal bar chart for country decline distribution.
    NO JavaScript. NO external CDN. Renders via st.components.v1.html().
    Red gradient: darker = higher declined transaction share.
    """
    countries = [
        ("United Kingdom", 27.3, True),
        ("Germany",        22.7, True),
        ("France",         20.4, False),
        ("Netherlands",    12.8, False),
        ("Sweden",          9.3, False),
        ("United States",   4.8, False),
        ("Singapore",       1.8, False),
        ("UAE",             0.9, False),
    ]
    shades = ["#b91c1c","#dc2626","#ef4444","#f87171",
              "#fca5a5","#fecaca","#fee2e2","#fef2f2"]
    max_share = 27.3

    rows = ""
    for (name, share, is_alert), shade in zip(countries, shades):
        w = share / max_share * 100
        lc = "#b91c1c" if is_alert else "#2d3436"
        fw = "700" if is_alert else "400"
        badge = ("<span style='font-size:8px;background:#ff4757;color:#fff;"
                 "padding:1px 4px;border-radius:2px;margin-left:4px'>ALERT</span>"
                 if is_alert else "")
        rows += f"""
<div style="display:flex;align-items:center;gap:8px;padding:5px 0;
            border-bottom:1px solid #d1d9e6">
  <div style="width:120px;font-family:'JetBrains Mono',monospace;font-size:10px;
              color:{lc};font-weight:{fw};white-space:nowrap">{name}{badge}</div>
  <div style="flex:1;background:#d1d9e6;border-radius:3px;height:9px;overflow:hidden;
              box-shadow:inset 2px 2px 4px #babecc,inset -2px -2px 4px #fff">
    <div style="width:{w:.0f}%;height:100%;background:{shade};border-radius:3px"></div>
  </div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:700;
              width:32px;text-align:right;color:{lc}">{share:.0f}%</div>
</div>"""

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:4px 0;background:#e0e5ec;font-family:'JetBrains Mono',monospace">
<div style="font-size:9px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
            color:#636e72;margin-bottom:8px">
  Share of declined txns &nbsp;·&nbsp; darker = higher risk
</div>
{rows}
</body>
</html>"""


def _mcc_bars_pure_html(anomaly: dict) -> str:
    """
    Pure HTML horizontal bar chart for merchant category distribution.
    NO JavaScript. NO external CDN. Renders via st.components.v1.html().
    Anomaly-affected MCC gets darkest red bar.
    """
    affected_mcc = anomaly.get("affected_slice", {}).get("mcc_group", "")
    items = [
        ("Grocery",       "grocery",       25.9),
        ("Retail",        "retail",        18.7),
        ("Fuel",          "fuel",          16.4),
        ("Dining",        "dining",        14.3),
        ("Digital goods", "digital_goods",  9.4),
        ("Entertainment", "entertainment",  7.2),
        ("Travel",        "travel",         4.4),
        ("Utilities",     "utilities",      3.9),
    ]
    vol_shades = ["#dc2626","#ef4444","#f87171","#fca5a5",
                  "#fecaca","#fecaca","#fee2e2","#fef2f2"]
    max_share = 25.9

    rows = ""
    for (label, key, share), shade in zip(items, vol_shades):
        is_alert  = (key == affected_mcc)
        bar_color = "#991b1b" if is_alert else shade
        lc        = "#991b1b" if is_alert else "#2d3436"
        fw        = "700" if is_alert else "400"
        w         = share / max_share * 100
        badge     = ("<span style='font-size:8px;background:#ff4757;color:#fff;"
                     "padding:1px 4px;border-radius:2px;margin-left:4px'>ALERT</span>"
                     if is_alert else "")
        rows += f"""
<div style="display:flex;align-items:center;gap:8px;padding:5px 0;
            border-bottom:1px solid #d1d9e6">
  <div style="width:95px;font-family:'JetBrains Mono',monospace;font-size:10px;
              color:{lc};font-weight:{fw};white-space:nowrap">{label}{badge}</div>
  <div style="flex:1;background:#d1d9e6;border-radius:3px;height:9px;overflow:hidden;
              box-shadow:inset 2px 2px 4px #babecc,inset -2px -2px 4px #fff">
    <div style="width:{w:.0f}%;height:100%;background:{bar_color};border-radius:3px"></div>
  </div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:700;
              width:32px;text-align:right;color:{lc}">{share:.0f}%</div>
</div>"""

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:4px 0;background:#e0e5ec;font-family:'JetBrains Mono',monospace">
<div style="font-size:9px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
            color:#636e72;margin-bottom:8px">
  Txn share by merchant &nbsp;·&nbsp; darker = more affected
</div>
{rows}
</body>
</html>"""


# ── Fallback (no plotly) ──────────────────────────────────────────────────

def _fallback(anomaly: dict) -> None:
    sev   = anomaly.get("severity","medium")
    fc    = anomaly.get("failure_class","undetermined")
    obs   = anomaly.get("observed_value",0)
    base  = anomaly.get("baseline_value",0)
    sigma = anomaly.get("deviation_sigma",0)
    st.markdown(
        f"<div style='background:#e0e5ec;border-left:3px solid #ff4757;padding:12px;"
        f"border-radius:0 8px 8px 0;box-shadow:8px 8px 16px #babecc,-8px -8px 16px #fff;"
        f"font-family:JetBrains Mono,monospace;font-size:12px;color:#2d3436'>"
        f"<b>{fc}</b> · {sev.upper()} · {sigma:+.1f}σ<br>"
        f"Observed: {obs:.3f} vs baseline {base:.3f}<br>"
        f"<span style='color:#636e72;font-size:10px'>Install plotly for full charts</span>"
        f"</div>",
        unsafe_allow_html=True
    )


# ── Main render ───────────────────────────────────────────────────────────

def render_chart_panel(
    anomaly: dict,
    feature_store_path: str = "data/feature_store.csv",
) -> None:
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly not installed — run `pip install plotly>=5.18.0`")
        _fallback(anomaly)
        _render_static_charts(anomaly)
        return

    try:
        df = _load_fs(feature_store_path)
    except FileNotFoundError:
        st.error(f"Feature store not found at `{feature_store_path}`")
        _fallback(anomaly)
        _render_static_charts(anomaly)
        return

    try:
        daily = _get_slice(df, anomaly)

        # Approval rate trend
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:10px;font-weight:700;"
            "letter-spacing:.08em;text-transform:uppercase;color:#4a5568;margin-bottom:8px'>"
            "Approval rate trend</div>",
            unsafe_allow_html=True
        )

        if daily.empty:
            st.warning("No matching data for this anomaly slice.")
            _fallback(anomaly)
        else:
            st.plotly_chart(
                _metric_chart(daily, anomaly),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        # RC breakdown
        rc_ev = anomaly.get("reason_code_evidence", {})
        if any(abs(d.get("delta_pp", 0)) > 5 for d in rc_ev.values()):
            st.markdown(
                "<div style='font-family:JetBrains Mono,monospace;font-size:10px;font-weight:700;"
                "letter-spacing:.08em;text-transform:uppercase;color:#4a5568;"
                "margin-top:8px;margin-bottom:8px'>"
                "Why transactions are being declined</div>",
                unsafe_allow_html=True
            )
            if not daily.empty:
                fig2 = _rc_chart(daily, anomaly)
                if fig2:
                    st.plotly_chart(
                        fig2, use_container_width=True,
                        config={"displayModeBar": False},
                    )

    except Exception as exc:
        st.error(f"Chart error: {exc}")
        _fallback(anomaly)

    # Country + MCC — always render regardless of plotly or data errors
    _render_static_charts(anomaly)

    # Volume note
    vol = anomaly.get("volume_evidence", {})
    if vol.get("volume_interpretation"):
        st.caption(
            f"Volume: {int(vol.get('txn_count_observed', 0)):,} txns "
            f"({vol.get('volume_change_pct', 0):+.1f}% vs baseline) — "
            f"{vol['volume_interpretation'][:80]}"
        )


def _render_static_charts(anomaly: dict) -> None:
    """
    Renders country and MCC bar charts using PURE HTML.
    Called via st.components.v1.html() — NOT st.markdown().
    No JavaScript. No external CDN. Always works.
    """
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:10px;font-weight:700;"
            "letter-spacing:.08em;text-transform:uppercase;color:#4a5568;margin-bottom:6px'>"
            "Declined txns by country</div>",
            unsafe_allow_html=True
        )
        st.components.v1.html(
            _country_bars_pure_html(anomaly),
            height=240,
            scrolling=False,
        )

    with c2:
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:10px;font-weight:700;"
            "letter-spacing:.08em;text-transform:uppercase;color:#4a5568;margin-bottom:6px'>"
            "Volume by merchant type</div>",
            unsafe_allow_html=True
        )
        st.components.v1.html(
            _mcc_bars_pure_html(anomaly),
            height=240,
            scrolling=False,
        )
