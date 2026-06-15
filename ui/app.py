"""
Transaction Anomaly Detection & Diagnostic Assistant
Streamlit entry point — Industrial Skeuomorphic theme.
"""
from __future__ import annotations
import json, logging, os
import streamlit as st

st.set_page_config(
    page_title  = "ANOMALY.DETECT",
    page_icon   = "🔴",
    layout      = "wide",
    initial_sidebar_state = "auto",
)

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


def _inject_api_key() -> bool:
    if os.environ.get("GROQ_API_KEY"):
        return True
    try:
        key = st.secrets.get("GROQ_API_KEY","")
        if key:
            os.environ["GROQ_API_KEY"] = key
            return True
    except Exception:
        pass
    return False


@st.cache_resource(show_spinner=False)
def _load_anomaly_objects() -> list:
    from run_pipeline import run_pipeline_if_needed
    return run_pipeline_if_needed(verbose=False)


@st.cache_resource(show_spinner=False)
def _get_llm_client():
    if not _inject_api_key():
        return None
    try:
        from llm.llm_client import LLMClient
        return LLMClient()
    except Exception as exc:
        logger.warning("LLM client init failed: %s", exc)
        return None


# ── Industrial CSS ─────────────────────────────────────────────────────────
def _inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* Hide Streamlit chrome entirely */
    [data-testid="stToolbar"]   { display:none !important; }
    header[data-testid="stHeader"] { display:none !important; }
    #MainMenu                   { visibility:hidden !important; }
    footer                      { visibility:hidden !important; }
    .stDeployButton             { display:none !important; }
    [data-testid="stDecoration"]{ display:none !important; }

    /* ── Base chassis ── */
    .stApp {
        background: #e0e5ec !important;
        font-family: 'Inter','Segoe UI',sans-serif !important;
    }
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }

    /* ── Neumorphic shadows ── */
    /* --sc: card, --sf: float, --sr: recessed, --sp: pressed */

    /* ── Metrics (native st.metric unused — we use HTML) ── */
    [data-testid="stMetric"] { display:none !important; }

    /* ── Tabs: industrial style ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #d1d9e6 !important;
        border-radius: 8px !important;
        padding: 3px !important;
        gap: 0 !important;
        box-shadow: inset 4px 4px 8px #babecc, inset -4px -4px 8px #ffffff !important;
        border-bottom: none !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px !important;
        padding: 5px 16px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        letter-spacing: .06em !important;
        text-transform: uppercase !important;
        color: #4a5568 !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: #e0e5ec !important;
        color: #2d3436 !important;
        box-shadow: 4px 4px 8px #babecc, -4px -4px 8px #ffffff !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display:none !important; }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 0.8rem !important; }

    /* ── Buttons ── */
    .stButton > button {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 10px !important;
        font-weight: 700 !important;
        letter-spacing: .06em !important;
        text-transform: uppercase !important;
        border-radius: 6px !important;
        background: #e0e5ec !important;
        border: none !important;
        color: #4a5568 !important;
        box-shadow: 4px 4px 8px #babecc, -4px -4px 8px #ffffff !important;
        transition: all 150ms !important;
        padding: 6px 14px !important;
    }
    .stButton > button:hover {
        color: #ff4757 !important;
        box-shadow: 6px 6px 12px #babecc, -6px -6px 12px #ffffff !important;
    }
    .stButton > button:active {
        box-shadow: inset 4px 4px 8px #babecc, inset -4px -4px 8px #ffffff !important;
        transform: translateY(1px) !important;
    }
    .stButton > button[kind="primary"] {
        background: #ff4757 !important;
        color: #ffffff !important;
        box-shadow: 4px 4px 8px rgba(166,50,60,.4), -4px -4px 8px rgba(255,100,110,.3) !important;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] > div {
        background: #d1d9e6 !important;
        border: none !important;
        border-radius: 8px !important;
        box-shadow: inset 4px 4px 8px #babecc, inset -4px -4px 8px #ffffff !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
    }
    [data-testid="stChatMessage"] {
        background: #e0e5ec !important;
        border: none !important;
        border-radius: 10px !important;
        box-shadow: 8px 8px 16px #babecc, -8px -8px 16px #ffffff !important;
        margin-bottom: 8px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 13px !important;
    }

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        background: #e0e5ec !important;
        border-radius: 6px !important;
        box-shadow: 4px 4px 8px #babecc, -4px -4px 8px #ffffff !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 10px !important;
        font-weight: 700 !important;
        letter-spacing: .06em !important;
        text-transform: uppercase !important;
        color: #4a5568 !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #d1d9e6; }
    ::-webkit-scrollbar-thumb { background: #babecc; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #a3b1c6; }

    /* ── Column layout: feed sticky + padding ── */
    [data-testid="stHorizontalBlock"] > div:first-child {
        position: sticky !important;
        top: 0 !important;
        align-self: flex-start !important;
        height: 100vh !important;
        overflow-y: auto !important;
        padding: 0 6px 0 10px !important;
        background: #d8dde5 !important;
        border-right: 1px solid #babecc !important;
    }
    [data-testid="stHorizontalBlock"] > div:last-child {
        height: 100vh !important;
        overflow-y: auto !important;
        padding: 0 12px !important;
    }

    /* ── Text inputs ── */
    .stTextInput > div > div {
        background: #d1d9e6 !important;
        border: none !important;
        border-radius: 8px !important;
        box-shadow: inset 4px 4px 8px #babecc, inset -4px -4px 8px #ffffff !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
        color: #2d3436 !important;
    }

    /* ── Captions ── */
    .stCaption {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 9px !important;
        color: #636e72 !important;
        letter-spacing: .04em !important;
    }

    /* ── Markdown body ── */
    .stMarkdown p, .stMarkdown li {
        font-size: 13px !important;
        color: #2d3436 !important;
        line-height: 1.65 !important;
    }
    .stMarkdown strong { font-weight: 700 !important; }

    /* ── Warning / info / error ── */
    [data-testid="stAlert"] {
        background: #e0e5ec !important;
        border-radius: 8px !important;
        box-shadow: 4px 4px 8px #babecc, -4px -4px 8px #ffffff !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ── Top bar (industrial) ───────────────────────────────────────────────────
def _render_topbar(anomalies: list, has_api: bool):
    from collections import Counter
    sev = Counter(a.get("severity") for a in anomalies)
    crit = sev.get("critical",0)
    high = sev.get("high",0)

    api_html = (
        '<span style="font-family:\'JetBrains Mono\',monospace;font-size:9px;font-weight:700;'
        'letter-spacing:.06em;text-transform:uppercase;padding:3px 9px;border-radius:3px;'
        'background:rgba(34,197,94,.15);color:#22c55e;border:1px solid rgba(34,197,94,.3)">'
        'ARIA ✓</span>'
        if has_api else
        '<span style="font-family:\'JetBrains Mono\',monospace;font-size:9px;font-weight:700;'
        'letter-spacing:.06em;text-transform:uppercase;padding:3px 9px;border-radius:3px;'
        'background:rgba(255,71,87,.15);color:#ff4757;border:1px solid rgba(255,71,87,.3)">'
        'NO API KEY</span>'
    )

    st.markdown(f"""
    <div style="background:#2d3436;height:46px;padding:0 18px;display:flex;align-items:center;
                justify-content:space-between;border-bottom:3px solid #ff4757;
                font-family:'JetBrains Mono',monospace">
      <div style="display:flex;align-items:center;gap:10px">
        <div style="width:8px;height:8px;border-radius:50%;background:#ff4757;
                    box-shadow:0 0 8px 2px rgba(255,71,87,.7)"></div>
        <span style="font-size:13px;font-weight:700;letter-spacing:.12em;
                     text-transform:uppercase;color:#e0e5ec">
          ANOMALY<span style="color:#ff4757">.</span>DETECT
        </span>
      </div>
      <div style="display:flex;align-items:center;gap:16px">
        <span style="font-size:10px;font-weight:700;letter-spacing:.08em;
                     text-transform:uppercase;color:#636e72">{len(anomalies)} alerts</span>
        <span style="font-size:10px;font-weight:700;letter-spacing:.08em;
                     text-transform:uppercase;padding:2px 8px;border-radius:3px;
                     background:#ff4757;color:#fff">{crit} critical</span>
        <span style="font-size:10px;font-weight:700;letter-spacing:.08em;
                     text-transform:uppercase;padding:2px 8px;border-radius:3px;
                     background:#f59e0b;color:#fff">{high} high</span>
        <div style="width:6px;height:6px;border-radius:50%;background:#22c55e;
                    box-shadow:0 0 6px 2px rgba(34,197,94,.6)"></div>
        <span style="font-size:9px;font-weight:700;letter-spacing:.08em;
                     text-transform:uppercase;color:#22c55e">Operational</span>
        {api_html}
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── KPI strip ─────────────────────────────────────────────────────────────
def _render_kpi_strip(anomalies: list):
    st.markdown("""
    <div style="background:#2d3436;display:grid;grid-template-columns:repeat(6,1fr);
                border-bottom:2px solid #3d4c4e;font-family:'JetBrains Mono',monospace">

      <div style="padding:22px 18px;border-right:1px solid #3d4c4e">
        <div style="font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
                    color:#888;margin-bottom:5px">Transactions</div>
        <div style="font-size:32px;font-weight:800;letter-spacing:-.03em;color:#e0e5ec;
                    line-height:1;text-shadow:0 0 20px rgba(255,255,255,.1)">3.42M</div>
        <div style="font-size:11px;color:#888;margin-top:4px">90-day window</div>
      </div>

      <div style="padding:22px 18px;border-right:1px solid #3d4c4e">
        <div style="font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
                    color:#888;margin-bottom:5px">Approval rate</div>
        <div style="font-size:32px;font-weight:800;letter-spacing:-.03em;color:#22c55e;
                    line-height:1;text-shadow:0 0 16px rgba(34,197,94,.4)">94.4%</div>
        <div style="font-size:11px;color:#22c55e;margin-top:4px">&#9660; &minus;1.2pp vs prior</div>
      </div>

      <div style="padding:22px 18px;border-right:1px solid #3d4c4e;position:relative">
        <div style="font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
                    color:#888;margin-bottom:5px">Decline rate</div>
        <div style="font-size:32px;font-weight:800;letter-spacing:-.03em;color:#ff4757;
                    line-height:1;text-shadow:0 0 16px rgba(255,71,87,.5)">5.6%</div>
        <div style="font-size:11px;color:#ff4757;margin-top:4px">&#9650; +1.2pp vs prior</div>
        <div style="position:absolute;bottom:0;left:0;right:0;height:2px;background:#ff4757"></div>
      </div>

      <div style="padding:22px 18px;border-right:1px solid #3d4c4e;position:relative">
        <div style="font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
                    color:#888;margin-bottom:5px">Fraud rate</div>
        <div style="font-size:32px;font-weight:800;letter-spacing:-.03em;color:#ff4757;
                    line-height:1;text-shadow:0 0 16px rgba(255,71,87,.5)">0.119%</div>
        <div style="font-size:11px;color:#ff4757;margin-top:4px">&#9650; +0.04pp alert period</div>
        <div style="position:absolute;bottom:0;left:0;right:0;height:2px;background:#ff4757"></div>
      </div>

      <div style="padding:22px 18px;border-right:1px solid #3d4c4e;position:relative">
        <div style="font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
                    color:#888;margin-bottom:5px">Revenue at risk</div>
        <div style="font-size:32px;font-weight:800;letter-spacing:-.03em;color:#ff4757;
                    line-height:1;text-shadow:0 0 16px rgba(255,71,87,.5)">&#163;7.4M</div>
        <div style="font-size:11px;color:#ff4757;margin-top:4px">Est. declined txn value</div>
        <div style="position:absolute;bottom:0;left:0;right:0;height:2px;background:#ff4757"></div>
      </div>

      <div style="padding:18px 16px">
        <div style="font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
                    color:#888;margin-bottom:5px">Fee income lost</div>
        <div style="font-size:32px;font-weight:800;letter-spacing:-.03em;color:#f59e0b;
                    line-height:1;text-shadow:0 0 16px rgba(245,158,11,.4)">&#163;134K</div>
        <div style="font-size:11px;color:#888;margin-top:4px">vs baseline approval</div>
      </div>

    </div>
    """, unsafe_allow_html=True)


# ── Status rail ───────────────────────────────────────────────────────────
def _render_status_rail(anomalies: list):
    from collections import Counter
    sev = Counter(a.get("severity") for a in anomalies)
    fc  = len(set(a.get("failure_class") for a in anomalies))

    INCIDENTS = [
        ("Processor outage","#3d4c4e","#a8b2d1"),
        ("3DS cascade","#1e3a5f","#93c5fd"),
        ("Fraud attack","#4a1515","#fca5a5"),
        ("Routing issue","#3d3000","#fcd34d"),
        ("Network rule","#1a2e1a","#86efac"),
    ]
    pills = "".join(
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:9px;font-weight:700;'
        f'letter-spacing:.04em;text-transform:uppercase;padding:2px 7px;border-radius:3px;'
        f'background:{bg};color:{fg};border:0.5px solid {fg}22">{label}</span>'
        for label, bg, fg in INCIDENTS
    )

    st.markdown(f"""
    <div style="background:#e0e5ec;border-bottom:1px solid #babecc;padding:7px 18px;
                display:flex;align-items:center;justify-content:space-between;
                box-shadow:0 2px 6px rgba(0,0,0,.06)">
      <div style="display:flex;align-items:center;gap:8px">
        <div style="width:6px;height:6px;border-radius:50%;background:#ff4757;
                    box-shadow:0 0 6px rgba(255,71,87,.6)"></div>
        <span style="font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:700;
                     letter-spacing:.1em;text-transform:uppercase;color:#4a5568">
          Detection summary
        </span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:700;
                     padding:2px 7px;border-radius:3px;background:#ff4757;color:#fff">
          {sev.get('critical',0)} critical
        </span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:700;
                     padding:2px 7px;border-radius:3px;background:#f59e0b;color:#fff">
          {sev.get('high',0)} high
        </span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:700;
                     padding:2px 7px;border-radius:3px;
                     background:#d1d9e6;color:#4a5568;
                     box-shadow:inset 1px 1px 3px #babecc,inset -1px -1px 3px #fff">
          {fc} issue types
        </span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:700;
                     padding:2px 7px;border-radius:3px;
                     background:#d1d9e6;color:#4a5568;
                     box-shadow:inset 1px 1px 3px #babecc,inset -1px -1px 3px #fff">
          {len(anomalies)} alerts
        </span>
      </div>
      <div style="display:flex;align-items:center;gap:6px">
        <span style="font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:700;
                     letter-spacing:.06em;text-transform:uppercase;color:#636e72">
          Incidents:
        </span>
        {pills}
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── API key bar (always visible, no sidebar needed) ──────────────────────
def _render_api_key_bar():
    """
    Fix 7: Visible API key input below the status rail.
    Analysts shouldn't need to hunt for the sidebar — key entry should be
    front and centre when the tool loads without a key.
    """
    import os
    has_key = bool(os.environ.get("GROQ_API_KEY",""))
    if has_key:
        return  # Already set — don't clutter the UI

    st.markdown(
        "<div style='background:#1a1a2e;border-bottom:1px solid #ff4757;padding:8px 18px;"
        "display:flex;align-items:center;gap:10px;font-family:JetBrains Mono,monospace'>",
        unsafe_allow_html=True
    )
    c1, c2, c3 = st.columns([1, 3, 1])
    with c1:
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:9px;font-weight:700;"
            "letter-spacing:.1em;text-transform:uppercase;color:#ff4757'>ARIA API KEY REQUIRED</div>",
            unsafe_allow_html=True
        )
    with c2:
        key_val = st.text_input(
            "groq_key_topbar",
            type="password",
            placeholder="Paste your Groq API key here (get free key at console.groq.com)...",
            label_visibility="collapsed",
            key="topbar_api_key",
        )
        if key_val and key_val != os.environ.get("GROQ_API_KEY",""):
            os.environ["GROQ_API_KEY"] = key_val
            _get_llm_client.clear()
            st.rerun()
    with c3:
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:9px;color:#636e72;"
            "text-align:right'>console.groq.com</div>",
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────
def _render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:700;
                    letter-spacing:.08em;text-transform:uppercase;color:#4a5568;margin-bottom:12px">
          System configuration
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:700;
                    letter-spacing:.08em;text-transform:uppercase;color:#636e72;margin-bottom:6px">
          Groq API key (for ARIA)
        </div>
        """, unsafe_allow_html=True)

        key_input = st.text_input(
            "API key",
            type="password",
            placeholder="Enter your Groq API key...",
            value=os.environ.get("GROQ_API_KEY",""),
            label_visibility="collapsed",
            help="Get a free key at https://console.groq.com"
        )
        if key_input and key_input != os.environ.get("GROQ_API_KEY",""):
            os.environ["GROQ_API_KEY"] = key_input
            _get_llm_client.clear()
            st.rerun()

        st.divider()
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#636e72;
                    line-height:1.6;margin-bottom:10px">
          ARIA — Anomaly Response &amp; Intelligence Assistant.<br>
          Detects via Z-score, STL, chi-squared, KL divergence.<br>
          LLM explains — never detects.
        </div>
        """, unsafe_allow_html=True)

        if st.button("Regenerate pipeline (full)", use_container_width=True):
            _load_anomaly_objects.clear()
            from llm.incident_memory import IncidentMemory
            IncidentMemory.reset()
            st.rerun()

        if st.button("Refresh alerts (incremental)", use_container_width=True):
            with st.spinner("Running incremental scan (last 7 days)..."):
                try:
                    from run_pipeline import run_incremental_pipeline
                    result = run_incremental_pipeline(lookback_days=7, verbose=False)
                    n_new  = len(result.get("anomalies_new",[]))
                    dur    = result.get("run_duration_s",0)
                    st.success(f"{n_new} alerts updated in {dur}s")
                    _load_anomaly_objects.clear()
                    from llm.incident_memory import IncidentMemory
                    IncidentMemory.reset()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Incremental scan failed: {exc}")


# ── Main layout ───────────────────────────────────────────────────────────
def _render_main(anomalies: list, llm_client) -> None:
    from ui.anomaly_feed     import render_anomaly_feed
    from ui.diagnostic_panel import render_diagnostic_panel
    from ui.chart_panel      import render_chart_panel
    from ui.chat_panel       import render_chat_panel

    col_feed, col_main = st.columns([1, 3], gap="small")

    with col_feed:
        selected = render_anomaly_feed(anomalies)

    with col_main:
        if selected is None:
            st.markdown("""
            <div style="display:flex;align-items:center;justify-content:center;
                        height:400px;font-family:'JetBrains Mono',monospace;
                        font-size:11px;font-weight:700;letter-spacing:.08em;
                        text-transform:uppercase;color:#636e72">
              ← Select an incident from the feed
            </div>
            """, unsafe_allow_html=True)
            return

        # Breadcrumb
        from ui.anomaly_feed import FAILURE_CLASS_LABELS, _format_slice
        fc_lbl   = FAILURE_CLASS_LABELS.get(selected.get("failure_class",""), "—")
        sev      = selected.get("severity","")
        sev_map  = {"critical":"#ff4757","high":"#f59e0b","medium":"#378add","low":"#22c55e"}
        sev_col  = sev_map.get(sev, "#636e72")
        slice_str = _format_slice(selected.get("affected_slice",{}))
        aid      = selected.get("anomaly_id","")[:12]

        st.markdown(
            f"<div style='font-family:\"JetBrains Mono\",monospace;font-size:10px;font-weight:700;"
            f"letter-spacing:.05em;text-transform:uppercase;color:#4a5568;"
            f"display:flex;align-items:center;gap:6px;margin-bottom:10px;padding:7px 11px;"
            f"background:#d1d9e6;border-radius:5px;"
            f"box-shadow:inset 4px 4px 8px #babecc,inset -4px -4px 8px #ffffff'>"
            f"<i class='ti ti-list' style='font-size:13px'></i>"
            f"Alerts <span style='color:#a3b1c6'>›</span>"
            f"<span style='color:#2d3436'>{fc_lbl}</span>"
            f"<span style='color:#a3b1c6'>·</span>"
            f"<span style='color:#636e72'>{slice_str}</span>"
            f"<span style='margin-left:auto;background:{sev_col};color:#fff;font-size:9px;"
            f"font-weight:700;letter-spacing:.08em;padding:2px 7px;border-radius:2px'>"
            f"{sev.upper()}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Tabs — ARIA expanded on click via session state
        if "aria_expanded" not in st.session_state:
            st.session_state["aria_expanded"] = False

        tab_labels = [
            "📋  Summary",
            "📈  Charts",
            "🤖  Ask ARIA — Anomaly Response & Intelligence Assistant"
            if st.session_state.get("aria_expanded") else
            "🤖  Ask ARIA",
        ]

        tab_diag, tab_chart, tab_aria = st.tabs(tab_labels)

        with tab_diag:
            render_diagnostic_panel(selected, llm_client)

        with tab_chart:
            render_chart_panel(selected)

        with tab_aria:
            # Expand ARIA on first visit
            st.session_state["aria_expanded"] = True
            render_chat_panel(selected, llm_client)


# ── Entry point ───────────────────────────────────────────────────────────
def main():
    _inject_css()
    _inject_api_key()

    with st.spinner("Loading anomaly data..."):
        try:
            anomalies = _load_anomaly_objects()
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            st.stop()

    if not anomalies:
        st.error("No anomaly objects found. Run `python run_pipeline.py`.")
        st.stop()

    has_api = bool(os.environ.get("GROQ_API_KEY",""))
    llm_client = _get_llm_client()

    _render_topbar(anomalies, has_api)
    _render_kpi_strip(anomalies)
    _render_status_rail(anomalies)
    _render_api_key_bar()
    _render_sidebar()

    if llm_client is None:
        st.warning(
            "No Groq API key — ARIA diagnostics unavailable. "
            "Add your key in the sidebar or in `.streamlit/secrets.toml`.",
            icon="⚠"
        )

    st.markdown("""
    <div style="background:#e0e5ec;padding:0">
    </div>
    """, unsafe_allow_html=True)

    _render_main(anomalies, llm_client)


if __name__ == "__main__":
    main()
