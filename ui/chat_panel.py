"""
ARIA Chat Panel — Industrial Layout Fix
==========================================
Renders the interactive analytical timeline layout with pinned layouts.
"""

import logging
import streamlit as st

logger = logging.getLogger("ARIA.chat_panel")


def render_aria_chat_panel(
    groq_api_key: str, model_choice: str, akey: str, ctx: dict
):
    """Renders the modular interaction terminal console panels cleanly."""
    from llm_client import GroqClient

    P = {
        "surface": "#0D1226",
        "surface2": "#111827",
        "border": "#1C2742",
        "border2": "#243050",
        "text_secondary": "#7A8BA8",
        "blue": "#38BDF8",
        "green": "#10B981",
    }

    hist_key = f"chat_{akey}"
    proc_key = f"proc_{akey}"

    if hist_key not in st.session_state:
        st.session_state[hist_key] = []
    if proc_key not in st.session_state:
        st.session_state[proc_key] = False

    history = st.session_state[hist_key]

    # ─────────────────────────────────────────────────────────────────────────
    # FIX: PERSISTENT HISTORICAL VIEW LAYER (PINNED AT THE TOP OUTRIGHT)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='margin-bottom: 6px; font-size: 0.65rem; font-weight:600; color:#3D506B; text-transform:uppercase;'>Operational Stream Timeline</div>",
        unsafe_allow_html=True,
    )

    if history:
        for msg in history:
            is_user = msg["role"] == "user"
            rc = P["blue"] if is_user else P["green"]
            rl = "OPERATIONS_ANALYST" if is_user else "ARIA_COMMANDER_AI"
            st.markdown(
                f'<div style="background:{P["surface2"]};border:1px solid {P["border2"]};'
                f'border-radius:6px;padding:8px 12px;margin-bottom:6px">'
                f'<div style="font-size:0.6rem;font-weight:600;color:{rc};'
                f'text-transform:uppercase;letter-spacing:0.5px;margin-bottom:3px">{rl}</div>'
                f'<div style="font-size:0.76rem;color:{P["text_secondary"]};line-height:1.5">'
                f'{msg["content"]}</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f"<div style='background:{P['surface2']}; border:1px dashed {P['border']}; border-radius:6px; padding:16px; text-align:center; font-size:0.75rem; color:{P['text_secondary']}'>Console stream idle. Submit transaction infrastructure queries below.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # FIX: SUB-COMPACT STATIC TRIAGE ACTIONS (PINNED STATICALLY ABOVE THE INPUT BOX)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.6rem;color:#3D506B;text-transform:uppercase;font-weight:600;letter-spacing:0.5px;margin-bottom:4px'>Triage Shortcuts</div>",
        unsafe_allow_html=True,
    )

    bq1, bq2 = st.columns(2)
    pending_q = ""
    with bq1:
        if st.button(
            "📊 Quantify Drop Volume",
            key=f"q1_{akey}",
            use_container_width=True,
        ):
            pending_q = "What is the global approval rate drop percentage and how many total transactions failed during this exact hour event window trace?"
    with bq2:
        if st.button(
            "🔒 Audit Auth Channels",
            key=f"q2_{akey}",
            use_container_width=True,
        ):
            pending_q = "Compare the exact metrics performance delta of ecom_3ds vs ecom_non3ds paths to verify active protocol verification failures."

    # ─────────────────────────────────────────────────────────────────────────
    # TEXT INPUT AND HANDLING FIELD
    # ─────────────────────────────────────────────────────────────────────────
    user_q = st.chat_input(
        "Query real-time ledger matrix structures...", key=f"ci_{akey}"
    )
    active_q = user_q or pending_q

    if active_q and not st.session_state[proc_key]:
        st.session_state[proc_key] = True
        history.append({"role": "user", "content": active_q})

        # Process response without streaming to align completely with layout rules
        try:
            client = GroqClient(api_key=groq_api_key)
            from llm_client import MODEL

            messages = [
                {
                    "role": "system",
                    "content": f"You are a helpful banking analyst system co-pilot. Ground your assessment strictly in the metrics context provided here:\n{ctx}",
                }
            ]
            for m in history[:-1]:
                messages.append({"role": m["role"], "content": m["content"]})
            messages.append({"role": "user", "content": active_q})

            # Fetch completion directly using client session context
            resp = client._client.chat.completions.create(
                model=model_choice if model_choice else MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=300,
            )
            ans_text = resp.choices[0].message.content.strip()
            history.append({"role": "assistant", "content": ans_text})

        except Exception as exc:
            logger.error("ARIA dialogue processing failure: %s", exc)
            history.append(
                {
                    "role": "assistant",
                    "content": f"Incident control channel interface failure error trace: `{str(exc)[:100]}`",
                }
            )

        st.session_state[hist_key] = history
        st.session_state[proc_key] = False
        st.rerun()

    # Clear Button Handler Core Position
    if history:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button(
            "🗑️ Reset Console History",
            key=f"clr_{akey}",
            use_container_width=True,
        ):
            st.session_state[hist_key] = []
            st.rerun()
