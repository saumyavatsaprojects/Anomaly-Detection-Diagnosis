"""
LLM Client — Groq Backend
==========================
Interfaces with the Groq API using the OpenAI-compatible SDK.

Model   : llama-3.1-8b-instant
API     : https://api.groq.com/openai/v1
SDK     : groq>=0.9.0  (pip install groq)

WHAT WAS WRONG
──────────────
The `GroqClient` class only had two internal methods:
  · run_triage_completion()   — structured JSON completion (used by run_pipeline)
  · _stream_with_retry()      — internal streaming helper

`chat_panel.py` and `diagnostic_panel.py` were written against a separate
`LLMClient` class (a different version that was lost) which exposed:
  · start_conversation(anomaly)           → ConversationState
  · get_initial_diagnostic_stream(state)  → Iterator[str]
  · ask_followup_stream(state, question)  → Iterator[str]

`app.py` passes the `GroqClient` instance directly to both panels,
so every call to `llm_client.start_conversation(...)` raised:
    AttributeError: 'GroqClient' object has no attribute 'start_conversation'

FIX: Added the full conversation interface onto `GroqClient`.
  · ConversationState dataclass holds the anomaly context + message history
  · start_conversation()           builds the system prompt + primes the state
  · get_initial_diagnostic_stream() streams the first structured narrative
  · ask_followup_stream()          streams grounded follow-up answers
  · GroqClient is now also importable as `LLMClient` (alias at bottom of file)
    so any `from llm.llm_client import LLMClient` import still works.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

from groq import APIStatusError, Groq, RateLimitError
from pydantic import BaseModel, Field

logger = logging.getLogger("ARIA.llm_client")

MODEL        = "llama-3.1-8b-instant"
MAX_RETRIES  = 5
RETRY_BASE_SECS = 2.0

# ── Failure-class → plain English ─────────────────────────────────────────────
FC_LABELS = {
    "3ds_acs_failure":      "3DS / ACS authentication failure",
    "processor_outage":     "Processor or switch outage",
    "fraud_attack":         "Fraud attack",
    "acquirer_routing":     "Acquirer routing issue",
    "network_rule_change":  "Card network rule change",
    "issuer_rules_misfire": "Issuer rule misconfiguration",
    "undetermined":         "Under investigation",
}

# Sentinel the UI panels parse to extract hallucination-risk metadata
SENTINEL_TEMPLATE = '<!--VERIFY:{json}-->'


# ── Structured output schema (used by run_pipeline triage step) ───────────────
class VerificationResult(BaseModel):
    """Structured response from the LLM triage validation step."""

    operational_severity: str = Field(
        description="P1 (Critical), P2 (High), P3 (Medium), P4 (Low)"
    )
    confidence_level: str = Field(
        description="High, Medium, or Low based on context match completeness"
    )
    technical_summary: str = Field(
        description="Technical assessment of the anomaly event"
    )
    primary_root_cause_mechanism: str = Field(
        description="The mechanical root-cause hypothesis"
    )
    ruled_out_hypotheses: str = Field(
        description="Hypotheses evaluated but discarded based on evidence"
    )
    recommended_mitigation_actions: list[str] = Field(
        default_factory=list,
        description="Ordered tactical remediations for operational engineering teams",
    )

    @property
    def summary(self) -> str:
        return self.technical_summary

    @property
    def explanation(self) -> str:
        return f"{self.primary_root_cause_mechanism} — {self.technical_summary}"


# ── Conversation state ────────────────────────────────────────────────────────
@dataclass
class ConversationState:
    """
    Holds everything needed to continue an ARIA conversation about one anomaly.
    Passed back to the caller after start_conversation() and threaded through
    every subsequent ask_followup_stream() call.
    """
    anomaly_id:     str
    system_prompt:  str
    messages:       list = field(default_factory=list)   # OpenAI-format dicts


# ── Main client ───────────────────────────────────────────────────────────────
class GroqClient:
    """
    Full ARIA LLM client.

    Exposes:
      · start_conversation(anomaly)            → ConversationState
      · get_initial_diagnostic_stream(state)   → Iterator[str]
      · ask_followup_stream(state, question)   → Iterator[str]
      · run_triage_completion(...)             → VerificationResult   (pipeline use)
    """

    def __init__(self, api_key: Optional[str] = None):
        target_key = api_key or os.environ.get("GROQ_API_KEY")
        if not target_key:
            raise ValueError(
                "GROQ_API_KEY not set. Add it in the sidebar or in "
                ".streamlit/secrets.toml as GROQ_API_KEY = '...'"
            )
        self._client = Groq(api_key=target_key)

    # ── Conversation interface (used by chat_panel + diagnostic_panel) ─────────

    def start_conversation(self, anomaly: dict) -> ConversationState:
        """
        Initialise a conversation context for one anomaly.
        Builds the grounding system prompt from the anomaly object and
        returns a ConversationState the caller must hold and pass back.
        """
        aid      = anomaly.get("anomaly_id", "unknown")
        fc       = anomaly.get("failure_class", "undetermined")
        fc_lbl   = FC_LABELS.get(fc, fc)
        sev      = anomaly.get("severity", "medium").upper()
        sigma    = anomaly.get("deviation_sigma", 0)
        dur      = anomaly.get("duration_hours", 0)
        obs      = anomaly.get("observed_value", 0)
        base     = anomaly.get("baseline_value", 0)
        conf     = anomaly.get("failure_class_confidence", "low")
        sl       = anomaly.get("affected_slice", {})
        evidence = anomaly.get("evidence", [])
        ruled_out    = anomaly.get("ruled_out", [])
        not_affected = anomaly.get("not_affected", [])
        rc_ev        = anomaly.get("reason_code_evidence", {})
        esc          = anomaly.get("recommended_escalation", "")

        # Format slice dimensions
        slice_parts = [
            str(v) for v in sl.values()
            if str(v).lower() not in ("all", "", "none")
        ]
        slice_str = " / ".join(slice_parts) if slice_parts else "all dimensions"

        # Format reason code evidence compactly
        rc_lines = []
        for code, d in sorted(
            rc_ev.items(),
            key=lambda x: abs(x[1].get("delta_pp", 0)),
            reverse=True,
        )[:6]:
            arrow = "▲" if d.get("delta_pp", 0) > 0 else "▼"
            rc_lines.append(
                f"  RC {code} ({d.get('label', '')}): "
                f"{d.get('current_share', 0):.1%} vs {d.get('baseline_share', 0):.1%} "
                f"{arrow} {abs(d.get('delta_pp', 0)):.1f}pp"
            )

        system_prompt = f"""You are ARIA — Anomaly Response & Intelligence Assistant for a card-issuing bank's real-time risk operations team.

INCIDENT GROUND TRUTH (use ONLY these values — never fabricate numbers):
  ID               : {aid}
  Classification   : {fc_lbl}  ({conf} confidence)
  Severity         : {sev}
  Affected segment : {slice_str}
  Observed value   : {obs:.3%} (approval rate or relevant metric)
  Baseline value   : {base:.3%}
  Deviation        : {sigma:+.1f}σ
  Duration         : {dur} hours

SUPPORTING EVIDENCE:
{chr(10).join(f'  · {e}' for e in evidence) or '  (none provided)'}

DECLINE REASON CODE SHIFTS:
{chr(10).join(rc_lines) or '  (none provided)'}

RULED-OUT HYPOTHESES:
{chr(10).join(f'  · {r}' for r in ruled_out) or '  (none)'}

NOT AFFECTED:
{chr(10).join(f'  · {n}' for n in not_affected) or '  (none)'}

RECOMMENDED ESCALATION:
  {esc or 'Not specified.'}

STRICT RULES:
1. Answer ONLY from the ground truth above. Never invent numbers, percentages, or country names not in the data.
2. If something is not in the data, say "not available in the current incident data."
3. Be concise and direct — analysts are triaging in real time.
4. After your answer, append this exact sentinel on a new line with your hallucination risk assessment:
   <!--VERIFY:{{"risk":"none","warnings":[]}}-->
   Set risk to "low" if you used approximate reasoning, "medium" if a value needed inference, "high" if you had to guess.
5. Do NOT say "based on the context provided" or similar preambles. Lead with the answer.
"""

        return ConversationState(
            anomaly_id=aid,
            system_prompt=system_prompt,
            messages=[],
        )

    def get_initial_diagnostic_stream(self, state: ConversationState) -> Iterator[str]:
        """
        Stream the first structured diagnostic narrative for an anomaly.
        Called by diagnostic_panel.py once per anomaly (result is cached).
        """
        prompt = (
            "Generate a concise incident diagnostic in exactly this structure:\n\n"
            "**WHAT HAPPENED**\n"
            "One paragraph. State the metric, the observed value vs baseline, deviation, "
            "duration, and affected segment. Use the exact numbers from the ground truth.\n\n"
            "**WHY IT HAPPENED**\n"
            "One paragraph. State the classified failure class, confidence level, and the "
            "top 2-3 pieces of supporting evidence. Reference reason code shifts if present.\n\n"
            "**WHAT TO DO NOW**\n"
            "2-4 bullet points. Use the recommended escalation path and evidence to suggest "
            "concrete next steps for the on-call analyst.\n\n"
            "End with the VERIFY sentinel."
        )
        yield from self._chat_stream(state, prompt, max_tokens=600)

    def ask_followup_stream(
        self, state: ConversationState, question: str
    ) -> Iterator[str]:
        """
        Stream an answer to a follow-up question.
        Called by chat_panel.py for every user question.
        Appends both the user turn and assistant response to state.messages
        so subsequent calls retain full conversation context.
        """
        # Classify the question type for the UI badge
        q_lower = question.lower()
        if any(w in q_lower for w in ["which country", "where", "region", "corridor"]):
            q_type = "slice_drilldown"
        elif any(w in q_lower for w in ["yesterday", "last week", "compare", "trend", "before"]):
            q_type = "time_comparison"
        elif any(w in q_lower for w in ["why", "cause", "reason", "explain"]):
            q_type = "causal_hypothesis"
        elif any(w in q_lower for w in ["do", "should", "action", "escalate", "fix", "now"]):
            q_type = "action_request"
        elif any(w in q_lower for w in ["impact", "revenue", "cost", "loss", "fee"]):
            q_type = "financial_impact"
        elif any(w in q_lower for w in ["affected", "scope", "all", "only", "other"]):
            q_type = "scope_check"
        elif any(w in q_lower for w in ["rc ", "reason code", "decline code", "code "]):
            q_type = "metric_detail"
        else:
            q_type = "metric_detail"

        # Inject question type into the sentinel so the UI can render the badge
        enhanced_q = (
            f"{question}\n\n"
            f"[Answer this question. After your answer, append the sentinel: "
            f'<!--VERIFY:{{"risk":"none","warnings":[],"question_type":"{q_type}"}}-->]'
        )

        yield from self._chat_stream(state, enhanced_q, max_tokens=400)

    def _chat_stream(
        self, state: ConversationState, user_content: str, max_tokens: int
    ) -> Iterator[str]:
        """
        Core streaming call. Maintains message history on `state` so
        multi-turn conversation context is preserved across calls.
        """
        # Build the full messages list: system + history + new user turn
        state.messages.append({"role": "user", "content": user_content})

        messages = (
            [{"role": "system", "content": state.system_prompt}]
            + state.messages
        )

        # Stream with retry
        collected: list[str] = []
        for chunk in self._stream_with_retry(messages, max_tokens=max_tokens):
            collected.append(chunk)
            yield chunk

        # Append the completed assistant turn to history for next call
        assistant_content = "".join(collected)
        state.messages.append({"role": "assistant", "content": assistant_content})

    # ── Structured triage completion (used by run_pipeline, not the UI) ────────

    def run_triage_completion(
        self,
        context_summary: dict,
        matched_hypotheses: list,
        retrieved_playbooks: str,
    ) -> VerificationResult:
        """
        Zero-shot structured JSON completion for the detection pipeline triage step.
        Returns a VerificationResult Pydantic model.
        """
        system_prompt = (
            "You are a Senior Payments Incident Commander at a card-issuing bank.\n"
            "Ingest the statistical anomaly data and playbook references provided, "
            "then output a verified triage assessment as valid JSON matching the schema.\n\n"
            "RULES:\n"
            "1. Do NOT calculate statistics — use only the pre-calculated values given.\n"
            "2. Output must be valid JSON only, no markdown fences.\n"
            "3. Populate every field in the schema.\n"
        )

        user_payload = {
            "telemetry_metrics": context_summary,
            "hypothesis_matches": matched_hypotheses,
            "playbook_references": retrieved_playbooks,
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, indent=2)},
        ]

        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = self._client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.05,
                    max_tokens=1024,
                    response_format={"type": "json_object"},
                )
                raw_txt     = resp.choices[0].message.content
                parsed_data = json.loads(raw_txt)
                return VerificationResult(**parsed_data)
            except RateLimitError as exc:
                wait = RETRY_BASE_SECS * (2 ** attempt)
                logger.warning(
                    "Groq rate limit (attempt %d/%d). Retrying in %.1fs.",
                    attempt + 1, MAX_RETRIES, wait,
                )
                time.sleep(wait)
                last_exc = exc
            except APIStatusError as exc:
                logger.error("Groq API error %s: %s", exc.status_code, exc.message)
                raise
        raise last_exc  # type: ignore[misc]

    # ── Internal streaming helper ──────────────────────────────────────────────

    def _stream_with_retry(
        self, messages: list, max_tokens: int
    ) -> Iterator[str]:
        """Streaming chat completion with exponential-backoff retry on rate limits."""
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                stream = self._client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.2,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
                return
            except RateLimitError as exc:
                wait = RETRY_BASE_SECS * (2 ** attempt)
                logger.warning(
                    "Groq rate limit (stream, attempt %d/%d). Retrying in %.1fs.",
                    attempt + 1, MAX_RETRIES, wait,
                )
                time.sleep(wait)
                last_exc = exc
            except APIStatusError as exc:
                logger.error("Groq stream API error: %s", exc)
                raise
        raise last_exc  # type: ignore[misc]


# ── Alias so `from llm.llm_client import LLMClient` still works ──────────────
LLMClient = GroqClient
