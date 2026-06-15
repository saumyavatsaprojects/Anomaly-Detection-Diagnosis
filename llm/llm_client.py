"""
LLM Client — Groq Backend
==========================
Interfaces with the Groq API using the OpenAI-compatible SDK.

Model   : llama-3.1-8b-instant
API     : https://api.groq.com/openai/v1
SDK     : groq>=0.9.0  (pip install groq)

Groq-specific notes:
  - Uses OpenAI-compatible chat completions: client.chat.completions.create()
  - Message format: [{"role": "system"|"user"|"assistant", "content": str}]
  - Streaming: stream=True yields chunks with .choices[0].delta.content
  - Rate limits (free tier): 30 req/min, 14,400 req/day on llama-3.3-70b
  - Retry logic handles 429 (rate limit) with exponential backoff

Colab setup:
    !pip install groq
    import os; os.environ["GROQ_API_KEY"] = "YOUR_GROQ_KEY_HERE"
    # or: from google.colab import userdata
    #     os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")

Streamlit setup:
    # .streamlit/secrets.toml  ->  GROQ_API_KEY = "YOUR_GROQ_KEY_HERE"
    # app.py injects: os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

from groq import Groq, RateLimitError, APIStatusError

from llm.context_builder import ContextBuilder, AssembledContext, FollowUpContext
from llm.output_verifier import OutputVerifier, VerificationResult
from llm.prompt_templates import (
    QUESTION_CLASSIFIER_PROMPT,
    TEMPLATE_VERSION,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

MODEL             = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_TOKENS        = 1024
CLASSIFIER_TOKENS = 10
MAX_HISTORY       = 10
MAX_RETRIES       = 3
RETRY_BASE_SECS   = 2.0

VALID_QUESTION_TYPES = {
    "slice_drilldown",
    "time_comparison",
    "causal_hypothesis",
    "action_request",
    "metric_detail",
    "scope_check",
    "out_of_scope",
}


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DiagnosticResponse:
    text:             str
    anomaly_id:       str
    question_type:    str
    verification:     VerificationResult
    is_out_of_scope:  bool = False
    model:            str  = MODEL
    input_tokens:     int  = 0
    output_tokens:    int  = 0
    latency_ms:       int  = 0
    template_version: str  = TEMPLATE_VERSION

    @property
    def hallucination_badge(self) -> str:
        return {
            "none":   "Grounded",
            "low":    "Low risk",
            "medium": "Review citations",
            "high":   "High risk — verify manually",
        }.get(self.verification.hallucination_risk, "Unknown")

    @property
    def hallucination_color(self) -> str:
        return {
            "none":   "green",
            "low":    "blue",
            "medium": "orange",
            "high":   "red",
        }.get(self.verification.hallucination_risk, "gray")


# ─────────────────────────────────────────────────────────────────────────────
# CONVERSATION STATE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConversationState:
    anomaly_id:        str
    anomaly_obj:       dict
    assembled_context: AssembledContext
    history:           list = field(default_factory=list)
    turn_count:        int  = 0

    def add_turn(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        self.turn_count += 1
        if len(self.history) > MAX_HISTORY:
            self.history = self.history[-MAX_HISTORY:]

    def get_messages(self) -> list:
        """
        Full messages array for Groq API:
          system prompt → initial brief (always anchored) → conversation history
        """
        return [
            {"role": "system", "content": self.assembled_context.system_prompt},
            {"role": "user",   "content": self.assembled_context.initial_brief},
            *self.history,
        ]


# ─────────────────────────────────────────────────────────────────────────────
# LLM CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Anomaly diagnostic assistant backed by Groq (meta-llama/llama-4-scout-17b-16e-instruct).

    Quick start (Colab):
        import os
        os.environ["GROQ_API_KEY"] = "YOUR_GROQ_KEY_HERE"
        from llm.llm_client import LLMClient
        client = LLMClient()
        state  = client.start_conversation(anomaly_obj)
        resp   = client.get_initial_diagnostic(state)
        print(resp.text)
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "GROQ_API_KEY not set.\n"
                "Colab     : os.environ['GROQ_API_KEY'] = 'YOUR_GROQ_KEY_HERE'\n"
                "Streamlit : set GROQ_API_KEY in .streamlit/secrets.toml\n"
                "Free key  : https://console.groq.com"
            )
        self._client   = Groq(api_key=key)
        self._builder  = ContextBuilder()
        self._verifier = OutputVerifier()

    # ── CONVERSATION MANAGEMENT ───────────────────────────────────────────────

    def start_conversation(self, anomaly_obj: dict) -> ConversationState:
        ctx = self._builder.build_initial_context(anomaly_obj)
        for w in ctx.assembly_warnings:
            logger.warning("Context assembly warning: %s", w)
        return ConversationState(
            anomaly_id        = ctx.anomaly_id,
            anomaly_obj       = anomaly_obj,
            assembled_context = ctx,
        )

    def reset_conversation(self, state: ConversationState) -> ConversationState:
        return ConversationState(
            anomaly_id        = state.anomaly_id,
            anomaly_obj       = state.anomaly_obj,
            assembled_context = state.assembled_context,
        )

    # ── INITIAL DIAGNOSTIC — non-streaming ───────────────────────────────────

    def get_initial_diagnostic(self, state: ConversationState) -> DiagnosticResponse:
        t0  = time.time()
        raw = self._call_with_retry(state.get_messages(), MAX_TOKENS)
        text = raw.choices[0].message.content or ""
        state.add_turn("assistant", text)
        verification = self._verifier.verify(text, state.assembled_context.supporting_stats)
        usage = raw.usage
        return DiagnosticResponse(
            text          = text,
            anomaly_id    = state.anomaly_id,
            question_type = "initial_diagnostic",
            verification  = verification,
            input_tokens  = getattr(usage, "prompt_tokens", 0),
            output_tokens = getattr(usage, "completion_tokens", 0),
            latency_ms    = int((time.time() - t0) * 1000),
        )

    # ── INITIAL DIAGNOSTIC — streaming ───────────────────────────────────────

    def get_initial_diagnostic_stream(self, state: ConversationState) -> Iterator[str]:
        """
        Yields text chunks for st.write_stream().
        Final yielded item is a hidden sentinel JSON for verification metadata.

        Streamlit usage:
            full = st.write_stream(client.get_initial_diagnostic_stream(state))
        """
        collected = []
        for chunk in self._stream_with_retry(state.get_messages(), MAX_TOKENS):
            collected.append(chunk)
            yield chunk

        complete     = "".join(collected)
        state.add_turn("assistant", complete)
        verification = self._verifier.verify(complete, state.assembled_context.supporting_stats)
        if verification.hallucination_risk not in ("none", "low"):
            logger.warning("Verification [%s initial]: %s", state.anomaly_id, verification.summary())
        yield (
            f"\n\n<!--VERIFY:"
            f"{json.dumps({'risk': verification.hallucination_risk, 'warnings': verification.warnings})}"
            f"-->"
        )

    # ── FOLLOW-UP — non-streaming ─────────────────────────────────────────────

    def ask_followup(self, state: ConversationState, user_question: str) -> DiagnosticResponse:
        t0            = time.time()
        question_type = self._classify_question(user_question, state.assembled_context.context_summary)
        followup_ctx  = self._builder.build_followup_context(state.anomaly_obj, user_question, question_type)

        if followup_ctx.is_out_of_scope:
            return DiagnosticResponse(
                text            = followup_ctx.out_of_scope_response,
                anomaly_id      = state.anomaly_id,
                question_type   = "out_of_scope",
                verification    = VerificationResult(hallucination_risk="none"),
                is_out_of_scope = True,
                latency_ms      = int((time.time() - t0) * 1000),
            )

        state.add_turn("user", followup_ctx.injected_data)
        raw  = self._call_with_retry(state.get_messages(), MAX_TOKENS)
        text = raw.choices[0].message.content or ""
        state.add_turn("assistant", text)
        verification = self._verifier.verify(text, state.assembled_context.supporting_stats)
        usage = raw.usage
        return DiagnosticResponse(
            text          = text,
            anomaly_id    = state.anomaly_id,
            question_type = question_type,
            verification  = verification,
            input_tokens  = getattr(usage, "prompt_tokens", 0),
            output_tokens = getattr(usage, "completion_tokens", 0),
            latency_ms    = int((time.time() - t0) * 1000),
        )

    # ── FOLLOW-UP — streaming ─────────────────────────────────────────────────

    def ask_followup_stream(self, state: ConversationState, user_question: str) -> Iterator[str]:
        question_type = self._classify_question(user_question, state.assembled_context.context_summary)
        followup_ctx  = self._builder.build_followup_context(state.anomaly_obj, user_question, question_type)

        if followup_ctx.is_out_of_scope:
            yield followup_ctx.out_of_scope_response
            return

        state.add_turn("user", followup_ctx.injected_data)
        collected = []
        for chunk in self._stream_with_retry(state.get_messages(), MAX_TOKENS):
            collected.append(chunk)
            yield chunk

        complete     = "".join(collected)
        state.add_turn("assistant", complete)
        verification = self._verifier.verify(complete, state.assembled_context.supporting_stats)
        if verification.hallucination_risk not in ("none", "low"):
            logger.warning("Verification [%s / %s]: %s", state.anomaly_id, question_type, verification.summary())
        yield (
            f"\n\n<!--VERIFY:"
            f"{json.dumps({'risk': verification.hallucination_risk, 'warnings': verification.warnings, 'question_type': question_type})}"
            f"-->"
        )

    # ── QUESTION CLASSIFICATION ───────────────────────────────────────────────

    def _classify_question(self, user_question: str, context_summary: str) -> str:
        prompt = QUESTION_CLASSIFIER_PROMPT.format(
            user_question   = user_question,
            context_summary = context_summary,
        )
        try:
            raw      = self._call_with_retry([{"role": "user", "content": prompt}], CLASSIFIER_TOKENS)
            category = (raw.choices[0].message.content or "").strip().lower()
            return category if category in VALID_QUESTION_TYPES else "metric_detail"
        except Exception as exc:
            logger.error("Question classification failed: %s", exc)
            return "metric_detail"

    # ── GROQ API WITH RETRY ───────────────────────────────────────────────────

    def _call_with_retry(self, messages: list, max_tokens: int):
        """Chat completion with exponential backoff on Groq rate limits."""
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                return self._client.chat.completions.create(
                    model       = MODEL,
                    messages    = messages,
                    max_tokens  = max_tokens,
                    temperature = 0.2,
                )
            except RateLimitError as exc:
                wait = RETRY_BASE_SECS * (2 ** attempt)
                logger.warning("Groq rate limit (attempt %d/%d). Retrying in %.1fs.", attempt + 1, MAX_RETRIES, wait)
                time.sleep(wait)
                last_exc = exc
            except APIStatusError as exc:
                logger.error("Groq API error %s: %s", exc.status_code, exc.message)
                raise
        raise last_exc  # type: ignore[misc]

    def _stream_with_retry(self, messages: list, max_tokens: int) -> Iterator[str]:
        """Streaming chat completion with retry. Re-starts on rate limit."""
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                stream = self._client.chat.completions.create(
                    model       = MODEL,
                    messages    = messages,
                    max_tokens  = max_tokens,
                    temperature = 0.2,
                    stream      = True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
                return
            except RateLimitError as exc:
                wait = RETRY_BASE_SECS * (2 ** attempt)
                logger.warning("Groq rate limit (stream, attempt %d/%d). Retrying in %.1fs.", attempt + 1, MAX_RETRIES, wait)
                time.sleep(wait)
                last_exc = exc
            except APIStatusError as exc:
                logger.error("Groq stream API error: %s", exc)
                raise
        raise last_exc  # type: ignore[misc]
