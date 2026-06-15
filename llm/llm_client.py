"""
LLM Client — Groq Backend
==========================
Interfaces with the Groq API using the OpenAI-compatible SDK.

Model   : llama-3.1-8b-instant
API     : https://api.groq.com/openai/v1
SDK     : groq>=0.9.0  (pip install groq)
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

MODEL = "llama-3.1-8b-instant"
MAX_RETRIES = 5
RETRY_BASE_SECS = 2.0


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

    # ─────────────────────────────────────────────────────────────────────────
    # SURGICAL FIX: Property aliases to map incoming telemetry calls cleanly
    # ─────────────────────────────────────────────────────────────────────────
    @property
    def summary(self) -> str:
        """Fallback alias for components expecting a .summary property."""
        return self.technical_summary

    @property
    def explanation(self) -> str:
        """Fallback alias for components expecting an .explanation property."""
        return f"{self.primary_root_cause_mechanism} — {self.technical_summary}"


class GroqClient:
    """Wrapped client wrapper around the Groq Python SDK with integrated resilience mechanics."""

    def __init__(self, api_key: Optional[str] = None):
        target_key = api_key or os.environ.get("GROQ_API_KEY")
        if not target_key:
            raise ValueError(
                "Missing Groq API Credential Key Token. Supply via environment variable or secrets storage configuration framework profiles."
            )
        self._client = Groq(api_key=target_key)

    def run_triage_completion(
        self,
        context_summary: dict,
        matched_hypotheses: list,
        retrieved_playbooks: str,
    ) -> VerificationResult:
        """Executes a zero-shot structured JSON completion schema targeting the core verification structure."""
        system_prompt = (
            "You are a Senior Principal Payments Incident Commander within a card-issuing bank's technical command operations center.\n"
            "Your objective is to ingest statistical anomalies flagged by deterministic detection logic, evaluate vector-retrieved playbooks, "
            "and construct a verified triage assessment data contract object structure.\n\n"
            "CRITICAL OPERATIONAL RULES:\n"
            "1. You do NOT calculate statistical parameters. Ground yourself strictly within provided pre-calculated metrics values data blocks.\n"
            "2. Ensure complete alignment with structural fields inside the output schema requirements configuration template.\n"
            "3. Output validation rules require valid standard JSON structure compilation format frames.\n"
        )

        user_payload = {
            "telemetry_metrics_aggregates": context_summary,
            "deterministic_rule_engine_matches": matched_hypotheses,
            "retrieved_vector_playbook_references": retrieved_playbooks,
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, indent=2)},
        ]

        # Use Groq's JSON Object routing protocol mechanisms
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
                raw_txt = resp.choices[0].message.content
                parsed_data = json.loads(raw_txt)
                return VerificationResult(**parsed_data)
            except RateLimitError as exc:
                wait = RETRY_BASE_SECS * (2 ** attempt)
                logger.warning(
                    "Groq rate limit (attempt %d/%d). Retrying in %.1fs.",
                    attempt + 1,
                    MAX_RETRIES,
                    wait,
                )
                time.sleep(wait)
                last_exc = exc
            except APIStatusError as exc:
                logger.error(
                    "Groq API error %s: %s", exc.status_code, exc.message
                )
                raise
        raise last_exc  # type: ignore[misc]

    def _stream_with_retry(
        self, messages: list, max_tokens: int
    ) -> Iterator[str]:
        """Streaming chat completion with retry. Re-starts on rate limit."""
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
                    attempt + 1,
                    MAX_RETRIES,
                    wait,
                )
                time.sleep(wait)
                last_exc = exc
            except APIStatusError as exc:
                logger.error("Groq stream API error: %s", exc)
                raise
        raise last_exc  # type: ignore[misc]
