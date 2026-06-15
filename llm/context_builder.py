"""
Context Builder
===============
Assembles the LLM prompt context from structured anomaly objects and the
feature store. This is the component that enforces grounding — the LLM
can only see what this module explicitly includes.

Key design decisions:
  - All numeric values are formatted to fixed precision before injection
    so the LLM cannot introduce rounding artifacts.
  - The supporting_stats dict is the authoritative record of every number
    the LLM is permitted to cite. The post-generation verifier checks against
    this same dict.
  - Every assembly method returns a dataclass, not a raw string, so callers
    can inspect what was included before sending it.
"""

import json
import re
import os as _os


def _load_config() -> dict:
    try:
        import yaml
        cfg_path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "config.yaml")
        if _os.path.exists(cfg_path):
            with open(cfg_path) as _f:
                return yaml.safe_load(_f) or {}
    except ImportError:
        pass
    return {}

_CFG = _load_config()
_FIN = _CFG.get("financial", {})
_INTERCHANGE_RATE = float(_FIN.get("interchange_rate", 0.018))
_DEFAULT_TICKET   = float(_FIN.get("default_avg_ticket_gbp", 42.0))
_CURRENCY         = str(_FIN.get("currency_symbol", "£"))
_FRAUD_COST       = float(_FIN.get("fraud_cost_per_event_gbp", 85.0))

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

from llm.prompt_templates import (
    ANOMALY_BRIEF_TEMPLATE,
    FOLLOWUP_TEMPLATE,
    OUT_OF_SCOPE_RESPONSE,
    CONTEXT_SUMMARY_TEMPLATE,
    SYSTEM_PROMPT,
    TEMPLATE_VERSION,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AssembledContext:
    """
    Complete context package ready to send to the LLM.
    Separates system prompt, brief, and conversation history so the
    caller can compose the messages array correctly.
    """
    anomaly_id: str
    system_prompt: str
    initial_brief: str
    supporting_stats: dict          # source of truth for verifier
    context_summary: str            # for question classification
    template_version: str = TEMPLATE_VERSION
    assembly_warnings: list = field(default_factory=list)


@dataclass
class FollowUpContext:
    """Context for a follow-up turn."""
    anomaly_id: str
    question_type: str
    injected_data: str              # the relevant stats for this question
    scope_notes: str
    is_out_of_scope: bool = False
    out_of_scope_response: Optional[str] = None
    template_version: str = TEMPLATE_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# QUESTION TYPE → CONTEXT FETCHER MAPPING
# Each question type specifies which keys from the anomaly brief are relevant.
# The LLM only sees these keys in the follow-up context.
# ─────────────────────────────────────────────────────────────────────────────

QUESTION_CONTEXT_MAP = {
    "slice_drilldown": {
        "keys": [
            "mcc_breakdown", "country_breakdown", "channel_breakdown",
            "auth_type_breakdown", "not_affected",
        ],
        "label": "Dimensional breakdown for this anomaly",
    },
    "time_comparison": {
        "keys": [
            "baseline_value", "observed_value", "baseline_period_days",
            "deviation_sigma", "first_seen_ts", "duration_hours",
            "prior_7d_same_slice",
        ],
        "label": "Time comparison data",
    },
    "causal_hypothesis": {
        "keys": [
            "failure_class", "failure_class_confidence", "evidence_items",
            "ruled_out_items", "reason_code_evidence", "fraud_evidence",
            "volume_evidence", "co_moving_signals",
        ],
        "label": "Causal attribution evidence",
    },
    "action_request": {
        "keys": [
            "recommended_escalation", "failure_class", "severity",
            "affected_slice", "duration_hours",
        ],
        "label": "Escalation and action data",
    },
    "metric_detail": {
        "keys": [
            "observed_value", "baseline_value", "deviation_sigma",
            "metric", "reason_code_evidence", "fraud_evidence",
            "volume_evidence", "baseline_period_days",
        ],
        "label": "Metric detail",
    },
    "scope_check": {
        "keys": [
            "data_window_start", "data_window_end", "baseline_period_days",
            "first_seen_ts", "duration_hours",
        ],
        "label": "Data availability",
    },
}

# Questions the scope guard catches before an LLM call
OUT_OF_SCOPE_PATTERNS = [
    {
        "pattern": r"last (quarter|year|month|Q[1-4])",
        "reason": "Data window is 90 days. Quarterly comparisons are not available.",
        "investigative_step": "Query the data warehouse for the equivalent period and compare approval rate trends.",
    },
    {
        "pattern": r"individual (transaction|cardholder|customer|merchant)",
        "reason": "Only aggregate hourly data is available. Individual transaction records are not in this system.",
        "investigative_step": "Use the transaction investigation system with the affected BIN range and time window.",
    },
    {
        "pattern": r"real.?time|live|right now|currently",
        "reason": "Data is batch-loaded. The most recent data point is the end of the anomaly window.",
        "investigative_step": "Check the real-time monitoring dashboard for current metrics.",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class ContextBuilder:
    """
    Assembles LLM context from a structured anomaly object.

    Usage:
        builder = ContextBuilder()
        ctx = builder.build_initial_context(anomaly_obj)
        followup = builder.build_followup_context(anomaly_obj, "Which MCCs?", "slice_drilldown")
    """

    def __init__(self, data_window_start: str = "2024-03-01",
                 data_window_end: str = "2024-05-30"):
        self.data_window_start = data_window_start
        self.data_window_end   = data_window_end

    # ── INITIAL CONTEXT ──────────────────────────────────────────────────────

    def build_initial_context(self, anomaly: dict) -> AssembledContext:
        """
        Build the complete context for the first message about an anomaly.
        The initial brief is injected as the first user turn, so the anomaly
        data is always present in history even as follow-ups accumulate.
        """
        warnings = []

        # Build supporting_stats — the authoritative allowlist for the verifier
        supporting_stats = self._build_supporting_stats(anomaly)

        # Format each section of the brief
        reason_code_section = self._format_reason_code_evidence(
            anomaly.get("reason_code_evidence", {}),
            anomaly.get("supporting_stats", {})
        )
        fraud_section  = self._format_fraud_evidence(anomaly)
        financial_impact_section = self._format_financial_impact(anomaly)
        volume_section = self._format_volume_evidence(anomaly)
        evidence_items = self._format_list("  - ", anomaly.get("evidence", []))
        ruled_out      = self._format_list("  - ", anomaly.get("ruled_out", []))
        supporting_fmt = self._format_supporting_stats(supporting_stats)

        # Validate required fields
        required = ["anomaly_id", "detector_type", "severity", "first_seen_ts",
                    "metric", "observed_value", "baseline_value", "deviation_sigma"]
        for req in required:
            if req not in anomaly:
                warnings.append(f"Missing required field: {req}")

        # Build the affected slice string
        affected_slice = anomaly.get("affected_slice", {})
        mcc_group  = affected_slice.get("mcc_group", anomaly.get("mcc_group", "all"))
        country    = affected_slice.get("country",   anomaly.get("country",   "all"))
        channel    = affected_slice.get("channel",   anomaly.get("channel",   "all"))
        auth_type  = affected_slice.get("auth_type", anomaly.get("auth_type", "all"))

        not_affected = ", ".join(anomaly.get("not_affected", ["not specified"]))

        # Assemble the brief
        # Extract batch rank from evidence
        batch_rank_str = "—"
        for ev in anomaly.get("evidence", []):
            if "Batch rank #" in ev:
                batch_rank_str = ev
                break

        # Fix 3: retrieve similar historical incidents (real RAG)
        similar_incidents_section = "No historical incidents found."
        try:
            from llm.incident_memory import IncidentMemory, format_similar_incidents_for_brief
            memory  = IncidentMemory.load()
            similar = memory.retrieve(
                anomaly,
                exclude_id=anomaly.get("anomaly_id",""),
                top_k=2, min_similarity=0.12,
            )
            similar_incidents_section = format_similar_incidents_for_brief(similar)
        except Exception as _mem_exc:
            import logging as _lg
            _lg.getLogger(__name__).warning("IncidentMemory failed: %s", _mem_exc)

        brief = ANOMALY_BRIEF_TEMPLATE.format(
            anomaly_id            = anomaly.get("anomaly_id", "UNKNOWN"),
            detector_type         = anomaly.get("detector_type", "unknown"),
            severity              = anomaly.get("severity", "unknown").upper(),
            batch_rank            = batch_rank_str,
            similar_incidents_section = similar_incidents_section,
            first_seen_ts         = anomaly.get("first_seen_ts", "unknown"),
            duration_hours        = anomaly.get("duration_hours", "unknown"),
            mcc_group             = mcc_group,
            country               = country,
            channel               = channel,
            auth_type             = auth_type,
            not_affected          = not_affected,
            metric                = anomaly.get("metric", "unknown"),
            observed_value        = self._fmt_rate(anomaly.get("observed_value")),
            baseline_value        = self._fmt_rate(anomaly.get("baseline_value")),
            deviation_sigma       = self._fmt_sigma(anomaly.get("deviation_sigma")),
            baseline_period_days  = anomaly.get("baseline_period_days", 7),
            reason_code_evidence  = reason_code_section,
            fraud_evidence        = fraud_section,
            volume_evidence       = volume_section,
            failure_class         = anomaly.get("failure_class", "undetermined"),
            failure_class_confidence = anomaly.get("failure_class_confidence", "low"),
            evidence_items        = evidence_items,
            ruled_out_items       = ruled_out,
            financial_impact_section = financial_impact_section,
            recommended_escalation = anomaly.get("recommended_escalation",
                                                  "Escalate to network operations"),
            supporting_stats_formatted = supporting_fmt,
            user_question         = "Please provide the initial diagnostic for this anomaly.",
        )

        # Build context summary for question classifier
        context_summary = CONTEXT_SUMMARY_TEMPLATE.format(
            detector_type    = anomaly.get("detector_type", "unknown"),
            mcc_group        = mcc_group,
            channel          = channel,
            country          = country,
            metric           = anomaly.get("metric", "unknown"),
            deviation_sigma  = self._fmt_sigma(anomaly.get("deviation_sigma")),
            failure_class    = anomaly.get("failure_class", "undetermined"),
            data_window_start = self.data_window_start,
            data_window_end   = self.data_window_end,
        )

        return AssembledContext(
            anomaly_id       = anomaly.get("anomaly_id", "UNKNOWN"),
            system_prompt    = SYSTEM_PROMPT,
            initial_brief    = brief,
            supporting_stats = supporting_stats,
            context_summary  = context_summary,
            assembly_warnings = warnings,
        )

    # ── FOLLOW-UP CONTEXT ────────────────────────────────────────────────────

    def build_followup_context(
        self,
        anomaly: dict,
        user_question: str,
        question_type: str,
    ) -> FollowUpContext:
        """
        Build context for a follow-up question.
        Only injects the stats relevant to the question type.
        Returns an out-of-scope context if the scope guard fires.
        """
        # Check scope guard first
        scope_block = self._check_scope(user_question, anomaly)
        if scope_block:
            return FollowUpContext(
                anomaly_id           = anomaly.get("anomaly_id", "UNKNOWN"),
                question_type        = "out_of_scope",
                injected_data        = "",
                scope_notes          = "",
                is_out_of_scope      = True,
                out_of_scope_response = scope_block,
            )

        # Get the keys relevant to this question type
        ctx_config   = QUESTION_CONTEXT_MAP.get(question_type, QUESTION_CONTEXT_MAP["metric_detail"])
        relevant_keys = ctx_config["keys"]
        supporting   = self._build_supporting_stats(anomaly)

        # Build the relevant data block — only the keys that apply
        relevant_lines = []
        for key in relevant_keys:
            if key in supporting:
                val = supporting[key]
                if isinstance(val, dict):
                    relevant_lines.append(f"\n{key}:")
                    for k, v in val.items():
                        relevant_lines.append(f"  {k}: {v}")
                elif isinstance(val, list):
                    relevant_lines.append(f"\n{key}:")
                    for item in val:
                        relevant_lines.append(f"  - {item}")
                else:
                    relevant_lines.append(f"{key}: {val}")
            elif key in anomaly:
                # Fallback: try the raw anomaly object
                val = anomaly[key]
                if isinstance(val, (dict, list)):
                    relevant_lines.append(f"\n{key}: {json.dumps(val, indent=2)}")
                else:
                    relevant_lines.append(f"{key}: {val}")

        relevant_data = "\n".join(relevant_lines) if relevant_lines else \
            "No additional data available for this question type."

        # Scope notes
        scope_notes = self._build_scope_notes(question_type, anomaly)

        injected = FOLLOWUP_TEMPLATE.format(
            question_type    = question_type,
            anomaly_id       = anomaly.get("anomaly_id", "UNKNOWN"),
            failure_class    = anomaly.get("failure_class", "undetermined"),
            severity         = anomaly.get("severity", "unknown").upper(),
            first_seen_ts    = anomaly.get("first_seen_ts", "unknown"),
            financial_impact_section = self._format_financial_impact(anomaly),
            relevant_data    = relevant_data,
            data_window_start = self.data_window_start,
            data_window_end   = self.data_window_end,
            scope_notes      = scope_notes,
            user_question    = user_question,
        )

        return FollowUpContext(
            anomaly_id    = anomaly.get("anomaly_id", "UNKNOWN"),
            question_type = question_type,
            injected_data = injected,
            scope_notes   = scope_notes,
        )

    # ── SUPPORTING STATS BUILDER ─────────────────────────────────────────────

    def _build_supporting_stats(self, anomaly: dict) -> dict:
        """
        Build the flat supporting_stats dict — every number the LLM is
        allowed to cite. Values are pre-formatted to fixed precision.
        This dict is also used by the post-generation verifier.
        """
        stats = {}
        affected_slice = anomaly.get("affected_slice", {})

        # Core metric values
        stats["approval_rate_observed"] = self._fmt_rate(anomaly.get("observed_value"))
        stats["approval_rate_baseline"] = self._fmt_rate(anomaly.get("baseline_value"))
        stats["deviation_sigma"]        = self._fmt_sigma(anomaly.get("deviation_sigma"))
        stats["duration_hours"]         = str(anomaly.get("duration_hours", "unknown"))
        stats["baseline_period_days"]   = str(anomaly.get("baseline_period_days", 7))
        stats["first_seen_ts"]          = str(anomaly.get("first_seen_ts", "unknown"))
        stats["severity"]               = str(anomaly.get("severity", "unknown"))
        stats["failure_class"]          = str(anomaly.get("failure_class", "undetermined"))
        stats["failure_class_confidence"] = str(anomaly.get("failure_class_confidence", "low"))

        # Affected slice dimensions
        stats["affected_mcc_group"]  = str(affected_slice.get("mcc_group", "all"))
        stats["affected_country"]    = str(affected_slice.get("country", "all"))
        stats["affected_channel"]    = str(affected_slice.get("channel", "all"))
        stats["affected_auth_type"]  = str(affected_slice.get("auth_type", "all"))

        # Not affected dimensions
        not_affected = anomaly.get("not_affected", [])
        stats["not_affected"] = ", ".join(not_affected) if not_affected else "not specified"

        # Reason code evidence
        rc_evidence = anomaly.get("reason_code_evidence", {})
        for rc_code, rc_data in rc_evidence.items():
            if isinstance(rc_data, dict):
                key_prefix = f"rc_{rc_code}"
                if "current_share" in rc_data:
                    stats[f"{key_prefix}_current_share"] = self._fmt_pct(rc_data["current_share"])
                if "baseline_share" in rc_data:
                    stats[f"{key_prefix}_baseline_share"] = self._fmt_pct(rc_data["baseline_share"])
                if "delta_pp" in rc_data:
                    stats[f"{key_prefix}_delta_pp"] = f"{rc_data['delta_pp']:+.1f}pp"
                if "label" in rc_data:
                    stats[f"{key_prefix}_label"] = str(rc_data["label"])

        # Fraud evidence
        fraud = anomaly.get("fraud_evidence", {})
        if fraud:
            if "fraud_rate_observed" in fraud:
                stats["fraud_rate_observed"] = f"{float(fraud['fraud_rate_observed']):.5f}"
            if "fraud_rate_baseline" in fraud:
                stats["fraud_rate_baseline"] = f"{float(fraud['fraud_rate_baseline']):.5f}"
            if "fraud_rate_multiple" in fraud:
                stats["fraud_rate_multiple"] = f"{float(fraud['fraud_rate_multiple']):.1f}x"
            if "avg_ticket_observed" in fraud:
                stats["avg_ticket_observed"] = f"£{float(fraud['avg_ticket_observed']):.2f}"
            if "avg_ticket_baseline" in fraud:
                stats["avg_ticket_baseline"] = f"£{float(fraud['avg_ticket_baseline']):.2f}"

        # Volume evidence
        volume = anomaly.get("volume_evidence", {})
        if volume:
            if "txn_count_observed" in volume:
                stats["txn_count_observed"] = f"{int(volume['txn_count_observed']):,}"
            if "txn_count_baseline" in volume:
                stats["txn_count_baseline"] = f"{int(volume['txn_count_baseline']):,}"
            if "volume_change_pct" in volume:
                stats["volume_change_pct"] = f"{float(volume['volume_change_pct']):+.1f}%"

        # Financial impact — compute revenue at risk and interchange loss
        # Revenue at risk = declined transactions * avg ticket size
        # Interchange loss = revenue at risk * 1.8% (typical interchange rate)
        fraud_ev = anomaly.get("fraud_evidence", {})
        vol_ev   = anomaly.get("volume_evidence", {})
        try:
            txn_obs  = float(vol_ev.get("txn_count_observed", 0))
            txn_base = float(vol_ev.get("txn_count_baseline", 0))
            # Declined txns = difference between observed and baseline approval
            obs_rate  = float(anomaly.get("observed_value", 0))
            base_rate = float(anomaly.get("baseline_value", 0))
            if base_rate > 0 and obs_rate < base_rate and txn_obs > 0:
                declined_excess = txn_obs * (base_rate - obs_rate)
                avg_ticket = float(
                    fraud_ev.get("avg_ticket_observed") or
                    fraud_ev.get("avg_ticket_baseline") or 42.0
                )
                revenue_at_risk = declined_excess * avg_ticket
                interchange_loss = revenue_at_risk * _INTERCHANGE_RATE
                stats["declined_transactions_excess"] = f"{int(declined_excess):,}"
                stats["avg_ticket_size"]   = f"£{avg_ticket:.2f}"
                stats["revenue_at_risk"]   = f"£{revenue_at_risk:,.2f}"
                stats["interchange_loss"]  = f"£{interchange_loss:,.2f}"
        except Exception:
            pass

        # Evidence and ruled-out lists (for causal hypothesis questions)
        stats["evidence_items"] = anomaly.get("evidence", [])
        stats["ruled_out_items"] = anomaly.get("ruled_out", [])

        # Escalation
        stats["recommended_escalation"] = str(
            anomaly.get("recommended_escalation", "Escalate to network operations")
        )

        # Dimensional breakdowns (if present)
        for key in ["mcc_breakdown", "country_breakdown", "channel_breakdown",
                    "auth_type_breakdown", "co_moving_signals", "prior_7d_same_slice"]:
            if key in anomaly:
                stats[key] = anomaly[key]

        # Data window
        stats["data_window_start"] = self.data_window_start
        stats["data_window_end"]   = self.data_window_end

        return stats

    # ── SCOPE GUARD ──────────────────────────────────────────────────────────

    def _check_scope(self, question: str, anomaly: dict) -> Optional[str]:
        """
        Check whether the question asks for data outside scope.
        Returns a pre-built refusal string if so, None otherwise.
        The LLM is NOT called when this returns a string.
        """
        q_lower = question.lower()
        for pattern_def in OUT_OF_SCOPE_PATTERNS:
            if re.search(pattern_def["pattern"], q_lower):
                return OUT_OF_SCOPE_RESPONSE.format(
                    what_i_have = (
                        f"90-day hourly aggregates from {self.data_window_start} "
                        f"to {self.data_window_end}, for the {anomaly.get('detector_type', 'detected')} "
                        f"anomaly at {anomaly.get('first_seen_ts', 'unknown')}"
                    ),
                    what_was_asked    = question,
                    reason            = pattern_def["reason"],
                    investigative_step = pattern_def["investigative_step"],
                )
        return None

    # ── FORMATTING HELPERS ───────────────────────────────────────────────────

    @staticmethod
    def _fmt_rate(value) -> str:
        if value is None:
            return "N/A"
        return f"{float(value):.1%}"

    @staticmethod
    def _fmt_sigma(value) -> str:
        if value is None:
            return "N/A"
        return f"{float(value):.2f}"

    @staticmethod
    def _fmt_pct(value) -> str:
        if value is None:
            return "N/A"
        return f"{float(value):.1%}"

    @staticmethod
    def _format_list(prefix: str, items: list) -> str:
        if not items:
            return f"{prefix}(none)"
        return "\n".join(f"{prefix}{item}" for item in items)

    def _format_reason_code_evidence(
        self, rc_evidence: dict, supporting_stats: dict
    ) -> str:
        if not rc_evidence:
            return "  No significant reason code shift detected."
        lines = []
        for code, data in rc_evidence.items():
            if isinstance(data, dict):
                label    = data.get("label", f"Code {code}")
                current  = self._fmt_pct(data.get("current_share"))
                baseline = self._fmt_pct(data.get("baseline_share"))
                delta    = data.get("delta_pp")
                delta_str = f"{delta:+.1f}pp" if delta is not None else ""
                lines.append(
                    f"  RC {code} ({label}): {current} (baseline: {baseline}) {delta_str}"
                )
        return "\n".join(lines) if lines else "  No reason code data."

    def _format_financial_impact(self, anomaly: dict) -> str:
        """Compute and format the financial impact of the anomaly for the LLM brief."""
        try:
            obs_rate  = float(anomaly.get("observed_value", 0))
            base_rate = float(anomaly.get("baseline_value", 0))
            vol_ev    = anomaly.get("volume_evidence", {})
            fraud_ev  = anomaly.get("fraud_evidence", {})
            txn_obs   = float(vol_ev.get("txn_count_observed", 0))

            avg_ticket = float(
                fraud_ev.get("avg_ticket_observed") or
                fraud_ev.get("avg_ticket_baseline") or _DEFAULT_TICKET
            )

            lines = [f"  Avg ticket size: £{avg_ticket:.2f}"]

            if base_rate > 0 and obs_rate < base_rate and txn_obs > 0:
                declined_excess = txn_obs * (base_rate - obs_rate)
                revenue_at_risk = declined_excess * avg_ticket
                interchange_loss = revenue_at_risk * _INTERCHANGE_RATE
                lines += [
                    f"  Declined transactions (excess vs baseline): {int(declined_excess):,}",
                    f"  Revenue at risk (declined txns × avg ticket): £{revenue_at_risk:,.2f}",
                    f"  Interchange loss (revenue at risk × 1.8%): £{interchange_loss:,.2f}",
                ]
            else:
                lines.append("  Revenue impact not quantifiable for this anomaly type.")

            return "\n".join(lines)
        except Exception:
            return "  Financial impact data unavailable."

    def _format_fraud_evidence(self, anomaly: dict) -> str:
        fraud = anomaly.get("fraud_evidence", {})
        if not fraud:
            return "  Fraud rate within normal bounds — not a fraud signal."
        lines = []
        if "fraud_rate_observed" in fraud:
            multiple = fraud.get("fraud_rate_multiple", "N/A")
            lines.append(
                f"  Fraud rate: {self._fmt_pct(fraud['fraud_rate_observed'])} "
                f"(baseline: {self._fmt_pct(fraud.get('fraud_rate_baseline'))}, "
                f"{multiple}x baseline)"
            )
        if "avg_ticket_observed" in fraud:
            lines.append(
                f"  Avg ticket: £{float(fraud['avg_ticket_observed']):.2f} "
                f"(baseline: £{float(fraud.get('avg_ticket_baseline', 0)):.2f})"
            )
        return "\n".join(lines) if lines else "  No fraud anomaly."

    def _format_volume_evidence(self, anomaly: dict) -> str:
        volume = anomaly.get("volume_evidence", {})
        if not volume:
            return "  Transaction volume within normal bounds."
        lines = []
        if "txn_count_observed" in volume:
            change = volume.get("volume_change_pct", 0)
            lines.append(
                f"  Transaction count: {int(volume['txn_count_observed']):,} "
                f"(baseline: {int(volume.get('txn_count_baseline', 0)):,}, "
                f"{float(change):+.1f}%)"
            )
        if "volume_interpretation" in volume:
            lines.append(f"  Interpretation: {volume['volume_interpretation']}")
        return "\n".join(lines) if lines else "  No volume anomaly."

    def _format_supporting_stats(self, stats: dict) -> str:
        lines = []
        for key, value in stats.items():
            if isinstance(value, list):
                lines.append(f"  {key}: {json.dumps(value)}")
            elif isinstance(value, dict):
                lines.append(f"  {key}: {json.dumps(value)}")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def _build_scope_notes(self, question_type: str, anomaly: dict) -> str:
        notes = []
        if question_type == "time_comparison":
            notes.append(
                f"The dataset covers {self.data_window_start} to {self.data_window_end}. "
                f"Comparisons beyond this window are not available."
            )
        if question_type == "slice_drilldown":
            not_affected = anomaly.get("not_affected", [])
            if not_affected:
                notes.append(
                    f"Confirmed NOT affected: {', '.join(not_affected)}. "
                    f"Do not speculate about unaffected dimensions."
                )
        return " ".join(notes) if notes else "No additional scope constraints."
