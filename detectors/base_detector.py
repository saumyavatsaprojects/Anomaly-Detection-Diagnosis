"""
Base Detector
=============
Abstract base class for all anomaly detectors.

Every detector in this package:
  1. Accepts the feature store DataFrame as input.
  2. Returns a list of AnomalyCandidate dataclass instances.
  3. Never calls the LLM — detection is purely statistical.
  4. Produces a named detector_type so the LLM grounding layer
     can always identify the source of any anomaly object.

Anomaly object structure
------------------------
Each AnomalyCandidate contains all fields required by the
root_cause.py attribution layer and the LLM context builder.
No downstream layer should need to re-query the feature store
to explain an anomaly — everything it needs is in the object.

Severity tiers
--------------
  critical  : |sigma| >= 5.0  OR  business_impact_score >= 0.8
  high      : |sigma| >= 3.5  OR  business_impact_score >= 0.5
  medium    : |sigma| >= 2.5
  low       : |sigma| >= 1.5  (informational only — not sent to LLM)

False-positive guard
--------------------
A candidate is promoted to a confirmed anomaly only when at least
one of the following is true:
  a) The affected txn_count during the window exceeds MIN_TXN_VOLUME
  b) The sigma deviation exceeds SIGMA_CONFIRM_THRESHOLD
  c) Two or more co-moving signals are present in the object
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# SHARED THRESHOLDS  (imported by all detectors)
# ─────────────────────────────────────────────────────────────────────────────

SIGMA_LOW             = 1.5
SIGMA_MEDIUM          = 2.5
SIGMA_HIGH            = 3.5
SIGMA_CRITICAL        = 5.0

MIN_TXN_VOLUME        = 100    # minimum transactions in window to confirm
SIGMA_CONFIRM         = 2.5   # minimum sigma to bypass volume gate

RC_CODES = ["05", "14", "51", "57", "59", "61", "65", "91", "96"]
RC_LABELS = {
    "05": "Do not honor",
    "14": "Invalid card number",
    "51": "Insufficient funds",
    "57": "Transaction not permitted",
    "59": "Suspected fraud",
    "61": "Exceeds withdrawal frequency limit",
    "65": "Soft decline — authentication required",
    "91": "Issuer/switch inoperative",
    "96": "System malfunction",
}


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY CANDIDATE DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnomalyCandidate:
    """
    A single flagged anomaly from one detector.

    All fields used downstream by root_cause.py and the LLM layer.
    Keep values as plain Python types (str, float, int, list, dict)
    so the object serialises cleanly to JSON.
    """
    # Identity
    anomaly_id:           str
    detector_type:        str   # "rate_drop" | "volume_spike" | "reason_shift" | "fraud_concentration"
    severity:             str   # "critical" | "high" | "medium" | "low"

    # Timing
    first_seen_ts:        str   # ISO-8601
    last_seen_ts:         str
    duration_hours:       int

    # Affected slice
    affected_slice:       dict  # {country, mcc_group, channel, auth_type, ...}
    not_affected:         list  # confirmed-unaffected dimensions

    # Core metric
    metric:               str
    observed_value:       float
    baseline_value:       float
    deviation_sigma:      float
    baseline_period_days: int   = 7

    # Supporting evidence
    evidence:             list  = field(default_factory=list)
    ruled_out:            list  = field(default_factory=list)
    co_moving_signals:    list  = field(default_factory=list)

    # Structured sub-evidence (populated by detector)
    reason_code_evidence: dict  = field(default_factory=dict)
    fraud_evidence:       dict  = field(default_factory=dict)
    volume_evidence:      dict  = field(default_factory=dict)

    # Breakdown tables (filled by root_cause.py)
    country_breakdown:    dict  = field(default_factory=dict)
    channel_breakdown:    dict  = field(default_factory=dict)
    mcc_breakdown:        dict  = field(default_factory=dict)

    # Root cause (filled by root_cause.py)
    failure_class:             str = "undetermined"
    failure_class_confidence:  str = "low"
    recommended_escalation:    str = ""

    # Validation flag
    confirmed:            bool  = False   # passes false-positive guard
    template_version:     str   = "1.1.0"

    def to_dict(self) -> dict:
        """Serialise to a plain dict (for JSON output)."""
        import dataclasses
        return dataclasses.asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# ABSTRACT BASE
# ─────────────────────────────────────────────────────────────────────────────

class BaseDetector(ABC):
    """
    Abstract base for all anomaly detectors.

    Subclasses implement detect() and return a list of AnomalyCandidate.
    The pipeline runner calls detect() and collects all candidates before
    passing them to root_cause.py.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def detect(self, df: pd.DataFrame) -> list[AnomalyCandidate]:
        """
        Run detection on the feature store DataFrame.

        Args:
            df: The full feature store, already sorted by timestamp.

        Returns:
            List of AnomalyCandidate — may be empty if nothing detected.
        """

    # ── SHARED UTILITIES ─────────────────────────────────────────────────────

    @staticmethod
    def make_id(detector_type: str) -> str:
        """Generate a short deterministic-ish anomaly ID."""
        short = uuid.uuid4().hex[:8].upper()
        prefix = {
            "rate_drop":            "RD",
            "volume_spike":         "VS",
            "reason_shift":         "RS",
            "fraud_concentration":  "FC",
        }.get(detector_type, "AN")
        return f"{prefix}-{short}"

    @staticmethod
    def sigma_to_severity(sigma: float) -> str:
        """Map absolute sigma to severity tier."""
        abs_s = abs(sigma)
        if abs_s >= SIGMA_CRITICAL:
            return "critical"
        if abs_s >= SIGMA_HIGH:
            return "high"
        if abs_s >= SIGMA_MEDIUM:
            return "medium"
        return "low"

    @staticmethod
    def passes_fp_guard(
        txn_count: int,
        sigma:     float,
        co_moving: list,
    ) -> bool:
        """
        False-positive guard.
        Returns True if the candidate should be confirmed.
        """
        if abs(sigma) >= SIGMA_CONFIRM:
            return True
        if txn_count >= MIN_TXN_VOLUME:
            return True
        if len(co_moving) >= 2:
            return True
        return False

    @staticmethod
    def fmt_ts(dt) -> str:
        """Format a timestamp to ISO-8601 string."""
        if isinstance(dt, str):
            return dt
        if hasattr(dt, "isoformat"):
            return dt.isoformat() + "Z" if "Z" not in str(dt) else str(dt)
        return str(dt)

    @staticmethod
    def compute_rc_evidence(
        window_df:   pd.DataFrame,
        baseline_df: pd.DataFrame,
    ) -> dict:
        """
        Build the reason_code_evidence dict from window vs baseline rows.
        Returns {rc_code: {label, current_share, baseline_share, delta_pp}}.
        """
        evidence = {}
        total_w = window_df["declined_count"].sum()
        total_b = baseline_df["declined_count"].sum()
        if total_w == 0:
            return evidence

        for code in RC_CODES:
            rc_col  = f"rc_{code}"
            if rc_col not in window_df.columns:
                continue
            curr_share = window_df[rc_col].sum() / max(total_w, 1)
            base_share = (
                baseline_df[rc_col].sum() / max(total_b, 1)
                if total_b > 0 else 0.0
            )
            delta = (curr_share - base_share) * 100
            if abs(delta) > 0.5 or curr_share > 0.05:
                evidence[code] = {
                    "label":          RC_LABELS.get(code, f"Code {code}"),
                    "current_share":  round(curr_share, 4),
                    "baseline_share": round(base_share, 4),
                    "delta_pp":       round(delta, 2),
                }
        return evidence

    @staticmethod
    def get_not_affected(
        df:           pd.DataFrame,
        window_start: pd.Timestamp,
        window_end:   pd.Timestamp,
        affected:     dict,
        metric_col:   str = "approval_rate_computed",
        baseline_col: str = "rolling_7d_approval_mean",
        sigma_col:    str = "approval_rate_zscore",
    ) -> list:
        """
        Identify dimensions that were NOT affected during the anomaly window.
        Returns a list of plain-language strings.
        """
        not_affected = []
        window = df[
            (df["timestamp"] >= window_start) &
            (df["timestamp"] <  window_end)
        ]

        # Check: other channels in same country
        affected_ch = affected.get("channel")
        if affected_ch:
            for ch in ["ecom", "pos", "contactless"]:
                if ch == affected_ch:
                    continue
                ch_rows = window[window["channel"] == ch]
                if len(ch_rows) == 0:
                    continue
                ch_z = ch_rows[sigma_col].mean() if sigma_col in ch_rows else 0
                ch_r = ch_rows[metric_col].mean()
                if abs(ch_z) < 1.5:
                    not_affected.append(
                        f"{ch.upper()} channel: "
                        f"{metric_col.replace('_',' ')} {ch_r:.1%} — unaffected"
                    )

        # Check: other countries
        affected_ctry = affected.get("country")
        if affected_ctry and "," not in str(affected_ctry):
            for ctry in df["country"].unique():
                if ctry == affected_ctry:
                    continue
                ctry_rows = window[window["country"] == ctry]
                if len(ctry_rows) < 5:
                    continue
                ctry_z = ctry_rows[sigma_col].mean() if sigma_col in ctry_rows else 0
                if abs(ctry_z) < 1.0:
                    ctry_r = ctry_rows[metric_col].mean()
                    not_affected.append(
                        f"{ctry}: {metric_col.replace('_',' ')} "
                        f"{ctry_r:.1%} — unaffected"
                    )

        return not_affected[:6]   # cap at 6 to keep brief manageable
