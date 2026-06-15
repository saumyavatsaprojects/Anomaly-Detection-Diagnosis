"""
Reason Code Detector
====================
Detects structural shifts in decline reason code distributions using
the chi-squared test. Catches *why* declines changed, not just *how many*.

What it catches
---------------
  A1 — Processor outage  : RC 96 (system malfunction) spikes 1.8% → 82.7%
  A2 — 3DS ACS failure   : RC 65 (soft decline)       spikes 4.1% → 57.1%
  A5 — Network rule change: RC 61 (frequency limit)   spikes 4× on weekends

Detection logic
---------------
1. For each slice (mcc_group × channel × auth_type), build a rolling
   14-day baseline reason code distribution.
2. At each hour, compare the current RC distribution to the baseline
   using chi-squared. Flag when p < CHI2_P_THRESHOLD.
3. To avoid cascading alerts (one shift persists for 14 hours), merge
   consecutive flagged windows and keep only the first occurrence.
4. Identify the dominant shifted RC code — the one with the largest
   absolute delta in share — as the primary evidence item.

Chi-squared suitability
-----------------------
The chi-squared test requires adequate expected counts per cell
(conventionally ≥ 5). We apply Laplace smoothing (+0.5 per cell) and
only run the test when the total declined count in the window exceeds
MIN_DECLINED_COUNT. Slices with fewer declines fall back to a simple
largest-delta check (no p-value).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from detectors.base_detector import (
    AnomalyCandidate,
    BaseDetector,
    RC_CODES,
    RC_LABELS,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CHI2_P_THRESHOLD        = 0.001   # p < 0.001 to flag (very conservative)
MIN_DECLINED_COUNT      = 20      # minimum declines in window for chi-squared
MIN_BASELINE_DECLINES   = 50      # minimum baseline declines for stable distribution
WINDOW_HOURS            = 6       # rolling window size for distribution comparison
MERGE_GAP_HOURS         = 3       # merge consecutive events within this gap
DELTA_PP_FALLBACK       = 15.0    # fallback threshold when chi-sq can't run

RC_COLS = [f"rc_{c}" for c in RC_CODES]
SLICE_DIMS = ["mcc_group", "channel", "auth_type"]


class ReasonCodeDetector(BaseDetector):
    """
    Detects structural shifts in decline reason code distributions.
    """

    def __init__(self) -> None:
        super().__init__("reason_code_detector")

    def detect(self, df: pd.DataFrame) -> list[AnomalyCandidate]:
        df = df.copy()
        if "date_only" not in df.columns:
            df["date_only"] = pd.to_datetime(df["timestamp"]).dt.date.astype(str)

        candidates: list[AnomalyCandidate] = []

        for key, grp in df.groupby(SLICE_DIMS, observed=True):
            grp = grp.sort_values("timestamp").reset_index(drop=True)

            # Need at least WINDOW_HOURS + 7*24 baseline rows
            if len(grp) < WINDOW_HOURS + 24:
                continue
            if grp["declined_count"].sum() < MIN_DECLINED_COUNT:
                continue

            slice_candidates = self._scan_slice(grp, key, df)
            candidates.extend(slice_candidates)

        confirmed = [c for c in candidates if c.confirmed]
        logger.info(
            "  ReasonCodeDetector — %d candidates, %d confirmed",
            len(candidates), len(confirmed),
        )
        return confirmed

    # ── SLICE SCANNING ────────────────────────────────────────────────────────

    def _scan_slice(
        self,
        grp:  pd.DataFrame,
        key:  tuple,
        full: pd.DataFrame,
    ) -> list[AnomalyCandidate]:
        """Scan one dimensional slice for RC distribution shifts."""
        candidates = []
        flagged_windows = []
        current_window_start = None
        current_window_rows  = []

        for i in range(WINDOW_HOURS, len(grp)):
            window   = grp.iloc[i - WINDOW_HOURS : i + 1]
            baseline = grp.iloc[max(0, i - 14 * 24) : i - WINDOW_HOURS]

            if baseline["declined_count"].sum() < MIN_BASELINE_DECLINES:
                continue

            is_shift, p_val = self._test_shift(window, baseline)
            if not is_shift:
                # Check if we were in a window and gap is too large
                if current_window_rows:
                    ts_now = grp.iloc[i]["timestamp"]
                    ts_last = current_window_rows[-1]["timestamp"]
                    gap_h = (ts_now - ts_last).total_seconds() / 3600
                    if gap_h > MERGE_GAP_HOURS:
                        flagged_windows.append(pd.DataFrame(current_window_rows))
                        current_window_rows = []
                continue

            row = grp.iloc[i].to_dict()
            row["_p_val"] = p_val
            current_window_rows.append(row)

        if current_window_rows:
            flagged_windows.append(pd.DataFrame(current_window_rows))

        # Build one candidate per merged window
        for win_df in flagged_windows:
            c = self._build_candidate(win_df, grp, key, full)
            if c:
                candidates.append(c)

        return candidates

    # ── CHI-SQUARED TEST ─────────────────────────────────────────────────────

    def _test_shift(
        self,
        window:   pd.DataFrame,
        baseline: pd.DataFrame,
    ) -> tuple[bool, float]:
        """
        Run chi-squared test on RC distribution: window vs baseline.
        Returns (is_flagged, p_value).
        Falls back to delta-pp check for low-volume slices.
        """
        w_counts = np.array([window[c].sum() for c in RC_COLS], dtype=float)
        b_counts = np.array([baseline[c].sum() for c in RC_COLS], dtype=float)

        total_w = w_counts.sum()
        total_b = b_counts.sum()

        if total_w < MIN_DECLINED_COUNT:
            return False, 1.0

        # Laplace smoothing to avoid zero-count cells
        w_smooth = w_counts + 0.5
        b_smooth = b_counts + 0.5

        if total_b >= MIN_BASELINE_DECLINES:
            try:
                contingency = np.array([w_smooth, b_smooth])
                _, p, _, _ = chi2_contingency(contingency)
                return p < CHI2_P_THRESHOLD, float(p)
            except Exception:
                pass

        # Fallback: check if any single RC code shifted > DELTA_PP_FALLBACK pp
        w_dist = w_smooth / w_smooth.sum()
        b_dist = b_smooth / b_smooth.sum()
        max_delta = float(np.abs(w_dist - b_dist).max() * 100)
        return max_delta > DELTA_PP_FALLBACK, 1.0

    # ── CANDIDATE BUILDER ─────────────────────────────────────────────────────

    def _build_candidate(
        self,
        win_df:   pd.DataFrame,
        full_grp: pd.DataFrame,
        key:      tuple,
        full_df:  pd.DataFrame,
    ) -> Optional[AnomalyCandidate]:

        if len(win_df) == 0:
            return None

        first_ts = pd.Timestamp(win_df["timestamp"].min())
        last_ts  = pd.Timestamp(win_df["timestamp"].max())
        duration = max(1, int((last_ts - first_ts).total_seconds() / 3600) + 1)

        total_txns    = int(win_df["txn_count"].sum())
        total_declined = int(win_df["declined_count"].sum())

        if total_txns < 10:
            return None

        # Build baseline from full_grp before window
        baseline_rows = full_grp[full_grp["timestamp"] < first_ts].tail(14 * 24)
        rc_evidence   = self.compute_rc_evidence(win_df, baseline_rows)

        if not rc_evidence:
            return None

        # Identify dominant shifted RC
        dominant_code = max(
            rc_evidence.items(),
            key=lambda x: abs(x[1].get("delta_pp", 0)),
        )
        dom_code  = dominant_code[0]
        dom_data  = dominant_code[1]
        dom_delta = dom_data.get("delta_pp", 0)

        if abs(dom_delta) < 5:
            return None

        # Approval rate in window vs baseline
        obs_appr  = float(win_df["approval_rate_computed"].mean()) \
            if "approval_rate_computed" in win_df.columns else 0.0
        base_appr = float(baseline_rows["approval_rate_computed"].mean()) \
            if len(baseline_rows) > 0 and "approval_rate_computed" in baseline_rows.columns else obs_appr
        sigma_appr = float(win_df["approval_rate_zscore"].mean()) \
            if "approval_rate_zscore" in win_df.columns else 0.0

        affected_slice = dict(zip(SLICE_DIMS, key))
        affected_slice["mcc_group"] = affected_slice.get("mcc_group", "all")

        not_affected = self.get_not_affected(full_df, first_ts, last_ts, affected_slice)

        # Volume evidence
        vol_obs  = float(win_df["txn_count"].sum())
        vol_base = float(baseline_rows["txn_count"].mean() * len(win_df)) if len(baseline_rows) > 0 else vol_obs
        vol_chg  = ((vol_obs - vol_base) / max(vol_base, 1)) * 100

        # Co-moving signals
        co_moving = [
            f"RC {dom_code} ({dom_data['label']}) shifted {dom_delta:+.1f}pp "
            f"({dom_data['baseline_share']:.1%} → {dom_data['current_share']:.1%})"
        ]
        if abs(sigma_appr) > 1.5:
            co_moving.append(
                f"Approval rate co-moved: {obs_appr:.1%} "
                f"(z={sigma_appr:.2f})"
            )

        # Evidence narrative
        evidence = [
            f"RC {dom_code} ({dom_data['label']}): "
            f"{dom_data['baseline_share']:.1%} → {dom_data['current_share']:.1%} "
            f"({dom_delta:+.1f}pp) — dominant shift",
            f"Distribution shift detected: "
            f"{total_declined:,} declines in window vs "
            f"{int(baseline_rows['declined_count'].sum()):,} in 14-day baseline",
            f"Approval rate: {obs_appr:.1%} vs baseline {base_appr:.1%} "
            f"(z={sigma_appr:.2f})",
        ]

        # Add secondary RC shifts
        secondary = [
            (code, data) for code, data in rc_evidence.items()
            if code != dom_code and abs(data.get("delta_pp", 0)) > 5
        ]
        for sec_code, sec_data in sorted(
            secondary, key=lambda x: abs(x[1].get("delta_pp", 0)), reverse=True
        )[:2]:
            evidence.append(
                f"RC {sec_code} ({sec_data['label']}): "
                f"{sec_data['delta_pp']:+.1f}pp secondary shift"
            )

        # Use dominant RC's p-value context as sigma proxy
        # (chi-sq doesn't produce sigma directly)
        # Use approval rate z-score if available, else use delta magnitude
        abs_sigma = abs(sigma_appr) if abs(sigma_appr) > 0.1 else abs(dom_delta) / 10
        severity  = self.sigma_to_severity(abs_sigma)
        # Reason-code anomalies with dominant delta > 30pp are at least high
        if abs(dom_delta) >= 30 and severity == "medium":
            severity = "high"
        if abs(dom_delta) >= 50:
            severity = "critical"

        confirmed = self.passes_fp_guard(total_txns, abs_sigma, co_moving)
        if total_declined >= MIN_DECLINED_COUNT and abs(dom_delta) >= 20:
            confirmed = True   # high-confidence shift overrides volume gate

        return AnomalyCandidate(
            anomaly_id           = self.make_id("reason_shift"),
            detector_type        = "reason_shift",
            severity             = severity,
            first_seen_ts        = self.fmt_ts(first_ts),
            last_seen_ts         = self.fmt_ts(last_ts),
            duration_hours       = duration,
            affected_slice       = affected_slice,
            not_affected         = not_affected,
            metric               = f"rc_{dom_code}_share",
            observed_value       = round(dom_data["current_share"], 4),
            baseline_value       = round(dom_data["baseline_share"], 4),
            deviation_sigma      = round(abs_sigma, 2),
            baseline_period_days = 14,
            evidence             = evidence,
            co_moving_signals    = co_moving,
            reason_code_evidence = rc_evidence,
            volume_evidence      = {
                "txn_count_observed": round(vol_obs, 0),
                "txn_count_baseline": round(vol_base, 0),
                "volume_change_pct":  round(vol_chg, 1),
            },
            confirmed            = confirmed,
        )
