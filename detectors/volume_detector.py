"""
Volume Detector
===============
Detects unexpected surges or drops in transaction volume using
STL decomposition residuals (or rolling-mean residuals as fallback).

What it catches
---------------
  A1 — Processor BIN outage: retry storm causes +38% volume spike
       on BIN 4531xx between 10:00–18:00 on Mar 23.
  A3 — Fraud attack: +47% volume surge on GB grocery ecom CNP
       as attackers enumerate card credentials.

Detection logic
---------------
1. Build a deduplicated daily series per {mcc_group × channel × bin_bucket?}.
2. Recompute a clean residual Z-score with a 14-day rolling std window
   (the feature_store stl_residual_zscore can have warm-up artefacts).
3. Flag any day where |residual_zscore| > VOLUME_SIGMA_THRESHOLD.
4. For flagged days, identify the peak hourly window (±6h around peak).
5. Compute the volume evidence block and co-moving approval rate signal.

Threshold rationale
-------------------
VOLUME_SIGMA_THRESHOLD = 2.5
  Chosen to catch genuine volume anomalies while avoiding noise from
  normal Friday-vs-Monday variation (the STL residual has already
  removed the weekly seasonal component, so this threshold is tighter
  than an equivalent raw-volume threshold).

BIN-level detection
-------------------
The processor outage (A1) is a BIN-scoped event. Transaction volume
at the mcc_group level is diluted because only 4531xx cards are affected
(~26% of issued cards). The volume detector therefore runs two passes:
  Pass 1 — mcc_group × channel   (catches fraud attack A3)
  Pass 2 — bin_bucket × channel  (catches processor outage A1)
"""

from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from detectors.base_detector import (
    AnomalyCandidate,
    BaseDetector,
    SIGMA_HIGH,
    SIGMA_CRITICAL,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

VOLUME_SIGMA_THRESHOLD  = 2.2    # flag when |residual_zscore| > this
VOLUME_SIGMA_CONFIRM    = 2.0    # minimum to pass FP guard (lower than rate)
STD_WINDOW_DAYS         = 14     # rolling std window for residual normalisation
MIN_STD_PERIODS         = 7      # minimum periods for rolling std
MIN_DAILY_TXN           = 50     # skip slices with very low daily volumes
MIN_SERIES_DAYS         = 10     # skip series too short for meaningful residual
PEAK_WINDOW_HOURS       = 6      # hours around daily peak to define the window
MIN_HOURS_PER_DAY       = 18     # days with fewer hours are incomplete — exclude
EWMA_SPAN_DAYS          = 7      # EWMA span for co-moving drift detection


class VolumeDetector(BaseDetector):
    """
    Detects volume anomalies using STL residuals.

    Two detection passes:
      Pass 1: mcc_group × channel          — broad-scope volume events
      Pass 2: bin_bucket × channel         — BIN-scoped events (outages)
    """

    def __init__(self) -> None:
        super().__init__("volume_detector")

    def detect(self, df: pd.DataFrame) -> list[AnomalyCandidate]:
        candidates: list[AnomalyCandidate] = []

        # Ensure date_only column is present
        df = df.copy()
        if "date_only" not in df.columns:
            df["date_only"] = pd.to_datetime(df["timestamp"]).dt.date.astype(str)

        logger.info("  VolumeDetector — pass 1: mcc_group × channel")
        candidates += self._detect_pass(
            df,
            group_dims  = ["mcc_group", "channel"],
            pass_label  = "mcc_channel",
        )

        logger.info("  VolumeDetector — pass 2: bin_bucket × channel")
        candidates += self._detect_pass(
            df,
            group_dims  = ["bin_bucket", "channel"],
            pass_label  = "bin_channel",
        )

        confirmed = [c for c in candidates if c.confirmed]
        logger.info(
            "  VolumeDetector — %d candidates, %d confirmed",
            len(candidates), len(confirmed),
        )
        return confirmed

    # ── DETECTION PASS ────────────────────────────────────────────────────────

    def _detect_pass(
        self,
        df:         pd.DataFrame,
        group_dims: list[str],
        pass_label: str,
    ) -> list[AnomalyCandidate]:
        """Run one detection pass over a set of grouping dimensions."""
        # Build deduplicated daily aggregates for this pass
        daily = self._build_daily(df, group_dims)
        if daily.empty:
            return []

        candidates = []
        for key, grp in daily.groupby(group_dims, observed=True):
            grp = grp.sort_values("date_only").reset_index(drop=True)
            if len(grp) < MIN_SERIES_DAYS:
                continue
            if grp["daily_txn_count"].max() < MIN_DAILY_TXN:
                continue

            # Recompute clean residual Z-score on this deduplicated series
            grp = self._compute_residual_zscore(grp)

            # Find flagged days
            flagged = grp[grp["residual_zscore"].abs() > VOLUME_SIGMA_THRESHOLD]
            if flagged.empty:
                continue

            # Merge consecutive flagged days into single events
            events = self._merge_events(flagged)

            for event_days in events:
                candidate = self._build_candidate(
                    df         = df,
                    grp        = grp,
                    event_days = event_days,
                    group_dims = group_dims,
                    key        = key if isinstance(key, tuple) else (key,),
                    pass_label = pass_label,
                )
                if candidate:
                    candidates.append(candidate)

        return candidates

    # ── DAILY AGGREGATION ─────────────────────────────────────────────────────

    def _build_daily(
        self, df: pd.DataFrame, group_dims: list[str]
    ) -> pd.DataFrame:
        """Aggregate hourly rows to daily totals for the given dimensions."""
        agg_cols = {
            "daily_txn_count":   ("txn_count",        "sum"),
            "daily_approved":    ("approved_count",    "sum"),
            "daily_declined":    ("declined_count",    "sum"),
            "daily_fraud":       ("fraud_count",       "sum"),
            "stl_residual":      ("stl_residual",      "mean"),  # already daily
        }
        # Only include cols that exist
        agg_dict = {
            k: v for k, v in agg_cols.items()
            if v[0] in df.columns
        }

        dims_plus_date = group_dims + ["date_only"]
        # Filter to cols we actually need
        needed = list(set(group_dims + ["date_only"] +
                          [v[0] for v in agg_dict.values()]))
        needed = [c for c in needed if c in df.columns]

        daily = (
            df[needed]
            .groupby(dims_plus_date, observed=True)
            .agg(**{k: v for k, v in agg_dict.items()
                    if v[0] in df.columns})
            .reset_index()
        )
        daily["date_only"] = pd.to_datetime(daily["date_only"])
        if "daily_approved" in daily.columns and "daily_txn_count" in daily.columns:
            daily["daily_approval_rate"] = (
                daily["daily_approved"] / daily["daily_txn_count"].clip(lower=1)
            ).round(6)

        # Drop days with fewer than MIN_HOURS_PER_DAY hours of data.
        # The last calendar day of the dataset often has < 24h of rows,
        # which creates a spurious -99% volume drop on every slice.
        hours_per_day = (
            df.groupby("date_only")["timestamp"]
            .nunique()
            .reset_index()
            .rename(columns={"timestamp": "hour_count"})
        )
        hours_per_day["date_only"] = pd.to_datetime(hours_per_day["date_only"])
        daily = daily.merge(hours_per_day, on="date_only", how="left")
        daily = daily[daily["hour_count"] >= MIN_HOURS_PER_DAY].drop(columns=["hour_count"])
        return daily

    # ── RESIDUAL Z-SCORE ─────────────────────────────────────────────────────

    def _compute_residual_zscore(self, grp: pd.DataFrame) -> pd.DataFrame:
        """
        Recompute a clean residual Z-score using a 14-day rolling std.
        Uses the stl_residual computed by feature_engineer.
        """
        residual = grp["stl_residual"].copy()

        _raw_std = (
            residual
            .rolling(STD_WINDOW_DAYS, min_periods=MIN_STD_PERIODS)
            .std()
        )
        roll_std = (
            _raw_std
            .bfill()
            .ffill()
            .fillna(residual.std())
            .clip(lower=max(1.0, residual.abs().median() * 0.1))
        )
        grp = grp.copy()
        grp["residual_zscore"] = (residual / roll_std).round(4)
        return grp

    # ── EVENT MERGING ─────────────────────────────────────────────────────────

    def _merge_events(self, flagged: pd.DataFrame) -> list[list]:
        """
        Merge consecutive flagged days into single events.
        Days within 2 calendar days of each other are merged.
        Returns a list of lists, each containing the flagged rows for one event.
        """
        if flagged.empty:
            return []

        dates  = sorted(flagged["date_only"].tolist())
        events = []
        current = [dates[0]]

        for d in dates[1:]:
            gap = (d - current[-1]).days
            if gap <= 2:
                current.append(d)
            else:
                events.append(current)
                current = [d]
        events.append(current)

        # Return the flagged rows for each event
        return [
            flagged[flagged["date_only"].isin(event_dates)]
            for event_dates in events
        ]

    # ── CANDIDATE BUILDER ─────────────────────────────────────────────────────

    def _build_candidate(
        self,
        df:         pd.DataFrame,
        grp:        pd.DataFrame,
        event_days: pd.DataFrame,
        group_dims: list[str],
        key:        tuple,
        pass_label: str,
    ) -> AnomalyCandidate | None:
        """Build an AnomalyCandidate from a flagged event."""
        if event_days.empty:
            return None

        # Get the peak day (highest absolute residual zscore)
        peak_idx    = event_days["residual_zscore"].abs().idxmax()
        peak_row    = event_days.loc[peak_idx]
        peak_date   = peak_row["date_only"]
        peak_zscore = float(peak_row["residual_zscore"])

        # Baseline: 14 days before the event
        event_start = event_days["date_only"].min()
        baseline_df = grp[grp["date_only"] < event_start].tail(14)
        if baseline_df.empty:
            return None

        # Observed and baseline volume
        observed_vol = float(event_days["daily_txn_count"].mean())
        baseline_vol = float(baseline_df["daily_txn_count"].mean())
        if baseline_vol < 1:
            return None

        vol_change_pct = ((observed_vol - baseline_vol) / baseline_vol) * 100

        # Duration
        duration_days = (
            event_days["date_only"].max() - event_days["date_only"].min()
        ).days + 1

        # Build affected slice dict from group key
        affected_slice = dict(zip(group_dims, key))
        affected_slice["mcc_group"] = affected_slice.get("mcc_group", "all")
        affected_slice["channel"]   = affected_slice.get("channel",   "all")

        # Get hourly window rows for evidence
        window_start = pd.Timestamp(peak_date)
        window_end   = window_start + pd.Timedelta(days=1)

        # Filter hourly df to this event window and slice
        filt = df.copy()
        for dim, val in affected_slice.items():
            if dim in filt.columns:
                filt = filt[filt[dim] == val]
        window_rows = filt[
            (filt["timestamp"] >= window_start) &
            (filt["timestamp"] <  window_end)
        ]

        # Approval rate co-movement
        approval_comoving = ""
        if len(window_rows) > 0 and "approval_rate_computed" in window_rows.columns:
            win_appr  = window_rows["approval_rate_computed"].mean()
            base_appr = (
                filt[filt["timestamp"] < window_start]
                .tail(7 * 24)["approval_rate_computed"].mean()
            )
            if not pd.isna(base_appr) and not pd.isna(win_appr):
                appr_delta = win_appr - base_appr
                if abs(appr_delta) > 0.02:
                    direction = "drop" if appr_delta < 0 else "rise"
                    approval_comoving = (
                        f"Approval rate co-moved: "
                        f"{win_appr:.1%} vs baseline {base_appr:.1%} "
                        f"({appr_delta*100:+.1f}pp {direction})"
                    )

        # RC evidence
        if len(window_rows) > 0:
            baseline_rows = filt[filt["timestamp"] < window_start].tail(14 * 24)
            rc_evidence = self.compute_rc_evidence(window_rows, baseline_rows)
        else:
            rc_evidence = {}

        # Co-moving signals
        co_moving = []
        if approval_comoving:
            co_moving.append(approval_comoving)
        if any(
            abs(rc_evidence.get(c, {}).get("delta_pp", 0)) > 10
            for c in ["96", "91", "59"]
        ):
            top_rc = max(
                rc_evidence.items(),
                key=lambda x: abs(x[1].get("delta_pp", 0)),
                default=(None, {})
            )
            if top_rc[0]:
                co_moving.append(
                    f"RC {top_rc[0]} ({RC_LABELS_SHORT.get(top_rc[0], '')}) "
                    f"shifted {top_rc[1].get('delta_pp', 0):+.1f}pp"
                )

        # Evidence list
        direction_word = "surge" if peak_zscore > 0 else "drop"
        evidence = [
            f"Transaction volume {direction_word}: "
            f"{observed_vol:,.0f} vs baseline {baseline_vol:,.0f} "
            f"({vol_change_pct:+.1f}%)",
            f"STL residual Z-score: {peak_zscore:+.2f}σ on {peak_date.date()}",
            f"Event duration: {duration_days} day(s)",
        ]
        if approval_comoving:
            evidence.append(approval_comoving)

        # Volume evidence block
        vol_interp = _volume_interpretation(vol_change_pct, approval_comoving)
        volume_evidence = {
            "txn_count_observed":  round(observed_vol, 0),
            "txn_count_baseline":  round(baseline_vol, 0),
            "volume_change_pct":   round(vol_change_pct, 1),
            "volume_interpretation": vol_interp,
        }

        # Severity
        severity = self.sigma_to_severity(peak_zscore)

        # FP guard
        total_txns = int(event_days["daily_txn_count"].sum())
        confirmed  = self.passes_fp_guard(total_txns, peak_zscore, co_moving)

        return AnomalyCandidate(
            anomaly_id           = self.make_id("volume_spike"),
            detector_type        = "volume_spike",
            severity             = severity,
            first_seen_ts        = self.fmt_ts(window_start),
            last_seen_ts         = self.fmt_ts(window_end),
            duration_hours       = duration_days * 24,
            affected_slice       = affected_slice,
            not_affected         = [],      # filled by root_cause.py
            metric               = "txn_count",
            observed_value       = round(observed_vol, 0),
            baseline_value       = round(baseline_vol, 0),
            deviation_sigma      = round(peak_zscore, 2),
            baseline_period_days = STD_WINDOW_DAYS,
            evidence             = evidence,
            co_moving_signals    = co_moving,
            reason_code_evidence = rc_evidence,
            volume_evidence      = volume_evidence,
            confirmed            = confirmed,
        )


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

RC_LABELS_SHORT = {
    "96": "system malfunction",
    "91": "issuer inoperative",
    "59": "suspected fraud",
    "65": "soft decline",
    "05": "do not honor",
}


def _volume_interpretation(vol_change_pct: float, appr_comoving: str) -> str:
    """Generate a human-readable interpretation of the volume change."""
    if vol_change_pct > 30:
        if "drop" in appr_comoving.lower():
            return (
                f"Volume up {vol_change_pct:+.1f}% while approval rate declined — "
                "consistent with a retry storm (cardholders retrying declined transactions)."
            )
        if "rise" in appr_comoving.lower() or not appr_comoving:
            return (
                f"Volume surge of {vol_change_pct:+.1f}% with elevated approval rate — "
                "consistent with card-testing fraud (high submission volumes by attackers)."
            )
    if vol_change_pct > 10:
        return f"Moderate volume increase of {vol_change_pct:+.1f}% above baseline."
    if vol_change_pct < -20:
        return (
            f"Volume down {vol_change_pct:+.1f}% — "
            "consistent with cardholder abandonment following repeated declines."
        )
    if vol_change_pct < -10:
        return f"Moderate volume decline of {vol_change_pct:+.1f}% below baseline."
    return f"Volume change of {vol_change_pct:+.1f}% — within normal variation bounds."
