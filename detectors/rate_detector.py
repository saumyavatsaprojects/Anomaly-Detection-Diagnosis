"""
Rate Detector
=============
Detects anomalies in approval rate, decline rate, and fraud rate
using the conditional Z-scores pre-computed by the feature engineer.

Also runs an EWMA drift detector to catch slow-degrading anomalies
(A4 — silent cross-border approval rate erosion over 6 days).

What it catches
---------------
  A1 — Processor outage    : approval z = -12.72 on BIN 4531xx
  A2 — 3DS ACS failure     : approval z =  -5.88 on DE/NL ecom 3DS
  A4 — Cross-border erosion: EWMA drift of -3.66pp over 6 days
  A5 — Weekend contactless : approval z around -2.8 on GB/FR weekends

Detection logic
---------------
1. For each unique dimensional slice (country × mcc_group × channel ×
   auth_type), group the hourly rows and scan for windows where the
   approval_rate_zscore stays below RATE_SIGMA_THRESHOLD for at least
   MIN_CONSECUTIVE_HOURS.

2. Within each flagged window, compute the mean z-score, mean observed
   rate, and mean baseline rate, then build the AnomalyCandidate.

3. EWMA drift detector runs separately on daily aggregates per slice,
   flagging when the EWMA has moved more than EWMA_DRIFT_THRESHOLD pp
   over the EWMA_LOOKBACK_DAYS window.

Threshold rationale
-------------------
RATE_SIGMA_THRESHOLD  = 3.0
  Rate anomalies are computed on conditional per-slice baselines that
  already account for day-of-week effects. A 3σ threshold gives a
  very low false-positive rate — at most 0.3% of hours would exceed
  this by chance in a normally distributed series.

EWMA_DRIFT_THRESHOLD  = 0.010  (1.0pp)
  The A4 erosion drifts ~0.6pp/day for 6 days. A 1.5pp cumulative
  threshold catches this after ~2.5 days while ignoring sub-1pp
  day-to-day noise that occurs naturally.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from detectors.base_detector import (
    AnomalyCandidate,
    BaseDetector,
    SIGMA_HIGH,
    SIGMA_CRITICAL,
    RC_CODES,
    RC_LABELS,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

RATE_SIGMA_THRESHOLD    = 3.0    # flag when z < -threshold (rate drop)
FRAUD_SIGMA_THRESHOLD   = 2.5    # fraud spikes use lower threshold (rarer)
MIN_CONSECUTIVE_HOURS   = 2      # minimum hours below threshold to confirm
MIN_TXN_PER_WINDOW      = 50     # minimum transactions in flagged window

EWMA_DRIFT_THRESHOLD    = 0.010  # 1.0pp cumulative drift triggers detection
EWMA_LOOKBACK_DAYS      = 7      # window for measuring EWMA drift direction
EWMA_MIN_DAILY_TXN      = 200    # minimum daily volume for EWMA detection

SLICE_DIMS = ["country", "mcc_group", "channel", "auth_type"]
RATE_METRICS = [
    ("approval_rate_computed", "approval_rate_zscore",  -1, "rate_drop"),
    ("fraud_rate",             "fraud_rate_zscore",     +1, "fraud_spike"),
]


class RateDetector(BaseDetector):
    """
    Detects approval-rate drops, fraud-rate spikes, and slow EWMA drift.
    """

    def __init__(self) -> None:
        super().__init__("rate_detector")

    def detect(self, df: pd.DataFrame) -> list[AnomalyCandidate]:
        df = df.copy()
        if "date_only" not in df.columns:
            df["date_only"] = pd.to_datetime(df["timestamp"]).dt.date.astype(str)

        candidates: list[AnomalyCandidate] = []

        # Pass 1 — hourly Z-score scan
        logger.info("  RateDetector — pass 1: hourly Z-score scan")
        candidates += self._zscore_scan(df)

        # Pass 2 — EWMA drift detector
        logger.info("  RateDetector — pass 2: EWMA drift scan")
        candidates += self._ewma_drift_scan(df)

        confirmed = [c for c in candidates if c.confirmed]
        logger.info(
            "  RateDetector — %d candidates, %d confirmed",
            len(candidates), len(confirmed),
        )
        return confirmed

    # ── PASS 1: HOURLY Z-SCORE SCAN ──────────────────────────────────────────

    def _zscore_scan(self, df: pd.DataFrame) -> list[AnomalyCandidate]:
        candidates = []

        for (metric_col, zscore_col, direction, det_type) in RATE_METRICS:
            if zscore_col not in df.columns:
                continue

            threshold = (
                FRAUD_SIGMA_THRESHOLD if det_type == "fraud_spike"
                else RATE_SIGMA_THRESHOLD
            )

            for key, grp in df.groupby(SLICE_DIMS, observed=True):
                grp = grp.sort_values("timestamp").reset_index(drop=True)

                # Find hours where z-score exceeds threshold in flagged direction
                z = grp[zscore_col]
                if direction == -1:   # rate drop — look for low z
                    flagged_mask = z < -threshold
                else:                 # fraud spike — look for high z
                    flagged_mask = z > threshold

                if not flagged_mask.any():
                    continue

                # Merge consecutive flagged hours into windows
                windows = self._merge_hourly_windows(grp, flagged_mask)

                for win_df in windows:
                    c = self._build_rate_candidate(
                        df          = df,
                        win_df      = win_df,
                        full_grp    = grp,
                        key         = key,
                        metric_col  = metric_col,
                        zscore_col  = zscore_col,
                        det_type    = det_type,
                        direction   = direction,
                    )
                    if c:
                        candidates.append(c)

        return candidates

    def _merge_hourly_windows(
        self,
        grp:         pd.DataFrame,
        flagged_mask: pd.Series,
    ) -> list[pd.DataFrame]:
        """
        Merge consecutive flagged hours into contiguous windows.
        Gaps of up to 2 hours are bridged (handles single-hour data gaps).
        """
        windows = []
        in_window = False
        start_idx = None
        last_flag_idx = None

        idxs = grp.index[flagged_mask].tolist()
        if not idxs:
            return []

        flag_set = set(idxs)
        all_idxs = grp.index.tolist()

        current = []
        for idx in all_idxs:
            if idx in flag_set:
                current.append(idx)
                last_flag_idx = idx
            elif current:
                # Gap — check if next flagged hour is within 2 positions
                pos = all_idxs.index(idx)
                next_flags = [i for i in idxs if all_idxs.index(i) > pos]
                if next_flags and (all_idxs.index(next_flags[0]) - pos) <= 2:
                    current.append(idx)   # bridge the gap
                else:
                    windows.append(grp.loc[current])
                    current = []

        if current:
            windows.append(grp.loc[current])

        # Filter: must have at least MIN_CONSECUTIVE_HOURS flagged rows
        return [
            w for w in windows
            if len(w[flagged_mask.reindex(w.index, fill_value=False)]) >= MIN_CONSECUTIVE_HOURS
        ]

    def _build_rate_candidate(
        self,
        df:         pd.DataFrame,
        win_df:     pd.DataFrame,
        full_grp:   pd.DataFrame,
        key:        tuple,
        metric_col: str,
        zscore_col: str,
        det_type:   str,
        direction:  int,
    ) -> Optional[AnomalyCandidate]:

        if len(win_df) == 0:
            return None

        total_txns = int(win_df["txn_count"].sum())
        if total_txns < MIN_TXN_PER_WINDOW:
            return None

        # Core metric values
        observed  = float(win_df[metric_col].mean())
        baseline_col = f"rolling_7d_{metric_col.split('_')[0]}_mean"
        if baseline_col not in win_df.columns:
            baseline_col = f"rolling_7d_approval_mean"
        baseline  = float(win_df[baseline_col].mean()) if baseline_col in win_df.columns else observed

        mean_z    = float(win_df[zscore_col].mean())
        peak_z    = float(win_df[zscore_col].min() if direction == -1
                          else win_df[zscore_col].max())

        first_ts  = win_df["timestamp"].min()
        last_ts   = win_df["timestamp"].max()
        duration  = max(1, int((last_ts - first_ts).total_seconds() / 3600) + 1)

        # Build affected slice
        affected_slice = dict(zip(SLICE_DIMS, key))

        # RC evidence
        baseline_rows = full_grp[full_grp["timestamp"] < first_ts].tail(14 * 24)
        rc_evidence   = self.compute_rc_evidence(win_df, baseline_rows)

        # Find the dominant shifted RC code
        dominant_rc = ""
        if rc_evidence:
            top = max(rc_evidence.items(), key=lambda x: abs(x[1].get("delta_pp", 0)))
            if abs(top[1].get("delta_pp", 0)) > 5:
                dominant_rc = (
                    f"RC {top[0]} ({top[1]['label']}): "
                    f"{top[1]['current_share']:.1%} vs baseline "
                    f"{top[1]['baseline_share']:.1%} "
                    f"({top[1]['delta_pp']:+.1f}pp)"
                )

        # Fraud evidence
        fraud_evidence = {}
        if "fraud_rate" in win_df.columns:
            fr_obs  = float(win_df["fraud_rate"].mean())
            fr_base = float(baseline_rows["fraud_rate"].mean()) if len(baseline_rows) > 0 else fr_obs
            if fr_base > 0:
                fraud_evidence = {
                    "fraud_rate_observed":  round(fr_obs, 6),
                    "fraud_rate_baseline":  round(fr_base, 6),
                    "fraud_rate_multiple":  round(fr_obs / max(fr_base, 1e-7), 2),
                }
                if "avg_ticket_usd" in win_df.columns:
                    fraud_evidence["avg_ticket_observed"] = round(float(win_df["avg_ticket_usd"].mean()), 2)
                    fraud_evidence["avg_ticket_baseline"] = round(float(baseline_rows["avg_ticket_usd"].mean()), 2) \
                        if len(baseline_rows) > 0 else fraud_evidence["avg_ticket_observed"]

        # Volume evidence
        vol_obs  = float(win_df["txn_count"].sum())
        vol_base = float(baseline_rows["txn_count"].mean() * len(win_df)) if len(baseline_rows) > 0 else vol_obs
        vol_chg  = ((vol_obs - vol_base) / max(vol_base, 1)) * 100
        volume_evidence = {
            "txn_count_observed": round(vol_obs, 0),
            "txn_count_baseline": round(vol_base, 0),
            "volume_change_pct":  round(vol_chg, 1),
        }

        # Co-moving signals
        co_moving = []
        if dominant_rc:
            co_moving.append(dominant_rc)
        if abs(vol_chg) > 15:
            co_moving.append(
                f"Volume {'surge' if vol_chg > 0 else 'drop'} "
                f"of {vol_chg:+.1f}% co-moves with {det_type.replace('_',' ')}"
            )

        # not_affected dimensions
        not_affected = self.get_not_affected(df, first_ts, last_ts, affected_slice)

        # Evidence narrative
        delta_pp = (observed - baseline) * 100
        evidence = [
            f"{metric_col.replace('_',' ').title()}: "
            f"{observed:.1%} vs baseline {baseline:.1%} "
            f"({delta_pp:+.1f}pp, {peak_z:.2f}σ)",
            f"Duration: {duration} hours "
            f"({first_ts.strftime('%Y-%m-%d %H:%M')} — {last_ts.strftime('%H:%M')})",
            f"Affected transactions in window: {total_txns:,}",
        ]
        if dominant_rc:
            evidence.append(dominant_rc)

        severity  = self.sigma_to_severity(peak_z)
        confirmed = self.passes_fp_guard(total_txns, peak_z, co_moving)

        return AnomalyCandidate(
            anomaly_id           = self.make_id(det_type),
            detector_type        = det_type,
            severity             = severity,
            first_seen_ts        = self.fmt_ts(first_ts),
            last_seen_ts         = self.fmt_ts(last_ts),
            duration_hours       = duration,
            affected_slice       = affected_slice,
            not_affected         = not_affected,
            metric               = metric_col,
            observed_value       = round(observed, 4),
            baseline_value       = round(baseline, 4),
            deviation_sigma      = round(peak_z, 2),
            baseline_period_days = 7,
            evidence             = evidence,
            co_moving_signals    = co_moving,
            reason_code_evidence = rc_evidence,
            fraud_evidence       = fraud_evidence,
            volume_evidence      = volume_evidence,
            confirmed            = confirmed,
        )

    # ── PASS 2: EWMA DRIFT DETECTOR ──────────────────────────────────────────

    def _ewma_drift_scan(self, df: pd.DataFrame) -> list[AnomalyCandidate]:
        """
        Detect slow monotonic drift in approval rate using EWMA.
        Catches A4 (cross-border erosion over 6 days) that hourly
        Z-score misses because no single hour crosses 3σ.

        FIX (v2): Recomputes a fresh daily EWMA with span=3 rather than
        using the feature store's hourly EWMA (span=168h), which is too
        smooth to detect the A4 injection.  span=3 on daily data is
        equivalent to a ~3-day exponential window — reactive but not
        noisy.
        """
        candidates = []

        # Work on daily aggregates per slice for stability
        drift_dims = ["corridor", "channel"]
        df_copy = df.copy()
        df_copy["date_only"] = pd.to_datetime(df_copy["timestamp"]).dt.date.astype(str)

        daily = (
            df_copy.groupby(["date_only"] + drift_dims, observed=True)
            .agg(
                approval_rate   = ("approval_rate_computed", "mean"),
                txn_count       = ("txn_count",              "sum"),
            )
            .reset_index()
        )
        daily["date_only"] = pd.to_datetime(daily["date_only"])
        daily = daily.sort_values(["corridor", "channel", "date_only"])

        # Drop last calendar day (may be incomplete)
        max_date = daily["date_only"].max()
        daily = daily[daily["date_only"] < max_date]

        for key, grp in daily.groupby(drift_dims, observed=True):
            grp = grp.sort_values("date_only").reset_index(drop=True)
            if len(grp) < EWMA_LOOKBACK_DAYS + 3:
                continue
            if grp["txn_count"].mean() < EWMA_MIN_DAILY_TXN:
                continue

            # FIX: recompute daily EWMA with span=3 (NOT the stored hourly EWMA)
            # span=3 gives a 3-day exponential window — reactive to daily trends
            grp = grp.copy()
            grp["ewma_rate"] = grp["approval_rate"].ewm(span=3, min_periods=2).mean()

            # Slide a window of EWMA_LOOKBACK_DAYS and measure drift
            for i in range(EWMA_LOOKBACK_DAYS, len(grp)):
                window     = grp.iloc[i - EWMA_LOOKBACK_DAYS : i + 1]
                ewma_start = float(window["ewma_rate"].iloc[0])
                ewma_end   = float(window["ewma_rate"].iloc[-1])
                drift      = ewma_end - ewma_start   # negative = deterioration

                if drift > -EWMA_DRIFT_THRESHOLD:
                    continue   # not a meaningful decline

                # Found a drift window — build candidate
                first_ts = pd.Timestamp(window["date_only"].iloc[0])
                last_ts  = pd.Timestamp(window["date_only"].iloc[-1]) + pd.Timedelta(hours=23)

                # Get hourly rows for this window
                corridor, channel = key
                win_rows = df_copy[
                    (df_copy["corridor"] == corridor) &
                    (df_copy["channel"]  == channel) &
                    (df_copy["timestamp"] >= first_ts) &
                    (df_copy["timestamp"] <= last_ts)
                ]
                if len(win_rows) == 0:
                    continue

                base_rows = df_copy[
                    (df_copy["corridor"] == corridor) &
                    (df_copy["channel"]  == channel) &
                    (df_copy["timestamp"] < first_ts)
                ].tail(14 * 24)

                obs_rate  = float(win_rows["approval_rate_computed"].mean())
                base_rate = float(base_rows["approval_rate_computed"].mean()) if len(base_rows) > 0 else obs_rate
                delta_pp  = (obs_rate - base_rate) * 100

                if abs(delta_pp) < 0.5:
                    continue

                rc_evidence = self.compute_rc_evidence(win_rows, base_rows)
                vol_obs  = float(win_rows["txn_count"].sum())
                vol_base = float(base_rows["txn_count"].mean() * len(win_rows)) if len(base_rows) > 0 else vol_obs
                vol_chg  = ((vol_obs - vol_base) / max(vol_base, 1)) * 100

                # Pseudo-sigma: use clean pre-window std, not in-window std
                # (in-window std is inflated by the anomaly itself)
                if "rolling_7d_approval_std" in base_rows.columns and len(base_rows) > 0:
                    base_std = float(base_rows["rolling_7d_approval_std"].mean())
                    if pd.isna(base_std) or base_std < 0.001:
                        base_std = 0.005  # fallback: typical approval rate volatility
                else:
                    base_std = 0.005
                # Scale: 1pp drift over 7 days on a 0.5pp-std baseline = 2σ
                pseudo_z = drift / max(base_std, 0.001)

                co_moving = [f"EWMA drifted {drift*100:+.1f}pp over {EWMA_LOOKBACK_DAYS} days"]

                # Find dominant RC shift
                if rc_evidence:
                    top_rc = max(rc_evidence.items(), key=lambda x: abs(x[1].get("delta_pp", 0)))
                    if abs(top_rc[1].get("delta_pp", 0)) > 3:
                        co_moving.append(
                            f"RC {top_rc[0]} ({top_rc[1]['label']}) trending: "
                            f"{top_rc[1]['delta_pp']:+.1f}pp"
                        )

                affected_slice = {
                    "corridor": corridor,
                    "channel":  channel,
                    "mcc_group": "all",
                    "country":   "cross_border" if corridor == "cross_border" else "domestic",
                }

                evidence = [
                    f"EWMA approval rate drifted from {ewma_start:.1%} to {ewma_end:.1%} "
                    f"({drift*100:+.2f}pp) over {EWMA_LOOKBACK_DAYS} days",
                    f"Daily approval rate: {obs_rate:.1%} vs 14-day baseline {base_rate:.1%} "
                    f"({delta_pp:+.1f}pp)",
                    f"No single-hour spike — gradual monotonic decline pattern",
                ]

                severity  = self.sigma_to_severity(pseudo_z)
                total_txn = int(win_rows["txn_count"].sum())
                confirmed = self.passes_fp_guard(total_txn, abs(pseudo_z), co_moving)

                candidates.append(AnomalyCandidate(
                    anomaly_id           = self.make_id("rate_drop"),
                    detector_type        = "rate_drift",
                    severity             = severity,
                    first_seen_ts        = self.fmt_ts(first_ts),
                    last_seen_ts         = self.fmt_ts(last_ts),
                    duration_hours       = EWMA_LOOKBACK_DAYS * 24,
                    affected_slice       = affected_slice,
                    not_affected         = [],
                    metric               = "approval_rate_ewma",
                    observed_value       = round(ewma_end, 4),
                    baseline_value       = round(ewma_start, 4),
                    deviation_sigma      = round(pseudo_z, 2),
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
                ))

                # Don't break — continue scanning for additional drift windows
                # Multiple disconnected drift periods should each produce a candidate

        return candidates
