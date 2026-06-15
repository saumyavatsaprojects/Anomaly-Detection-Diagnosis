"""
Fraud Concentration Detector
=============================
Detects concentrated fraud attacks using KL divergence on the daily
fraud distribution across MCC groups per country.

What it catches
---------------
  A3 — GB grocery CNP fraud attack: grocery's share of GB fraud rises
       from 14.2% (baseline) to 33.0% (attack period) over 3 days.
       KL divergence = 0.23 (well above the 0.10 threshold).

Why KL divergence
-----------------
A fraud *rate* detector misses attacks that are spread across multiple
transaction types or that inflate the aggregate fraud rate only modestly.
KL divergence measures whether the *distribution* of fraud across MCCs
has become unusual — a fraud attack concentrated in one MCC will shift
the distribution even if the total fraud count is only mildly elevated.

Detection logic
---------------
1. For each country, build a daily distribution of fraud counts
   across MCC groups.
2. Compare the current day's distribution to a 14-day rolling baseline
   using KL divergence (Q ‖ P, where P = baseline).
3. Flag days where KL > KL_THRESHOLD.
4. On flagged days, identify the over-represented MCC group and check
   the slice-level fraud rate for confirmation.
5. Build the AnomalyCandidate with fraud evidence including the rate
   multiple and average ticket size (card-testing signature).

Laplace smoothing
-----------------
Zero counts in either distribution would make KL divergence infinite.
We add LAPLACE_ALPHA to all counts before normalising. This is standard
practice and does not materially affect the KL value when counts are large.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy

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

KL_THRESHOLD            = 0.10    # KL divergence above this is flagged
FRAUD_RATE_MULTIPLE_MIN = 1.8     # minimum fraud rate multiple to confirm
BASELINE_DAYS           = 14      # days of history for baseline distribution
MIN_FRAUD_IN_WINDOW     = 3       # minimum fraud events in window to run KL
MIN_DAILY_TXN           = 100     # minimum transactions per country-day
LAPLACE_ALPHA           = 0.5     # Laplace smoothing constant

MCC_GROUPS = [
    "grocery", "retail", "dining", "fuel",
    "digital_goods", "travel", "entertainment", "utilities",
]


class FraudConcentrationDetector(BaseDetector):
    """
    Detects MCC-concentrated fraud attacks using KL divergence.
    """

    def __init__(self) -> None:
        super().__init__("fraud_concentration_detector")

    def detect(self, df: pd.DataFrame) -> list[AnomalyCandidate]:
        df = df.copy()
        df["date_only"] = pd.to_datetime(df["timestamp"]).dt.date.astype(str)

        candidates: list[AnomalyCandidate] = []

        # Build daily fraud distribution per country
        daily = self._build_daily_fraud(df)
        if daily.empty:
            logger.info("  FraudConcentrationDetector — no data")
            return []

        # Drop last calendar day (may be incomplete)
        max_date = daily["date_only"].max()
        daily = daily[daily["date_only"] < max_date]

        for country, grp in daily.groupby("country"):
            grp = grp.sort_values("date_only").reset_index(drop=True)
            if len(grp) < BASELINE_DAYS + 1:
                continue

            country_candidates = self._scan_country(grp, country, df)
            candidates.extend(country_candidates)

        confirmed = [c for c in candidates if c.confirmed]
        logger.info(
            "  FraudConcentrationDetector — %d candidates, %d confirmed",
            len(candidates), len(confirmed),
        )
        return confirmed

    # ── DAILY FRAUD DISTRIBUTION ─────────────────────────────────────────────

    def _build_daily_fraud(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a daily fraud count matrix:
        rows = (country, date), columns = one per MCC group.
        """
        daily = (
            df.groupby(["date_only", "country", "mcc_group"], observed=True)
            .agg(
                fraud_count = ("fraud_count", "sum"),
                txn_count   = ("txn_count",   "sum"),
            )
            .reset_index()
        )
        daily["date_only"] = pd.to_datetime(daily["date_only"])

        # Pivot to wide format: one column per MCC group
        pivot = daily.pivot_table(
            index   = ["date_only", "country"],
            columns = "mcc_group",
            values  = "fraud_count",
            aggfunc = "sum",
            fill_value = 0,
        ).reset_index()

        # Ensure all MCC group columns are present
        for mcc in MCC_GROUPS:
            if mcc not in pivot.columns:
                pivot[mcc] = 0

        # Total transactions per country-day (for volume gate)
        txn_daily = (
            daily.groupby(["date_only", "country"])["txn_count"]
            .sum()
            .reset_index()
            .rename(columns={"txn_count": "total_txn"})
        )
        pivot = pivot.merge(txn_daily, on=["date_only", "country"], how="left")
        return pivot

    # ── COUNTRY SCAN ──────────────────────────────────────────────────────────

    def _scan_country(
        self,
        grp:     pd.DataFrame,
        country: str,
        full_df: pd.DataFrame,
    ) -> list[AnomalyCandidate]:
        """Scan daily fraud distributions for one country."""
        candidates = []
        flagged_days = []

        for i in range(BASELINE_DAYS, len(grp)):
            today    = grp.iloc[i]
            baseline = grp.iloc[max(0, i - BASELINE_DAYS) : i]

            # Total fraud in window
            today_fraud = int(sum(today.get(mcc, 0) for mcc in MCC_GROUPS))
            if today_fraud < MIN_FRAUD_IN_WINDOW:
                continue
            if today.get("total_txn", 0) < MIN_DAILY_TXN:
                continue

            # Build distributions with Laplace smoothing
            p_dist, q_dist = self._build_distributions(today, baseline)

            # KL divergence: how much does today differ from baseline?
            kl = float(entropy(p_dist, q_dist))

            if kl < KL_THRESHOLD:
                continue

            # Identify the over-represented MCC
            deltas = p_dist - q_dist
            dominant_mcc_idx = int(np.argmax(deltas))
            dominant_mcc     = MCC_GROUPS[dominant_mcc_idx]

            flagged_days.append({
                "date_only":    today["date_only"],
                "kl":           kl,
                "dominant_mcc": dominant_mcc,
                "p_dist":       p_dist,
                "q_dist":       q_dist,
                "today_fraud":  today_fraud,
            })

        if not flagged_days:
            return []

        # Merge consecutive flagged days into events
        events = self._merge_flagged_days(flagged_days)

        for event in events:
            c = self._build_candidate(event, country, full_df)
            if c:
                candidates.append(c)

        return candidates

    # ── DISTRIBUTION BUILDER ─────────────────────────────────────────────────

    def _build_distributions(
        self,
        today:    pd.Series,
        baseline: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build normalised fraud distributions with Laplace smoothing."""
        # Today's distribution
        p_counts = np.array(
            [float(today.get(mcc, 0)) + LAPLACE_ALPHA for mcc in MCC_GROUPS]
        )
        # Baseline distribution (mean across baseline days)
        q_counts = np.array(
            [float(baseline[mcc].mean()) + LAPLACE_ALPHA
             if mcc in baseline.columns else LAPLACE_ALPHA
             for mcc in MCC_GROUPS]
        )
        p_dist = p_counts / p_counts.sum()
        q_dist = q_counts / q_counts.sum()
        return p_dist, q_dist

    # ── EVENT MERGER ──────────────────────────────────────────────────────────

    def _merge_flagged_days(
        self, flagged_days: list[dict]
    ) -> list[list[dict]]:
        """Merge consecutive flagged days into events (gap ≤ 2 days)."""
        if not flagged_days:
            return []
        events  = []
        current = [flagged_days[0]]
        for fd in flagged_days[1:]:
            gap = (fd["date_only"] - current[-1]["date_only"]).days
            if gap <= 2:
                current.append(fd)
            else:
                events.append(current)
                current = [fd]
        events.append(current)
        return events

    # ── CANDIDATE BUILDER ─────────────────────────────────────────────────────

    def _build_candidate(
        self,
        event:   list[dict],
        country: str,
        full_df: pd.DataFrame,
    ) -> Optional[AnomalyCandidate]:

        if not event:
            return None

        dominant_mcc = event[0]["dominant_mcc"]   # first day's dominant MCC
        first_date   = event[0]["date_only"]
        last_date    = event[-1]["date_only"]
        peak_kl      = max(e["kl"] for e in event)
        total_fraud  = sum(e["today_fraud"] for e in event)

        first_ts     = pd.Timestamp(first_date)
        last_ts      = pd.Timestamp(last_date) + pd.Timedelta(hours=23)

        # Get hourly rows for this event window and dominant MCC + country
        # Capture all channels for the affected MCC/country — don't pre-filter to ecom
        # because the KL signal may come from mixed channels, and the fraud
        # rate calculation should reflect the full picture.
        win_rows = full_df[
            (full_df["country"]    == country) &
            (full_df["mcc_group"]  == dominant_mcc) &
            (full_df["timestamp"]  >= first_ts) &
            (full_df["timestamp"]  <= last_ts)
        ]
        if len(win_rows) == 0:
            return None

        base_rows = full_df[
            (full_df["country"]   == country) &
            (full_df["mcc_group"] == dominant_mcc) &
            (full_df["timestamp"] <  first_ts)
        ].tail(BASELINE_DAYS * 24)

        if len(base_rows) == 0:
            return None

        # Fraud rate evidence
        fr_obs  = float(win_rows["fraud_rate"].mean())
        fr_base = float(base_rows["fraud_rate"].mean())
        if fr_base < 1e-8:
            return None
        fr_mult = fr_obs / fr_base

        if fr_mult < FRAUD_RATE_MULTIPLE_MIN:
            return None   # KL flagged but fraud rate not elevated enough

        # Ticket size (card-testing signature: small tickets)
        ticket_obs  = float(win_rows["avg_ticket_usd"].mean()) \
            if "avg_ticket_usd" in win_rows.columns else None
        ticket_base = float(base_rows["avg_ticket_usd"].mean()) \
            if "avg_ticket_usd" in base_rows.columns and len(base_rows) > 0 else None

        # RC evidence
        rc_evidence = self.compute_rc_evidence(win_rows, base_rows)

        # Volume
        vol_obs = float(win_rows["txn_count"].sum())
        vol_base_est = float(base_rows["txn_count"].mean() * len(win_rows)) if len(base_rows) > 0 else vol_obs
        vol_chg = ((vol_obs - vol_base_est) / max(vol_base_est, 1)) * 100

        # Dominant KL distribution shares
        p_mean = np.mean([e["p_dist"] for e in event], axis=0)
        q_mean = np.mean([e["q_dist"] for e in event], axis=0)
        dom_idx = MCC_GROUPS.index(dominant_mcc)
        dom_curr_share  = float(p_mean[dom_idx])
        dom_base_share  = float(q_mean[dom_idx])
        dom_delta       = (dom_curr_share - dom_base_share) * 100

        # Not affected dimensions (other MCCs in same country)
        not_affected = []
        for mcc in MCC_GROUPS:
            if mcc == dominant_mcc:
                continue
            mcc_rows = full_df[
                (full_df["country"]   == country) &
                (full_df["mcc_group"] == mcc) &
                (full_df["timestamp"] >= first_ts) &
                (full_df["timestamp"] <= last_ts)
            ]
            if len(mcc_rows) < 5:
                continue
            mcc_fr = float(mcc_rows["fraud_rate"].mean())
            mcc_base_fr = float(
                full_df[(full_df["country"]==country) & (full_df["mcc_group"]==mcc) &
                        (full_df["timestamp"] < first_ts)]
                .tail(BASELINE_DAYS * 24)["fraud_rate"].mean()
            )
            if mcc_base_fr > 0 and (mcc_fr / mcc_base_fr) < 1.5:
                not_affected.append(f"{mcc}: fraud rate {mcc_fr:.5f} — unaffected")

        # Co-moving signals
        co_moving = [
            f"KL divergence {peak_kl:.4f} — {dominant_mcc} fraud concentration "
            f"over-represented {dom_curr_share:.1%} vs {dom_base_share:.1%} baseline"
        ]
        ticket_str = ""
        if ticket_obs and ticket_base and ticket_obs < ticket_base * 0.6:
            ticket_str = (
                f"Avg ticket: £{ticket_obs:.2f} vs baseline £{ticket_base:.2f} "
                f"— card-testing signature (micro-transaction pattern)"
            )
            co_moving.append(ticket_str)

        # RC 59 (suspected fraud) co-movement
        rc59 = rc_evidence.get("59", {})
        if rc59.get("delta_pp", 0) > 5:
            co_moving.append(
                f"RC 59 (suspected fraud): {rc59['baseline_share']:.1%} → "
                f"{rc59['current_share']:.1%} (+{rc59['delta_pp']:.1f}pp)"
            )

        # Evidence narrative
        duration = max(1, (last_ts - first_ts).days + 1)
        evidence = [
            f"Fraud rate: {fr_obs:.5f} vs baseline {fr_base:.5f} "
            f"({fr_mult:.1f}× baseline elevation)",
            f"KL divergence: {peak_kl:.4f} — {dominant_mcc} disproportionately "
            f"concentrated ({dom_curr_share:.1%} vs {dom_base_share:.1%} baseline)",
            f"Total fraud events in window: {total_fraud}",
            f"Event duration: {duration} day(s) "
            f"({first_date.strftime('%Y-%m-%d')} — {last_date.strftime('%Y-%m-%d')})",
        ]
        if ticket_str:
            evidence.append(ticket_str)

        affected_slice = {
            "country":   country,
            "mcc_group": dominant_mcc,
            "channel":   "ecom",    # card-testing attacks are almost exclusively CNP
            "auth_type": "non-3DS",
        }

        # Fraud evidence block
        fraud_evidence = {
            "fraud_rate_observed":  round(fr_obs, 6),
            "fraud_rate_baseline":  round(fr_base, 6),
            "fraud_rate_multiple":  round(fr_mult, 2),
        }
        if ticket_obs:
            fraud_evidence["avg_ticket_observed"] = round(ticket_obs, 2)
        if ticket_base:
            fraud_evidence["avg_ticket_baseline"] = round(ticket_base, 2)

        # Sigma proxy: use fraud rate z-score if available
        if "fraud_rate_zscore" in win_rows.columns:
            pseudo_z = float(win_rows["fraud_rate_zscore"].mean())
        else:
            # Convert rate multiple to pseudo-sigma (rough heuristic)
            pseudo_z = min(15.0, (fr_mult - 1.0) * 2.0)

        severity  = self.sigma_to_severity(pseudo_z)
        if fr_mult >= 5 and severity in ("low", "medium"):
            severity = "high"
        if fr_mult >= 8:
            severity = "critical"

        confirmed = (
            fr_mult >= FRAUD_RATE_MULTIPLE_MIN and
            total_fraud >= MIN_FRAUD_IN_WINDOW and
            peak_kl >= KL_THRESHOLD
        )

        return AnomalyCandidate(
            anomaly_id           = self.make_id("fraud_concentration"),
            detector_type        = "fraud_concentration",
            severity             = severity,
            first_seen_ts        = self.fmt_ts(first_ts),
            last_seen_ts         = self.fmt_ts(last_ts),
            duration_hours       = duration * 24,
            affected_slice       = affected_slice,
            not_affected         = not_affected[:5],
            metric               = "fraud_rate",
            observed_value       = round(fr_obs, 6),
            baseline_value       = round(fr_base, 6),
            deviation_sigma      = round(pseudo_z, 2),
            baseline_period_days = BASELINE_DAYS,
            evidence             = evidence,
            co_moving_signals    = co_moving,
            reason_code_evidence = rc_evidence,
            fraud_evidence       = fraud_evidence,
            volume_evidence      = {
                "txn_count_observed": round(vol_obs, 0),
                "txn_count_baseline": round(vol_base_est, 0),
                "volume_change_pct":  round(vol_chg, 1),
            },
            confirmed            = confirmed,
        )
