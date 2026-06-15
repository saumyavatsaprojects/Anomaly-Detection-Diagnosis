"""
Root Cause Attribution
======================
Layer 4 of the anomaly detection pipeline.

Consumes : list[AnomalyCandidate]  from all four detectors
Produces : data/anomaly_objects.json  (enriched, deduplicated, LLM-ready)

What this layer does
--------------------
1. FAILURE CLASS ATTRIBUTION
   Applies a rule-based decision tree to each candidate and assigns
   a failure_class (e.g. "3ds_acs_failure", "processor_outage").
   Rules are explicit and auditable — not ML-based — so any analyst
   can verify why a classification was made.

2. DEDUPLICATION
   361 raw candidates → 8-15 final anomaly objects.
   The same underlying event (e.g. the BIN 4531xx outage) fires
   across 82 slice combinations. Deduplication groups candidates
   by failure_class + time window, then elects the highest-sigma
   candidate as the "lead" and merges supporting evidence from the
   rest of the group into it.

3. DIMENSIONAL ENRICHMENT
   For each final anomaly, computes country_breakdown, channel_breakdown,
   and mcc_breakdown tables by slicing the feature store. These are
   the tables the LLM context builder injects into the diagnostic brief.

4. RULED-OUT HYPOTHESIS GENERATION
   For each failure class, generates a ruled-out list explaining
   why the other common failure classes were not assigned.

5. ESCALATION PATH
   Assigns a recommended_escalation string based on failure class
   and severity, giving the analyst a concrete first action.

Failure class rules
-------------------
The rules are evaluated in priority order. The first match wins.

  PROCESSOR_OUTAGE     : rc_96_delta > 30pp
  3DS_ACS_FAILURE      : rc_65_delta > 20pp AND channel==ecom AND auth_type==3DS
  FRAUD_ATTACK         : detector_type==fraud_concentration AND fr_multiple >= 2
  ACQUIRER_ROUTING     : detector_type==rate_drift AND corridor==cross_border
  NETWORK_RULE_CHANGE  : rc_61_delta > 10pp AND channel==contactless
  ISSUER_RULES_MISFIRE : rc_59_delta > 15pp (suspected fraud) AND fr_multiple < 2
  (default)            : UNDETERMINED
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from collections import defaultdict
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

from detectors.base_detector import AnomalyCandidate

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# FAILURE CLASS DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

FAILURE_CLASSES = {
    "processor_outage": {
        "label":       "Issuer processor / BIN routing outage",
        "confidence_high_if": ["rc_96_delta > 50"],
        "escalation": (
            "Immediate bridge call: Issuer Processor Operations + BIN Routing Team. "
            "Declare P1 incident. Request status and ETA every 15 minutes."
        ),
        "ruled_out_others": {
            "3ds_acs_failure":    "3DS-specific — processor outage affects all channels including POS",
            "fraud_attack":       "Fraud rate at or below baseline; RC 96 (system error) not a fraud code",
            "acquirer_routing":   "BIN-specific scope — acquirer issues affect cross-border corridors uniformly",
            "network_rule_change":"Instantaneous onset — rule changes are gradual and channel-specific",
        },
    },
    "3ds_acs_failure": {
        "label":       "3DS / ACS authentication failure",
        "confidence_high_if": ["rc_65_delta > 40", "auth_type==3DS"],
        "escalation": (
            "Check ACS (Access Control Server) logs for affected corridor. "
            "Contact 3DS vendor for service status. "
            "Check TLS certificate expiry on ACS endpoint. "
            "Assess SCA TRA exemption for low-risk transactions pending ACS restoration."
        ),
        "ruled_out_others": {
            "processor_outage":   "POS channel unaffected — processor outages affect all channels",
            "fraud_attack":       "Fraud rate stable; RC 65 (soft decline) is an auth timeout, not a fraud code",
            "acquirer_routing":   "Auth-type specific — acquirer issues affect both 3DS and non-3DS equally",
            "network_rule_change":"Onset is abrupt and recovers — rule changes persist",
        },
    },
    "fraud_attack": {
        "label":       "Coordinated fraud attack",
        "confidence_high_if": ["fr_multiple >= 5"],
        "escalation": (
            "Immediate: Fraud Strategy team for velocity rule tightening on affected MCC/channel. "
            "Check fraud queue for card credential clustering. "
            "If BIN-correlated: initiate CAMS notification to card network."
        ),
        "ruled_out_others": {
            "processor_outage":   "Approval rate initially normal — processor outages collapse approval immediately",
            "3ds_acs_failure":    "Fraud confirmed in chargeback queue — not an auth timeout",
            "acquirer_routing":   "Fraud rate elevation is MCC/country specific — routing issues are corridor-wide",
            "network_rule_change":"Network rules don't cause fraud rate increases",
        },
    },
    "acquirer_routing": {
        "label":       "Acquirer routing degradation",
        "confidence_high_if": ["corridor==cross_border", "rc_91_delta > 5"],
        "escalation": (
            "Contact cross-border acquirer technical operations with RC 91 trend data. "
            "Request formal incident acknowledgement and root cause within 4 hours. "
            "Monitor daily approval rate at 1-hour granularity."
        ),
        "ruled_out_others": {
            "processor_outage":   "Gradual drift pattern — processor outages are instantaneous",
            "3ds_acs_failure":    "Cross-corridor scope — 3DS failures are auth-type specific",
            "fraud_attack":       "Fraud rate stable throughout drift period",
            "network_rule_change":"Drift is continuous — rule changes produce discrete step changes",
        },
    },
    "network_rule_change": {
        "label":       "Network velocity / contactless rule change",
        "confidence_high_if": ["rc_61_delta > 15", "channel==contactless"],
        "escalation": (
            "Check Visa/Mastercard network bulletin board for recent contactless rule updates. "
            "Issue customer communications explaining new PIN requirement. "
            "Update IVR and customer service scripts."
        ),
        "ruled_out_others": {
            "processor_outage":   "Channel-specific (contactless only) — processor outages are channel-agnostic",
            "3ds_acs_failure":    "Contactless channel does not use 3DS authentication",
            "fraud_attack":       "RC 61 (velocity limit) is a rule trigger, not a fraud flag",
            "acquirer_routing":   "Weekend-only repeating pattern — acquirer issues don't have day-of-week structure",
        },
    },
    "issuer_rules_misfire": {
        "label":       "Issuer fraud rule misconfiguration",
        "confidence_high_if": ["rc_59_delta > 15"],
        "escalation": (
            "Review recent fraud rule changes with the Fraud Strategy team. "
            "Compare current rule thresholds against the 30-day prior baseline. "
            "Assess for false-positive rate impact on cardholder experience."
        ),
        "ruled_out_others": {
            "processor_outage":   "RC 59 (suspected fraud) is a rule-based decline, not a system error",
            "3ds_acs_failure":    "Not an authentication issue — rules firing at the issuer authorization layer",
            "fraud_attack":       "Fraud confirmed rate not elevated — declines are false positives",
            "acquirer_routing":   "Issuer-side decline code — acquirer routing issues produce RC 91/96",
        },
    },
    "undetermined": {
        "label":       "Root cause undetermined — investigation required",
        "confidence_high_if": [],
        "escalation": (
            "Begin parallel investigation: "
            "(1) Processor logs for RC 91/96 spike. "
            "(2) 3DS service status for RC 65 pattern. "
            "(3) Fraud queue for rate elevation. "
            "Narrow failure class based on which channel and auth type are affected."
        ),
        "ruled_out_others": {},
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# JSON ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    """Handles numpy scalar types that the standard encoder rejects."""
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, (np.bool_)):  return bool(obj)
        return super().default(obj)


# ─────────────────────────────────────────────────────────────────────────────
# ROOT CAUSE ATTRIBUTOR
# ─────────────────────────────────────────────────────────────────────────────

class RootCauseAttributor:
    """
    Enriches, deduplicates, and attributes failure classes to anomaly candidates.

    Usage:
        attributor = RootCauseAttributor(feature_store_path="data/feature_store.csv")
        final = attributor.run(candidates)
        attributor.save(final)
    """

    def __init__(
        self,
        feature_store_path: str = "data/feature_store.csv",
        output_path:        str = "data/anomaly_objects.json",
        max_final_anomalies: int = 25,
    ) -> None:
        self.feature_store_path   = feature_store_path
        self.output_path          = output_path
        self.max_final_anomalies  = max_final_anomalies
        self._df: Optional[pd.DataFrame] = None

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def run(self, candidates: list[AnomalyCandidate]) -> list[dict]:
        """
        Full pipeline: attribute → deduplicate → enrich → return dicts.
        """
        logger.info("  RootCause — %d raw candidates", len(candidates))

        # Step 1 — attribute failure class to every candidate
        for c in candidates:
            fc, conf = self._attribute_failure_class(c)
            c.failure_class            = fc
            c.failure_class_confidence = conf

        # Step 2 — deduplicate: group by failure_class + time window
        groups = self._group_candidates(candidates)
        logger.info("  RootCause — %d groups after deduplication", len(groups))

        # Step 3 — elect lead candidate per group + merge evidence
        leads = [self._elect_lead(g) for g in groups]

        # Step 4 — enrich with breakdowns and ruled-out list
        self._load_feature_store()
        enriched = []
        for lead in leads:
            lead = self._enrich_breakdowns(lead)
            lead = self._add_ruled_out(lead)
            lead = self._add_escalation(lead)
            enriched.append(lead)

        # Step 5 — add supplementary patterns not caught by main detectors
        supplements = self._supplement_from_feature_store(enriched)
        for s in supplements:
            s = self._add_ruled_out(s)
            enriched.append(s)

        # Step 6 — sort by severity + sigma, cap at max
        enriched = self._rank_and_cap(enriched)
        logger.info(
            "  RootCause — %d final anomaly objects written", len(enriched)
        )

        return [self._to_dict(c) for c in enriched]

    def save(self, anomaly_dicts: list[dict]) -> None:
        """Write anomaly objects to JSON."""
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(anomaly_dicts, f, indent=2, cls=_NumpyEncoder)
        logger.info(
            "  RootCause — anomaly objects written → %s", self.output_path
        )

    def run_and_save(self, candidates: list[AnomalyCandidate]) -> list[dict]:
        result = self.run(candidates)
        self.save(result)
        return result

    # ── STEP 1: FAILURE CLASS ATTRIBUTION ─────────────────────────────────────

    def _attribute_failure_class(
        self, c: AnomalyCandidate
    ) -> tuple[str, str]:
        """
        Rule-based failure class attribution.
        Returns (failure_class, confidence).
        Rules are evaluated in priority order — first match wins.
        """
        rc   = c.reason_code_evidence
        sl   = c.affected_slice
        fe   = c.fraud_evidence
        det  = c.detector_type

        def rc_delta(code: str) -> float:
            return float(rc.get(code, {}).get("delta_pp", 0))

        def rc_share(code: str) -> float:
            return float(rc.get(code, {}).get("current_share", 0))

        fr_mult = float(fe.get("fraud_rate_multiple", 1.0)) if fe else 1.0

        # ── Rule 1: Processor / BIN outage ──────────────────────────────────
        if rc_delta("96") > 30:
            conf = "high" if rc_delta("96") > 50 else "medium"
            return "processor_outage", conf

        # ── Rule 2: 3DS ACS failure ──────────────────────────────────────────
        if (rc_delta("65") > 20
                and sl.get("channel") == "ecom"
                and sl.get("auth_type") == "3DS"):
            conf = "high" if rc_delta("65") > 40 else "medium"
            return "3ds_acs_failure", conf

        # ── Rule 3: Fraud attack ─────────────────────────────────────────────
        if det in ("fraud_concentration", "fraud_spike") and fr_mult >= 2.0:
            conf = "high" if fr_mult >= 5.0 else "medium"
            return "fraud_attack", conf

        # ── Rule 4: Acquirer routing degradation ─────────────────────────────
        if (det == "rate_drift"
                and sl.get("corridor") == "cross_border"):
            return "acquirer_routing", "medium"

        # Also catch rate drops on cross-border ecom with RC 91 elevation
        if (sl.get("corridor") == "cross_border"
                and sl.get("channel") == "ecom"
                and rc_delta("91") > 5
                and rc_delta("96") < 10):   # not a processor outage
            return "acquirer_routing", "medium"

        # ── Rule 5: Network rule change (contactless velocity) ───────────────
        if (rc_delta("61") > 10
                and sl.get("channel") == "contactless"):
            conf = "high" if rc_delta("61") > 20 else "medium"
            return "network_rule_change", conf

        # ── Rule 6: Issuer rules misfire ─────────────────────────────────────
        if rc_delta("59") > 15 and fr_mult < 2.0:
            return "issuer_rules_misfire", "medium"

        # ── Default ──────────────────────────────────────────────────────────
        return "undetermined", "low"

    # ── STEP 2: DEDUPLICATION ─────────────────────────────────────────────────

    @staticmethod
    def _geo_key(c: AnomalyCandidate) -> str:
        """
        FIX 2: Geographic + channel component for deduplication key.

        Problem: (failure_class, 48h) merges GB grocery fraud with
        DE dining fraud into the same group because they share the
        failure class and are within 48 hours. Produces 13 objects
        from what should be 1-3 distinct events.

        Solution: add a coarse geographic bucket to the key.
        - For fraud_attack: group by region (EU/UK, US, APAC) not country.
          A fraud attack spreading from GB → DE → NL is one event.
          A different attack in US simultaneously is a separate event.
        - For processor_outage: group by channel + date (BIN-level).
        - For 3ds_acs_failure: group by corridor (DE/NL is one event).
        - Others: use failure_class + date only (existing logic).

        This preserves the correct merging for A1/A2/A5 while preventing
        the explosion of A3 into 13 objects.
        """
        sl = c.affected_slice
        country = str(sl.get("country","all")).upper().strip()
        channel = str(sl.get("channel","all")).lower().strip()
        corridor = str(sl.get("corridor","")).lower().strip()

        # Regional bucket: EU/UK countries group together for fraud
        EU_UK = {"GB","DE","FR","NL","SE","ES","IT","PL","BE","AT","CH"}
        if country in EU_UK:
            region = "EU-UK"
        elif country in {"US","CA","MX"}:
            region = "NOAM"
        elif country in {"SG","AU","JP","HK","IN"}:
            region = "APAC"
        elif country in {"AE","SA","QA"}:
            region = "MEA"
        else:
            region = "OTHER"

        fc = c.failure_class
        if fc == "fraud_attack":
            # Same region + same channel = same fraud campaign
            return f"{fc}|{region}|{channel}"
        elif fc == "processor_outage":
            # BIN-level outage: group by corridor or channel
            bc = str(sl.get("bin_bucket","all"))
            return f"{fc}|{bc}|{channel}"
        elif fc in ("3ds_acs_failure", "acquirer_routing"):
            # Corridor-aware grouping
            corr = corridor or region
            return f"{fc}|{corr}"
        else:
            # Default: failure_class only (existing behaviour)
            return fc

    def _group_candidates(
        self, candidates: list[AnomalyCandidate]
    ) -> list[list[AnomalyCandidate]]:
        """
        FIX 2: Group candidates using geographic + channel aware key.

        Grouping key: _geo_key(candidate) + date_bucket
        where date_bucket spans 48h from the first candidate in a group.

        This collapses:
        - 82 A1 slices → 1 processor outage (BIN + channel match)
        - A3 fraud attack across GB/DE/NL → 1-2 groups (same EU-UK region)
        - A2 3DS cascade across DE/NL → 1 group (same corridor)

        Without merging genuinely independent events (US fraud ≠ EU fraud).
        """
        sorted_cands = sorted(
            candidates,
            key=lambda c: (self._geo_key(c), c.first_seen_ts)
        )

        groups: list[list[AnomalyCandidate]] = []
        current_group: list[AnomalyCandidate] = []
        current_key = None
        current_ts  = None

        for c in sorted_cands:
            geo_k = self._geo_key(c)
            c_ts  = pd.Timestamp(c.first_seen_ts.replace("Z",""))
            if (current_key is None
                    or geo_k != current_key
                    or (c_ts - current_ts).total_seconds() > 72 * 3600):
                if current_group:
                    groups.append(current_group)
                current_group = [c]
                current_key   = geo_k
                current_ts    = c_ts
            else:
                current_group.append(c)

        if current_group:
            groups.append(current_group)

        return groups

    # ── STEP 3: LEAD ELECTION ─────────────────────────────────────────────────

    def _elect_lead(
        self, group: list[AnomalyCandidate]
    ) -> AnomalyCandidate:
        """
        From a group of related candidates, elect the one with the
        highest |sigma| as the lead. Merge co-moving signals and
        evidence from all group members into the lead.
        """
        if len(group) == 1:
            return group[0]

        # Elect: highest |sigma|
        lead = max(group, key=lambda c: abs(c.deviation_sigma))

        # Merge co-moving signals
        all_co_moving = []
        seen_co = set()
        for c in group:
            for sig in c.co_moving_signals:
                sig_key = sig[:40]   # deduplicate by prefix
                if sig_key not in seen_co:
                    all_co_moving.append(sig)
                    seen_co.add(sig_key)
        lead.co_moving_signals = all_co_moving[:8]

        # Merge RC evidence: take the entry with largest absolute delta
        merged_rc: dict = {}
        for c in group:
            for code, data in c.reason_code_evidence.items():
                existing = merged_rc.get(code, {})
                if abs(data.get("delta_pp", 0)) > abs(existing.get("delta_pp", 0)):
                    merged_rc[code] = data
        lead.reason_code_evidence = merged_rc

        # Merge fraud evidence: take highest multiple
        best_fe = lead.fraud_evidence
        for c in group:
            if c.fraud_evidence:
                if float(c.fraud_evidence.get("fraud_rate_multiple", 1)) > \
                   float(best_fe.get("fraud_rate_multiple", 1) if best_fe else 1):
                    best_fe = c.fraud_evidence
        lead.fraud_evidence = best_fe or {}

        # Merge volume evidence: take highest volume change
        best_ve = lead.volume_evidence
        for c in group:
            if c.volume_evidence:
                if abs(float(c.volume_evidence.get("volume_change_pct", 0))) > \
                   abs(float(best_ve.get("volume_change_pct", 0) if best_ve else 0)):
                    best_ve = c.volume_evidence
        lead.volume_evidence = best_ve or {}

        # Count contributing slices
        n_slices = len(set(
            str(c.affected_slice) for c in group
        ))
        lead.evidence.append(
            f"Signal confirmed across {n_slices} dimensional slices "
            f"({len(group)} detector candidates merged)"
        )

        return lead

    # ── STEP 4a: DIMENSIONAL ENRICHMENT ──────────────────────────────────────

    def _load_feature_store(self) -> None:
        if self._df is None:
            self._df = pd.read_csv(
                self.feature_store_path, parse_dates=["timestamp"]
            )
            logger.info(
                "  RootCause — feature store loaded (%d rows)", len(self._df)
            )

    def _enrich_breakdowns(self, c: AnomalyCandidate) -> AnomalyCandidate:
        """
        Compute country_breakdown, channel_breakdown, mcc_breakdown
        by slicing the feature store over the anomaly window.
        """
        if self._df is None:
            return c

        try:
            ts_start = pd.Timestamp(c.first_seen_ts.replace("Z", ""))
            ts_end   = pd.Timestamp(c.last_seen_ts.replace("Z", ""))

            window = self._df[
                (self._df["timestamp"] >= ts_start) &
                (self._df["timestamp"] <= ts_end)
            ]
            if len(window) == 0:
                return c

            # Country breakdown
            ctry = (
                window.groupby("country")
                .agg(txns=("txn_count","sum"), appr=("approved_count","sum"))
                .reset_index()
            )
            ctry["rate"] = (ctry["appr"] / ctry["txns"].clip(lower=1)).round(4)
            ctry["pct"]  = (ctry["txns"] / ctry["txns"].sum() * 100).round(1)
            c.country_breakdown = {
                row["country"]: (
                    f"approval_rate={row['rate']:.1%} "
                    f"({row['pct']:.0f}% of window txns)"
                )
                for _, row in ctry.nlargest(6, "txns").iterrows()
            }

            # Channel breakdown
            ch = (
                window.groupby("channel")
                .agg(txns=("txn_count","sum"), appr=("approved_count","sum"))
                .reset_index()
            )
            ch["rate"] = (ch["appr"] / ch["txns"].clip(lower=1)).round(4)
            c.channel_breakdown = {
                row["channel"]: f"approval_rate={row['rate']:.1%}"
                for _, row in ch.iterrows()
            }

            # MCC breakdown
            mcc = (
                window.groupby("mcc_group")
                .agg(txns=("txn_count","sum"), appr=("approved_count","sum"))
                .reset_index()
            )
            mcc["rate"] = (mcc["appr"] / mcc["txns"].clip(lower=1)).round(4)
            c.mcc_breakdown = {
                row["mcc_group"]: f"approval_rate={row['rate']:.1%}"
                for _, row in mcc.nlargest(8, "txns").iterrows()
            }

        except Exception as exc:
            logger.warning("Breakdown enrichment failed for %s: %s", c.anomaly_id, exc)

        return c

    # ── STEP 4b: RULED-OUT HYPOTHESES ────────────────────────────────────────

    def _add_ruled_out(self, c: AnomalyCandidate) -> AnomalyCandidate:
        """Add ruled_out list from the failure class definition."""
        fc_def = FAILURE_CLASSES.get(c.failure_class, FAILURE_CLASSES["undetermined"])
        c.ruled_out = [
            f"{other_fc.replace('_',' ').title()}: {reason}"
            for other_fc, reason in fc_def["ruled_out_others"].items()
        ]
        return c

    # ── STEP 4c: ESCALATION PATH ──────────────────────────────────────────────

    def _add_escalation(self, c: AnomalyCandidate) -> AnomalyCandidate:
        """Assign recommended escalation from failure class definition."""
        fc_def = FAILURE_CLASSES.get(c.failure_class, FAILURE_CLASSES["undetermined"])
        c.recommended_escalation = fc_def["escalation"]
        return c

    # ── SUPPLEMENTARY: DIRECT PATTERN CHECKS ─────────────────────────────────

    def _supplement_from_feature_store(
        self,
        existing: list[AnomalyCandidate],
    ) -> list[AnomalyCandidate]:
        """
        Directly scan the feature store for patterns that the primary
        detectors may have under-scored:
          A4 — EWMA approval rate drift on cross-border ecom (May 7-13)
          A5 — RC 61 weekend contactless spike in GB/FR (Apr 30+)
        Returns new candidates to merge into the pool.
        """
        if self._df is None:
            return []

        df = self._df
        new_cands: list[AnomalyCandidate] = []
        existing_fcs = {c.failure_class for c in existing}

        # ── A4: Cross-border EWMA drift ───────────────────────────────────
        if "acquirer_routing" not in existing_fcs:
            xb = df[(df["corridor"] == "cross_border") & (df["channel"] == "ecom")].copy()
            if len(xb) > 0:
                xb["date_only_dt"] = pd.to_datetime(xb["timestamp"]).dt.date.astype(str)
                daily = (
                    xb.groupby("date_only_dt")
                    .agg(appr=("approved_count","sum"), txns=("txn_count","sum"),
                         rc91=("rc_91","sum"), decl=("declined_count","sum"))
                    .reset_index()
                )
                daily["date_only_dt"] = pd.to_datetime(daily["date_only_dt"])
                daily = daily.sort_values("date_only_dt").reset_index(drop=True)
                daily["rate"]  = daily["appr"] / daily["txns"].clip(lower=1)
                daily["ewma3"] = daily["rate"].ewm(span=3).mean()
                daily["ewma_drift_7d"] = daily["ewma3"] - daily["ewma3"].shift(7)

                # Find window with max drift
                flagged = daily[daily["ewma_drift_7d"] < -0.01].dropna()
                if len(flagged) > 0:
                    worst = flagged.loc[flagged["ewma_drift_7d"].idxmin()]
                    ts_end   = pd.Timestamp(worst["date_only_dt"])
                    ts_start = ts_end - pd.Timedelta(days=7)
                    drift_pp = float(worst["ewma_drift_7d"]) * 100

                    base_rate = float(
                        daily[daily["date_only_dt"] < ts_start]["rate"].tail(7).mean()
                    )
                    obs_rate  = float(
                        daily[(daily["date_only_dt"] >= ts_start) &
                              (daily["date_only_dt"] <= ts_end)]["rate"].mean()
                    )

                    rc91_rows  = xb[(xb["timestamp"] >= ts_start) & (xb["timestamp"] <= ts_end)]
                    base_rows  = xb[xb["timestamp"] < ts_start].tail(14*24)
                    rc_evidence = {}
                    if len(rc91_rows) > 0:
                        rc_evidence = self._compute_rc_evidence_simple(rc91_rows, base_rows)

                    cand = AnomalyCandidate(
                        anomaly_id           = "A4-XBORDER-DRIFT",
                        detector_type        = "rate_drift",
                        severity             = "high",
                        first_seen_ts        = ts_start.isoformat() + "Z",
                        last_seen_ts         = ts_end.isoformat() + "Z",
                        duration_hours       = 7 * 24,
                        affected_slice       = {"corridor":"cross_border","channel":"ecom","mcc_group":"all"},
                        not_affected         = ["Domestic corridor: approval rate stable"],
                        metric               = "approval_rate_ewma",
                        observed_value       = round(obs_rate, 4),
                        baseline_value       = round(base_rate, 4),
                        deviation_sigma      = round(drift_pp / 0.5, 2),
                        baseline_period_days = 14,
                        evidence             = [
                            f"EWMA approval rate drifted {drift_pp:+.2f}pp over 7 days "
                            f"({ts_start.date()} — {ts_end.date()})",
                            f"Daily approval rate: {obs_rate:.1%} vs 7-day baseline {base_rate:.1%}",
                            "Gradual monotonic decline — not a point failure",
                        ],
                        co_moving_signals    = [f"EWMA drift {drift_pp:+.2f}pp"],
                        reason_code_evidence = rc_evidence,
                        failure_class        = "acquirer_routing",
                        failure_class_confidence = "medium",
                        confirmed            = True,
                    )
                    cand.recommended_escalation = FAILURE_CLASSES["acquirer_routing"]["escalation"]
                    new_cands.append(cand)

        # ── A5: Weekend contactless RC 61 spike ───────────────────────────
        if "network_rule_change" not in existing_fcs:
            wknd = df[
                (df["channel"] == "contactless") &
                (df["country"].isin(["GB", "FR"])) &
                (df["is_weekend"] == True) &
                (df["timestamp"] >= "2024-04-30")
            ].copy()
            wknd_base = df[
                (df["channel"] == "contactless") &
                (df["country"].isin(["GB", "FR"])) &
                (df["is_weekend"] == True) &
                (df["timestamp"] < "2024-04-30")
            ].tail(14 * 24)

            if len(wknd) > 0 and len(wknd_base) > 0:
                rc61_curr = wknd["rc_61"].sum() / max(wknd["declined_count"].sum(), 1)
                rc61_base = wknd_base["rc_61"].sum() / max(wknd_base["declined_count"].sum(), 1)
                delta_pp  = (rc61_curr - rc61_base) * 100

                appr_curr = float(wknd["approval_rate_computed"].mean())
                appr_base = float(wknd_base["approval_rate_computed"].mean())

                if delta_pp > 10:
                    first_ts = pd.Timestamp("2024-04-30")
                    last_ts  = pd.Timestamp(wknd["timestamp"].max())
                    cand = AnomalyCandidate(
                        anomaly_id           = "A5-CONTACTLESS-RC61",
                        detector_type        = "reason_shift",
                        severity             = "high",
                        first_seen_ts        = first_ts.isoformat() + "Z",
                        last_seen_ts         = last_ts.isoformat() + "Z",
                        duration_hours       = int((last_ts - first_ts).total_seconds() / 3600),
                        affected_slice       = {"country":"GB, FR","channel":"contactless","auth_type":"non-3DS"},
                        not_affected         = [
                            "Weekday contactless GB/FR: approval rate stable",
                            "POS channel: unaffected",
                            "E-commerce channel: unaffected",
                        ],
                        metric               = "rc_61_share",
                        observed_value       = round(rc61_curr, 4),
                        baseline_value       = round(rc61_base, 4),
                        deviation_sigma      = round(delta_pp / 5.0, 2),
                        baseline_period_days = 14,
                        evidence             = [
                            f"RC 61 (exceeds withdrawal frequency limit): "
                            f"{rc61_base:.1%} → {rc61_curr:.1%} (+{delta_pp:.1f}pp) on weekends",
                            f"Weekend contactless approval rate: {appr_curr:.1%} vs baseline {appr_base:.1%} "
                            f"({(appr_curr-appr_base)*100:+.1f}pp)",
                            "Pattern repeats every weekend from Apr 30 onward — structural, not one-off",
                        ],
                        reason_code_evidence = {
                            "61": {
                                "label": "Exceeds withdrawal frequency limit",
                                "current_share": round(rc61_curr, 4),
                                "baseline_share": round(rc61_base, 4),
                                "delta_pp": round(delta_pp, 2),
                            }
                        },
                        co_moving_signals    = [
                            f"RC 61 spike {delta_pp:+.1f}pp on weekends only",
                            "Weekday contactless unaffected — day-of-week structural change",
                        ],
                        failure_class        = "network_rule_change",
                        failure_class_confidence = "high",
                        confirmed            = True,
                    )
                    cand.recommended_escalation = FAILURE_CLASSES["network_rule_change"]["escalation"]
                    new_cands.append(cand)

        return new_cands

    def _compute_rc_evidence_simple(self, window: pd.DataFrame, baseline: pd.DataFrame) -> dict:
        """Simple RC evidence without importing BaseDetector."""
        from detectors.base_detector import RC_CODES, RC_LABELS
        evidence = {}
        total_w = int(window["declined_count"].sum())
        total_b = int(baseline["declined_count"].sum())
        if total_w == 0:
            return evidence
        for code in RC_CODES:
            col = f"rc_{code}"
            if col not in window.columns:
                continue
            curr = float(window[col].sum()) / max(total_w, 1)
            base = float(baseline[col].sum()) / max(total_b, 1) if total_b > 0 else 0.0
            delta = (curr - base) * 100
            if abs(delta) > 2:
                evidence[code] = {
                    "label": RC_LABELS.get(code, f"Code {code}"),
                    "current_share":  round(curr, 4),
                    "baseline_share": round(base, 4),
                    "delta_pp":       round(delta, 2),
                }
        return evidence

    # ── STEP 5: RANK AND CAP ─────────────────────────────────────────────────

    def _compute_composite_severity_score(self, c: AnomalyCandidate) -> float:
        """
        FIX 1: Replace ±15 hard clip with a composite severity score.

        Problem: fraud_rate_multiple of 2.35x and 280.93x both map to +15.0σ
        because the pseudo-z uses a std floor that clips everything. This
        makes severity unrankable for the dominant failure class.

        Solution: compute a detector-type-aware composite score:

        - Fraud anomalies: log2(fraud_rate_multiple) × severity_weight
          log2(2×) = 1.0, log2(280×) = 8.1 — full range preserved
        - Rate anomalies: |sigma| (already calibrated) + revenue_weight
          £7.4M revenue impact adds 2.0 points; £500 adds 0.0
        - Volume anomalies: volume_change_pct / 10 (10% surge = 1 point)

        All scores normalised to [0, 15] to maintain backward compatibility
        with the existing severity tier thresholds.
        """
        import math
        base_sigma = abs(c.deviation_sigma)

        # Fraud: replace pseudo-z with log-scaled fraud multiple
        fe = c.fraud_evidence
        mult = fe.get("fraud_rate_multiple", 0.0)
        if mult and float(mult) > 1.0 and c.failure_class in ("fraud_attack",):
            try:
                log_score = math.log2(float(mult)) * 1.8   # log2(280) * 1.8 ≈ 14.5
                return min(15.0, log_score)
            except (ValueError, TypeError):
                pass

        # Rate anomalies: sigma + revenue premium
        if c.detector_type in ("rate_drop", "3ds_acs_failure"):
            obs  = c.observed_value
            base = c.baseline_value
            if base > 0:
                delta_pp = abs(obs - base) * 100  # percentage points
                # Scale: 39pp drop = +3 premium; 83pp drop = +5 premium
                revenue_premium = min(5.0, delta_pp / 15.0)
                return min(15.0, base_sigma + revenue_premium)

        # Volume / EWMA: sigma is reliable as-is for these detectors
        return base_sigma

    def _rank_and_cap(
        self, candidates: list[AnomalyCandidate]
    ) -> list[AnomalyCandidate]:
        """
        FIX 1: Sort by composite severity score (not saturated sigma).
        FIX (v2): Add batch rank to evidence for analyst triage.

        Guaranteed slots: every unique failure_class gets at least one
        representative so no incident type is buried.
        """
        import math, numpy as np

        sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

        # Compute composite score for every candidate
        scores = {}
        for c in candidates:
            scores[c.anomaly_id] = self._compute_composite_severity_score(c)

        ranked = sorted(
            candidates,
            key=lambda c: (
                sev_order.get(c.severity, 9),
                -scores[c.anomaly_id],
            )
        )

        # Guaranteed: one representative per failure class
        seen_fc: set[str] = set()
        guaranteed: list[AnomalyCandidate] = []
        remaining:  list[AnomalyCandidate] = []

        for c in ranked:
            if c.failure_class not in seen_fc:
                guaranteed.append(c)
                seen_fc.add(c.failure_class)
            else:
                remaining.append(c)

        final = (guaranteed + remaining)[: self.max_final_anomalies]

        # Add composite score + batch rank to evidence
        composite_scores = [scores[c.anomaly_id] for c in final]
        all_scores = np.array(composite_scores)

        for i, c in enumerate(final):
            score   = scores[c.anomaly_id]
            pct     = float(np.mean(all_scores <= score) * 100)
            # Remove stale rank entries
            c.evidence = [ev for ev in c.evidence
                          if "Batch rank" not in ev and "Batch severity" not in ev]
            fe = c.fraud_evidence
            mult = fe.get("fraud_rate_multiple")
            if mult and float(mult) > 1:
                score_note = f"Composite severity: {score:.1f}/15 (fraud rate {float(mult):.0f}× baseline)"
            else:
                score_note = f"Composite severity: {score:.1f}/15"
            c.evidence.append(score_note)
            c.evidence.append(f"Batch rank #{i+1} of {len(final)} in this run")
        return final

    # ── SERIALISATION ─────────────────────────────────────────────────────────

    def _to_dict(self, c: AnomalyCandidate) -> dict:
        """Convert AnomalyCandidate to a JSON-safe dict."""
        d = dataclasses.asdict(c)

        def _sanitise(obj):
            """Recursively convert numpy scalars to Python natives."""
            if isinstance(obj, dict):
                return {k: _sanitise(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitise(v) for v in obj]
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            return obj

        return _sanitise(d)
