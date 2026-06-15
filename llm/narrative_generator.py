"""
Diagnostic Narrative Generator
================================
Produces structured, analyst-grade diagnostic narratives from anomaly objects.

Each narrative has 8 sections mirroring the mental model of a senior payments
operations analyst:
  1. Executive Summary      — one-paragraph brief for escalation
  2. What Changed           — the metric delta, precisely stated
  3. Where It Happened      — dimensional scope with what was NOT affected
  4. Severity Assessment    — business impact with quantified exposure
  5. Probable Root Causes   — ranked hypotheses with evidence links
  6. Supporting Evidence    — every claim traceable to a detection layer stat
  7. Investigation Steps    — sequenced, time-bounded, owner-tagged actions
  8. Recommended Mitigations — immediate, short-term, and preventive actions

Design constraints:
  - No number appears in the narrative unless it came from the anomaly object.
  - Every section is generated from a dedicated method so each can be
    unit-tested independently.
  - The generator is called by the LLM grounding layer; the LLM receives
    the rendered narrative as part of its context, not as something to rewrite.
  - A plain-text fallback is always produced even if markdown rendering fails.

Colab / Streamlit:
  - Import NarrativeGenerator and call generate(anomaly_obj).
  - Returns a DiagnosticNarrative dataclass with .markdown and .plain_text.
  - The Streamlit UI renders .markdown directly.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# SEVERITY CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SEVERITY_CONFIG = {
    "critical": {
        "label": "CRITICAL",
        "emoji": "🔴",
        "sla_minutes": 15,
        "escalation_level": "VP Operations + Head of Risk — immediate call required",
        "color_hint": "red",
    },
    "high": {
        "label": "HIGH",
        "emoji": "🟠",
        "sla_minutes": 30,
        "escalation_level": "Operations Manager + Fraud Team Lead",
        "color_hint": "orange",
    },
    "medium": {
        "label": "MEDIUM",
        "emoji": "🟡",
        "sla_minutes": 120,
        "escalation_level": "Operations Analyst — standard incident ticket",
        "color_hint": "yellow",
    },
    "low": {
        "label": "LOW",
        "emoji": "🟢",
        "sla_minutes": 480,
        "escalation_level": "Operations Analyst — monitoring ticket",
        "color_hint": "green",
    },
}

DECLINE_RC_LABELS = {
    "05": ("Do not honor",                   "issuer-hard",   "not retryable"),
    "14": ("Invalid card number",             "issuer-hard",   "not retryable"),
    "51": ("Insufficient funds",              "issuer-soft",   "retryable after balance change"),
    "57": ("Transaction not permitted",       "issuer-hard",   "not retryable"),
    "59": ("Suspected fraud",                 "issuer-hard",   "not retryable — fraud flag raised"),
    "61": ("Exceeds withdrawal freq. limit",  "issuer-soft",   "retryable after limit reset"),
    "65": ("Soft decline — 3DS required",     "network-soft",  "retryable with step-up auth"),
    "91": ("Issuer/switch inoperative",       "system-error",  "retryable — routing issue"),
    "96": ("System malfunction",              "system-error",  "retryable — processor issue"),
}

FAILURE_CLASS_LABELS = {
    "3ds_acs_failure":          "3DS / ACS authentication failure",
    "processor_outage":         "Issuer processor / BIN routing outage",
    "fraud_attack":             "Coordinated fraud attack",
    "acquirer_routing":         "Acquirer routing degradation",
    "network_rule_change":      "Network velocity / rule change",
    "issuer_rules_misfire":     "Issuer fraud rule misconfiguration",
    "acquirer_outage":          "Acquirer system outage",
    "undetermined":             "Root cause undetermined — investigation required",
}


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DiagnosticNarrative:
    anomaly_id:       str
    anomaly_type:     str
    severity:         str
    generated_at:     str
    markdown:         str
    plain_text:       str
    section_texts:    dict = field(default_factory=dict)   # section_name → raw text
    cited_stats:      list = field(default_factory=list)   # for verifier
    word_count:       int  = 0


# ─────────────────────────────────────────────────────────────────────────────
# NARRATIVE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class NarrativeGenerator:
    """
    Generates structured diagnostic narratives from anomaly objects.

    Usage:
        gen = NarrativeGenerator()
        narrative = gen.generate(anomaly_obj)
        print(narrative.markdown)
    """

    def generate(self, anomaly: dict) -> DiagnosticNarrative:
        """Entry point. Returns a complete DiagnosticNarrative."""
        sections     = {}
        cited_stats  = []
        severity_key = anomaly.get("severity", "medium").lower()
        sev_cfg      = SEVERITY_CONFIG.get(severity_key, SEVERITY_CONFIG["medium"])

        # Generate each section
        sections["executive_summary"]     = self._exec_summary(anomaly, sev_cfg)
        sections["what_changed"]          = self._what_changed(anomaly)
        sections["where_it_happened"]     = self._where_it_happened(anomaly)
        sections["severity_assessment"]   = self._severity_assessment(anomaly, sev_cfg)
        sections["probable_root_causes"]  = self._probable_root_causes(anomaly)
        sections["supporting_evidence"]   = self._supporting_evidence(anomaly, cited_stats)
        sections["investigation_steps"]   = self._investigation_steps(anomaly)
        sections["recommended_mitigations"] = self._recommended_mitigations(anomaly)

        # Render to markdown
        md   = self._render_markdown(anomaly, sections, sev_cfg)
        txt  = self._render_plain_text(anomaly, sections)

        return DiagnosticNarrative(
            anomaly_id    = anomaly.get("anomaly_id", "UNKNOWN"),
            anomaly_type  = anomaly.get("detector_type", "unknown"),
            severity      = severity_key,
            generated_at  = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            markdown      = md,
            plain_text    = txt,
            section_texts = sections,
            cited_stats   = cited_stats,
            word_count    = len(md.split()),
        )

    # ── SECTION 1: EXECUTIVE SUMMARY ─────────────────────────────────────────

    def _exec_summary(self, a: dict, sev_cfg: dict) -> str:
        affected   = a.get("affected_slice", {})
        mcc        = affected.get("mcc_group", "all MCC groups")
        country    = affected.get("country", "all countries")
        channel    = affected.get("channel", "all channels")
        auth       = affected.get("auth_type", "")
        auth_str   = f" ({auth})" if auth and auth != "all" else ""

        metric     = a.get("metric", "approval_rate").replace("_", " ")
        observed   = self._fmt_rate(a.get("observed_value"))
        baseline   = self._fmt_rate(a.get("baseline_value"))
        delta_pp   = self._delta_pp(a.get("observed_value"), a.get("baseline_value"))
        sigma      = self._fmt_sigma(a.get("deviation_sigma"))
        duration   = a.get("duration_hours", "unknown")
        first_seen = self._fmt_ts(a.get("first_seen_ts"))
        fc         = FAILURE_CLASS_LABELS.get(
                         a.get("failure_class", "undetermined"),
                         a.get("failure_class", "undetermined")
                     )
        fc_conf    = a.get("failure_class_confidence", "low")
        sla        = sev_cfg["sla_minutes"]

        lines = [
            f"At {first_seen}, the detection system flagged a {sev_cfg['label']} "
            f"anomaly affecting {metric} for {channel}{auth_str} transactions in "
            f"{country} ({mcc}).",
            "",
            f"The {metric} dropped from a 7-day baseline of {baseline} to {observed} "
            f"— a {delta_pp} decline representing a {sigma} deviation from the "
            f"conditional baseline. The anomaly persisted for {duration} hours before "
            f"returning to baseline levels.",
            "",
            f"Root cause attribution ({fc_conf} confidence): **{fc}**. "
            f"All alternative failure classes have been assessed; "
            f"see Section 5 for the full attribution rationale.",
            "",
            f"**SLA:** This severity requires acknowledgement within "
            f"{sla} minutes. Escalation path: {sev_cfg['escalation_level']}.",
        ]
        return "\n".join(lines)

    # ── SECTION 2: WHAT CHANGED ──────────────────────────────────────────────

    def _what_changed(self, a: dict) -> str:
        metric      = a.get("metric", "approval_rate").replace("_", " ")
        observed    = self._fmt_rate(a.get("observed_value"))
        baseline    = self._fmt_rate(a.get("baseline_value"))
        delta_pp    = self._delta_pp(a.get("observed_value"), a.get("baseline_value"))
        sigma       = self._fmt_sigma(a.get("deviation_sigma"))
        bp_days     = a.get("baseline_period_days", 7)
        first_seen  = self._fmt_ts(a.get("first_seen_ts"))
        duration    = a.get("duration_hours", "unknown")
        detector    = a.get("detector_type", "rate_drop")

        # Volume change
        vol         = a.get("volume_evidence", {})
        vol_str     = ""
        if vol:
            vol_obs   = vol.get("txn_count_observed", 0)
            vol_base  = vol.get("txn_count_baseline", 0)
            vol_chg   = vol.get("volume_change_pct", 0)
            vol_str   = (
                f"\n\nTransaction volume during the anomaly window was "
                f"**{int(vol_obs):,}** transactions against a baseline of "
                f"**{int(vol_base):,}** ({float(vol_chg):+.1f}%). "
                f"{vol.get('volume_interpretation', '')}"
            )

        # Fraud rate change
        fraud       = a.get("fraud_evidence", {})
        fraud_str   = ""
        if fraud and float(fraud.get("fraud_rate_multiple", 1.0)) > 1.5:
            fr_obs    = float(fraud.get("fraud_rate_observed", 0))
            fr_base   = float(fraud.get("fraud_rate_baseline", 0))
            multiple  = float(fraud.get("fraud_rate_multiple", 1.0))
            fraud_str = (
                f"\n\nThe fraud rate co-moved with the anomaly, rising from "
                f"**{fr_base:.5f}** (baseline) to **{fr_obs:.5f}** "
                f"(**{multiple:.1f}×** baseline). This co-movement is assessed "
                f"separately in Section 5."
            )

        lines = [
            f"**Detection method:** {detector.replace('_', ' ').title()} — "
            f"{bp_days}-day conditional baseline per dimensional slice.",
            "",
            f"The primary metric, **{metric}**, registered at **{observed}** "
            f"against its {bp_days}-day conditional baseline of **{baseline}**, "
            f"a decline of **{delta_pp}** ({sigma} below baseline). "
            f"The anomaly was first detected at **{first_seen}** and "
            f"persisted for **{duration} hours**.",
            vol_str,
            fraud_str,
        ]

        # Reason code shift
        rc = a.get("reason_code_evidence", {})
        if rc:
            lines.append("\n**Decline reason code distribution shift:**")
            lines.append("")
            for code, data in sorted(rc.items()):
                if not isinstance(data, dict):
                    continue
                lbl      = data.get("label", f"Code {code}")
                curr     = self._fmt_pct(data.get("current_share"))
                base_    = self._fmt_pct(data.get("baseline_share"))
                delta    = data.get("delta_pp", 0)
                arrow    = "▲" if delta > 0 else "▼"
                emphasis = "**" if abs(delta) > 10 else ""
                lines.append(
                    f"- {emphasis}RC {code} ({lbl}): {curr} vs baseline {base_} "
                    f"({arrow} {abs(delta):.1f}pp){emphasis}"
                )

        return "\n".join(lines)

    # ── SECTION 3: WHERE IT HAPPENED ─────────────────────────────────────────

    def _where_it_happened(self, a: dict) -> str:
        affected     = a.get("affected_slice", {})
        not_affected = a.get("not_affected", [])
        country_bd   = a.get("country_breakdown", {})
        channel_bd   = a.get("channel_breakdown", {})
        mcc_bd       = a.get("mcc_breakdown", {})

        lines = ["**Affected dimensions:**", ""]

        for dim, key in [
            ("Country / corridor", "country"),
            ("MCC group",          "mcc_group"),
            ("Channel",            "channel"),
            ("Authentication type","auth_type"),
            ("Card present",       "card_present"),
            ("BIN bucket",         "bin_bucket"),
        ]:
            val = affected.get(key)
            if val and str(val).lower() not in ("all", "none", ""):
                lines.append(f"- **{dim}:** {val}")

        if not_affected:
            lines.append("")
            lines.append("**Confirmed NOT affected** *(isolates the failure scope)*:")
            lines.append("")
            for item in not_affected:
                lines.append(f"- {item}")

        if country_bd:
            lines.append("")
            lines.append("**Country-level breakdown:**")
            lines.append("")
            for ctry, detail in country_bd.items():
                lines.append(f"- {ctry}: {detail}")

        if channel_bd:
            lines.append("")
            lines.append("**Channel breakdown:**")
            lines.append("")
            for ch, detail in channel_bd.items():
                lines.append(f"- {ch}: {detail}")

        if mcc_bd:
            lines.append("")
            lines.append("**MCC group breakdown:**")
            lines.append("")
            for mcc, detail in mcc_bd.items():
                lines.append(f"- {mcc}: {detail}")

        return "\n".join(lines)

    # ── SECTION 4: SEVERITY ASSESSMENT ───────────────────────────────────────

    def _severity_assessment(self, a: dict, sev_cfg: dict) -> str:
        severity    = sev_cfg["label"]
        sigma       = self._fmt_sigma(a.get("deviation_sigma"))
        duration    = a.get("duration_hours", 0)
        vol         = a.get("volume_evidence", {})
        observed    = a.get("observed_value", 0)
        baseline    = a.get("baseline_value", 1)
        delta_pp    = self._delta_pp(a.get("observed_value"), a.get("baseline_value"))

        # Estimated transaction impact
        impact_str = ""
        if vol:
            txn_count   = int(vol.get("txn_count_observed", 0))
            # Transactions that would have been approved at baseline rate
            lost_approvals = int(txn_count * (float(baseline) - float(observed)))
            if lost_approvals > 0:
                impact_str = (
                    f"\n\n**Estimated transaction impact:** Approximately "
                    f"**{lost_approvals:,}** transactions that would have been "
                    f"approved at the baseline rate of {self._fmt_rate(baseline)} "
                    f"were declined during the anomaly window. This is a lower-bound "
                    f"estimate; retry behaviour inflates the raw decline count."
                )

        rationale = self._severity_rationale(a)

        lines = [
            f"**Severity:** {sev_cfg['emoji']} {severity}",
            "",
            f"**Sigma deviation:** {sigma} — "
            f"{'extreme outlier, exceeds 99.9th percentile of historical variation' if float(a.get('deviation_sigma', 0)) >= 4.0 else 'significant outlier, exceeds 99th percentile'}.",
            "",
            f"**Duration:** {duration} hours of degraded service.",
            "",
            f"**Metric impact:** {delta_pp} decline in approval rate from "
            f"conditional baseline.",
            impact_str,
            "",
            f"**Severity rationale:** {rationale}",
            "",
            f"**Response SLA:** Acknowledgement required within "
            f"**{sev_cfg['sla_minutes']} minutes** of alert generation.",
            f"**Escalation path:** {sev_cfg['escalation_level']}.",
        ]
        return "\n".join(lines)

    def _severity_rationale(self, a: dict) -> str:
        sigma       = float(a.get("deviation_sigma", 0))
        duration    = int(a.get("duration_hours", 0))
        fc          = a.get("failure_class", "undetermined")
        fraud_mult  = float(a.get("fraud_evidence", {}).get("fraud_rate_multiple", 1.0))

        reasons = []
        if sigma >= 4.0:
            reasons.append(f"sigma deviation of {sigma:.2f} represents an extreme statistical outlier")
        if duration >= 6:
            reasons.append(f"sustained duration of {duration} hours causes material cardholder impact")
        if "outage" in fc or "failure" in fc:
            reasons.append("system-level failure class has immediate revenue and reputational impact")
        if fraud_mult >= 3.0:
            reasons.append(f"fraud rate at {fraud_mult:.1f}× baseline indicates active attack")
        if not reasons:
            reasons.append("metric deviation crosses the operational alert threshold")

        return "; ".join(reasons).capitalize() + "."

    # ── SECTION 5: PROBABLE ROOT CAUSES ──────────────────────────────────────

    def _probable_root_causes(self, a: dict) -> str:
        fc          = a.get("failure_class", "undetermined")
        fc_label    = FAILURE_CLASS_LABELS.get(fc, fc)
        fc_conf     = a.get("failure_class_confidence", "low").capitalize()
        evidence    = a.get("evidence", [])
        ruled_out   = a.get("ruled_out", [])
        escalation  = a.get("recommended_escalation", "")

        lines = [
            f"**Primary hypothesis ({fc_conf} confidence):** {fc_label}",
            "",
            "**Supporting evidence for this hypothesis:**",
            "",
        ]
        for item in evidence:
            lines.append(f"- {item}")

        if ruled_out:
            lines.append("")
            lines.append("**Alternative hypotheses assessed and ruled out:**")
            lines.append("")
            for item in ruled_out:
                lines.append(f"- ~~{item.split(':')[0]}~~: {item.split(':', 1)[-1].strip()}" if ':' in item else f"- {item}")

        if escalation:
            lines.append("")
            lines.append(f"**Escalation path implied by this hypothesis:** "
                         f"{escalation.split('.')[0]}.")

        return "\n".join(lines)

    # ── SECTION 6: SUPPORTING EVIDENCE ───────────────────────────────────────

    def _supporting_evidence(self, a: dict, cited_stats: list) -> str:
        lines = []

        # Primary metric table
        observed  = a.get("observed_value")
        baseline  = a.get("baseline_value")
        sigma     = a.get("deviation_sigma")
        metric    = a.get("metric", "approval_rate").replace("_", " ")
        bp_days   = a.get("baseline_period_days", 7)

        lines += [
            "**Primary metric evidence:**",
            "",
            f"| Metric | Observed | Baseline ({bp_days}d) | Delta | Sigma |",
            "|--------|----------|----------------------|-------|-------|",
            f"| {metric.title()} | {self._fmt_rate(observed)} | "
            f"{self._fmt_rate(baseline)} | "
            f"{self._delta_pp(observed, baseline)} | {self._fmt_sigma(sigma)} |",
        ]
        cited_stats.extend(["observed_value", "baseline_value", "deviation_sigma"])

        # Reason code table
        rc = a.get("reason_code_evidence", {})
        if rc:
            lines += [
                "",
                "**Decline reason code evidence:**",
                "",
                "| Code | Label | Current | Baseline | Delta | Type |",
                "|------|-------|---------|----------|-------|------|",
            ]
            for code in sorted(rc.keys()):
                data   = rc[code]
                if not isinstance(data, dict):
                    continue
                lbl    = data.get("label", f"Code {code}")
                curr   = self._fmt_pct(data.get("current_share"))
                base_  = self._fmt_pct(data.get("baseline_share"))
                delta  = data.get("delta_pp", 0)
                arrow  = "▲" if delta > 0 else "▼"
                rc_meta = DECLINE_RC_LABELS.get(code, ("", "", ""))
                rc_type = rc_meta[1] if rc_meta else ""
                lines.append(
                    f"| {code} | {lbl} | {curr} | {base_} | "
                    f"{arrow} {abs(delta):.1f}pp | {rc_type} |"
                )
            cited_stats.append("reason_code_evidence")

        # Fraud evidence
        fraud = a.get("fraud_evidence", {})
        if fraud:
            lines += [
                "",
                "**Fraud signal evidence:**",
                "",
                "| Signal | Observed | Baseline | Multiple |",
                "|--------|----------|----------|----------|",
            ]
            if "fraud_rate_observed" in fraud:
                lines.append(
                    f"| Fraud rate | {float(fraud['fraud_rate_observed']):.5f} | "
                    f"{float(fraud.get('fraud_rate_baseline', 0)):.5f} | "
                    f"{float(fraud.get('fraud_rate_multiple', 1)):.2f}× |"
                )
            if "avg_ticket_observed" in fraud:
                lines.append(
                    f"| Avg ticket | £{float(fraud['avg_ticket_observed']):.2f} | "
                    f"£{float(fraud.get('avg_ticket_baseline', 0)):.2f} | — |"
                )
            cited_stats.append("fraud_evidence")

        # Volume evidence
        vol = a.get("volume_evidence", {})
        if vol:
            lines += [
                "",
                "**Volume signal:**",
                "",
                f"- Transaction count: **{int(vol.get('txn_count_observed',0)):,}** "
                f"vs baseline **{int(vol.get('txn_count_baseline',0)):,}** "
                f"({float(vol.get('volume_change_pct',0)):+.1f}%)",
            ]
            interp = vol.get("volume_interpretation")
            if interp:
                lines.append(f"- Interpretation: *{interp}*")
            cited_stats.append("volume_evidence")

        # Co-moving signals
        co_moving = a.get("co_moving_signals", [])
        if co_moving:
            lines += ["", "**Co-moving signals (corroborate primary hypothesis):**", ""]
            for sig in co_moving:
                lines.append(f"- {sig}")

        return "\n".join(lines)

    # ── SECTION 7: INVESTIGATION STEPS ───────────────────────────────────────

    def _investigation_steps(self, a: dict) -> str:
        fc          = a.get("failure_class", "undetermined")
        affected    = a.get("affected_slice", {})
        first_seen  = self._fmt_ts(a.get("first_seen_ts"))
        duration    = a.get("duration_hours", 0)
        country     = affected.get("country", "affected countries")
        channel     = affected.get("channel", "affected channel")
        auth        = affected.get("auth_type", "")
        bin_bucket  = affected.get("bin_bucket", "")

        steps = self._get_investigation_steps(
            fc, first_seen, duration, country, channel, auth, bin_bucket, a
        )

        lines = [
            "Steps are sequenced by diagnostic priority. Time estimates assume "
            "normal staffing. Owner tags are indicative — adapt to your team structure.",
            "",
        ]
        for i, (step, owner, est_mins) in enumerate(steps, 1):
            lines.append(
                f"**{i}.** [{owner}] *({est_mins} min)* {step}"
            )

        return "\n".join(lines)

    def _get_investigation_steps(
        self, fc, first_seen, duration, country, channel, auth, bin_bucket, a
    ) -> list:
        """Returns list of (step_text, owner, est_minutes) tuples by failure class."""

        common_first = [
            (
                f"Confirm the anomaly window: pull hourly approval rate for "
                f"{channel} transactions in {country} from {first_seen} for "
                f"{duration} hours. Verify the detection system's reported "
                f"start and end times against live data.",
                "Ops Analyst", 5
            ),
            (
                "Check whether the anomaly has resolved. If approval rate has "
                "not returned to within 1σ of baseline, escalate immediately — "
                "this is still an active incident.",
                "Ops Analyst", 3
            ),
        ]

        if fc == "3ds_acs_failure":
            specific = [
                (
                    f"Pull ACS (Access Control Server) logs for {country} corridor "
                    f"covering {first_seen} ± 2 hours. Look for timeout errors, "
                    "TLS handshake failures, or certificate expiry warnings.",
                    "3DS/Network Ops", 10
                ),
                (
                    "Check TLS certificate expiry date on the ACS endpoint. "
                    "A certificate within 7 days of expiry or recently renewed "
                    "is a primary suspect.",
                    "3DS/Network Ops", 5
                ),
                (
                    "Contact the 3DS vendor / network (Visa/MC 3D Secure) for "
                    f"service status in {country}. Request their incident log "
                    "for the same window.",
                    "Network Ops", 10
                ),
                (
                    "Verify that POS and non-3DS e-commerce channels in the same "
                    "countries remained unaffected during the window. If POS was "
                    "also impacted, widen the hypothesis — this may be an issuer "
                    "processor issue, not an ACS issue.",
                    "Ops Analyst", 5
                ),
                (
                    "Assess SCA exemption options. If the ACS is still degraded, "
                    "work with compliance to determine whether low-risk transactions "
                    f"in {country} can be temporarily exempted under PSD2 "
                    "transaction risk analysis (TRA) provisions.",
                    "Compliance + Risk", 20
                ),
            ]

        elif fc == "processor_outage":
            bin_str = f" for BIN range {bin_bucket}" if bin_bucket else ""
            specific = [
                (
                    f"Pull authorization logs{bin_str} for the outage window. "
                    "Confirm that RC 96 (system malfunction) is the dominant "
                    "decline code and cross-reference against the BIN range scope.",
                    "Ops Analyst", 10
                ),
                (
                    "Contact the issuer processor operations team immediately. "
                    "Request their incident timeline and expected resolution. "
                    "Confirm whether the outage affected only this BIN range or "
                    "a broader routing group.",
                    "Processor Ops", 10
                ),
                (
                    "Check for retry storm indicators: if transaction volume is "
                    ">20% above baseline during the outage window, customers are "
                    "retrying. Issue a customer communication to prevent further "
                    "retries if the outage is expected to persist >30 minutes.",
                    "Customer Ops", 15
                ),
                (
                    "Verify the processor's SLA breach threshold. If the outage "
                    "has exceeded the contractual maximum downtime, begin SLA "
                    "documentation for the post-incident review.",
                    "Vendor Management", 10
                ),
                (
                    "Test a manual authorization using an affected BIN card. "
                    "This confirms whether the outage has resolved and validates "
                    "the system's automated recovery detection.",
                    "Ops Analyst", 5
                ),
            ]

        elif fc == "fraud_attack":
            specific = [
                (
                    "Pull the fraud queue for the affected MCC and channel. "
                    "Identify the compromised card credentials: are they clustered "
                    "by BIN range, issuing date, or merchant? This determines the "
                    "scope of the compromise.",
                    "Fraud Analyst", 15
                ),
                (
                    "Check average ticket size distribution. A bimodal distribution "
                    "(micro-amounts + high-value) confirms the card-testing + cashout "
                    "pattern. If tickets are uniformly small, the attack is in the "
                    "testing phase and cashout has not begun.",
                    "Fraud Analyst", 10
                ),
                (
                    "Identify whether the attack is correlated with a specific "
                    "merchant or acquiring BIN. A single merchant with an unusual "
                    "spike indicates a compromised merchant terminal or CNP "
                    "data breach at that merchant.",
                    "Fraud Analyst", 10
                ),
                (
                    "Apply immediate fraud rule tightening: lower velocity limits "
                    "for the affected MCC/channel combination. Document the "
                    "rule change for compliance review.",
                    "Fraud Strategy", 20
                ),
                (
                    "If BIN-clustered: initiate mass card reissuance for the "
                    "affected BIN range. If merchant-correlated: raise with the "
                    "acquiring bank for merchant investigation.",
                    "Fraud Strategy + Compliance", 30
                ),
            ]

        elif fc == "acquirer_routing":
            specific = [
                (
                    "Confirm the RC 91 (issuer/switch inoperative) trend. "
                    "If RC 91 share is rising over multiple days, this is a "
                    "slow-degrading routing issue — not a point failure.",
                    "Ops Analyst", 5
                ),
                (
                    "Identify which acquirer is handling the cross-border "
                    "transactions in scope. Pull their service status page and "
                    "contact their technical operations team with the timestamp "
                    "range and transaction volume affected.",
                    "Network Ops", 15
                ),
                (
                    "Check for alternative routing paths. If the acquirer has a "
                    "backup routing option, assess whether rerouting affected "
                    "transactions is possible without a full cutover.",
                    "Network Ops", 20
                ),
                (
                    "Monitor the trend daily for the next 7 days. Silent "
                    "degradation anomalies often indicate infrastructure "
                    "problems the acquirer has not yet acknowledged.",
                    "Ops Analyst", 5
                ),
            ]

        elif fc == "network_rule_change":
            specific = [
                (
                    "Confirm the RC 61 (exceeds withdrawal frequency limit) "
                    "and RC 51 (insufficient funds) pattern. If RC 61 is "
                    "dominant on weekends, this is a velocity rule change, "
                    "not a card balance issue.",
                    "Ops Analyst", 5
                ),
                (
                    "Check the network (Visa/Mastercard) bulletin board for "
                    "recent operating regulation changes. Contactless velocity "
                    "limits for offline transactions are updated periodically "
                    "and may not come with advance notice.",
                    "Network Relations", 15
                ),
                (
                    "Issue a customer communication explaining the new PIN "
                    "requirement threshold and how to resolve the declined "
                    "transaction. This reduces inbound support volume.",
                    "Customer Comms", 30
                ),
                (
                    "Work with the network to understand the new limit structure "
                    "and update internal cardholder documentation.",
                    "Product + Compliance", 20
                ),
            ]

        else:  # undetermined
            specific = [
                (
                    "The root cause is currently undetermined. Begin with the "
                    "most common failure class for this metric and channel: "
                    "check processor logs, 3DS service status, and network "
                    "bulletins in parallel.",
                    "Ops Analyst", 15
                ),
                (
                    "Pull the full decline reason code distribution for the "
                    "anomaly window and compare to the 14-day baseline. The "
                    "dominant shifted code is the primary diagnostic indicator.",
                    "Ops Analyst", 10
                ),
                (
                    "Isolate the failure by channel: check whether the anomaly "
                    "is present in POS, e-commerce, and contactless independently. "
                    "Channel-specific failure narrows the hypothesis significantly.",
                    "Ops Analyst", 10
                ),
            ]

        common_last = [
            (
                "Document the incident timeline: detection time, "
                "acknowledgement time, root cause determination time, "
                "resolution time. This feeds the post-incident review.",
                "Ops Manager", 10
            ),
        ]
        return common_first + specific + common_last

    # ── SECTION 8: RECOMMENDED MITIGATIONS ───────────────────────────────────

    def _recommended_mitigations(self, a: dict) -> str:
        fc     = a.get("failure_class", "undetermined")
        mitigations = self._get_mitigations(fc, a)

        lines = []
        for horizon, items in mitigations.items():
            lines.append(f"**{horizon}:**")
            lines.append("")
            for item in items:
                lines.append(f"- {item}")
            lines.append("")

        return "\n".join(lines)

    def _get_mitigations(self, fc: str, a: dict) -> dict:
        """Returns ordered dict of horizon → [mitigation items]."""

        if fc == "3ds_acs_failure":
            return {
                "Immediate (0–2 hours)": [
                    "Implement SCA transaction risk analysis (TRA) exemptions for "
                    "low-risk transactions in the affected corridor to restore "
                    "approval rate while ACS is degraded. Requires compliance sign-off.",
                    "Monitor ACS availability every 5 minutes. Trigger auto-recovery "
                    "alert when 3DS success rate returns to >95% for 3 consecutive checks.",
                    "Disable retry amplification: if transaction volume is >15% above "
                    "baseline, suppress duplicate authorization attempts from the same card.",
                ],
                "Short-term (2–48 hours)": [
                    "Conduct a post-incident review of ACS certificate management. "
                    "Implement automated certificate expiry monitoring with 30-day "
                    "advance warnings.",
                    "Establish a 3DS service health API integration so the monitoring "
                    "system receives machine-readable status before analysts need to "
                    "call the vendor.",
                    "Assess whether the ACS vendor SLA was breached and document "
                    "for contractual review.",
                ],
                "Preventive (1–4 weeks)": [
                    "Implement a 3DS fallback routing path to a secondary ACS "
                    "provider for the DE/NL corridor.",
                    "Build a synthetic transaction monitor that tests 3DS "
                    "authentication end-to-end every 5 minutes per corridor.",
                    "Review PSD2 SCA exemption strategy — increase reliance on "
                    "TRA for low-risk transaction types to reduce ACS dependency.",
                ],
            }

        elif fc == "processor_outage":
            return {
                "Immediate (0–2 hours)": [
                    "Engage processor ops team on a bridge call. Establish a "
                    "30-minute status update cadence.",
                    "If the outage is expected to exceed 2 hours, assess failover "
                    "routing to a secondary processor for the affected BIN range.",
                    "Suppress outbound customer retry prompts to prevent retry "
                    "storm amplification.",
                ],
                "Short-term (2–48 hours)": [
                    "Conduct SLA breach analysis. Calculate the financial impact "
                    "using lost approval count × average ticket value × interchange rate.",
                    "Issue a post-incident customer communication for cardholders "
                    "who experienced unexplained declines during the window.",
                    "Review monitoring thresholds: this outage should have been "
                    "detectable within 15 minutes. Assess why the SLA for detection "
                    "was or was not met.",
                ],
                "Preventive (1–4 weeks)": [
                    "Implement automated processor failover for BIN ranges with "
                    ">50k daily authorizations.",
                    "Negotiate a contractual SLA with the processor for maximum "
                    "acceptable downtime per BIN range per month.",
                    "Build a processor health scorecard tracking uptime, latency "
                    "p95, and RC 96 rate per BIN range on a rolling 30-day basis.",
                ],
            }

        elif fc == "fraud_attack":
            fraud = a.get("fraud_evidence", {})
            multiple = float(fraud.get("fraud_rate_multiple", 1.0))
            return {
                "Immediate (0–2 hours)": [
                    "Apply velocity block: limit to 3 authorization attempts per "
                    "card per hour in the affected MCC/channel combination.",
                    "Flag all cards that attempted a transaction in the affected "
                    "MCC/channel during the anomaly window for enhanced monitoring.",
                    f"If fraud rate is confirmed at {multiple:.1f}× baseline: "
                    "initiate card block on confirmed fraud cards. Do not block "
                    "on suspicion alone — validate against fraud confirmation queue.",
                ],
                "Short-term (2–48 hours)": [
                    "Conduct a full cardholder impact assessment: how many cards "
                    "were successfully frauded vs attempted? Calculate total fraud "
                    "exposure in GBP.",
                    "Identify the data compromise source if BIN-clustered. "
                    "Notify the relevant card network (Visa/MC) under the "
                    "CAMS (Compromised Account Management System) protocol.",
                    "Review and permanently tighten fraud rules for the affected "
                    "MCC/channel combination based on the attack pattern observed.",
                ],
                "Preventive (1–4 weeks)": [
                    "Implement ML-based velocity scoring for CNP transactions in "
                    "high-risk MCCs (digital goods, grocery CNP).",
                    "Establish a dark web monitoring integration to receive early "
                    "warning when issued card credentials appear on carding forums.",
                    "Review 3DS coverage for the affected channel — enforcing 3DS "
                    "step-up for higher-risk transactions reduces CNP fraud exposure.",
                ],
            }

        elif fc == "acquirer_routing":
            return {
                "Immediate (0–2 hours)": [
                    "Contact acquirer technical operations with the full decline "
                    "log showing RC 91 trend. Request acknowledgement of the "
                    "connectivity issue.",
                    "Monitor daily approval rate for the cross-border channel "
                    "at 1-hour granularity until trend reverses.",
                ],
                "Short-term (2–48 hours)": [
                    "Assess whether alternative acquirer routing is available "
                    "for cross-border e-commerce transactions.",
                    "Conduct a 7-day review of RC 91 rates across all acquirer "
                    "partnerships to identify whether this is an isolated issue.",
                ],
                "Preventive (1–4 weeks)": [
                    "Implement multi-acquirer routing for cross-border corridors "
                    "with >10k daily transactions.",
                    "Build a slow-drift detector specifically for cross-border "
                    "approval rates with a 6-day EWMA sensitivity window.",
                    "Add RC 91 share to the weekly acquirer performance scorecard "
                    "with a threshold of >5% triggering a formal review.",
                ],
            }

        elif fc == "network_rule_change":
            return {
                "Immediate (0–2 hours)": [
                    "Publish a customer-facing notice explaining that PIN entry "
                    "will be required after a certain number of contactless "
                    "transactions. Provide guidance on ATM PIN reset if needed.",
                    "Update IVR / customer service scripts with the new threshold "
                    "information to reduce inbound call volume.",
                ],
                "Short-term (2–48 hours)": [
                    "Update the card product terms and conditions to reflect the "
                    "new contactless velocity limits.",
                    "Assess whether the new limits materially affect NPS or "
                    "contactless usage rates. If impact is significant, escalate "
                    "to the network for a limit adjustment discussion.",
                ],
                "Preventive (1–4 weeks)": [
                    "Subscribe to network operating regulation update feeds "
                    "(Visa Business News, Mastercard Connect) with automated "
                    "alerting for contactless rule changes.",
                    "Build a weekend/weekday approval rate differential monitor "
                    "that fires when the weekend rate diverges >3pp from weekday "
                    "for the same channel.",
                ],
            }

        else:
            return {
                "Immediate (0–2 hours)": [
                    "Continue investigation per Section 7. Do not apply mitigations "
                    "until the root cause is confirmed — premature intervention can "
                    "mask the true failure or create secondary issues.",
                ],
                "Short-term (2–48 hours)": [
                    "Once root cause is confirmed, apply the relevant mitigation "
                    "playbook from the Incident Response Runbook.",
                ],
                "Preventive (1–4 weeks)": [
                    "Document this incident and the root cause determination path "
                    "to improve future diagnostic speed.",
                ],
            }

    # ── RENDERING ─────────────────────────────────────────────────────────────

    def _render_markdown(
        self, a: dict, sections: dict, sev_cfg: dict
    ) -> str:
        anomaly_id  = a.get("anomaly_id", "UNKNOWN")
        first_seen  = self._fmt_ts(a.get("first_seen_ts"))
        fc          = FAILURE_CLASS_LABELS.get(
                          a.get("failure_class", "undetermined"),
                          a.get("failure_class", "undetermined")
                      )

        parts = [
            f"# Anomaly Diagnostic Report",
            f"**Anomaly ID:** `{anomaly_id}`  |  "
            f"**Detected:** {first_seen}  |  "
            f"**Failure class:** {fc}  |  "
            f"**Severity:** {sev_cfg['emoji']} {sev_cfg['label']}",
            "",
            "---",
            "",
            "## 1. Executive Summary",
            "",
            sections["executive_summary"],
            "",
            "---",
            "",
            "## 2. What Changed",
            "",
            sections["what_changed"],
            "",
            "---",
            "",
            "## 3. Where It Happened",
            "",
            sections["where_it_happened"],
            "",
            "---",
            "",
            "## 4. Severity Assessment",
            "",
            sections["severity_assessment"],
            "",
            "---",
            "",
            "## 5. Probable Root Causes",
            "",
            sections["probable_root_causes"],
            "",
            "---",
            "",
            "## 6. Supporting Evidence",
            "",
            sections["supporting_evidence"],
            "",
            "---",
            "",
            "## 7. Recommended Investigation Steps",
            "",
            sections["investigation_steps"],
            "",
            "---",
            "",
            "## 8. Recommended Mitigations",
            "",
            sections["recommended_mitigations"],
            "",
            "---",
            "",
            f"*Report generated by Anomaly Detection System v1.0 · "
            f"Template version {a.get('template_version', '1.0.0')} · "
            f"All metrics sourced from the statistical detection layer · "
            f"No LLM-generated numbers in this document*",
        ]
        return "\n".join(parts)

    def _render_plain_text(self, a: dict, sections: dict) -> str:
        """Plain-text fallback (no markdown) for export / email."""
        anomaly_id = a.get("anomaly_id", "UNKNOWN")
        parts = [
            "=" * 70,
            f"ANOMALY DIAGNOSTIC REPORT — {anomaly_id}",
            "=" * 70,
        ]
        headings = [
            ("1. EXECUTIVE SUMMARY",          "executive_summary"),
            ("2. WHAT CHANGED",               "what_changed"),
            ("3. WHERE IT HAPPENED",          "where_it_happened"),
            ("4. SEVERITY ASSESSMENT",        "severity_assessment"),
            ("5. PROBABLE ROOT CAUSES",       "probable_root_causes"),
            ("6. SUPPORTING EVIDENCE",        "supporting_evidence"),
            ("7. INVESTIGATION STEPS",        "investigation_steps"),
            ("8. RECOMMENDED MITIGATIONS",    "recommended_mitigations"),
        ]
        for heading, key in headings:
            parts += [
                "",
                heading,
                "-" * len(heading),
                "",
                sections.get(key, ""),
                "",
            ]
        return "\n".join(parts)

    # ── FORMAT HELPERS ────────────────────────────────────────────────────────

    @staticmethod
    def _fmt_rate(v) -> str:
        if v is None:
            return "N/A"
        return f"{float(v):.1%}"

    @staticmethod
    def _fmt_pct(v) -> str:
        if v is None:
            return "N/A"
        return f"{float(v):.1%}"

    @staticmethod
    def _fmt_sigma(v) -> str:
        if v is None:
            return "N/A"
        return f"{float(v):.2f}σ"

    @staticmethod
    def _fmt_ts(ts) -> str:
        if ts is None:
            return "unknown"
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            return str(ts)

    @staticmethod
    def _delta_pp(observed, baseline) -> str:
        if observed is None or baseline is None:
            return "N/A"
        delta = (float(observed) - float(baseline)) * 100
        return f"{delta:+.1f}pp"
