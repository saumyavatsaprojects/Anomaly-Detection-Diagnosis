"""
Detection test suite — proves the system works, not just "seems to work".

Run with:  pytest tests/ -v

Tests cover:
  - All 5 injected anomalies detected in final output (A1-A5)
  - A4 detected by the EWMA detector, not a hardcoded scan
  - Individual detector recall for each anomaly type
  - False positive rate on clean (pre-injection) window
  - Sigma not saturated (informative distribution)
  - Batch rank present on all anomaly objects
  - Verifier catches value mismatches, not just missing citations
  - Financial context (revenue_at_risk, interchange_loss) computed
  - Config drives interchange rate
"""
import json, logging, os, sys, types
from pathlib import Path

import pandas as pd
import pytest

logging.disable(logging.CRITICAL)

# ── Mock Streamlit ─────────────────────────────────────────────────────────
def _mock_st():
    st = types.ModuleType("streamlit")
    st.cache_data = st.cache_resource = lambda **kw: (lambda f: f)
    st.session_state = {}
    for a in ["markdown","caption","warning","error","info","success","metric",
              "button","write","expander","divider","rerun","stop","spinner",
              "empty","columns","tabs","sidebar","chat_message","chat_input",
              "text_input","components"]:
        setattr(st, a, lambda *a,**kw: None)
    class _C:
        def __enter__(self): return self
        def __exit__(self,*a): return False
        def html(self,*a,**kw): pass
    for a in ["chat_message","expander","spinner","sidebar","columns","tabs"]:
        setattr(st, a, lambda *a,**kw: _C())
    st.columns = lambda n,**kw: [_C()]*(n if isinstance(n,int) else len(n))
    st.components = types.ModuleType("st.components")
    st.components.v1 = _C()
    return st

for mod in ["streamlit","plotly","plotly.graph_objects","plotly.subplots","groq"]:
    sys.modules[mod] = types.ModuleType(mod)
sys.modules["streamlit"] = _mock_st()
sys.modules["groq"].Groq = type("Groq",(),{"__init__":lambda s,**k:None})
sys.modules["groq"].RateLimitError = Exception
sys.modules["groq"].APIStatusError = Exception

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def feature_store():
    p = ROOT / "data" / "feature_store.csv"
    if not p.exists():
        pytest.skip("Feature store not found — run run_pipeline.py first")
    return pd.read_csv(p, parse_dates=["timestamp"])


@pytest.fixture(scope="session")
def anomaly_objects():
    p = ROOT / "data" / "anomaly_objects.json"
    if not p.exists():
        pytest.skip("anomaly_objects.json not found — run run_pipeline.py first")
    with open(p) as f:
        return json.load(f)


# ── A1-A5 end-to-end ──────────────────────────────────────────────────────────

def test_all_five_failure_classes_detected(anomaly_objects):
    fcs = {a["failure_class"] for a in anomaly_objects}
    missing = {"processor_outage","3ds_acs_failure","fraud_attack",
               "acquirer_routing","network_rule_change"} - fcs
    assert not missing, f"Missing failure classes: {missing}"


def test_a1_processor_outage_on_mar23(anomaly_objects):
    a1 = [a for a in anomaly_objects
          if a["failure_class"]=="processor_outage" and "2024-03-23" in a["first_seen_ts"]]
    assert a1, "A1 processor outage not in final anomalies"
    assert a1[0]["severity"] in ("critical","high")


def test_a2_3ds_cascade_rc65(anomaly_objects):
    a2 = [a for a in anomaly_objects
          if a["failure_class"]=="3ds_acs_failure" and "2024-04-10" in a["first_seen_ts"]]
    assert a2, "A2 3DS cascade not found"
    assert a2[0]["reason_code_evidence"].get("65",{}).get("delta_pp",0) > 20


def test_a3_fraud_attack_in_april(anomaly_objects):
    a3 = [a for a in anomaly_objects
          if a["failure_class"]=="fraud_attack"
          and a["first_seen_ts"][:7] in ("2024-04","2024-05")]
    assert a3, "No fraud_attack found in Apr-May"


def test_a4_ewma_detector_finds_may_drift(feature_store):
    """A4 must be detected by the EWMA detector in May, not by a hardcoded scan."""
    from detectors.rate_detector import RateDetector
    cands = RateDetector().detect(feature_store)
    may_drift = [c for c in cands
                 if c.detector_type=="rate_drift"
                 and c.affected_slice.get("corridor")=="cross_border"
                 and c.first_seen_ts[:7]=="2024-05"]
    assert may_drift, "A4 cross-border drift not detected by EWMA in May"
    assert abs(may_drift[0].deviation_sigma) > 0.2, "EWMA sigma is trivially small for slow-drift signal"


def test_a5_network_rule_rc61(anomaly_objects):
    a5 = [a for a in anomaly_objects if a["failure_class"]=="network_rule_change"]
    assert a5, "A5 not found"
    assert a5[0]["reason_code_evidence"].get("61",{}).get("delta_pp",0) > 10


# ── Detector-level recall ──────────────────────────────────────────────────────

def test_rate_detector_a1_a2(feature_store):
    from detectors.rate_detector import RateDetector
    cands = RateDetector().detect(feature_store)
    assert any("2024-03-23" in c.first_seen_ts and c.deviation_sigma < -4
               for c in cands), "A1 not in rate detector output"
    assert any("2024-04-10" in c.first_seen_ts
               and c.affected_slice.get("auth_type")=="3DS"
               for c in cands), "A2 not in rate detector output"


def test_rc_detector_rc96_rc65(feature_store):
    from detectors.reason_code_detector import ReasonCodeDetector
    cands = ReasonCodeDetector().detect(feature_store)
    assert any(c.reason_code_evidence.get("96",{}).get("delta_pp",0)>30
               for c in cands), "RC 96 spike not detected"
    assert any(c.reason_code_evidence.get("65",{}).get("delta_pp",0)>20
               for c in cands), "RC 65 spike not detected"


def test_volume_detector_bin4531_retry_storm(feature_store):
    from detectors.volume_detector import VolumeDetector
    cands = VolumeDetector().detect(feature_store)
    hit = [c for c in cands
           if c.affected_slice.get("bin_bucket")=="4531xx"
           and "2024-03-23" in c.first_seen_ts]
    assert hit, "A1 retry storm not in volume detector"
    assert hit[0].volume_evidence.get("volume_change_pct",0) > 10


def test_fraud_detector_a3_concentration(feature_store):
    from detectors.fraud_concentration import FraudConcentrationDetector
    cands = FraudConcentrationDetector().detect(feature_store)
    assert any(c.first_seen_ts[:7] in ("2024-04","2024-05") for c in cands), \
        "No fraud concentration found in Apr-May window"


# ── Quality & calibration ─────────────────────────────────────────────────────

def test_false_positive_rate_clean_window(feature_store):
    """Pre-injection window (before Mar 15) must have ≤3 FPs total."""
    # Start from day 15 (baselines need 14 days to warm up) and end before A1 injection
    clean = feature_store[(feature_store["timestamp"] >= "2024-03-15") & (feature_store["timestamp"] < "2024-03-20")].copy()
    if len(clean) < 500:
        pytest.skip("Insufficient clean data")
    from detectors.volume_detector import VolumeDetector
    from detectors.rate_detector import RateDetector
    fps = len(VolumeDetector().detect(clean))
    fps += len([c for c in RateDetector().detect(clean)
                if c.detector_type=="rate_drop"])
    assert fps <= 3, f"Too many false positives on clean window: {fps}"


def test_sigma_not_saturated(anomaly_objects):
    """At least 30% of anomaly objects must have distinct |sigma| (rounding to 1dp)."""
    import numpy as np
    sigmas = [abs(a["deviation_sigma"]) for a in anomaly_objects]
    unique = len(set(round(s,1) for s in sigmas))
    assert unique/len(sigmas) >= 0.30, \
        f"Sigma saturated: only {unique}/{len(sigmas)} distinct values"


def test_batch_rank_present(anomaly_objects):
    """Every anomaly object must have a batch rank in its evidence."""
    missing = [a["anomaly_id"] for a in anomaly_objects
               if not any("Batch rank" in ev for ev in a.get("evidence",[]))]
    assert not missing, f"Missing batch rank: {missing}"


# ── Verifier accuracy ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def ctx(anomaly_objects):
    from llm.context_builder import ContextBuilder
    return ContextBuilder().build_initial_context(anomaly_objects[0])


def test_verifier_correct_value_passes(ctx):
    from llm.output_verifier import OutputVerifier
    ov  = OutputVerifier()
    obs = ctx.supporting_stats["approval_rate_observed"]
    r   = ov.verify(f"Approval rate was {obs} [approval_rate_observed].",
                    ctx.supporting_stats)
    assert r.hallucination_risk == "none", f"Correct value flagged: {r.warnings}"


def test_verifier_large_wrong_value_flagged(ctx):
    from llm.output_verifier import OutputVerifier
    ov = OutputVerifier()
    r  = ov.verify("Revenue at risk is £500,000 [revenue_at_risk].",
                   ctx.supporting_stats)
    assert r.hallucination_risk in ("medium","high"), \
        f"Wrong value not flagged: {r.hallucination_risk}"
    assert r.value_mismatches, "No value mismatch recorded"


def test_verifier_close_value_passes(ctx):
    from llm.output_verifier import OutputVerifier, _parse_to_float
    ov  = OutputVerifier()
    rev = ctx.supporting_stats.get("revenue_at_risk","£3000")
    av  = _parse_to_float(str(rev))
    if av is None:
        pytest.skip("revenue_at_risk not numeric")
    close = f"£{av*1.15:,.0f}"  # 15% off — within 25% tolerance
    r = ov.verify(f"Revenue at risk is {close} [revenue_at_risk].",
                  ctx.supporting_stats)
    assert r.hallucination_risk == "none", \
        f"Close value incorrectly flagged: {r.warnings}"


def test_verifier_uncited_is_low_not_high(ctx):
    from llm.output_verifier import OutputVerifier
    ov = OutputVerifier()
    r  = ov.verify("Revenue at risk is £500,000.", ctx.supporting_stats)
    assert r.hallucination_risk in ("low","medium"), \
        f"Uncited number should be low/medium, got {r.hallucination_risk}"


# ── Financial context ─────────────────────────────────────────────────────────

def test_revenue_at_risk_in_context(anomaly_objects):
    from llm.context_builder import ContextBuilder
    cb = ContextBuilder()
    rate_drops = [a for a in anomaly_objects
                  if a["observed_value"] < a["baseline_value"]]
    assert rate_drops, "No rate-drop anomalies found"
    ctx = cb.build_initial_context(rate_drops[0])
    assert "revenue_at_risk" in ctx.supporting_stats
    assert "interchange_loss" in ctx.supporting_stats


def test_financial_section_in_brief(anomaly_objects):
    from llm.context_builder import ContextBuilder
    ctx = ContextBuilder().build_initial_context(anomaly_objects[0])
    assert "FINANCIAL IMPACT" in ctx.initial_brief


def test_interchange_rate_from_config():
    from llm.context_builder import _INTERCHANGE_RATE
    assert 0 < _INTERCHANGE_RATE <= 0.05, \
        f"Interchange rate invalid: {_INTERCHANGE_RATE}"
