"""
Pipeline Runner
===============
End-to-end CLI for the anomaly detection pipeline.

Runs all four stages in sequence:
  [1/4] Data generation      → data/raw_transactions.csv
  [2/4] Feature engineering  → data/feature_store.csv
  [3/4] Anomaly detection    → (in memory)
  [4/4] Root cause attribution → data/anomaly_objects.json

Usage
-----
  # Full pipeline (first run or to regenerate from scratch)
  python run_pipeline.py

  # Skip data generation (reuse existing raw_transactions.csv)
  python run_pipeline.py --skip-generate

  # Skip generation + feature engineering (reuse feature_store.csv)
  python run_pipeline.py --skip-features

  # Verbose logging
  python run_pipeline.py --verbose

Colab usage
-----------
  !python run_pipeline.py

  Or cell by cell:
    from run_pipeline import run_generate, run_features, run_detectors, run_attribution
    run_generate()
    run_features()
    candidates = run_detectors()
    run_attribution(candidates)

Streamlit auto-run
------------------
  app.py calls run_pipeline_if_needed() on startup, which runs the
  full pipeline only if data/anomaly_objects.json does not exist.
  Subsequent loads skip the pipeline entirely (fast startup).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR              = "data"
RAW_TRANSACTIONS_PATH = os.path.join(DATA_DIR, "raw_transactions.csv")
FEATURE_STORE_PATH    = os.path.join(DATA_DIR, "feature_store.csv")
ANOMALY_OBJECTS_PATH  = os.path.join(DATA_DIR, "anomaly_objects.json")


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL STAGE RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def run_generate(output_path: str = RAW_TRANSACTIONS_PATH) -> None:
    """Stage 1 — generate synthetic transaction data."""
    from pipeline.data_generator import generate_dataset

    print("\n[1/4] Generating synthetic transaction data...")
    t0 = time.time()

    # Patch output to use CSV (parquet not available in all envs)
    df = generate_dataset(output_path=output_path)

    elapsed = time.time() - t0
    print(f"      Rows: {len(df):,} | Txns: {df['txn_count'].sum():,.0f} | "
          f"Time: {elapsed:.1f}s")


def run_features(
    input_path:  str = RAW_TRANSACTIONS_PATH,
    output_path: str = FEATURE_STORE_PATH,
) -> None:
    """Stage 2 — feature engineering."""
    from pipeline.feature_engineer import FeatureEngineer, summarise_feature_store

    print("\n[2/4] Engineering features...")
    t0 = time.time()

    fe = FeatureEngineer(input_path=input_path, output_path=output_path)
    df = fe.run_and_save()

    elapsed = time.time() - t0
    print(f"      Rows: {len(df):,} | Columns: {len(df.columns)} | "
          f"Time: {elapsed:.1f}s")


def run_detectors(
    feature_store_path: str = FEATURE_STORE_PATH,
) -> list:
    """Stage 3 — run all four detectors and return candidate list."""
    import pandas as pd
    from detectors.rate_detector         import RateDetector
    from detectors.reason_code_detector  import ReasonCodeDetector
    from detectors.fraud_concentration   import FraudConcentrationDetector
    from detectors.volume_detector       import VolumeDetector

    print("\n[3/4] Running anomaly detectors...")
    t0 = time.time()

    df = pd.read_csv(feature_store_path, parse_dates=["timestamp"])
    print(f"      Feature store: {len(df):,} rows")

    all_candidates = []
    detector_results = {}

    for name, Detector in [
        ("Volume detector    ", VolumeDetector),
        ("Rate detector      ", RateDetector),
        ("Reason code detector", ReasonCodeDetector),
        ("Fraud detector     ", FraudConcentrationDetector),
    ]:
        t_det = time.time()
        cands = Detector().detect(df)
        elapsed_det = time.time() - t_det
        print(f"      {name}: {len(cands):>3} anomalies  ({elapsed_det:.1f}s)")
        detector_results[name.strip()] = len(cands)
        all_candidates.extend(cands)

    elapsed = time.time() - t0
    print(f"      Total candidates: {len(all_candidates)} | Time: {elapsed:.1f}s")
    return all_candidates


def run_attribution(
    candidates:         list,
    feature_store_path: str = FEATURE_STORE_PATH,
    output_path:        str = ANOMALY_OBJECTS_PATH,
    max_anomalies:      int = 20,
) -> list[dict]:
    """Stage 4 — root cause attribution and JSON output."""
    from pipeline.root_cause import RootCauseAttributor

    print("\n[4/4] Root cause attribution...")
    t0 = time.time()

    attributor = RootCauseAttributor(
        feature_store_path  = feature_store_path,
        output_path         = output_path,
        max_final_anomalies = max_anomalies,
    )
    result = attributor.run_and_save(candidates)

    elapsed = time.time() - t0
    print(f"      Final anomaly objects: {len(result)} | Time: {elapsed:.1f}s")
    print(f"      Written → {output_path}")

    # Print summary table
    _print_anomaly_summary(result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def _print_anomaly_summary(anomaly_dicts: list[dict]) -> None:
    """Print a clean table of the final anomaly objects."""
    sev_icon = {
        "critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"
    }
    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  FINAL ANOMALY OBJECTS                                      │")
    print("  ├──────────────┬──────────┬────────────┬──────────────────────┤")
    print("  │ ID           │ Severity │ First seen │ Failure class        │")
    print("  ├──────────────┼──────────┼────────────┼──────────────────────┤")
    for a in anomaly_dicts:
        sev  = a.get("severity", "?")
        icon = sev_icon.get(sev, " ")
        aid  = a.get("anomaly_id", "?")[:12]
        ts   = str(a.get("first_seen_ts", "?"))[:10]
        fc   = a.get("failure_class", "?").replace("_", " ")[:20]
        print(f"  │ {aid:<12s} │ {icon} {sev:<6s} │ {ts:<10s} │ {fc:<20s} │")
    print("  └──────────────┴──────────┴────────────┴──────────────────────┘")


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT AUTO-RUN  (called by app.py on startup)
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline_if_needed(
    force:      bool = False,
    skip_gen:   bool = False,
    skip_feat:  bool = False,
    verbose:    bool = False,
) -> list[dict]:
    """
    Run the full pipeline only if anomaly_objects.json does not exist.
    Called by app.py at startup.

    Returns the loaded anomaly_objects list.
    """
    import json

    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="  %(message)s")

    os.makedirs(DATA_DIR, exist_ok=True)

    needs_run = (
        force
        or not os.path.exists(ANOMALY_OBJECTS_PATH)
        or os.path.getsize(ANOMALY_OBJECTS_PATH) < 100
    )

    if needs_run:
        print("Pipeline: generating data and running detectors...")
        _run_full(skip_gen=skip_gen, skip_feat=skip_feat)

    with open(ANOMALY_OBJECTS_PATH) as f:
        return json.load(f)


def _run_full(
    skip_gen:  bool = False,
    skip_feat: bool = False,
) -> None:
    """Run all pipeline stages."""
    if not skip_gen and not os.path.exists(RAW_TRANSACTIONS_PATH):
        run_generate()
    elif not skip_gen:
        print(f"      Reusing existing {RAW_TRANSACTIONS_PATH}")
    else:
        print(f"      Skipping data generation (--skip-generate)")

    if not skip_feat and not os.path.exists(FEATURE_STORE_PATH):
        run_features()
    elif not skip_feat and skip_gen:
        run_features()   # always re-run features if explicitly requested
    else:
        print(f"      Skipping feature engineering (--skip-features or reusing)")

    # Always re-run detection and attribution when pipeline runs
    candidates = run_detectors()
    run_attribution(candidates)


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Anomaly Detection Pipeline — end-to-end runner",
    )
    parser.add_argument(
        "--skip-generate", action="store_true",
        help="Skip data generation (reuse existing raw_transactions.csv)",
    )
    parser.add_argument(
        "--skip-features", action="store_true",
        help="Skip feature engineering (reuse existing feature_store.csv)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-run even if outputs already exist",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--max-anomalies", type=int, default=20,
        help="Maximum final anomaly objects to output (default: 20)",
    )
    args = parser.parse_args()

    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="  %(message)s")

    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 65)
    print("  Transaction Anomaly Detection — Pipeline Runner")
    print("=" * 65)

    t_total = time.time()

    # Stage 1
    if args.skip_generate and os.path.exists(RAW_TRANSACTIONS_PATH):
        print(f"\n[1/4] Skipping data generation (file exists)")
    else:
        run_generate()

    # Stage 2
    if args.skip_features and os.path.exists(FEATURE_STORE_PATH):
        print(f"\n[2/4] Skipping feature engineering (file exists)")
    else:
        run_features()

    # Stage 3 + 4 — always run
    candidates = run_detectors()
    run_attribution(candidates, max_anomalies=args.max_anomalies)

    total = time.time() - t_total
    print(f"\n{'=' * 65}")
    print(f"  Pipeline complete — total time: {total:.1f}s")
    print(f"  Output: {ANOMALY_OBJECTS_PATH}")
    print(f"{'=' * 65}\n")


def run_incremental_pipeline(
    lookback_days: int = 7,
    verbose: bool = False,
) -> dict:
    """
    Fix 4: Incremental pipeline — process only the last N days.

    Architecture demonstration for sub-5-minute alert latency.
    In production this would be triggered by a Kafka consumer
    or a scheduled Lambda/Cloud Function every 5 minutes.

    On Streamlit Community Cloud:
    - First run: full pipeline (8-12 min, cached)
    - Subsequent: incremental run on last 7 days (~45 seconds)
    - UI "Refresh alerts" button calls this function

    Returns
    -------
    dict with keys:
        anomalies_new      : list of new/changed anomaly objects
        anomalies_unchanged: count of unchanged carry-forward
        run_duration_s     : wall-clock seconds
        window_start       : ISO timestamp of window start
        window_end         : ISO timestamp of window end
    """
    import time
    from datetime import datetime, timedelta

    t0 = time.time()
    log = logging.getLogger(__name__)

    # Load existing baseline
    baseline_path = DATA_DIR / "anomaly_objects.json"
    feature_path  = DATA_DIR / "feature_store.csv"

    if not baseline_path.exists() or not feature_path.exists():
        log.warning("Incremental: no baseline found — running full pipeline")
        anomalies = run_pipeline_if_needed(force=True, verbose=verbose)
        return {
            "anomalies_new":       anomalies,
            "anomalies_unchanged": 0,
            "run_duration_s":      round(time.time() - t0, 1),
            "window_start":        "",
            "window_end":          datetime.utcnow().isoformat(),
        }

    # Load existing anomalies for carry-forward
    with open(baseline_path) as f:
        existing = json.load(f)

    # Load feature store and filter to incremental window
    import pandas as pd
    df = pd.read_csv(feature_path, parse_dates=["timestamp"])
    window_end   = df["timestamp"].max()
    window_start = window_end - pd.Timedelta(days=lookback_days)

    df_window = df[df["timestamp"] >= window_start].copy()
    if verbose:
        print(f"  Incremental window: {window_start.date()} → {window_end.date()}")
        print(f"  Window rows: {len(df_window):,} / {len(df):,} total")

    # Run detectors only on the incremental window
    # Provide the full dataset as baseline context but only detect on window
    from detectors.rate_detector       import RateDetector
    from detectors.volume_detector     import VolumeDetector
    from detectors.reason_code_detector import ReasonCodeDetector
    from detectors.fraud_concentration import FraudConcentrationDetector
    from pipeline.root_cause           import RootCauseAttributor

    all_candidates = []
    for DetectorClass in [RateDetector, VolumeDetector,
                          ReasonCodeDetector, FraudConcentrationDetector]:
        try:
            # Pass full df so baselines are stable; detector filters internally
            cands = DetectorClass().detect(df)
            # Keep only candidates whose first_seen_ts is in the window
            window_cands = [
                c for c in cands
                if pd.Timestamp(c.first_seen_ts.replace("Z","")) >= window_start
            ]
            all_candidates.extend(window_cands)
        except Exception as exc:
            log.warning("Detector %s failed: %s", DetectorClass.__name__, exc)

    if verbose:
        print(f"  New candidates in window: {len(all_candidates)}")

    # Run root cause attribution on new candidates
    new_anomalies = []
    if all_candidates:
        try:
            attributor = RootCauseAttributor(feature_store_path=str(feature_path))
            new_anomalies = attributor.attribute(all_candidates)
        except Exception as exc:
            log.error("Root cause attribution failed: %s", exc)

    # Reset incident memory so it picks up new anomalies
    try:
        from llm.incident_memory import IncidentMemory
        IncidentMemory.reset()
    except Exception:
        pass

    duration = round(time.time() - t0, 1)
    if verbose:
        print(f"  Incremental run complete: {len(new_anomalies)} new anomalies in {duration}s")

    return {
        "anomalies_new":       new_anomalies,
        "anomalies_unchanged": len([
            a for a in existing
            if not any(n.get("anomaly_id")==a.get("anomaly_id") for n in new_anomalies)
        ]),
        "run_duration_s":      duration,
        "window_start":        window_start.isoformat(),
        "window_end":          window_end.isoformat(),
    }


if __name__ == "__main__":
    main()
