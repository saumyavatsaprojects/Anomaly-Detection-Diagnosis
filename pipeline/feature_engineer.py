"""
Feature Engineer
================
Layer 2 of the anomaly detection pipeline.

Consumes : data/raw_transactions.csv  (327k rows, hourly aggregates)
Produces : data/feature_store.csv     (same rows + engineered features)

Features computed
-----------------
Per-row (hourly slice level):
  approval_rate_computed     — approved / txn_count
  decline_rate               — declined / txn_count
  fraud_rate                 — fraud_count / txn_count

Per-slice rolling baselines  (slice = country × mcc_group × channel × auth_type):
  rolling_7d_approval_mean   — 7-day trailing mean of approval_rate
  rolling_7d_approval_std    — 7-day trailing std  of approval_rate
  approval_rate_zscore       — (observed - mean) / std
  rolling_7d_fraud_mean      — 7-day trailing mean of fraud_rate
  rolling_7d_fraud_std
  fraud_rate_zscore

Per slice × date daily aggregates:
  daily_txn_count            — total transactions per slice per day
  daily_approval_rate        — weighted approval rate per slice per day
  daily_fraud_rate

STL decomposition (per mcc_group × channel daily series):
  stl_trend                  — long-run trend component
  stl_seasonal               — weekly seasonal component
  stl_residual               — surprise component (what detectors act on)
  stl_residual_zscore        — residual normalised by its own std

Reason code distribution baseline (per mcc_group × channel × auth_type):
  rc_{code}_share            — share of declines for each RC code this hour
  rc_{code}_baseline_share   — 14-day rolling mean share (baseline)
  rc_{code}_delta_pp         — current - baseline in percentage points

STL fallback
------------
Slices with fewer than 336 data points (14 days × 24 hours) cannot
run STL with a weekly period. These slices fall back to a simple
rolling-mean-based residual using a 7-day window.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*STL.*")

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

RC_CODES          = ["05", "14", "51", "57", "59", "61", "65", "91", "96"]
RC_COLS           = [f"rc_{c}" for c in RC_CODES]

ROLLING_WINDOW_H  = 7 * 24        # 7 days in hours  — rate baseline
RC_BASELINE_H     = 14 * 24       # 14 days in hours — reason code baseline
STL_MIN_POINTS    = 336           # 14 × 24 — minimum for weekly STL
STL_PERIOD        = 24            # daily period for hourly data
STL_SEASONAL      = 7             # seasonal smoother (days)

SLICE_DIMS        = ["country", "mcc_group", "channel", "auth_type"]
DAILY_SLICE_DIMS  = ["mcc_group", "channel"]  # STL runs on broader aggregates


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Computes all features required by the four anomaly detectors.

    Usage:
        fe  = FeatureEngineer("data/raw_transactions.csv")
        out = fe.run()           # returns enriched DataFrame
        fe.save(out)             # writes data/feature_store.csv

    Or via run_pipeline.py:
        fe = FeatureEngineer()
        fe.run_and_save()
    """

    def __init__(
        self,
        input_path:  str = "data/raw_transactions.csv",
        output_path: str = "data/feature_store.csv",
    ) -> None:
        self.input_path  = input_path
        self.output_path = output_path
        self._df: Optional[pd.DataFrame] = None

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """Run all feature engineering steps. Returns enriched DataFrame."""
        logger.info("Feature engineering — loading %s", self.input_path)
        df = self._load()

        logger.info("  Step 1/5 — base rate features")
        df = self._base_rates(df)

        logger.info("  Step 2/5 — slice rolling baselines (Z-scores)")
        df = self._rolling_baselines(df)

        logger.info("  Step 3/5 — reason code share + baseline delta")
        df = self._reason_code_features(df)

        logger.info("  Step 4/5 — daily aggregates")
        df = self._daily_aggregates(df)

        logger.info("  Step 5/5 — STL decomposition (volume residuals)")
        df = self._stl_features(df)

        logger.info("  Feature store ready — %d rows × %d cols", *df.shape)
        self._df = df
        return df

    def save(self, df: Optional[pd.DataFrame] = None) -> None:
        """Write feature store to CSV."""
        out = df if df is not None else self._df
        if out is None:
            raise RuntimeError("No DataFrame to save — call run() first.")
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        out.to_csv(self.output_path, index=False)
        logger.info("  Feature store written → %s  (%d rows)", self.output_path, len(out))

    def run_and_save(self) -> pd.DataFrame:
        df = self.run()
        self.save(df)
        return df

    # ── STEP 1 — BASE RATES ───────────────────────────────────────────────────

    def _load(self) -> pd.DataFrame:
        df = pd.read_csv(self.input_path, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        # Clip edge cases
        df["txn_count"]    = df["txn_count"].clip(lower=1)
        df["approval_rate_computed"] = (
            df["approved_count"] / df["txn_count"]
        ).clip(0, 1)
        df["fraud_rate"]   = (
            df["fraud_count"] / df["txn_count"]
        ).clip(0, 1)
        df["decline_rate"] = (
            df["declined_count"] / df["txn_count"]
        ).clip(0, 1)
        return df

    def _base_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recompute rates from counts to ensure consistency."""
        df["approval_rate_computed"] = (
            df["approved_count"] / df["txn_count"].clip(lower=1)
        ).round(6)
        df["fraud_rate"] = (
            df["fraud_count"] / df["txn_count"].clip(lower=1)
        ).round(6)
        df["decline_rate"] = (
            df["declined_count"] / df["txn_count"].clip(lower=1)
        ).round(6)
        return df

    # ── STEP 2 — ROLLING BASELINES & Z-SCORES ────────────────────────────────

    def _rolling_baselines(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute 7-day rolling conditional baselines and Z-scores per slice.
        'Conditional' means the baseline is computed separately for each
        unique {country × mcc_group × channel × auth_type} combination,
        eliminating day-of-week and channel-specific patterns from the baseline.

        Uses min_periods=48 (2 days) so slices with limited history still
        get a baseline rather than producing NaN for the first week.
        """
        df = df.sort_values(["country", "mcc_group", "channel", "auth_type", "timestamp"])

        results = []
        groups  = df.groupby(SLICE_DIMS, observed=True)
        n_groups = len(groups)

        for i, (key, grp) in enumerate(groups):
            grp = grp.copy().sort_values("timestamp")

            for metric, col in [
                ("approval", "approval_rate_computed"),
                ("fraud",    "fraud_rate"),
                ("decline",  "decline_rate"),
            ]:
                roll   = grp[col].rolling(
                    window      = ROLLING_WINDOW_H,
                    min_periods = 48,
                )
                mean_col  = f"rolling_7d_{metric}_mean"
                std_col   = f"rolling_7d_{metric}_std"
                zscore_col = f"{metric}_rate_zscore"

                grp[mean_col]   = roll.mean().shift(1)   # shift 1 to avoid look-ahead
                raw_std         = roll.std().shift(1)
                # Floor: use the larger of the computed std OR 0.5% of the mean.
                # This prevents division-by-near-zero on ultra-stable sparse slices
                # (e.g. a route that is always approved at 100%) from producing
                # z-scores in the millions when a single off reading occurs.
                mean_floor      = (grp[mean_col].abs() * 0.005).clip(lower=0.001)
                grp[std_col]    = raw_std.clip(lower=1e-6).combine(mean_floor, max).fillna(0.005)
                grp[zscore_col] = (
                    (grp[col] - grp[mean_col]) / grp[std_col]
                ).clip(-15, 15).round(4)

            # EWMA for slow-drift detection (used by the drift detector)
            grp["ewma_approval_rate"] = (
                grp["approval_rate_computed"]
                .ewm(span=ROLLING_WINDOW_H, min_periods=24)
                .mean()
                .round(6)
            )

            results.append(grp)

        df = pd.concat(results).sort_values("timestamp").reset_index(drop=True)
        logger.info("    Rolling baselines: %d slices processed", n_groups)
        return df

    # ── STEP 3 — REASON CODE SHARE + BASELINE DELTA ──────────────────────────

    def _reason_code_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each RC code, compute:
          rc_{code}_share         — share of declines this hour
          rc_{code}_14d_baseline  — 14-day rolling mean share (shift-1)
          rc_{code}_delta_pp      — (current - baseline) × 100

        Only rows with declined_count > 0 get meaningful shares;
        rows with no declines get 0.0 for all RC shares.
        """
        df = df.sort_values(["mcc_group", "channel", "auth_type", "timestamp"])

        # Compute current shares
        denom = df["declined_count"].clip(lower=1)
        for code in RC_CODES:
            col            = f"rc_{code}"
            share_col      = f"rc_{code}_share"
            df[share_col]  = (df[col] / denom).round(6)
            # Zero out share where there are genuinely no declines
            df.loc[df["declined_count"] == 0, share_col] = 0.0

        # 14-day rolling baseline per {mcc_group × channel × auth_type}
        baseline_dims = ["mcc_group", "channel", "auth_type"]
        groups = df.groupby(baseline_dims, observed=True)

        baseline_frames = []
        for key, grp in groups:
            grp = grp.copy().sort_values("timestamp")
            for code in RC_CODES:
                share_col    = f"rc_{code}_share"
                base_col     = f"rc_{code}_14d_baseline"
                delta_col    = f"rc_{code}_delta_pp"
                grp[base_col] = (
                    grp[share_col]
                    .rolling(window=RC_BASELINE_H, min_periods=48)
                    .mean()
                    .shift(1)
                    .round(6)
                )
                grp[delta_col] = (
                    (grp[share_col] - grp[base_col]) * 100
                ).round(4)
            baseline_frames.append(grp)

        df = pd.concat(baseline_frames).sort_values("timestamp").reset_index(drop=True)
        logger.info("    Reason code features: %d RC codes × %d rows", len(RC_CODES), len(df))
        return df

    # ── STEP 4 — DAILY AGGREGATES ─────────────────────────────────────────────

    def _daily_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute daily aggregates per {mcc_group × channel} and join back
        onto the hourly rows. Used by STL decomposition and the volume detector.
        """
        df["date_only"] = pd.to_datetime(df["timestamp"]).dt.date

        daily = (
            df.groupby(["date_only", "mcc_group", "channel"], observed=True)
            .agg(
                daily_txn_count     = ("txn_count",             "sum"),
                daily_approved      = ("approved_count",        "sum"),
                daily_declined      = ("declined_count",        "sum"),
                daily_fraud         = ("fraud_count",           "sum"),
                daily_amount_usd    = ("txn_amount_usd",        "sum"),
            )
            .reset_index()
        )
        daily["daily_approval_rate"] = (
            daily["daily_approved"] / daily["daily_txn_count"].clip(lower=1)
        ).round(6)
        daily["daily_fraud_rate"] = (
            daily["daily_fraud"] / daily["daily_txn_count"].clip(lower=1)
        ).round(6)

        df = df.merge(daily, on=["date_only", "mcc_group", "channel"], how="left")
        logger.info("    Daily aggregates: %d day × slice combinations", len(daily))
        return df

    # ── STEP 5 — STL DECOMPOSITION ────────────────────────────────────────────

    def _stl_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run STL decomposition on the daily txn_count series per
        {mcc_group × channel} aggregate. Produces:
          stl_trend, stl_seasonal, stl_residual, stl_residual_zscore

        Falls back to a rolling-mean residual for series with < STL_MIN_POINTS.
        The STL features are computed at the daily level and broadcast
        back onto hourly rows via the date join.
        """
        # Build one daily series per mcc_group × channel
        daily_key = df.drop_duplicates(["date_only", "mcc_group", "channel"])[
            ["date_only", "mcc_group", "channel",
             "daily_txn_count", "daily_approval_rate"]
        ].copy()
        daily_key = daily_key.sort_values(["mcc_group", "channel", "date_only"])

        stl_results = []
        groups = daily_key.groupby(["mcc_group", "channel"], observed=True)

        stl_available = self._check_stl_available()

        for (mcc, ch), grp in groups:
            grp = grp.copy().sort_values("date_only").reset_index(drop=True)
            n   = len(grp)

            if stl_available and n >= STL_MIN_POINTS:
                grp = self._run_stl(grp, mcc, ch)
            else:
                grp = self._rolling_residual(grp, mcc, ch, n)

            stl_results.append(grp)

        stl_daily = pd.concat(stl_results, ignore_index=True)

        # Broadcast daily STL features back to hourly rows
        df = df.merge(
            stl_daily[[
                "date_only", "mcc_group", "channel",
                "stl_trend", "stl_seasonal", "stl_residual", "stl_residual_zscore",
                "stl_method",
            ]],
            on=["date_only", "mcc_group", "channel"],
            how="left",
        )
        n_stl     = (stl_daily["stl_method"] == "stl").sum()
        n_rolling = (stl_daily["stl_method"] == "rolling").sum()
        logger.info(
            "    STL decomposition: %d series via STL, %d via rolling fallback",
            n_stl, n_rolling,
        )
        return df

    def _check_stl_available(self) -> bool:
        """Check whether statsmodels STL is importable."""
        try:
            from statsmodels.tsa.seasonal import STL  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "statsmodels not found — using rolling residual fallback for all series. "
                "Install with: pip install statsmodels>=0.14.0"
            )
            return False

    def _run_stl(self, grp: pd.DataFrame, mcc: str, ch: str) -> pd.DataFrame:
        """Run statsmodels STL on a daily txn_count series."""
        from statsmodels.tsa.seasonal import STL

        series = grp["daily_txn_count"].values.astype(float)
        # Replace zeros with small positive value to avoid STL instability
        series = np.where(series <= 0, 0.1, series)

        try:
            stl = STL(
                series,
                period         = 7,       # weekly seasonality in daily data
                seasonal       = STL_SEASONAL,
                trend          = None,    # auto
                robust         = True,    # robust to outliers (anomalies)
            )
            res = stl.fit()

            grp["stl_trend"]    = res.trend.round(4)
            grp["stl_seasonal"] = res.seasonal.round(4)
            grp["stl_residual"] = res.resid.round(4)
            grp["stl_method"]   = "stl"

        except Exception as exc:
            logger.warning("STL failed for %s/%s: %s — using rolling fallback", mcc, ch, exc)
            grp = self._rolling_residual(grp, mcc, ch, len(grp))
            return grp

        # Z-score the residual using its own rolling std (30-day window)
        resid_std = (
            pd.Series(grp["stl_residual"])
            .rolling(30, min_periods=7)
            .std()
            .fillna(method="bfill")
            .clip(lower=1.0)
        )
        grp["stl_residual_zscore"] = (grp["stl_residual"] / resid_std).round(4)
        return grp

    def _rolling_residual(
        self, grp: pd.DataFrame, mcc: str, ch: str, n: int
    ) -> pd.DataFrame:
        """
        Fallback when STL cannot run. Computes residual as:
            observed - 7-day rolling mean (treat rolling mean as trend+seasonal).
        """
        series = pd.Series(grp["daily_txn_count"].values.astype(float))
        window = min(7, max(2, n // 2))

        trend    = series.rolling(window, min_periods=1, center=True).mean()
        residual = series - trend

        grp["stl_trend"]    = trend.round(4)
        grp["stl_seasonal"] = 0.0
        grp["stl_residual"] = residual.round(4)
        grp["stl_method"]   = "rolling"

        resid_std = residual.rolling(30, min_periods=3).std().fillna(1.0).clip(lower=1.0)
        grp["stl_residual_zscore"] = (residual / resid_std).round(4)
        return grp


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY STATS  (called by run_pipeline.py for progress reporting)
# ─────────────────────────────────────────────────────────────────────────────

def summarise_feature_store(df: pd.DataFrame) -> None:
    """Print a human-readable summary of the feature store."""
    print(f"  Rows          : {len(df):,}")
    print(f"  Columns       : {len(df.columns)}")
    print(f"  Date range    : {df['timestamp'].min()} → {df['timestamp'].max()}")

    # Coverage of key engineered features
    for col in [
        "rolling_7d_approval_mean",
        "approval_rate_zscore",
        "rc_65_14d_baseline",
        "stl_residual",
        "stl_residual_zscore",
        "ewma_approval_rate",
    ]:
        if col in df.columns:
            non_null = df[col].notna().sum()
            pct      = 100 * non_null / len(df)
            print(f"  {col:<35s}: {non_null:>8,} non-null ({pct:.1f}%)")
        else:
            print(f"  {col:<35s}: MISSING")

    # STL method breakdown
    if "stl_method" in df.columns:
        counts = df.drop_duplicates(["date_only","mcc_group","channel"])["stl_method"].value_counts()
        print(f"  STL method breakdown: {counts.to_dict()}")

    # Z-score range check (sanity — extreme values indicate an issue)
    zs = df["approval_rate_zscore"].dropna()
    print(f"  approval_rate_zscore range: [{zs.min():.2f}, {zs.max():.2f}]")

    # Anomaly preview: rows with |zscore| > 3
    high_z = df[df["approval_rate_zscore"].abs() > 3.0]
    print(f"  Rows with |approval z| > 3.0: {len(high_z):,}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    inp = sys.argv[1] if len(sys.argv) > 1 else "data/raw_transactions.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/feature_store.csv"

    fe  = FeatureEngineer(inp, out)
    df  = fe.run_and_save()

    print("\nFeature store summary:")
    summarise_feature_store(df)
