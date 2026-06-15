"""
Synthetic Transaction Data Generator
=====================================
Generates 90 days of hourly card transaction aggregates for a mid-sized
European card issuer. Produces realistic seasonality across day-of-week,
hour-of-day, country timezone, and MCC-specific patterns.

Injects 5 operationally realistic anomalies covering:
  1. Issuer processor BIN range outage
  2. 3DS ACS cascade failure (DE/NL corridors)
  3. MCC-targeted fraud attack (GB grocery CNP)
  4. Silent cross-border approval rate erosion
  5. Weekend contactless network rule change

Output: data/raw_transactions.parquet
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# SEED & DATE RANGE
# ─────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)

START_DATE = datetime(2024, 3, 1)
END_DATE   = datetime(2024, 5, 30)   # 90 days
FREQ       = "h"                     # hourly aggregates


# ─────────────────────────────────────────────
# DIMENSION DEFINITIONS
# ─────────────────────────────────────────────

COUNTRIES = {
    # country: (tz_offset_hours, home_market, base_volume_weight)
    "GB": ( 0, True,  1.00),
    "DE": ( 1, True,  0.85),
    "FR": ( 1, True,  0.78),
    "NL": ( 1, True,  0.52),
    "SE": ( 1, True,  0.41),
    "US": (-5, False, 0.22),   # cross-border; UTC-5 but card is EU-issued
    "SG": ( 8, False, 0.14),
    "AE": ( 4, False, 0.11),
}

MCC_GROUPS = {
    # mcc_group: (base_volume_weight, base_fraud_rate, base_approval_rate,
    #             avg_ticket_gbp, channel_profile, time_profile)
    "grocery":      (1.00, 0.0008, 0.955, 42,  "mixed",   "retail"),
    "fuel":         (0.62, 0.0006, 0.961, 68,  "cp_heavy","retail"),
    "dining":       (0.55, 0.0012, 0.941, 35,  "mixed",   "retail"),
    "retail":       (0.72, 0.0014, 0.938, 78,  "mixed",   "retail"),
    "digital_goods":(0.38, 0.0045, 0.891, 29,  "cnp_only","digital"),
    "travel":       (0.18, 0.0021, 0.907, 312, "cnp_heavy","digital"),
    "entertainment":(0.29, 0.0018, 0.929, 55,  "mixed",   "digital"),
    "utilities":    (0.15, 0.0003, 0.971, 95,  "cnp_only","retail"),
}

CHANNELS = ["ecom", "pos", "contactless"]

DECLINE_REASON_CODES = {
    # code: (label, base_share_by_mcc_group)
    "05": "Do not honor",
    "14": "Invalid card number",
    "51": "Insufficient funds",
    "57": "Transaction not permitted",
    "59": "Suspected fraud",
    "61": "Exceeds withdrawal frequency limit",
    "65": "Soft decline – authentication required",
    "91": "Issuer inoperative",
    "96": "System malfunction",
}

# Base decline reason distribution per MCC group (must sum to 1.0 per group)
MCC_DECLINE_DIST = {
    "grocery":       {"05":0.12,"14":0.05,"51":0.55,"57":0.03,"59":0.08,"61":0.04,"65":0.06,"91":0.04,"96":0.03},
    "fuel":          {"05":0.10,"14":0.04,"51":0.52,"57":0.05,"59":0.07,"61":0.06,"65":0.05,"91":0.06,"96":0.05},
    "dining":        {"05":0.14,"14":0.06,"51":0.48,"57":0.04,"59":0.10,"61":0.03,"65":0.07,"91":0.04,"96":0.04},
    "retail":        {"05":0.13,"14":0.07,"51":0.44,"57":0.05,"59":0.11,"61":0.04,"65":0.09,"91":0.04,"96":0.03},
    "digital_goods": {"05":0.08,"14":0.14,"51":0.28,"57":0.08,"59":0.18,"61":0.03,"65":0.12,"91":0.05,"96":0.04},
    "travel":        {"05":0.18,"14":0.08,"51":0.22,"57":0.12,"59":0.12,"61":0.02,"65":0.14,"91":0.07,"96":0.05},
    "entertainment": {"05":0.11,"14":0.09,"51":0.35,"57":0.06,"59":0.14,"61":0.04,"65":0.11,"91":0.05,"96":0.05},
    "utilities":     {"05":0.09,"14":0.04,"51":0.45,"57":0.06,"59":0.06,"61":0.08,"65":0.05,"91":0.11,"96":0.06},
}

BIN_BUCKETS = ["4111xx","4531xx","5204xx","5490xx","4929xx"]


# ─────────────────────────────────────────────
# SEASONALITY FACTORS
# ─────────────────────────────────────────────

DOW_FACTOR = {0:0.92, 1:0.91, 2:0.88, 3:0.90, 4:1.12, 5:1.30, 6:0.97}
# Mon=0 … Sun=6

HOUR_PROFILES = {
    "retail": {
        0:0.08, 1:0.05, 2:0.04, 3:0.03, 4:0.04, 5:0.07,
        6:0.18, 7:0.35, 8:0.55, 9:0.72, 10:0.85, 11:0.92,
        12:1.00, 13:0.95, 14:0.88, 15:0.90, 16:0.95, 17:0.98,
        18:1.00, 19:0.92, 20:0.80, 21:0.65, 22:0.45, 23:0.25
    },
    "digital": {
        0:0.22, 1:0.15, 2:0.10, 3:0.08, 4:0.08, 5:0.10,
        6:0.18, 7:0.28, 8:0.40, 9:0.52, 10:0.60, 11:0.68,
        12:0.72, 13:0.70, 14:0.68, 15:0.72, 16:0.78, 17:0.82,
        18:0.90, 19:0.98, 20:1.00, 21:0.95, 22:0.78, 23:0.52
    },
}

CHANNEL_PROFILE_MAP = {
    # channel_profile: {channel: share}
    "mixed":     {"ecom":0.32, "pos":0.50, "contactless":0.18},
    "cp_heavy":  {"ecom":0.08, "pos":0.72, "contactless":0.20},
    "cnp_heavy": {"ecom":0.72, "pos":0.22, "contactless":0.06},
    "cnp_only":  {"ecom":0.88, "pos":0.10, "contactless":0.02},
}

AUTH_TYPE_MAP = {
    # channel: {country_type: {auth_type: share}}
    "ecom": {
        "home":   {"3DS":0.68, "non-3DS":0.32},  # EU SCA compliance
        "abroad": {"3DS":0.38, "non-3DS":0.62},  # non-EU merchants
    },
    "pos":  {
        "home":   {"3DS":0.00, "non-3DS":1.00},
        "abroad": {"3DS":0.00, "non-3DS":1.00},
    },
    "contactless": {
        "home":   {"3DS":0.00, "non-3DS":1.00},
        "abroad": {"3DS":0.00, "non-3DS":1.00},
    },
}

CARD_PRESENT_MAP = {
    "ecom":        False,
    "pos":         True,
    "contactless": True,
}


# ─────────────────────────────────────────────
# ANOMALY DEFINITIONS
# ─────────────────────────────────────────────

def anomaly_windows():
    """Return dict of anomaly_id → (start_dt, end_dt) for easy lookup."""
    d = START_DATE
    return {
        "A1_processor_outage": (
            d + timedelta(days=22, hours=10),
            d + timedelta(days=22, hours=18)
        ),
        "A2_3ds_cascade": (
            d + timedelta(days=40, hours=8),
            d + timedelta(days=40, hours=22)
        ),
        "A3_fraud_attack": (
            d + timedelta(days=54),
            d + timedelta(days=57)
        ),
        "A4_erosion": (
            d + timedelta(days=67),
            d + timedelta(days=73)
        ),
        # A5 is structural: every weekend from day 60 onward
        "A5_weekend_contactless": (
            d + timedelta(days=60),
            END_DATE
        ),
    }


# ─────────────────────────────────────────────
# CORE GENERATION LOGIC
# ─────────────────────────────────────────────

def base_volume(hour_dt, country, mcc_group, channel):
    """Compute expected txn_count for a given slice before anomaly injection."""
    c_tz_offset, c_home, c_weight = COUNTRIES[country]
    (m_vol, m_fraud, m_appr, m_ticket,
     m_ch_profile, m_time_profile) = MCC_GROUPS[mcc_group]

    local_hour = (hour_dt.hour + c_tz_offset) % 24
    dow        = hour_dt.weekday()

    hour_f = HOUR_PROFILES[m_time_profile][local_hour]
    dow_f  = DOW_FACTOR[dow]

    # Month-end boost for utilities + fuel
    dom = hour_dt.day
    monthend_f = 1.0
    if mcc_group in ("utilities", "fuel") and dom >= 28:
        monthend_f = 1.15
    elif mcc_group == "grocery" and 14 <= dom <= 16:
        monthend_f = 0.94   # mid-month grocery dip

    ch_shares = CHANNEL_PROFILE_MAP[m_ch_profile]
    ch_share  = ch_shares.get(channel, 0.01)

    raw = 180 * c_weight * m_vol * ch_share * hour_f * dow_f * monthend_f

    # Add Gaussian noise (σ = 8% of signal)
    noise = rng.normal(1.0, 0.08)
    return max(0, raw * noise)


def compute_approval_rate(channel, auth_type, country, mcc_group):
    """Baseline approval rate for a slice."""
    is_home = COUNTRIES[country][1]
    base    = MCC_GROUPS[mcc_group][2]

    # Channel penalty
    if channel == "ecom" and auth_type == "non-3DS":
        base -= 0.018
    elif channel == "ecom" and not is_home:
        base -= 0.024   # cross-border e-com

    return float(np.clip(base + rng.normal(0, 0.005), 0.70, 0.995))


def sample_decline_reason_codes(mcc_group, n_declines, custom_dist=None):
    """
    Draw decline reason codes for n_declines transactions.
    Returns a dict {code: count}.
    """
    if n_declines == 0:
        return {c: 0 for c in DECLINE_REASON_CODES}

    dist = custom_dist if custom_dist else MCC_DECLINE_DIST[mcc_group]
    codes  = list(dist.keys())
    probs  = np.array([dist[c] for c in codes])
    probs /= probs.sum()

    draws  = rng.multinomial(n_declines, probs)
    return dict(zip(codes, draws))


# ─────────────────────────────────────────────
# ROW BUILDER
# ─────────────────────────────────────────────

def build_row(hour_dt, country, mcc_group, channel,
              auth_type, card_present, bin_bucket,
              anomaly_flags):
    """
    Build one aggregate row for a given dimension combination and hour.
    anomaly_flags: dict of active anomaly effects to apply.
    """
    is_home    = COUNTRIES[country][1]
    corridor   = "domestic" if is_home else "cross_border"
    mcc_data   = MCC_GROUPS[mcc_group]
    base_fraud = mcc_data[1]

    # ── Base volume ──────────────────────────
    txn_count = int(round(base_volume(hour_dt, country, mcc_group, channel)))
    if txn_count == 0:
        return None

    # ── Base approval rate ───────────────────
    approval_rate = compute_approval_rate(channel, auth_type, country, mcc_group)

    # ── Fraud rate ───────────────────────────
    fraud_rate = base_fraud * (1 + rng.normal(0, 0.15))
    fraud_rate = max(0.0001, fraud_rate)

    # ── Decline reason distribution ──────────
    decline_dist = {k: v for k, v in MCC_DECLINE_DIST[mcc_group].items()}

    # ── Ticket size ──────────────────────────
    avg_ticket = mcc_data[3] * (1 + rng.normal(0, 0.12))

    # ═══════════════════════════════════════════
    # ANOMALY INJECTIONS
    # ═══════════════════════════════════════════

    # ─ A1: Processor BIN outage ─────────────
    if anomaly_flags.get("A1_processor_outage") and bin_bucket == "4531xx":
        # Approval rate collapses; system error dominates declines
        approval_rate = rng.uniform(0.08, 0.18)
        # Retry storm: volume spikes 35–45%
        txn_count = int(txn_count * rng.uniform(1.35, 1.45))
        decline_dist = {
            "05":0.03,"14":0.02,"51":0.03,"57":0.02,
            "59":0.01,"61":0.01,"65":0.02,"91":0.04,"96":0.82
        }

    # ─ A2: 3DS ACS cascade failure ──────────
    if (anomaly_flags.get("A2_3ds_cascade")
            and country in ("DE", "NL")
            and channel == "ecom"
            and auth_type == "3DS"):
        approval_rate = rng.uniform(0.52, 0.64)
        decline_dist = {
            "05":0.06,"14":0.03,"51":0.08,"57":0.04,
            "59":0.04,"61":0.02,"65":0.58,"91":0.10,"96":0.05
        }

    # ─ A3: Fraud attack — GB grocery CNP ────
    if (anomaly_flags.get("A3_fraud_attack")
            and country == "GB"
            and mcc_group == "grocery"
            and channel == "ecom"
            and not card_present):
        # Fraud rate spikes; approval rate initially clean then degrades
        day_offset = (hour_dt - (START_DATE + timedelta(days=54))).days
        fraud_rate = base_fraud * rng.uniform(7.5, 10.2)
        avg_ticket  = rng.uniform(9, 16)      # card-testing micro-amounts
        if day_offset >= 2:                   # rules fire on day 3
            approval_rate *= rng.uniform(0.78, 0.88)
            decline_dist["59"] = min(0.35, decline_dist["59"] * 5)
            # renormalize
            total = sum(decline_dist.values())
            decline_dist = {k: v/total for k, v in decline_dist.items()}

    # ─ A4: Silent cross-border erosion ──────
    if (anomaly_flags.get("A4_erosion")
            and corridor == "cross_border"
            and channel == "ecom"):
        elapsed_days = (hour_dt - (START_DATE + timedelta(days=67))).days
        # Linear drift: -0.7pp per day over 6 days
        drift = min(elapsed_days, 6) * 0.007
        approval_rate = max(0.75, approval_rate - drift)
        # Reason code 91 rises gradually
        rc91_boost = min(elapsed_days, 6) * 0.012
        decline_dist["91"] = min(0.22, decline_dist["91"] + rc91_boost)
        total = sum(decline_dist.values())
        decline_dist = {k: v/total for k, v in decline_dist.items()}

    # ─ A5: Weekend contactless velocity rule ─
    if (anomaly_flags.get("A5_weekend_contactless")
            and channel == "contactless"
            and country in ("GB", "FR")
            and card_present
            and hour_dt.weekday() in (5, 6)):   # Sat/Sun
        approval_rate = max(0.82, approval_rate - rng.uniform(0.04, 0.07))
        decline_dist["61"] = min(0.28, decline_dist["61"] * 4.5)
        decline_dist["51"] = min(0.35, decline_dist["51"] * 1.4)
        total = sum(decline_dist.values())
        decline_dist = {k: v/total for k, v in decline_dist.items()}

    # ═══════════════════════════════════════════
    # COMPUTE FINAL COUNTS
    # ═══════════════════════════════════════════

    approval_rate = float(np.clip(approval_rate, 0.05, 0.999))
    approved_count  = int(round(txn_count * approval_rate))
    declined_count  = txn_count - approved_count

    rc_counts = sample_decline_reason_codes(mcc_group, declined_count,
                                            custom_dist=decline_dist)

    fraud_count = int(np.clip(
        rng.poisson(fraud_rate * txn_count),
        0, int(txn_count * 0.15)
    ))

    txn_amount_usd = round(avg_ticket * txn_count * rng.uniform(0.95, 1.05), 2)

    # ── Assemble row ─────────────────────────
    row = {
        "timestamp":         hour_dt,
        "country":           country,
        "corridor":          corridor,
        "mcc_group":         mcc_group,
        "channel":           channel,
        "auth_type":         auth_type,
        "card_present":      card_present,
        "bin_bucket":        bin_bucket,
        "txn_count":         txn_count,
        "approved_count":    approved_count,
        "declined_count":    declined_count,
        "approval_rate":     round(approval_rate, 4),
        "fraud_count":       fraud_count,
        "txn_amount_usd":    txn_amount_usd,
        "avg_ticket_usd":    round(avg_ticket, 2),
        **{f"rc_{c}": rc_counts.get(c, 0) for c in DECLINE_REASON_CODES},
    }
    return row


# ─────────────────────────────────────────────
# DIMENSION EXPANSION
# ─────────────────────────────────────────────

def get_auth_type(channel, country):
    """Sample auth type for a channel × country combination."""
    is_home  = COUNTRIES[country][1]
    ctype    = "home" if is_home else "abroad"
    dist     = AUTH_TYPE_MAP[channel][ctype]
    choices  = list(dist.keys())
    probs    = [dist[k] for k in choices]
    idx      = rng.choice(len(choices), p=probs)
    return choices[idx]


def get_bin_bucket(country):
    """Assign bin bucket; 4531xx is the 'problem' BIN for anomaly A1."""
    # 4531xx is slightly more common in GB/DE (domestic issuer BINs)
    if country in ("GB", "DE"):
        probs = [0.22, 0.30, 0.20, 0.15, 0.13]
    else:
        probs = [0.22, 0.16, 0.24, 0.20, 0.18]
    idx = rng.choice(len(BIN_BUCKETS), p=probs)
    return BIN_BUCKETS[idx]


# ─────────────────────────────────────────────
# MAIN GENERATION LOOP
# ─────────────────────────────────────────────

def generate_dataset(output_path: str = "data/raw_transactions.csv"):
    """
    Generate the full 90-day hourly synthetic dataset and write to parquet.
    Returns the DataFrame.
    """
    print("=" * 60)
    print("Synthetic Transaction Dataset Generator")
    print("=" * 60)

    hours = pd.date_range(START_DATE, END_DATE, freq=FREQ)
    windows = anomaly_windows()

    rows = []
    total_hours = len(hours)

    for i, hour_dt in enumerate(hours):
        if i % 500 == 0:
            print(f"  Progress: {i}/{total_hours} hours ({100*i//total_hours}%)")

        # Determine which anomalies are active this hour
        flags = {}
        for anom_id, (a_start, a_end) in windows.items():
            flags[anom_id] = a_start <= hour_dt < a_end

        for country in COUNTRIES:
            for mcc_group, mcc_data in MCC_GROUPS.items():
                ch_profile = mcc_data[4]
                ch_shares  = CHANNEL_PROFILE_MAP[ch_profile]

                for channel, ch_weight in ch_shares.items():
                    if ch_weight < 0.02:
                        continue  # skip negligible channel/mcc combos

                    auth_type   = get_auth_type(channel, country)
                    card_present = CARD_PRESENT_MAP[channel]
                    bin_bucket  = get_bin_bucket(country)

                    row = build_row(
                        hour_dt, country, mcc_group, channel,
                        auth_type, card_present, bin_bucket, flags
                    )
                    if row:
                        rows.append(row)

    print(f"\nBuilding DataFrame from {len(rows):,} rows...")
    df = pd.DataFrame(rows)

    # ── Post-processing ──────────────────────
    df["timestamp"]   = pd.to_datetime(df["timestamp"])
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek   # 0=Mon
    df["day_of_week_name"] = df["timestamp"].dt.day_name()
    df["date"]        = df["timestamp"].dt.date
    df["week"]        = df["timestamp"].dt.isocalendar().week.astype(int)
    df["is_weekend"]  = df["day_of_week"].isin([5, 6])

    # Computed approval rate (ground truth from counts)
    df["approval_rate_computed"] = (
        df["approved_count"] / df["txn_count"].clip(lower=1)
    ).round(4)

    df["fraud_rate"] = (
        df["fraud_count"] / df["txn_count"].clip(lower=1)
    ).round(6)

    df["decline_rate"] = (
        df["declined_count"] / df["txn_count"].clip(lower=1)
    ).round(4)

    # Dominant decline reason (most frequent reason code per row)
    rc_cols = [f"rc_{c}" for c in DECLINE_REASON_CODES]
    df["dominant_decline_rc"] = df[rc_cols].idxmax(axis=1).str.replace("rc_", "")
    df.loc[df["declined_count"] == 0, "dominant_decline_rc"] = None

    # ── Sort and write ───────────────────────
    df = df.sort_values(["timestamp","country","mcc_group","channel"]).reset_index(drop=True)

    import os
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n✓ Dataset written → {output_path}")
    print(f"  Rows:          {len(df):,}")
    print(f"  Date range:    {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Total txns:    {df['txn_count'].sum():,.0f}")
    print(f"  Total approved:{df['approved_count'].sum():,.0f}")
    print(f"  Total fraud:   {df['fraud_count'].sum():,.0f}")
    print(f"  Avg appr rate: {df['approval_rate_computed'].mean():.3f}")
    print(f"  Countries:     {sorted(df['country'].unique())}")
    print(f"  MCCs:          {sorted(df['mcc_group'].unique())}")
    print(f"  Channels:      {sorted(df['channel'].unique())}")
    print()

    # ── Anomaly summary ───────────────────────
    print("Anomaly injection summary:")
    print("─" * 50)
    _print_anomaly_summary(df)

    return df


def _print_anomaly_summary(df):
    windows = anomaly_windows()

    # A1 — BIN outage
    a1s, a1e = windows["A1_processor_outage"]
    a1 = df[(df["timestamp"] >= a1s) & (df["timestamp"] < a1e) & (df["bin_bucket"] == "4531xx")]
    if len(a1):
        print(f"\nA1 Processor outage (4531xx BIN, {a1s.strftime('%b %d %H:%M')}–{a1e.strftime('%H:%M')})")
        print(f"   Avg approval rate: {a1['approval_rate_computed'].mean():.3f}  (baseline ~0.94)")
        print(f"   rc_96 share:       {a1['rc_96'].sum() / a1['declined_count'].sum():.3f}")
        print(f"   Rows affected:     {len(a1)}")

    # A2 — 3DS cascade
    a2s, a2e = windows["A2_3ds_cascade"]
    a2 = df[(df["timestamp"] >= a2s) & (df["timestamp"] < a2e)
            & (df["country"].isin(["DE","NL"])) & (df["channel"]=="ecom") & (df["auth_type"]=="3DS")]
    if len(a2):
        print(f"\nA2 3DS cascade (DE/NL ecom 3DS, {a2s.strftime('%b %d %H:%M')}–{a2e.strftime('%H:%M')})")
        print(f"   Avg approval rate: {a2['approval_rate_computed'].mean():.3f}  (baseline ~0.91)")
        print(f"   rc_65 share:       {a2['rc_65'].sum() / a2['declined_count'].sum():.3f}")
        print(f"   Rows affected:     {len(a2)}")

    # A3 — Fraud attack
    a3s, a3e = windows["A3_fraud_attack"]
    a3 = df[(df["timestamp"] >= a3s) & (df["timestamp"] < a3e)
            & (df["country"]=="GB") & (df["mcc_group"]=="grocery")
            & (df["channel"]=="ecom") & (~df["card_present"])]
    if len(a3):
        print(f"\nA3 Fraud attack (GB grocery ecom, {a3s.strftime('%b %d')}–{a3e.strftime('%b %d')})")
        print(f"   Avg fraud rate:    {a3['fraud_rate'].mean():.5f}  (baseline ~0.00080)")
        print(f"   Avg ticket:        £{a3['avg_ticket_usd'].mean():.2f}   (baseline ~£42)")
        print(f"   Rows affected:     {len(a3)}")

    # A4 — Silent erosion
    a4s, a4e = windows["A4_erosion"]
    a4 = df[(df["timestamp"] >= a4s) & (df["timestamp"] < a4e)
            & (df["corridor"]=="cross_border") & (df["channel"]=="ecom")]
    if len(a4):
        print(f"\nA4 Cross-border erosion ({a4s.strftime('%b %d')}–{a4e.strftime('%b %d')})")
        print(f"   Avg approval rate: {a4['approval_rate_computed'].mean():.3f}  (baseline ~0.86)")
        print(f"   rc_91 share:       {a4['rc_91'].sum() / a4['declined_count'].sum():.3f}")
        print(f"   Rows affected:     {len(a4)}")

    # A5 — Weekend contactless
    a5s, a5e = windows["A5_weekend_contactless"]
    a5 = df[(df["timestamp"] >= a5s) & (df["timestamp"] < a5e)
            & (df["channel"]=="contactless") & (df["country"].isin(["GB","FR"]))
            & (df["is_weekend"])]
    if len(a5):
        print(f"\nA5 Weekend contactless rule ({a5s.strftime('%b %d')} onward, GB/FR weekends)")
        print(f"   Avg approval rate: {a5['approval_rate_computed'].mean():.3f}  (baseline ~0.95)")
        rc61_share = a5['rc_61'].sum() / a5['declined_count'].sum() if a5['declined_count'].sum() > 0 else 0
        print(f"   rc_61 share:       {rc61_share:.3f}")
        print(f"   Rows affected:     {len(a5)}")

    print()


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "data/raw_transactions.parquet"
    df = generate_dataset(output_path=out)
