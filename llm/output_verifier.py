"""
Output Verifier — v2
====================
Verifies LLM responses against the supporting_stats allowlist.

v2 improvements over v1:
  1. VALUE-RANGE MATCHING: for every cited statistic, extracts the numeric
     value stated by the LLM and checks it falls within ±25% of the
     authoritative value in supporting_stats.  Catches hallucinated
     numbers that cite the correct key but state a wrong figure.

  2. IMPROVED NUMBER EXTRACTION: handles £, %, σ, x (multiples), K/M/B
     suffixes, and comma-separated integers.

  3. GRANULAR RISK TIERS:
     none   — all citations verified, all values match
     low    — some numbers appear without citation (common LLM behaviour)
     medium — a cited value is off by >25% (suspicious)
     high   — a cited value is off by >10x, or multiple value mismatches

Verification algorithm
----------------------
Step 1: find all [key] citation markers in the response text.
Step 2: for each cited key, extract the numeric value the LLM stated
        immediately before the citation marker.
Step 3: compare that value to supporting_stats[key] — flag if >25% apart.
Step 4: find uncited numbers and check if they appear anywhere in
        supporting_stats values (leniency pass).
Step 5: aggregate into a risk tier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    hallucination_risk:  str          # "none" | "low" | "medium" | "high"
    warnings:            list[str]    = field(default_factory=list)
    value_mismatches:    list[dict]   = field(default_factory=list)
    uncited_numbers:     list[str]    = field(default_factory=list)
    cited_keys:          list[str]    = field(default_factory=list)
    coverage_pct:        float        = 0.0   # % of cited keys that matched a stat

    def to_sentinel(self) -> str:
        """Serialise to the <!--VERIFY:{json}--> sentinel string."""
        import json
        meta = {
            "risk":     self.hallucination_risk,
            "warnings": self.warnings[:3],
        }
        if self.value_mismatches:
            meta["value_mismatches"] = self.value_mismatches[:2]
        return f"\n\n<!--VERIFY:{json.dumps(meta)}-->"
    def summary(self) -> str:
    return (
        f"risk={self.hallucination_risk}, "
        f"warnings={len(self.warnings)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# NUMERIC EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

# Patterns ordered from most-specific to least-specific
_NUM_PATTERNS = [
    r'£[\d,]+(?:\.\d+)?[KMBkmb]?',   # £3,162.68  £500K  £2.1M
    r'[\d,]+(?:\.\d+)?%',             # 56.2%  99.0%
    r'[\d,]+(?:\.\d+)?[xX]',          # 6.5x  3x
    r'[\d.]+σ',                        # 15.0σ  -2.3σ (absolute)
    r'[\d,]+(?:\.\d+)?[KMBkmb]\b',   # 3.42M  134K (standalone)
    r'\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b',  # 3,421,968
    r'\b\d+(?:\.\d+)?\b',             # plain integers/decimals
]
_NUM_RE = re.compile('|'.join(_NUM_PATTERNS))
_CITATION_RE = re.compile(r'\[([a-zA-Z0-9_]+)\]')


def _parse_to_float(raw: str) -> Optional[float]:
    """Convert a raw number string to float. Returns None if unparsable."""
    s = raw.strip().replace(',', '')
    # Handle currency
    s = s.lstrip('£$€')
    # Handle suffixes
    multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9,
                   'k': 1e3, 'm': 1e6, 'b': 1e9}
    for suf, mult in multipliers.items():
        if s.endswith(suf):
            try:
                return float(s[:-1]) * mult
            except ValueError:
                return None
    # Strip trailing units
    for unit in ('%', 'x', 'X', 'σ'):
        s = s.rstrip(unit)
    try:
        return float(s)
    except ValueError:
        return None


def _extract_numbers(text: str) -> list[tuple[str, int]]:
    """Return list of (raw_number_str, position) tuples from text."""
    return [(m.group(), m.start()) for m in _NUM_RE.finditer(text)]


# ─────────────────────────────────────────────────────────────────────────────
# VALUE MATCHING
# ─────────────────────────────────────────────────────────────────────────────

def _classify_stat_unit(stat_key: str, stat_value: str) -> str:
    """
    Fix 5: Classify a stat as 'rate', 'currency', 'count', or 'other'
    so we can apply metric-type-aware tolerance.

    Rate (%) : ±5pp absolute tolerance  (56.2% ± 5pp = 51.2-61.2%)
    Currency : ±25% relative tolerance  (£3,162 ± 25%)
    Count    : ±20% relative tolerance
    Other    : ±25% relative tolerance (default)

    Why this matters: 56.2% ± 25% relative = 42-70%, which would pass
    a hallucinated 42% approval rate as "grounded". That's wrong for
    a risk tool where decline rates drive escalation decisions.
    """
    key_lower = stat_key.lower()
    val_str   = str(stat_value).strip()

    if val_str.endswith("%") or "rate" in key_lower or "share" in key_lower:
        return "rate"
    if val_str.startswith("£") or val_str.startswith("$") or val_str.startswith("€"):
        return "currency"
    if "count" in key_lower or "txn" in key_lower or "volume" in key_lower:
        return "count"
    return "other"


def _values_match(stated_raw: str, stat_value: str,
                  tolerance: float = 0.25,
                  stat_key: str = "") -> bool:
    """
    Fix 5: Metric-type-aware tolerance.

    - Rate metrics (%, approval_rate, fraud_rate):  ±5pp absolute
      56.2% ± 5pp = 51.2–61.2%. 42% stated vs 56.2% actual → FAIL.
    - Currency (£, $):                              ±25% relative
      £3,162 ± 25% = £2,372–£3,953. Standard financial rounding.
    - Count metrics:                                ±20% relative
    - Other:                                        ±25% relative (default)

    Falls through to True when either value is unparsable to avoid
    false positives on non-numeric stats like "High confidence".
    """
    sv = _parse_to_float(stated_raw)
    av = _parse_to_float(str(stat_value))
    if sv is None or av is None:
        return True
    if av == 0:
        return abs(sv) < 0.001

    unit = _classify_stat_unit(stat_key, stat_value)

    if unit == "rate":
        # Stated value may be in 0-100 range OR 0-1 range
        # Normalise both to 0-100 for comparison
        sv_norm = sv * 100 if sv <= 1.0 and "%" not in str(stated_raw) else sv
        av_norm = av * 100 if av <= 1.0 and "%" not in str(stat_value)  else av
        return abs(sv_norm - av_norm) <= 5.0   # ±5 percentage points absolute

    elif unit == "currency":
        return abs(sv - av) / abs(av) <= 0.25   # ±25% relative

    elif unit == "count":
        return abs(sv - av) / abs(av) <= 0.20   # ±20% relative

    else:
        return abs(sv - av) / abs(av) <= tolerance   # default ±25%


def _find_number_before_citation(text: str, citation_pos: int,
                                  lookback: int = 120) -> Optional[str]:
    """
    Find the last number that appears within `lookback` characters
    before a citation marker — this is the value the LLM stated for that key.
    """
    window = text[max(0, citation_pos - lookback): citation_pos]
    nums = _extract_numbers(window)
    if not nums:
        return None
    # Return the last (closest to citation) number found
    return nums[-1][0]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN VERIFIER
# ─────────────────────────────────────────────────────────────────────────────

class OutputVerifier:
    """
    Verifies LLM responses against the supporting_stats dictionary.

    Usage:
        verifier = OutputVerifier()
        result   = verifier.verify(llm_response_text, supporting_stats)
        sentinel = result.to_sentinel()   # appended to streaming response

    The verifier is intentionally lenient on non-financial text to avoid
    false positives on narrative language.  It is strict on cited numeric
    values because wrong financial figures are the primary risk.
    """

    def __init__(self, value_tolerance: float = 0.25,
                 high_risk_tolerance: float = 9.0) -> None:
        self.value_tolerance      = value_tolerance
        self.high_risk_tolerance  = high_risk_tolerance

    def verify(
        self,
        response_text:   str,
        supporting_stats: dict,
    ) -> VerificationResult:
        """
        Run all verification checks and return a VerificationResult.

        Args:
            response_text:    The LLM's full response (before sentinel stripping).
            supporting_stats: Dict of {key: value} from ContextBuilder.
        """
        warnings:         list[str] = []
        value_mismatches: list[dict] = []
        cited_keys:       list[str]  = []

        # ── Step 1: Find all [key] citations ─────────────────────────────────
        for m in _CITATION_RE.finditer(response_text):
            key = m.group(1)
            if key not in supporting_stats:
                warnings.append(
                    f"Citation [{key}] not found in supporting stats"
                )
                continue

            cited_keys.append(key)
            stat_value = supporting_stats[key]

            # ── Step 2: Extract the number stated for this key ────────────────
            stated_raw = _find_number_before_citation(
                response_text, m.start()
            )
            if stated_raw is None:
                # Citation exists but no number precedes it — common for
                # categorical stats like failure_class. Don't penalise.
                continue

            # ── Step 3: Value range check ─────────────────────────────────────
            if not _values_match(stated_raw, str(stat_value),
                                 self.value_tolerance, stat_key=key):
                stated_f = _parse_to_float(stated_raw)
                actual_f = _parse_to_float(str(stat_value))
                ratio = (abs(stated_f - actual_f) / max(abs(actual_f), 0.001)
                         if stated_f is not None and actual_f is not None
                         else None)

                mismatch = {
                    "key":    key,
                    "stated": stated_raw,
                    "actual": str(stat_value),
                    "off_by": f"{ratio*100:.0f}%" if ratio else "unknown",
                }
                value_mismatches.append(mismatch)
                warnings.append(
                    f"Value mismatch [{key}]: stated {stated_raw!r}, "
                    f"actual {stat_value!r} "
                    f"(off by {mismatch['off_by']})"
                )

        # ── Step 4: Find uncited numbers ──────────────────────────────────────
        all_nums_in_text = _extract_numbers(response_text)
        # Remove numbers that are part of citations already verified
        citation_positions = {m.start() for m in _CITATION_RE.finditer(response_text)}

        # Build a flat set of all numeric values in supporting_stats
        stat_values_flat: set[str] = set()
        for v in supporting_stats.values():
            stat_values_flat.add(str(v).lower())
            # Also add the raw numeric form for fuzzy matching
            fv = _parse_to_float(str(v))
            if fv is not None:
                stat_values_flat.add(str(round(fv, 2)))

        uncited_suspicious: list[str] = []
        for num_str, pos in all_nums_in_text:
            # Check it's not part of a citation (within 5 chars of a [key])
            near_citation = any(abs(pos - cp) < len(num_str) + 50
                                for cp in citation_positions)
            if near_citation:
                continue

            # Is this number in the stat allowlist? If so, fine.
            fv = _parse_to_float(num_str)
            in_stats = any(
                fv is not None and
                _parse_to_float(str(sv)) is not None and
                abs(fv - _parse_to_float(str(sv))) / max(abs(_parse_to_float(str(sv))), 0.001) < 0.01
                for sv in supporting_stats.values()
                if _parse_to_float(str(sv)) is not None
            )
            if not in_stats and fv is not None and abs(fv) > 1:
                uncited_suspicious.append(num_str)

        if uncited_suspicious:
            n = len(uncited_suspicious)
            warnings.append(
                f"{n} number(s) in the response could not be traced to "
                f"supporting_stats: {uncited_suspicious[:3]}"
            )

        # ── Step 5: Coverage percentage ───────────────────────────────────────
        numeric_stats = sum(
            1 for v in supporting_stats.values()
            if _parse_to_float(str(v)) is not None
        )
        coverage_pct = (
            len(cited_keys) / max(numeric_stats, 1) * 100
        )

        # ── Step 6: Determine risk tier ───────────────────────────────────────
        # High risk: any single value is off by >high_risk_tolerance (9x)
        high_risk_mismatches = [
            mm for mm in value_mismatches
            if _parse_to_float(mm.get("off_by","0%").rstrip('%') or "0") is not None
            and (_parse_to_float(mm.get("off_by","0%").rstrip('%') or "0") or 0) > self.high_risk_tolerance * 100
        ]

        if value_mismatches and high_risk_mismatches:
            risk = "high"
        elif value_mismatches:
            risk = "medium"
        elif uncited_suspicious:
            risk = "low"
        else:
            risk = "none"

        return VerificationResult(
            hallucination_risk = risk,
            warnings           = warnings,
            value_mismatches   = value_mismatches,
            uncited_numbers    = uncited_suspicious,
            cited_keys         = cited_keys,
            coverage_pct       = coverage_pct,
        )
