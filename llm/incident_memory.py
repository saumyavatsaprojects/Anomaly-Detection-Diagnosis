"""
Incident Memory — TF-IDF similarity retrieval
===============================================
Fix 3: Real retrieval-augmented generation using TF-IDF cosine similarity
on historical anomaly briefs. No vector DB required — runs in-process.

Architecture
------------
On first use, IncidentMemory:
  1. Loads all anomaly objects from data/anomaly_objects.json
  2. Renders each as a compact text document
  3. Fits a TfidfVectorizer (no API calls, no GPU)
  4. For each new anomaly, retrieves top-k most similar past incidents

What makes this real RAG vs context stuffing
--------------------------------------------
- RETRIEVAL: only 2 most similar incidents are injected (not all 20)
- AUGMENTED: retrieved examples ground ARIA's response in real precedent
- GENERATION: ARIA can cite "similar incident on Mar 23" with actual data

Design choice: TF-IDF over dense embeddings
--------------------------------------------
TF-IDF is more transparent and auditable for a financial tool.
An analyst can understand "similar because both had RC 96 spikes on ecom"
without needing to explain cosine distance in embedding space.
Adding sentence_transformers later is a drop-in upgrade.

Threat: retrieval from same dataset (training/test contamination)
-----------------------------------------------------------------
The retrieved incidents come from the same 90-day run. In production
this would be a rolling historical store of real past incidents.
For the POC, we exclude the exact same anomaly_id to avoid self-retrieval.
"""

from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_MEMORY_INSTANCE: Optional["IncidentMemory"] = None


@dataclass
class SimilarIncident:
    anomaly_id:   str
    failure_class: str
    first_seen_ts: str
    similarity:    float   # 0–1 cosine similarity
    summary:       str     # compact 2-line summary for injection
    resolution:    str     # what was done / what was ruled out


def _build_doc(anomaly: dict) -> str:
    """Build a searchable text document from an anomaly object."""
    sl    = anomaly.get("affected_slice", {})
    fc    = anomaly.get("failure_class", "undetermined")
    sev   = anomaly.get("severity", "unknown")
    ts    = anomaly.get("first_seen_ts", "")[:10]
    obs   = anomaly.get("observed_value", 0)
    base  = anomaly.get("baseline_value", 0)
    sigma = anomaly.get("deviation_sigma", 0)
    dur   = anomaly.get("duration_hours", 0)
    rc_ev = anomaly.get("reason_code_evidence", {})

    # Key signals as text
    rc_codes = " ".join(f"RC{code}" for code in rc_ev.keys())
    rc_spikes = " ".join(
        f"RC{code}_plus{abs(d.get('delta_pp',0)):.0f}pp"
        for code, d in rc_ev.items()
        if d.get("delta_pp", 0) > 5
    )
    rc_drops = " ".join(
        f"RC{code}_minus{abs(d.get('delta_pp',0)):.0f}pp"
        for code, d in rc_ev.items()
        if d.get("delta_pp", 0) < -5
    )

    fe   = anomaly.get("fraud_evidence", {})
    mult = fe.get("fraud_rate_multiple", 0)
    mcc  = str(sl.get("mcc_group", ""))
    country = str(sl.get("country", ""))
    channel = str(sl.get("channel", ""))
    auth    = str(sl.get("auth_type", ""))
    corridor = str(sl.get("corridor", ""))

    return (
        f"{fc} {sev} {ts} {mcc} {country} {channel} {auth} {corridor} "
        f"obs{obs:.3f} base{base:.3f} sigma{sigma:.1f} dur{dur}h "
        f"{rc_codes} {rc_spikes} {rc_drops} "
        f"fraud_mult{mult:.1f} "
        f"{' '.join(anomaly.get('evidence', [])[:3])}"
    )


def _build_summary(anomaly: dict) -> str:
    """One-line summary for injection into ARIA context."""
    sl   = anomaly.get("affected_slice", {})
    fc   = anomaly.get("failure_class", "undetermined")
    ts   = anomaly.get("first_seen_ts", "")[:10]
    obs  = anomaly.get("observed_value", 0)
    base = anomaly.get("baseline_value", 0)
    dur  = anomaly.get("duration_hours", 0)
    conf = anomaly.get("failure_class_confidence", "")
    mcc  = sl.get("mcc_group", "")
    country = sl.get("country", "")
    delta_pp = (obs - base) * 100 if base > 0 else 0
    return (
        f"On {ts}: {fc} ({conf} confidence) — "
        f"{country}/{mcc}, approval rate {obs:.1%} vs {base:.1%} "
        f"({delta_pp:+.1f}pp), lasted {dur}h"
    )


def _build_resolution(anomaly: dict) -> str:
    """What was found and done — for ARIA to learn from."""
    esc    = anomaly.get("recommended_escalation", "")
    ruled  = anomaly.get("ruled_out", [])
    evid   = anomaly.get("evidence", [])[:2]
    ruled_str = "; ".join(ruled[:2]) if ruled else "—"
    evid_str  = "; ".join(evid) if evid else "—"
    return f"Evidence: {evid_str}. Ruled out: {ruled_str}. Action: {esc[:120]}"


class IncidentMemory:
    """
    TF-IDF incident similarity store.

    Usage:
        memory = IncidentMemory.load()
        similar = memory.retrieve(current_anomaly, exclude_id="AN-123", top_k=2)
        for inc in similar:
            print(inc.summary)
    """

    def __init__(self, anomalies: list[dict]) -> None:
        self._anomalies = anomalies
        self._docs      = [_build_doc(a) for a in anomalies]
        self._vectorizer = None
        self._matrix     = None
        self._fit()

    def _fit(self) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import normalize
            self._vectorizer = TfidfVectorizer(
                max_features=500,
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
            )
            raw = self._vectorizer.fit_transform(self._docs)
            self._matrix = normalize(raw, norm="l2")
            logger.info("IncidentMemory: fitted %d docs, %d features",
                        len(self._docs), self._matrix.shape[1])
        except ImportError:
            logger.warning("IncidentMemory: sklearn not available — falling back to keyword match")
            self._vectorizer = None
            self._matrix     = None

    def retrieve(
        self,
        query_anomaly: dict,
        exclude_id:    str  = "",
        top_k:         int  = 2,
        min_similarity: float = 0.15,
    ) -> list[SimilarIncident]:
        """
        Return top_k most similar historical incidents.

        Parameters
        ----------
        query_anomaly : the current anomaly being investigated
        exclude_id    : exclude this anomaly_id (avoid self-retrieval)
        top_k         : max number of results
        min_similarity: minimum cosine similarity threshold
        """
        query_doc = _build_doc(query_anomaly)

        if self._vectorizer is not None:
            return self._sklearn_retrieve(query_doc, exclude_id, top_k, min_similarity)
        else:
            return self._keyword_retrieve(query_anomaly, exclude_id, top_k)

    def _sklearn_retrieve(
        self, query_doc: str, exclude_id: str,
        top_k: int, min_similarity: float
    ) -> list[SimilarIncident]:
        from sklearn.preprocessing import normalize
        import scipy.sparse as sp

        q_vec = normalize(self._vectorizer.transform([query_doc]), norm="l2")
        sims  = (self._matrix @ q_vec.T).toarray().flatten()

        results = []
        ranked  = np.argsort(sims)[::-1]

        for idx in ranked:
            a = self._anomalies[idx]
            if a.get("anomaly_id") == exclude_id:
                continue
            sim = float(sims[idx])
            if sim < min_similarity:
                break
            results.append(SimilarIncident(
                anomaly_id    = a.get("anomaly_id",""),
                failure_class = a.get("failure_class",""),
                first_seen_ts = a.get("first_seen_ts",""),
                similarity    = round(sim, 3),
                summary       = _build_summary(a),
                resolution    = _build_resolution(a),
            ))
            if len(results) >= top_k:
                break

        return results

    def _keyword_retrieve(
        self, query: dict, exclude_id: str, top_k: int
    ) -> list[SimilarIncident]:
        """Fallback: match by failure_class and affected slice."""
        fc  = query.get("failure_class","")
        sl  = query.get("affected_slice",{})
        mcc = sl.get("mcc_group","")
        ch  = sl.get("channel","")

        scored = []
        for a in self._anomalies:
            if a.get("anomaly_id") == exclude_id:
                continue
            score = 0
            if a.get("failure_class") == fc:
                score += 3
            asl = a.get("affected_slice",{})
            if asl.get("mcc_group") == mcc:
                score += 2
            if asl.get("channel") == ch:
                score += 1
            if score > 0:
                scored.append((score, a))

        scored.sort(key=lambda x: -x[0])
        return [
            SimilarIncident(
                anomaly_id    = a.get("anomaly_id",""),
                failure_class = a.get("failure_class",""),
                first_seen_ts = a.get("first_seen_ts",""),
                similarity    = score / 6.0,
                summary       = _build_summary(a),
                resolution    = _build_resolution(a),
            )
            for score, a in scored[:top_k]
        ]

    @classmethod
    def load(cls, path: str = "data/anomaly_objects.json") -> "IncidentMemory":
        global _MEMORY_INSTANCE
        if _MEMORY_INSTANCE is not None:
            return _MEMORY_INSTANCE
        try:
            with open(path) as f:
                anomalies = json.load(f)
            _MEMORY_INSTANCE = cls(anomalies)
            return _MEMORY_INSTANCE
        except Exception as exc:
            logger.error("IncidentMemory.load failed: %s", exc)
            return cls([])

    @classmethod
    def reset(cls) -> None:
        """Force reload on next call (e.g. after pipeline regeneration)."""
        global _MEMORY_INSTANCE
        _MEMORY_INSTANCE = None


def format_similar_incidents_for_brief(
    similar: list[SimilarIncident],
) -> str:
    """Format retrieved incidents for injection into the ARIA brief."""
    if not similar:
        return "No similar historical incidents found."
    lines = []
    for i, inc in enumerate(similar, 1):
        sim_pct = f"{inc.similarity*100:.0f}%"
        lines.append(
            f"  [{i}] {inc.summary}\n"
            f"      Similarity: {sim_pct} | ID: {inc.anomaly_id[:12]}\n"
            f"      {inc.resolution}"
        )
    return "\n\n".join(lines)
