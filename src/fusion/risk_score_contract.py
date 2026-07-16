"""Shared level-to-score contract for all warning modalities."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


ATTENTION_SCORE = 0.35
HIGH_SCORE = 0.70


@dataclass(frozen=True)
class NormalizedRiskScore:
    score: Optional[float]
    status: str


_FALLBACK_SCORES = {
    0: 0.0,
    1: ATTENTION_SCORE,
    2: HIGH_SCORE,
}


def normalize_risk_score(level: Optional[int], score) -> NormalizedRiskScore:
    """Keep the discrete safety level authoritative and constrain its score band."""
    if level not in (0, 1, 2):
        return NormalizedRiskScore(None, "unavailable")

    fallback = _FALLBACK_SCORES[level]
    if score is None:
        return NormalizedRiskScore(fallback, "missing_fallback")
    try:
        value = float(score)
    except (TypeError, ValueError):
        return NormalizedRiskScore(fallback, "invalid_fallback")
    if not math.isfinite(value):
        return NormalizedRiskScore(fallback, "nonfinite_fallback")

    lower = (0.0 if level == 0 else
             ATTENTION_SCORE if level == 1 else HIGH_SCORE)
    upper = (math.nextafter(ATTENTION_SCORE, 0.0) if level == 0 else
             math.nextafter(HIGH_SCORE, 0.0) if level == 1 else 1.0)
    normalized = max(lower, min(upper, value))
    status = "valid" if normalized == value else "clamped_to_level"
    return NormalizedRiskScore(normalized, status)

