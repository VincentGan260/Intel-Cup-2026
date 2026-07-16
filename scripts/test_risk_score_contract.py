"""Checks for the shared modality level-to-score contract."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fusion.risk_score_contract import normalize_risk_score


def main() -> None:
    for level, score in ((0, 0.20), (1, 0.50), (2, 0.90)):
        result = normalize_risk_score(level, score)
        assert result.score == score
        assert result.status == "valid"

    low = normalize_risk_score(0, 0.90)
    assert low.score is not None and 0.0 <= low.score < 0.35
    assert low.status == "clamped_to_level"

    medium_low = normalize_risk_score(1, 0.10)
    medium_high = normalize_risk_score(1, 0.90)
    assert medium_low.score == 0.35
    assert medium_high.score is not None and 0.35 <= medium_high.score < 0.70

    high = normalize_risk_score(2, 0.40)
    assert high.score == 0.70
    assert high.status == "clamped_to_level"

    assert normalize_risk_score(1, None).score == 0.35
    assert normalize_risk_score(2, None).score == 0.70
    assert normalize_risk_score(2, "bad").status == "invalid_fallback"
    assert normalize_risk_score(2, math.nan).status == "nonfinite_fallback"
    assert normalize_risk_score(None, 0.90).score is None

    print("risk score contract: all tests passed")


if __name__ == "__main__":
    main()

