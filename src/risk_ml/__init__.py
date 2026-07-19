"""Standalone learned-risk runtime.

This package is intentionally separate from ``src.fusion`` warning rules.  It
may reuse sensor data types and hardware readers, but it must not import or
modify the deterministic competition rule implementations.
"""

from .predictor import RiskPrediction, XGBoostRiskPredictor

__all__ = ["RiskPrediction", "XGBoostRiskPredictor"]
