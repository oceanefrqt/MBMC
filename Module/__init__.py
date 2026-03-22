"""
MBMC — Multi-class Bivariate Monotonic Classifiers
====================================================

Public API
----------
Core classifier:
    Data, MultiClassMonotonicClassifier

Error computation:
    error_matrix_multiclass, pred_multiclass

Pair selection:
    all_configurations, preselection_multiclass

Visualization:
    visualization
"""

from .mbmc import (
    Data,
    MultiClassMonotonicClassifier,
    error_matrix_multiclass,
    pred_multiclass,
    monotonic_model_MAE_multiclass,
    monotonic_model_MAE_CVE_multiclass,
)
from .selection_top_mbmc import all_configurations, preselection_multiclass
from .visualization_MBMC import visualization

__all__ = [
    "Data",
    "MultiClassMonotonicClassifier",
    "error_matrix_multiclass",
    "pred_multiclass",
    "monotonic_model_MAE_multiclass",
    "monotonic_model_MAE_CVE_multiclass",
    "all_configurations",
    "preselection_multiclass",
    "visualization",
]
