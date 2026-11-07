"""Phase 4b baseline models.

Each builder returns an *unfitted* :class:`~sklearn.pipeline.Pipeline` of
``preprocess`` (fitted on training data only) followed by ``model``. All
settings are fixed here; no hyperparameter search is performed.
"""

from __future__ import annotations

from typing import Callable, Sequence

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.preprocessing import build_preprocessor

MAJORITY_BASELINE = "majority_class"
LOGISTIC_REGRESSION = "logistic_regression_balanced"

#: Fixed, untuned Logistic Regression settings (scikit-learn defaults except
#: ``class_weight`` and an increased ``max_iter`` to allow convergence).
LOGISTIC_REGRESSION_PARAMS = {
    "class_weight": "balanced",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
}


def build_majority_baseline(feature_columns: Sequence[str], random_state: int) -> Pipeline:
    """Majority-class baseline.

    ``strategy="prior"`` always predicts the most frequent training class and
    returns the training class prior as its probability for every row, so the
    scores carry no ranking information (ROC-AUC 0.5, average precision equal
    to the evaluation prevalence). ``random_state`` is accepted for a uniform
    builder interface; the model is deterministic.
    """
    del random_state
    return Pipeline(
        [
            ("preprocess", build_preprocessor(feature_columns)),
            ("model", DummyClassifier(strategy="prior")),
        ]
    )


def build_logistic_regression(feature_columns: Sequence[str], random_state: int) -> Pipeline:
    """Standardized, class-weighted (``class_weight="balanced"``) Logistic Regression."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor(feature_columns)),
            ("model", LogisticRegression(random_state=random_state, **LOGISTIC_REGRESSION_PARAMS)),
        ]
    )


ModelBuilder = Callable[[Sequence[str], int], Pipeline]

BASELINE_BUILDERS: dict[str, ModelBuilder] = {
    MAJORITY_BASELINE: build_majority_baseline,
    LOGISTIC_REGRESSION: build_logistic_regression,
}
