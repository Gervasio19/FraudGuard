"""Sigmoid (Platt) calibration and its evaluation (protocol Sections 6.3-6.5; Addendum C5, C8), Phase 5B-3.

**Fitter (mentor decision D6).** :func:`fit_platt` fits ``z = A * s + B``,
``p = expit(z)`` by minimizing the cross-entropy against Platt's (2000) smoothed
targets, ``t+ = (N+ + 1) / (N+ + 2)`` and ``t- = 1 / (N- + 2)``. It uses a damped
Newton method with a backtracking line search, as in Lin, Lin and Weng (2007),
"A note on Platt's probabilistic outputs for support vector machines", on
internally standardized scores. Only NumPy and SciPy public functions are used;
no scikit-learn private API. The objective is the one scikit-learn's public
``CalibratedClassifierCV(method="sigmoid")`` minimizes (with L-BFGS-B), and it is
strictly convex whenever the scores are not constant, so both reach the same
optimum up to optimizer tolerance. The runner and the tests measure that
difference instead of assuming it.

**Technical-failure rule (Section 6.4).** Decided on ``T_cal`` alone: fit error,
non-convergence or non-finite parameters; slope ``A <= 0``; numerically constant
calibrated logits (population SD < 1e-8). A failed model keeps its uncalibrated
scores; there is no switch to another method.

**Evaluation (Section 6.5).** O/E with the exact Poisson (Garwood) interval,
calibration intercept and slope with profile-likelihood intervals, Brier score,
Brier skill against the constant prevalence, log loss (from logits), and the
reliability table with Clopper-Pearson intervals. ECE is deliberately absent.

Calibration never changes a ranking metric: the sigmoid is strictly increasing
when ``A > 0``. Calibration results are reported, never acted upon.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import optimize, stats
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

from src.evaluation.block_bootstrap import iter_replicate_weights, period_offsets
from src.models.phase5_settings import (
    CALIBRATION_CONSTANT_LOGIT_SD, CALIBRATION_MIN_VALID_SHARE, RELIABILITY_BIN_EDGES,
)
from src.models.ranking_score import (
    calibrated_logits, is_numerically_constant, ranking_order, reliability_bin_index,
)

VALID = "valid"
INVALID_NO_POSITIVES = "invalid_no_positives"
INVALID_FIT_FAILURE = "invalid_fit_failure"
INVALID_NON_POSITIVE_SLOPE = "invalid_non_positive_slope"
INVALID_DEGENERATE = "invalid_degenerate"
INVALID_CATEGORIES = (INVALID_NO_POSITIVES, INVALID_FIT_FAILURE, INVALID_NON_POSITIVE_SLOPE, INVALID_DEGENERATE)
LOW_POSITIVE_MAX = 4
#: Probability clip used only when a log loss must be computed from probabilities that
#: can be exactly 0 or 1 (the isotonic sensitivity). Sigmoid log loss uses logits.
PROBABILITY_CLIP = 1e-15
_CHI2_95 = float(stats.chi2.ppf(0.95, 1))


# ---------------------------------------------------------------------------
# Newton solver shared by the Platt fit and the calibration intercept / slope
# ---------------------------------------------------------------------------

def _loss(eta, target, weight) -> float:
    return float(weight @ (np.logaddexp(0.0, eta) - target * eta))


def newton_logistic(X, target, weight=None, offset=None, start=None, max_iter: int = 100,
                    gradient_tolerance: float = 1e-9, decrement_tolerance: float = 1e-13) -> dict:
    """Minimize ``sum w * (log(1 + exp(eta)) - t * eta)``, ``eta = X @ beta + offset``.

    ``target`` may be fractional (Platt's smoothed targets). Damped Newton with
    step halving. The iteration stops when the Newton decrement ``g' H^-1 g``
    falls below ``decrement_tolerance``: the loss is then within about half that
    amount of its minimum. ``converged`` requires that stop (or a gradient
    infinity-norm, relative to the total weight, below ``gradient_tolerance``)
    and a finite solution. Complete separation is reported as non-convergence.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    target = np.asarray(target, dtype=np.float64)
    weight = np.ones(len(target)) if weight is None else np.asarray(weight, dtype=np.float64)
    offset = np.zeros(len(target)) if offset is None else np.asarray(offset, dtype=np.float64)
    beta = np.zeros(X.shape[1]) if start is None else np.array(start, dtype=np.float64)
    total = float(weight.sum())
    scale = max(total, 1.0)
    eta = X @ beta + offset
    loss = _loss(eta, target, weight)
    gradient_norm, n_iter, stalled, at_optimum = math.inf, 0, False, False
    hessian = np.eye(X.shape[1])
    for n_iter in range(1, max_iter + 1):
        p = expit(eta)
        gradient = X.T @ (weight * (p - target))
        hessian = (X * (weight * p * (1.0 - p))[:, None]).T @ X
        gradient_norm = float(np.abs(gradient).max()) / scale
        if not np.isfinite(gradient).all() or not np.isfinite(hessian).all():
            break
        try:
            step = np.linalg.solve(hessian + 1e-12 * scale * np.eye(len(beta)), gradient)
        except np.linalg.LinAlgError:
            break
        if float(gradient @ step) <= decrement_tolerance:
            at_optimum = True
            break
        size, improved = 1.0, False
        while size >= 1e-10:
            candidate = beta - size * step
            candidate_eta = X @ candidate + offset
            candidate_loss = _loss(candidate_eta, target, weight)
            if np.isfinite(candidate_loss) and candidate_loss <= loss - 1e-4 * size * float(gradient @ step):
                beta, eta, loss, improved = candidate, candidate_eta, candidate_loss, True
                break
            size /= 2.0
        if not improved:  # floating-point floor reached: no further decrease is possible
            stalled = True
            break
    finite = bool(np.isfinite(beta).all() and np.isfinite(loss))
    # Under complete separation the loss and the gradient both vanish as beta grows without a
    # finite optimum; a near-zero loss is therefore reported as non-convergence, not as a fit.
    separated = loss <= 1e-7 * scale
    converged = bool(finite and not separated and (at_optimum or gradient_norm <= gradient_tolerance)
                     and np.abs(beta).max() < 1e6)
    return {"beta": beta, "loss": loss, "converged": converged, "n_iter": n_iter,
            "gradient_inf_norm_relative": gradient_norm, "stalled": stalled, "hessian": hessian}


# ---------------------------------------------------------------------------
# Platt fit and the technical-failure rule
# ---------------------------------------------------------------------------

def fit_platt(scores, y_true, sample_weight=None, positive_label=1) -> dict:
    """Fit the protocol sigmoid ``p = expit(A * s + B)`` (Platt targets). Never raises on a bad sample.

    Returns ``status`` = ``"ok"`` or a failure reason, with ``A``, ``B``
    (original score units), ``converged``, ``n_iter`` and the weighted class
    counts. Samples without a positive (or without a negative), with constant
    scores or with non-finite inputs are reported as failures, not fitted.
    """
    s = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(y_true) == positive_label
    w = np.ones(len(s)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if not (s.ndim == 1 and s.shape == positive.shape == w.shape):
        raise ValueError("scores, y_true and sample_weight must be one-dimensional and of equal length.")
    if (w < 0).any():
        raise ValueError("sample_weight must be non-negative.")
    n1, n0 = float(w[positive].sum()), float(w[~positive].sum())
    base = {"A": math.nan, "B": math.nan, "converged": False, "n_iter": 0,
            "n_positive": n1, "n_negative": n0}
    if not np.isfinite(s).all():
        return {**base, "status": "non-finite scores"}
    if n1 <= 0:
        return {**base, "status": "no positives"}
    if n0 <= 0:
        return {**base, "status": "no negatives"}
    total = n1 + n0
    mean = float(w @ s) / total
    sd = math.sqrt(float(w @ (s - mean) ** 2) / total)
    if not sd > 0:
        return {**base, "status": "constant scores"}
    x = (s - mean) / sd
    target = np.where(positive, (n1 + 1.0) / (n1 + 2.0), 1.0 / (n0 + 2.0))
    start = np.array([0.0, math.log((n1 + 1.0) / (n0 + 1.0))])
    fit = newton_logistic(np.column_stack([x, np.ones_like(x)]), target, w, start=start)
    a, b = fit["beta"]
    A, B = float(a / sd), float(b - a * mean / sd)
    ok = fit["converged"] and math.isfinite(A) and math.isfinite(B)
    return {**base, "A": A, "B": B, "converged": bool(fit["converged"]), "n_iter": fit["n_iter"],
            "gradient_inf_norm_relative": fit["gradient_inf_norm_relative"], "loss": fit["loss"],
            "status": "ok" if ok else "fit failure (non-convergence or non-finite parameters)"}


def weighted_logit_sd(logits, weight=None) -> float:
    z = np.asarray(logits, dtype=np.float64)
    if weight is None:
        return float(np.std(z))
    w = np.asarray(weight, dtype=np.float64)
    mean = float(w @ z) / float(w.sum())
    return math.sqrt(float(w @ (z - mean) ** 2) / float(w.sum()))


def classify_calibrator(fit: dict, scores, sample_weight=None) -> str:
    """Section 6.4 / 6.5 (A) classification of a fitted sigmoid on the rows it was fitted on."""
    if fit["status"] == "no positives":
        return INVALID_NO_POSITIVES
    if fit["status"] != "ok":
        return INVALID_FIT_FAILURE
    if fit["A"] <= 0:
        return INVALID_NON_POSITIVE_SLOPE
    z = calibrated_logits(scores, fit["A"], fit["B"])
    if sample_weight is None:
        constant = is_numerically_constant(z)
    else:
        w = np.asarray(sample_weight, dtype=np.float64)
        kept = w > 0
        constant = (not np.isfinite(z[kept]).all()
                    or weighted_logit_sd(z[kept], w[kept]) < CALIBRATION_CONSTANT_LOGIT_SD)
    return INVALID_DEGENERATE if constant else VALID


def primary_calibrator(scores, y_true, positive_label=1) -> dict:
    """Fit and check one model's C-A calibrator on its ``T_cal`` scores (labels of ``T_cal`` only)."""
    fit = fit_platt(scores, y_true, positive_label=positive_label)
    classification = classify_calibrator(fit, scores)
    out = {"method": "sigmoid (Platt)", "fit_status": fit["status"], "classification": classification,
           "calibrated": classification == VALID, "A": fit["A"], "B": fit["B"],
           "converged": fit["converged"], "n_iter": fit["n_iter"],
           "n_rows": int(len(np.asarray(scores))), "n_positive": int(fit["n_positive"])}
    if classification == VALID:
        z = calibrated_logits(scores, fit["A"], fit["B"])
        out["t_cal_logit_sd"] = float(np.std(z))
        out["t_cal_logit_range"] = [float(z.min()), float(z.max())]
    else:
        out["consequence"] = ("technical failure: this model continues with its uncalibrated scores; "
                              "probability-based outputs are 'not available (no valid calibrator)'")
    return out


def isotonic_probabilities(cal_scores, cal_y, scores, positive_label=1) -> np.ndarray:
    """Isotonic sensitivity (never the primary method): fitted on ``T_cal`` scores, applied to ``scores``."""
    model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    model.fit(np.asarray(cal_scores, dtype=np.float64), (np.asarray(cal_y) == positive_label).astype(float))
    return np.asarray(model.predict(np.asarray(scores, dtype=np.float64)), dtype=np.float64)


# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------

def log_loss_terms(positive, logits=None, probabilities=None) -> np.ndarray:
    """Per-row negative log-likelihood, from logits when available (always finite)."""
    positive = np.asarray(positive, dtype=bool)
    if logits is not None:
        z = np.asarray(logits, dtype=np.float64)
        return np.where(positive, np.logaddexp(0.0, -z), np.logaddexp(0.0, z))
    p = np.clip(np.asarray(probabilities, dtype=np.float64), PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    return np.where(positive, -np.log(p), -np.log1p(-p))


def garwood_interval(observed: float, level: float = 0.95) -> tuple[float, float]:
    """Exact Poisson (Garwood) interval for a count."""
    alpha = 1.0 - level
    lower = 0.0 if observed == 0 else float(stats.chi2.ppf(alpha / 2, 2 * observed) / 2)
    upper = float(stats.chi2.ppf(1 - alpha / 2, 2 * observed + 2) / 2)
    return lower, upper


def clopper_pearson(successes: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    alpha = 1.0 - level
    lower = 0.0 if successes == 0 else float(stats.beta.ppf(alpha / 2, successes, n - successes + 1))
    upper = 1.0 if successes == n else float(stats.beta.ppf(1 - alpha / 2, successes + 1, n - successes))
    return lower, upper


def _profile_interval(profile_loss, estimate: float, minimum: float, scale: float) -> tuple[float, float]:
    """Both roots of ``2 * (profile_loss(x) - minimum) = chi2_95`` around ``estimate``."""
    def excess(x):
        return 2.0 * (profile_loss(x) - minimum) - _CHI2_95

    bounds = []
    for direction in (-1.0, 1.0):
        step, far = max(scale, 1e-6), None
        for _ in range(60):
            candidate = estimate + direction * step
            value = excess(candidate)
            if not math.isfinite(value):
                break
            if value > 0:
                far = candidate
                break
            step *= 2.0
        if far is None:
            bounds.append(math.nan)
            continue
        low, high = sorted((estimate, far))
        bounds.append(float(optimize.brentq(excess, low, high, xtol=1e-10, rtol=1e-12)))
    return bounds[0], bounds[1]


def calibration_intercept(positive, logits, weight=None, with_interval: bool = False) -> dict:
    """Logistic regression of y on the offset ``logit(p)`` with the slope fixed at 1."""
    y = np.asarray(positive, dtype=np.float64)
    z = np.asarray(logits, dtype=np.float64)
    w = np.ones(len(y)) if weight is None else np.asarray(weight, dtype=np.float64)
    if not np.isfinite(z).all():
        return {"status": "not estimable (non-finite calibrated logits)"}
    if float(w @ y) <= 0:
        return {"status": "not estimable (zero frauds)"}
    ones = np.ones((len(y), 1))
    fit = newton_logistic(ones, y, w, offset=z)
    if not fit["converged"]:
        return {"status": "not estimable (fit failure)"}
    estimate = float(fit["beta"][0])
    out = {"status": "ok", "estimate": estimate}
    if with_interval:
        se = 1.0 / math.sqrt(float(fit["hessian"][0, 0]))
        lower, upper = _profile_interval(lambda a: _loss(z + a, y, w), estimate, fit["loss"], se)
        out.update({"profile_likelihood_95": [lower, upper], "wald_standard_error": se})
    return out


def calibration_slope(positive, logits, weight=None, with_interval: bool = False) -> dict:
    """Logistic regression of y on ``logit(p)`` (free intercept); the slope is the estimate.

    A zero or negative slope is a valid evaluation result, not a failure.
    """
    y = np.asarray(positive, dtype=np.float64)
    z = np.asarray(logits, dtype=np.float64)
    w = np.ones(len(y)) if weight is None else np.asarray(weight, dtype=np.float64)
    if not np.isfinite(z).all():
        return {"status": "not estimable (non-finite calibrated logits)"}
    if float(w @ y) <= 0:
        return {"status": "not estimable (zero frauds)"}
    total = float(w.sum())
    mean = float(w @ z) / total
    sd = math.sqrt(float(w @ (z - mean) ** 2) / total)
    if not sd > 0:
        return {"status": "not estimable (constant calibrated logits)"}
    x = (z - mean) / sd
    ones = np.ones(len(y))
    start = np.array([0.0, math.log(float(w @ y) / (total - float(w @ y)))]) if total > float(w @ y) else None
    fit = newton_logistic(np.column_stack([x, ones]), y, w, start=start)
    if not fit["converged"]:
        return {"status": "not estimable (fit failure)"}
    slope = float(fit["beta"][0] / sd)
    out = {"status": "ok", "estimate": slope,
           "intercept_of_slope_model": float(fit["beta"][1] - fit["beta"][0] * mean / sd)}
    if with_interval:
        covariance = np.linalg.inv(fit["hessian"])
        se = math.sqrt(float(covariance[0, 0])) / sd

        def profile(b):
            inner = newton_logistic(ones[:, None], y, w, offset=b * (z - mean), start=[fit["beta"][1]])
            return inner["loss"] if inner["converged"] else math.inf

        lower, upper = _profile_interval(profile, slope, fit["loss"], se)
        out.update({"profile_likelihood_95": [lower, upper], "wald_standard_error": se})
    return out


def calibration_metrics(positive, probabilities, logits=None, with_intervals: bool = True) -> dict:
    """Full-partition calibration point estimates (Section 6.5) for one probability vector.

    ``logits`` are the calibrated logits when they exist (sigmoid); without
    them (isotonic) the log loss uses clipped probabilities and the intercept
    and slope are estimable only if no probability is exactly 0 or 1.
    """
    y = np.asarray(positive, dtype=bool)
    p = np.asarray(probabilities, dtype=np.float64)
    n, observed, expected = len(y), int(y.sum()), float(p.sum())
    prevalence = observed / n
    brier = float(np.mean((p - y) ** 2))
    reference = prevalence * (1.0 - prevalence)
    garwood = garwood_interval(observed)
    out = {
        "n_rows": n, "n_positive": observed, "prevalence": prevalence,
        "mean_predicted_probability": expected / n,
        "observed_expected": {"observed": observed, "expected": expected, "ratio": observed / expected,
                              "garwood_95_on_observed": list(garwood),
                              "garwood_95_ratio": [garwood[0] / expected, garwood[1] / expected]},
        "brier_score": brier,
        "brier_reference_constant_prevalence": reference,
        "brier_skill": (1.0 - brier / reference) if reference > 0 else math.nan,
        "log_loss": float(np.mean(log_loss_terms(y, logits, p))),
        "log_loss_source": "calibrated logits" if logits is not None else
                           f"probabilities clipped to [{PROBABILITY_CLIP}, 1 - {PROBABILITY_CLIP}]",
        "n_probability_exactly_zero": int((p == 0).sum()), "n_probability_exactly_one": int((p == 1).sum()),
        "n_frauds_with_probability_zero": int((y & (p == 0)).sum()),
    }
    if logits is None:
        with np.errstate(divide="ignore"):
            logits = np.log(p) - np.log1p(-p)
    out["calibration_intercept"] = calibration_intercept(y, logits, with_interval=with_intervals)
    out["calibration_slope"] = calibration_slope(y, logits, with_interval=with_intervals)
    return out


# ---------------------------------------------------------------------------
# Reliability tables
# ---------------------------------------------------------------------------

def decade_bin_labels(edges=RELIABILITY_BIN_EDGES) -> list[str]:
    labels = [f"[{edges[i]:g}, {edges[i + 1]:g})" for i in range(len(edges) - 1)]
    labels[-1] = f"[{edges[-2]:g}, {edges[-1]:g}]"
    return labels


def reliability_table(positive, probabilities, bin_index, labels) -> list[dict]:
    """Rows, observed frauds, expected frauds and the Clopper-Pearson interval of each bin.

    A bin with fewer than 5 observed or fewer than 5 expected frauds is flagged
    "not interpretable". Bins are never summarized into a single number.
    """
    y = np.asarray(positive, dtype=bool)
    p = np.asarray(probabilities, dtype=np.float64)
    bin_index = np.asarray(bin_index)
    table = []
    for k, label in enumerate(labels):
        inside = bin_index == k
        n, observed, expected = int(inside.sum()), int(y[inside].sum()), float(p[inside].sum())
        lower, upper = clopper_pearson(observed, n)
        table.append({
            "bin": label, "n_rows": n, "observed_frauds": observed, "expected_frauds": expected,
            "observed_rate": observed / n if n else math.nan,
            "mean_predicted_probability": expected / n if n else math.nan,
            "clopper_pearson_95": [lower, upper],
            "interpretable": bool(observed >= 5 and expected >= 5),
        })
    return table


def decade_reliability(positive, probabilities) -> list[dict]:
    return reliability_table(positive, probabilities, reliability_bin_index(probabilities), decade_bin_labels())


def k_derived_score_edges(scores, times, positions, ks) -> list[dict]:
    """Score of the K-th ranked row for each K under the Section 9.2 key. **Label-free by construction.**"""
    scores = np.asarray(scores, dtype=np.float64)
    order = ranking_order(scores, times, positions)
    return [{"k": int(k), "score_at_k": float(scores[order[int(k) - 1]])} for k in ks]


def k_band_index(scores, edges: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Band of each row between consecutive K-derived score edges; band 0 holds the highest scores."""
    ordered = sorted(edges, key=lambda e: e["k"])
    ascending = np.array([e["score_at_k"] for e in ordered][::-1], dtype=np.float64)
    if np.any(np.diff(ascending) < 0):
        raise ValueError("K-derived edges must be non-increasing in K.")
    index = len(ordered) - np.searchsorted(ascending, np.asarray(scores, dtype=np.float64), side="right")
    labels = [f"score >= score@K={ordered[0]['k']}"]
    labels += [f"score@K={ordered[i + 1]['k']} <= score < score@K={ordered[i]['k']}"
               for i in range(len(ordered) - 1)]
    labels.append(f"score < score@K={ordered[-1]['k']}")
    return index, labels


# ---------------------------------------------------------------------------
# (A) Calibrator-fit bootstrap on T_cal (Section 6.5 (A); Addendum C3)
# ---------------------------------------------------------------------------

def tcal_calibrator_bootstrap(scores, y_true, times, start_time: int, end_time_inclusive: int,
                              block_seconds: int, n_replicates: int, seed: int, positive_label=1,
                              min_valid_share: float = CALIBRATION_MIN_VALID_SHARE) -> dict:
    """CBB of ``T_cal`` that refits only the 2-parameter sigmoid of one model.

    Each call creates its own generator from ``seed``, so two models get
    identical resampled rows. Returns the classification counts, the
    estimability decision (>= ``min_valid_share`` valid) and, when estimable,
    the percentile intervals of A and B. ``parameters`` (in-memory only) holds
    the valid replicates' ``(A, B)`` for propagation.
    """
    s = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(y_true) == positive_label
    offsets, T = period_offsets(times, start_time, end_time_inclusive)
    counts = {VALID: 0, **{c: 0 for c in INVALID_CATEGORIES}}
    parameters, low_positive = [], []
    for weights in iter_replicate_weights(offsets, T, "cbb", block_seconds, n_replicates, seed):
        w = weights.astype(np.float64)
        fit = fit_platt(s, positive, sample_weight=w, positive_label=True)
        category = classify_calibrator(fit, s, sample_weight=w)
        counts[category] += 1
        if category == VALID:
            parameters.append((fit["A"], fit["B"]))
            low_positive.append(fit["n_positive"] <= LOW_POSITIVE_MAX)
    parameters = np.asarray(parameters, dtype=np.float64).reshape(-1, 2)
    low_positive = np.asarray(low_positive, dtype=bool)
    valid_share = counts[VALID] / n_replicates
    estimable = valid_share >= min_valid_share
    out = {
        "scheme": "cbb", "block_seconds": int(block_seconds), "n_blocks_per_replicate": math.ceil(T / block_seconds),
        "period_seconds": T, "n_replicates": int(n_replicates), "seed": int(seed),
        "counts": counts, "n_valid_low_positive_1_to_4": int(low_positive.sum()),
        "valid_share": valid_share, "min_valid_share": min_valid_share, "estimable": bool(estimable),
        "note": ("invalid replicates are excluded; exclusion conditions the distribution on fittable "
                 "samples and tends to understate uncertainty"),
        "parameters": parameters, "low_positive": low_positive,
    }
    if not estimable:
        out["status"] = "calibrator parameter uncertainty not estimable under the pre-registered rule"
        return out
    def interval(values):
        lower, upper = np.percentile(values, [2.5, 97.5])
        return {"percentile_95": [float(lower), float(upper)], "median": float(np.median(values)),
                "standard_error": float(np.std(values, ddof=1))}
    out["status"] = "estimable"
    out["A"], out["B"] = interval(parameters[:, 0]), interval(parameters[:, 1])
    if low_positive.any() and (~low_positive).sum() > 1:
        kept = parameters[~low_positive]
        out["sensitivity_without_low_positive"] = {"n": int(len(kept)), "A": interval(kept[:, 0]),
                                                   "B": interval(kept[:, 1])}
    return out


# ---------------------------------------------------------------------------
# (B) Calibration-evaluation bootstrap of a frozen calibrated model
# ---------------------------------------------------------------------------

def calibration_evaluation_bootstrap(vectors: dict, y_true, times, start_time: int, end_time_inclusive: int,
                                     block_seconds: int, n_replicates: int, seed: int, pairs=(),
                                     positive_label=1,
                                     min_valid_share: float = CALIBRATION_MIN_VALID_SHARE) -> dict:
    """Block-bootstrap intervals of the calibration metrics of frozen probability vectors.

    ``vectors[name]`` = ``{"probabilities": p, "logits": z or None, "fit_metrics": bool}``.
    No calibrator is refitted. Direct metrics (O/E, Brier, log loss) are never
    excluded; Brier skill is undefined in a zero-fraud replicate; the intercept
    and slope need a fit per replicate and are reported only if at least
    ``min_valid_share`` of the replicates fit. A negative slope is a kept result.
    ``pairs`` lists ``(a, b)`` names for paired differences ``a - b``.
    """
    positive = np.asarray(y_true) == positive_label
    yf = positive.astype(np.float64)
    offsets, T = period_offsets(times, start_time, end_time_inclusive)
    names = list(vectors)
    squared = {n: (np.asarray(vectors[n]["probabilities"], dtype=np.float64) - yf) ** 2 for n in names}
    loss_terms = {n: log_loss_terms(positive, vectors[n].get("logits"), vectors[n]["probabilities"]) for n in names}
    # Fitted metrics are warm-started from the full-partition estimates (same optimum, fewer iterations).
    fitted, warm = [], {}
    for n in names:
        z = vectors[n].get("logits")
        if not vectors[n].get("fit_metrics") or z is None or not np.isfinite(z).all():
            continue
        a_full, b_full = calibration_intercept(yf, z), calibration_slope(yf, z)
        if a_full["status"] == "ok" and b_full["status"] == "ok":
            fitted.append(n)
            warm[n] = {"design": np.column_stack([np.asarray(z, dtype=np.float64), np.ones(len(yf))]),
                       "intercept": [a_full["estimate"]],
                       "slope": [b_full["estimate"], b_full["intercept_of_slope_model"]]}
    draws = {n: {k: np.full(n_replicates, np.nan) for k in
                 ("observed_expected", "brier_score", "log_loss", "brier_skill")} for n in names}
    for n in fitted:
        draws[n]["calibration_intercept"] = np.full(n_replicates, np.nan)
        draws[n]["calibration_slope"] = np.full(n_replicates, np.nan)
    n_zero_fraud = 0
    for b, weights in enumerate(iter_replicate_weights(offsets, T, "cbb", block_seconds, n_replicates, seed)):
        w = weights.astype(np.float64)
        total, observed = float(w.sum()), float(w @ yf)
        n_zero_fraud += observed == 0
        prevalence = observed / total
        reference = prevalence * (1.0 - prevalence)
        for n in names:
            d = draws[n]
            d["observed_expected"][b] = observed / float(w @ vectors[n]["probabilities"])
            d["brier_score"][b] = float(w @ squared[n]) / total
            d["log_loss"][b] = float(w @ loss_terms[n]) / total
            if reference > 0:
                d["brier_skill"][b] = 1.0 - d["brier_score"][b] / reference
        if observed > 0:
            drawn = w > 0
            wd, yd = w[drawn], yf[drawn]
            for n in fitted:
                design = warm[n]["design"][drawn]
                a = newton_logistic(design[:, 1:], yd, wd, offset=design[:, 0], start=warm[n]["intercept"])
                if a["converged"]:
                    draws[n]["calibration_intercept"][b] = a["beta"][0]
                s = newton_logistic(design, yd, wd, start=warm[n]["slope"])
                if s["converged"]:
                    draws[n]["calibration_slope"][b] = s["beta"][0]

    def summarize(values, fitted_metric: bool) -> dict:
        finite = values[np.isfinite(values)]
        base = {"n_replicates": int(n_replicates), "n_valid": int(finite.size),
                "valid_share": finite.size / n_replicates}
        if fitted_metric and base["valid_share"] < min_valid_share:
            return {**base, "status": "not estimable under the pre-registered rule"}
        if finite.size < 2:
            return {**base, "status": "not estimable (fewer than two valid replicates)"}
        lower, upper = np.percentile(finite, [2.5, 97.5])
        return {**base, "status": "ok", "percentile_95": [float(lower), float(upper)],
                "median": float(np.median(finite)), "standard_error": float(finite.std(ddof=1))}

    out = {"scheme": "cbb", "block_seconds": int(block_seconds), "n_replicates": int(n_replicates),
           "seed": int(seed), "n_zero_fraud_replicates": int(n_zero_fraud), "by_vector": {}, "paired": {}}
    for n in names:
        block = {k: summarize(draws[n][k], False) for k in ("observed_expected", "brier_score", "log_loss")}
        block["brier_skill"] = {**summarize(draws[n]["brier_skill"], False),
                                "n_undefined_zero_fraud": int(n_zero_fraud)}
        if n in fitted:
            for k in ("calibration_intercept", "calibration_slope"):
                block[k] = summarize(draws[n][k], True)
            slopes = draws[n]["calibration_slope"]
            valid = np.isfinite(slopes)
            block["calibration_slope"]["share_slope_at_or_below_zero"] = (
                float((slopes[valid] <= 0).mean()) if valid.any() else None)
        else:
            reason = "not available: the intercept and slope need finite calibrated logits"
            block["calibration_intercept"] = block["calibration_slope"] = {"status": reason}
        out["by_vector"][n] = block
    for a, b in pairs:
        out["paired"][f"{a} minus {b}"] = {
            k: summarize(draws[a][k] - draws[b][k], False) for k in ("brier_score", "log_loss")}
    return out
