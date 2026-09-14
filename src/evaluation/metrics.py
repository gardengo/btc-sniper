"""Forecast metrics required by VALIDATION_SPEC.md section 7.

Model-agnostic on purpose: everything here takes aligned arrays of actuals and
predictions, so the same code scores a naive baseline, a LightGBM candidate and
a logged production forecast. That matters because a metric implemented twice is
a metric that will eventually disagree with itself.

Conventions
-----------
* **Return space** metrics operate on log returns, which is what the model
  predicts. **Price space** metrics are reconstructed with
  ``price = origin_close * exp(log_return)`` (MODEL_SPEC.md section 1).
* Every function drops pairs where either side is ``NaN`` and returns ``NaN``
  for an empty sample rather than raising. A horizon with no scoreable origins
  is a normal state (VALIDATION_SPEC.md section 12), not an error.
* Nothing here silently fills a missing actual. A missing actual means the
  target date has not arrived.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPSILON: float = 1e-12


def _paired(actual: pd.Series, predicted: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Aligned, finite (actual, predicted) pairs as float arrays."""
    frame = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    if frame.empty:
        return np.array([]), np.array([])
    return (
        frame["actual"].to_numpy(dtype="float64"),
        frame["predicted"].to_numpy(dtype="float64"),
    )


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    a, p = _paired(actual, predicted)
    return float(np.mean(np.abs(a - p))) if a.size else float("nan")


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    a, p = _paired(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2))) if a.size else float("nan")


def bias(actual: pd.Series, predicted: pd.Series) -> float:
    """Mean signed error. Separates a systematic tilt from raw dispersion."""
    a, p = _paired(actual, predicted)
    return float(np.mean(p - a)) if a.size else float("nan")


def smape(actual: pd.Series, predicted: pd.Series) -> float:
    """Symmetric MAPE in percent, ``200|p-a| / (|a|+|p|)``.

    Used in price space where values are strictly positive, so the pathological
    near-zero denominator of MAPE does not arise.
    """
    a, p = _paired(actual, predicted)
    if not a.size:
        return float("nan")
    denominator = np.abs(a) + np.abs(p)
    safe = denominator > EPSILON
    if not safe.any():
        return float("nan")
    return float(np.mean(200.0 * np.abs(p - a)[safe] / denominator[safe]))


def mase(
    actual: pd.Series, predicted: pd.Series, reference_predicted: pd.Series
) -> float:
    """Mean absolute error scaled by a reference forecast's MAE.

    This is the MASE idea -- an error ratio that is unitless and comparable
    across horizons -- with the denominator taken from the **no-change baseline
    evaluated on the same origins and horizon** rather than from in-sample
    naive-1 errors. At a 365-day horizon an in-sample one-step denominator would
    be roughly 19x too small and the resulting number would be meaningless.

    Below 1.0 means the forecast beats the reference.
    """
    frame = pd.DataFrame(
        {"actual": actual, "predicted": predicted, "reference": reference_predicted}
    ).dropna()
    if frame.empty:
        return float("nan")
    reference_error = float(np.mean(np.abs(frame["actual"] - frame["reference"])))
    if reference_error <= EPSILON:
        return float("nan")
    model_error = float(np.mean(np.abs(frame["actual"] - frame["predicted"])))
    return model_error / reference_error


def directional_accuracy(
    actual: pd.Series, predicted: pd.Series
) -> dict[str, float]:
    """Directional hit rates, counting only origins where a call was made.

    A forecast whose median is exactly zero -- the `no_change` baseline -- makes
    no directional claim. Scoring it as wrong every time would read as 0%
    accuracy, which misrepresents an abstention as a failure. Such origins are
    excluded and ``direction_calls`` reports how many remain, so an abstaining
    forecast shows ``nan`` accuracy over ``0`` calls instead.
    """
    a, p = _paired(actual, predicted)
    empty = {
        "direction_accuracy": float("nan"),
        "direction_accuracy_up": float("nan"),
        "direction_accuracy_down": float("nan"),
        "direction_calls": 0.0,
    }
    if not a.size:
        return empty

    called = (np.abs(p) > EPSILON) & (np.abs(a) > EPSILON)
    if not called.any():
        return empty

    actual_sign = np.sign(a[called])
    predicted_sign = np.sign(p[called])
    correct = actual_sign == predicted_sign

    up = actual_sign > 0
    down = actual_sign < 0
    return {
        "direction_accuracy": float(np.mean(correct)),
        "direction_accuracy_up": float(np.mean(correct[up])) if up.any() else float("nan"),
        "direction_accuracy_down": float(np.mean(correct[down]))
        if down.any()
        else float("nan"),
        "direction_calls": float(called.sum()),
    }


def pinball_loss(actual: pd.Series, predicted: pd.Series, level: float) -> float:
    """Quantile (pinball) loss at ``level``. Lower is better; 0 is perfect."""
    if not 0.0 < level < 1.0:
        raise ValueError(f"quantile level must be in (0, 1), got {level}")
    a, p = _paired(actual, predicted)
    if not a.size:
        return float("nan")
    delta = a - p
    return float(np.mean(np.maximum(level * delta, (level - 1.0) * delta)))


def interval_coverage(
    actual: pd.Series, lower: pd.Series, upper: pd.Series
) -> float:
    """Fraction of actuals falling inside ``[lower, upper]``."""
    frame = pd.DataFrame({"actual": actual, "lower": lower, "upper": upper}).dropna()
    if frame.empty:
        return float("nan")
    inside = (frame["actual"] >= frame["lower"]) & (frame["actual"] <= frame["upper"])
    return float(inside.mean())


def interval_width(lower: pd.Series, upper: pd.Series) -> float:
    """Mean interval width in log-return space."""
    frame = pd.DataFrame({"lower": lower, "upper": upper}).dropna()
    if frame.empty:
        return float("nan")
    return float((frame["upper"] - frame["lower"]).mean())


def relative_price_width(lower: pd.Series, upper: pd.Series) -> float:
    """Mean interval width as a fraction of the origin close.

    ``exp(log_return)`` is the price multiple, so this reads directly: 0.4 means
    the band spans 40% of today's price. Comparable across horizons and across
    price levels in a way the log-return width is not.
    """
    frame = pd.DataFrame({"lower": lower, "upper": upper}).dropna()
    if frame.empty:
        return float("nan")
    return float((np.exp(frame["upper"]) - np.exp(frame["lower"])).mean())


def interval_score(
    actual: pd.Series, lower: pd.Series, upper: pd.Series, interval: float
) -> float:
    """Winkler interval score -- the coverage/width tradeoff as one number.

    ``width + (2/alpha) * miss_distance``. A forecast cannot win by reporting an
    absurdly wide band (width is penalised) or an absurdly narrow one (misses
    are penalised), which is exactly the tradeoff VALIDATION_SPEC.md section 7
    asks to be measured. Lower is better.
    """
    if not 0.0 < interval < 1.0:
        raise ValueError(f"interval must be in (0, 1), got {interval}")
    frame = pd.DataFrame({"actual": actual, "lower": lower, "upper": upper}).dropna()
    if frame.empty:
        return float("nan")
    alpha = 1.0 - interval
    width = frame["upper"] - frame["lower"]
    below = np.maximum(frame["lower"] - frame["actual"], 0.0)
    above = np.maximum(frame["actual"] - frame["upper"], 0.0)
    return float((width + (2.0 / alpha) * (below + above)).mean())


def dispersion(values: pd.Series) -> dict[str, float]:
    """Stability summary across folds or regimes (VALIDATION_SPEC.md section 7).

    A model with a good mean and a terrible worst fold is not a good model; the
    worst value is reported alongside the spread so it cannot be averaged away.
    """
    clean = pd.Series(values).dropna()
    if clean.empty:
        return {"mean": float("nan"), "std": float("nan"), "worst": float("nan"), "folds": 0.0}
    return {
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)),
        "worst": float(clean.max()),
        "folds": float(clean.size),
    }
