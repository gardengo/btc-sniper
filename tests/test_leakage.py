"""Look-ahead leakage tests.

These are the most important tests in the repository. CLAUDE.md section 2.1 and
DATA_SPEC.md section 5 make point-in-time correctness non-negotiable, and a
leak is silent: it shows up as an implausibly good model, not as an error.

The central technique is the truncation equivalence check -- build features on
the full series, build them again on the series cut at date ``T``, and require
that every value at or before ``T`` is identical. A feature that peeks at
``T+1`` cannot satisfy that.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import indicators as ind
from src.features.pipeline import assert_no_lookahead, build_features
from src.features.regime import compute_regime_labels
from src.utils.config import AppConfig

CUTOFFS: tuple[str, ...] = ("2018-09-30", "2019-06-15", "2020-01-31")


def test_features_do_not_change_when_future_is_removed(
    ohlcv: pd.DataFrame, app_config: AppConfig
) -> None:
    for cutoff in CUTOFFS:
        mismatched = assert_no_lookahead(ohlcv, app_config.features, cutoff=cutoff)
        assert mismatched.empty, (
            f"look-ahead leakage at cutoff {cutoff} in columns "
            f"{list(mismatched.columns)}"
        )


def test_appending_a_future_row_never_changes_past_features(
    ohlcv: pd.DataFrame, app_config: AppConfig
) -> None:
    """A single new candle must only affect the row it belongs to."""
    base = ohlcv.iloc[:-1]
    extended = ohlcv

    base_features = build_features(base, app_config.features).features
    extended_features = build_features(extended, app_config.features).features

    overlap = base_features.index
    difference = (
        extended_features.loc[overlap] - base_features
    ).abs().max().max()
    assert difference == pytest.approx(0.0, abs=1e-12), (
        "appending one future candle changed an earlier feature value"
    )


def test_extreme_future_value_does_not_bleed_backwards(
    ohlcv: pd.DataFrame, app_config: AppConfig
) -> None:
    """A 10x spike on the last day must not alter any earlier feature.

    This catches accidental use of a whole-series statistic (a global mean, a
    global min/max, a non-causal normalisation) which a gentle synthetic series
    might not reveal.
    """
    original = build_features(ohlcv, app_config.features).features

    spiked = ohlcv.copy()
    last = spiked.index[-1]
    for column in ("open", "high", "low", "close"):
        spiked.loc[last, column] = spiked.loc[last, column] * 10.0
    spiked.loc[last, "volume"] = spiked.loc[last, "volume"] * 100.0

    after_spike = build_features(spiked, app_config.features).features
    earlier = original.index[:-1]

    difference = (after_spike.loc[earlier] - original.loc[earlier]).abs().max().max()
    assert difference == pytest.approx(0.0, abs=1e-12), (
        "a future spike changed past feature values; a feature is not causal"
    )


def test_regime_labels_are_causal(ohlcv: pd.DataFrame, app_config: AppConfig) -> None:
    """Regime labels drive the evaluation breakdown, so they must be causal too."""
    params = dict(app_config.features.regime)
    cutoff = pd.Timestamp("2019-06-15")

    full = compute_regime_labels(ohlcv, params).loc[:cutoff]
    truncated = compute_regime_labels(ohlcv.loc[:cutoff], params).loc[:cutoff]

    for axis in ("direction", "volatility"):
        # Compare the null masks and the labelled values separately: NaN != NaN
        # under pandas comparison semantics, which would make this vacuously fail.
        assert full[axis].isna().equals(truncated[axis].isna()), (
            f"{axis} regime label becomes available at a different date"
        )
        labelled = full[axis].notna()
        mismatch = (full.loc[labelled, axis] != truncated.loc[labelled, axis]).sum()
        assert mismatch == 0, f"{axis} regime label depends on future data"

    for column in ("trend_score", "drawdown", "direction_return", "volatility_rank"):
        difference = (full[column] - truncated[column]).abs().max()
        assert difference == pytest.approx(0.0, abs=1e-12), (
            f"regime statistic {column} depends on future data"
        )


@pytest.mark.parametrize(
    "name, builder",
    [
        ("sma", lambda s: ind.sma(s, 20)),
        ("ema", lambda s: ind.ema(s, 20)),
        ("wilder_ema", lambda s: ind.wilder_ema(s, 14)),
        ("rolling_std", lambda s: ind.rolling_std(s, 20)),
        ("rolling_max", lambda s: ind.rolling_max(s, 20)),
        ("rolling_min", lambda s: ind.rolling_min(s, 20)),
        ("rsi", lambda s: ind.rsi(s, 14)),
        ("zscore", lambda s: ind.zscore(s, 30)),
        ("rolling_percentile_rank", lambda s: ind.rolling_percentile_rank(s, 60, 30)),
    ],
)
def test_indicator_is_causal(name: str, builder, ohlcv: pd.DataFrame) -> None:
    """Each primitive must produce identical values on a truncated series."""
    close = ohlcv["close"]
    cut = 600

    full = builder(close).iloc[:cut]
    truncated = builder(close.iloc[:cut])

    pd.testing.assert_series_equal(
        full, truncated, check_names=False, rtol=1e-12, atol=1e-12,
        obj=f"{name} is not causal",
    )


# Every module that must be causal. `src/models/targets.py` is deliberately
# absent: a label has to read a future price, and that is the one place allowed
# to. Keeping the exemption to a single named file is what stops it spreading.
CAUSAL_SOURCE_ROOTS: tuple[str, ...] = (
    "src/features",
    "src/evaluation",
    "src/models/baselines.py",
    "src/validation",
)
FORWARD_LOOKING_EXEMPTION: str = "src/models/targets.py"


def _causal_source_files() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    files: list[Path] = []
    for entry in CAUSAL_SOURCE_ROOTS:
        target = root / entry
        files.extend(sorted(target.rglob("*.py")) if target.is_dir() else [target])
    return files


def _non_causal_calls(path: Path) -> list[str]:
    import ast

    offenders: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else ""
        if name in {"shift", "tshift"}:
            for argument in node.args:
                if (
                    isinstance(argument, ast.UnaryOp)
                    and isinstance(argument.op, ast.USub)
                ) or (
                    isinstance(argument, ast.Constant)
                    and isinstance(argument.value, int)
                    and argument.value < 0
                ):
                    offenders.append(f"{path.name}:{node.lineno} negative shift")
        if name in {"bfill", "backfill"}:
            offenders.append(f"{path.name}:{node.lineno} backward fill")
        for keyword in node.keywords:
            if keyword.arg == "center" and getattr(keyword.value, "value", False) is True:
                offenders.append(f"{path.name}:{node.lineno} centred window")
            if keyword.arg == "method" and getattr(keyword.value, "value", "") in {
                "bfill",
                "backfill",
            }:
                offenders.append(f"{path.name}:{node.lineno} backward fill")
    return offenders


def test_no_non_causal_constructs_in_causal_source() -> None:
    """Guard against a future `shift(-n)` or `center=True` creeping in.

    The source is parsed with `ast` rather than grepped, so the prose in the
    module docstrings that *describes* these forbidden constructs does not trip
    the check.
    """
    offenders: list[str] = []
    for path in _causal_source_files():
        offenders.extend(_non_causal_calls(path))
    assert not offenders, f"non-causal constructs found: {offenders}"


def test_the_only_forward_looking_module_is_the_target_builder() -> None:
    """The exemption must stay exactly one file.

    If a second module starts shifting forwards, this fails -- which is the
    point. A forward shift outside target construction is a leak by definition.
    """
    root = Path(__file__).resolve().parents[1]
    exempt = root / FORWARD_LOOKING_EXEMPTION
    assert _non_causal_calls(exempt), (
        f"{FORWARD_LOOKING_EXEMPTION} no longer shifts forward; "
        "if targets moved elsewhere, move the exemption too"
    )
    assert exempt not in _causal_source_files()


def test_true_range_uses_previous_close_only(ohlcv: pd.DataFrame) -> None:
    """True range at t may use close[t-1] but never close[t+1]."""
    high, low, close = ohlcv["high"], ohlcv["low"], ohlcv["close"]
    computed = ind.true_range(high, low, close)

    index = 100
    expected = max(
        high.iloc[index] - low.iloc[index],
        abs(high.iloc[index] - close.iloc[index - 1]),
        abs(low.iloc[index] - close.iloc[index - 1]),
    )
    assert computed.iloc[index] == pytest.approx(expected)
    # Row 0 has no previous close, so true range degenerates to the intraday
    # range. ATR still starts as NaN because its smoothing needs a full window.
    assert computed.iloc[0] == pytest.approx(high.iloc[0] - low.iloc[0])
    assert np.isnan(ind.atr(high, low, close, 14).iloc[0])
