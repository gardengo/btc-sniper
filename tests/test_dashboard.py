"""Dashboard data access, charts, and a smoke test of the real pages.

The pages themselves are deliberately thin, so almost everything worth asserting
lives in `app.data_access` and `app.charts`, which import no Streamlit. The smoke
test then runs the actual app through `streamlit.testing` against a synthetic
database, which is the only way to catch a page that raises on an empty table --
the state this project is in for most of its first year.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import charts
from app import data_access as data
from app.streamlit_app import CONFIG_ENV, enabled_pages
from src.forecast.generate import Forecast
from src.models import registry
from src.models.base import ModelMetadata
from src.monitoring.realization import realize_points
from src.storage import repositories as repo
from src.storage.db import connect, init_db, transaction
from src.utils.config import AppConfig, load_config
from tests.conftest import frame_to_candles, make_ohlcv

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
APP_SCRIPT: str = str(REPO_ROOT / "app" / "streamlit_app.py")

HORIZONS: tuple[int, ...] = (1, 7, 30, 90, 180, 365)
LEVELS: tuple[float, ...] = (0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975)
MODEL_VERSION: str = "lgbmq-expanding-20240101-v1-abc123"


def _forecast(
    origin: pd.Timestamp,
    origin_close: float,
    *,
    forecast_id: str,
    drift: float = 0.0,
) -> Forecast:
    """A forecast whose bands widen with horizon, like a real one."""
    points: list[dict] = []
    quantiles: list[dict] = []
    for horizon in HORIZONS:
        median = drift * horizon / 365.0
        spread = 0.05 * np.sqrt(horizon)
        points.append(
            {
                "horizon_days": horizon,
                "target_date": (origin + pd.Timedelta(days=horizon)).strftime("%Y-%m-%d"),
                "predicted_log_return": median,
                "predicted_price": origin_close * float(np.exp(median)),
                "direction_predicted": int(np.sign(median)),
                "model_weight": 1.0 if horizon == 1 else (0.0 if horizon >= 30 else 0.5),
                "source": "model" if horizon == 1 else ("baseline" if horizon >= 30 else "blend"),
            }
        )
        for level in LEVELS:
            value = median + spread * (level - 0.50) * 4.0
            quantiles.append(
                {
                    "horizon_days": horizon,
                    "quantile": level,
                    "quantile_label": f"q{int(level * 1000)}",
                    "predicted_log_return": value,
                    "predicted_price": origin_close * float(np.exp(value)),
                    "crossing_adjusted": 0,
                }
            )
    return Forecast(
        forecast_id=forecast_id,
        origin_date=origin,
        origin_close=origin_close,
        model_version=MODEL_VERSION,
        feature_version="v1",
        horizon_grid_version="v1",
        config_version="v1",
        code_commit="abc1234",
        created_at="2020-06-18T00:00:00Z",
        points=pd.DataFrame(points),
        quantiles=pd.DataFrame(quantiles),
        current_price=origin_close * 1.004,
    )


def _metadata() -> ModelMetadata:
    return ModelMetadata(
        model_id="model-1",
        model_version=MODEL_VERSION,
        algorithm="lightgbm_quantile",
        feature_version="v1",
        horizon_grid_version="v1",
        training_cutoff="2023-12-30",
        status="candidate",
        config_version="v1",
        code_commit="abc1234",
        training_window_strategy="expanding",
        training_start="2018-08-16",
        training_rows=1_900,
        horizons=HORIZONS,
        quantiles=LEVELS,
        feature_names=("ret_1d",),
        hyperparameters={"num_leaves": 3},
        random_seed=42,
        created_at="2020-06-01T00:00:00Z",
    )


@pytest.fixture(scope="module")
def ohlcv_frame() -> pd.DataFrame:
    return make_ohlcv()


@pytest.fixture(scope="module")
def populated_db(tmp_path_factory, ohlcv_frame) -> Path:
    """A database holding candles, a model, two forecasts and their realizations."""
    path = init_db(tmp_path_factory.mktemp("dashboard") / "btc_forecast.db")
    connection = connect(path)
    try:
        with transaction(connection):
            repo.upsert_candles(connection, frame_to_candles(ohlcv_frame))
            registry.register(connection, _metadata())

        close = ohlcv_frame["close"]
        # Two origins a month apart so the revision chart has something to draw,
        # and the older one is far enough back that its 1-day target resolved.
        for offset, tag, drift in ((120, "older", 0.10), (30, "newer", -0.05)):
            origin = close.index[-offset]
            with transaction(connection):
                repo.upsert_forecast(
                    connection,
                    _forecast(
                        origin, float(close.loc[origin]), forecast_id=tag, drift=drift
                    ),
                    source="binance",
                    symbol="BTCUSDT",
                    timeframe="1d",
                )

        points = repo.load_forecast_points_for_realization(
            connection, source="binance", symbol="BTCUSDT", timeframe="1d"
        )
        quantiles = repo.load_forecast_quantiles_long(
            connection, points["forecast_id"].unique().tolist()
        )
        rows = realize_points(points, quantiles, close)
        with transaction(connection):
            repo.upsert_realizations(connection, rows)
    finally:
        connection.close()
    return path


@pytest.fixture(scope="module")
def dashboard_config(tmp_path_factory, populated_db) -> AppConfig:
    """The real config.yaml, pointed at the synthetic database."""
    source = (REPO_ROOT / "config.yaml").read_text(encoding="utf-8")
    patched = source.replace("  data_dir: data\n", f"  data_dir: {populated_db.parent.as_posix()}\n")
    assert patched != source
    target = tmp_path_factory.mktemp("config") / "config.yaml"
    target.write_text(patched, encoding="utf-8")
    config = load_config(target)
    assert config.storage.database_path == populated_db
    return config


@pytest.fixture
def db(populated_db) -> sqlite3.Connection:
    connection = connect(populated_db, read_only=True)
    try:
        yield connection
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# data access
# ---------------------------------------------------------------------------


class TestReadOnly:
    def test_the_dashboard_cannot_write(self, dashboard_config) -> None:
        # The whole point of the read-only URI: a viewer clicking around must not
        # be able to change a stored forecast or promote a model.
        connection = data.read_only_connection(dashboard_config)
        try:
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                connection.execute("DELETE FROM forecasts")
        finally:
            connection.close()

    def test_a_missing_database_is_reported_not_created(self, tmp_path) -> None:
        from src.storage.db import connect as raw_connect

        with pytest.raises(FileNotFoundError):
            raw_connect(tmp_path / "absent.db", read_only=True)


class TestForecastView:
    def test_the_newest_forecast_is_the_one_shown(self, db, dashboard_config, ohlcv_frame) -> None:
        view = data.load_forecast_view(db, dashboard_config)
        assert view is not None
        assert view.forecast_id == "newer"
        assert view.origin_date == ohlcv_frame.index[-30]
        assert view.horizons == HORIZONS

    def test_the_curve_is_daily_and_anchored_at_the_origin(self, db, dashboard_config) -> None:
        view = data.load_forecast_view(db, dashboard_config)
        assert len(view.curve) == view.max_horizon + 1
        first = view.curve.iloc[0]
        band = [first[level] for level in LEVELS]
        assert all(value == pytest.approx(view.origin_close) for value in band)

    def test_a_candidate_forecast_is_not_marked_production(self, db, dashboard_config) -> None:
        # Nothing is promoted in this database, so the page must not imply one is.
        assert data.production_version(db) is None
        assert data.load_forecast_view(db, dashboard_config).is_production is False

    def test_provenance_survives_the_round_trip(self, db, dashboard_config) -> None:
        provenance = data.load_forecast_view(db, dashboard_config).provenance()
        by_horizon = provenance.set_index("horizon_days")
        assert by_horizon.loc[1, "blend_source"] == "model"
        assert by_horizon.loc[365, "blend_source"] == "baseline"
        assert by_horizon.loc[365, "model_weight"] == pytest.approx(0.0)

    def test_the_summary_covers_one_three_six_and_twelve_months(self, db, dashboard_config) -> None:
        summary = data.horizon_summary(data.load_forecast_view(db, dashboard_config))
        assert summary["horizon"].tolist() == ["1M", "3M", "6M", "12M"]
        assert (summary["low_95"] < summary["median"]).all()
        assert (summary["high_95"] > summary["median"]).all()

    def test_no_forecast_at_all_returns_none(self, tmp_path, dashboard_config) -> None:
        empty = init_db(tmp_path / "empty.db")
        connection = connect(empty)
        try:
            assert data.load_forecast_view(connection, dashboard_config) is None
        finally:
            connection.close()


class TestPredictionLog:
    def test_the_log_pairs_each_prediction_with_its_outcome(self, db, dashboard_config) -> None:
        frame = data.prediction_log(db, dashboard_config)
        assert not frame.empty
        assert "predicted_median" in frame.columns
        resolved = frame[frame["evaluation_status"] == data.STATUS_FULL]
        assert not resolved.empty
        assert resolved["predicted_median"].notna().all()
        assert resolved["actual_close"].notna().all()

    def test_pending_rows_are_listed_not_hidden(self, db, dashboard_config) -> None:
        # A log that drops unresolved rows is a log of the answers you already have.
        frame = data.prediction_log(db, dashboard_config)
        assert (frame["evaluation_status"] == "pending").any()

    def test_filters_narrow_the_rows(self, db, dashboard_config) -> None:
        everything = data.prediction_log(db, dashboard_config)
        one_horizon = data.prediction_log(db, dashboard_config, horizon_days=7)
        assert set(one_horizon["horizon_days"]) == {7}
        assert len(one_horizon) < len(everything)

        resolved = data.prediction_log(db, dashboard_config, status=data.STATUS_FULL)
        assert set(resolved["evaluation_status"]) == {data.STATUS_FULL}

    def test_a_date_window_excludes_the_older_origin(self, db, dashboard_config) -> None:
        options = data.log_filter_options(db)
        latest = options["origins"][-1]
        frame = data.prediction_log(db, dashboard_config, origin_from=latest)
        assert set(frame["forecast_origin_date"]) == {latest}

    def test_filter_options_come_from_stored_rows(self, db) -> None:
        options = data.log_filter_options(db)
        assert set(options["horizons"]) == set(HORIZONS)
        assert len(options["origins"]) == 2
        assert "pending" in options["statuses"]

    def test_an_impossible_filter_returns_an_empty_frame_not_an_error(
        self, db, dashboard_config
    ) -> None:
        frame = data.prediction_log(db, dashboard_config, origin_from="2099-01-01")
        assert frame.empty
        assert list(frame.columns) == list(data.LOG_COLUMNS)


class TestPerformance:
    def test_scopes_are_kept_apart(self, db) -> None:
        performance = data.load_performance(db)
        assert performance.production.empty
        assert performance.outer_test.empty
        assert performance.realized > 0

    def test_production_evidence_is_counted_from_resolved_rows_only(self, db) -> None:
        performance = data.load_performance(db)
        log = data.load_performance(db)
        assert performance.realized + performance.pending == len(
            repo.load_realizations(db, limit=None)
        )
        assert log.has_production_evidence is True

    def test_metric_table_is_one_row_per_horizon(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "horizon_days": horizon,
                    "regime": "all",
                    "metric_name": name,
                    "metric_value": 0.5,
                }
                for horizon in (1, 7)
                for name in ("pinball_mean", "coverage_95")
            ]
        )
        table = data.metric_table(metrics)
        assert table["horizon_days"].tolist() == [1, 7]
        assert set(table.columns) == {"horizon_days", "pinball_mean", "coverage_95"}

    def test_metric_table_of_nothing_is_empty_not_an_error(self) -> None:
        assert data.metric_table(pd.DataFrame()).empty

    def test_regime_table_excludes_the_overall_row(self) -> None:
        metrics = pd.DataFrame(
            [
                {"horizon_days": 1, "regime": regime, "metric_name": "pinball_mean",
                 "metric_value": 0.1}
                for regime in ("all", "bull", "bear")
            ]
        )
        table = data.regime_table(metrics)
        assert set(table.columns) == {"horizon_days", "bull", "bear"}

    def test_model_history_shows_status(self, db) -> None:
        history = data.model_history(db)
        assert history.loc[0, "model_version"] == MODEL_VERSION
        assert history.loc[0, "status"] == "candidate"

    def test_revision_history_is_ordered_oldest_first(self, db, dashboard_config) -> None:
        revisions = data.revision_history(db, dashboard_config, horizon_days=365)
        assert len(revisions) == 2
        assert revisions["forecast_origin_date"].is_monotonic_increasing
        assert revisions["change_pct"].iloc[0] > 0  # the older forecast drifted up
        assert revisions["change_pct"].iloc[1] < 0


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------


class TestCharts:
    def test_the_forecast_chart_draws_history_bands_and_a_now_marker(
        self, db, dashboard_config
    ) -> None:
        view = data.load_forecast_view(db, dashboard_config)
        history = data.price_history(db, dashboard_config, days=200)
        figure = charts.forecast_chart(
            history,
            view.curve,
            origin_date=view.origin_date,
            origin_close=view.origin_close,
            current_price=view.current_price,
        )
        names = [trace.name for trace in figure.data]
        assert "actual close" in names
        assert "median forecast" in names
        for interval in (50, 80, 95):
            assert f"{interval}% interval" in names
        assert any(
            annotation.text == "NOW" for annotation in figure.layout.annotations
        )

    def test_the_forecast_chart_says_so_when_there_is_nothing_to_draw(self) -> None:
        figure = charts.forecast_chart(
            pd.DataFrame(),
            pd.DataFrame(),
            origin_date=pd.Timestamp("2024-01-01"),
            origin_close=1.0,
        )
        assert figure.data == ()
        assert "No forecast" in figure.layout.annotations[0].text

    def test_the_revision_chart_needs_two_origins_to_say_anything(self) -> None:
        one = pd.DataFrame(
            [{"forecast_origin_date": "2024-01-01", "origin_close": 1.0,
              "predicted_price": 1.1}]
        )
        figure = charts.revision_chart(one, horizon_days=365)
        assert figure.data == ()
        assert "at least two origins" in figure.layout.annotations[0].text

    def test_the_revision_chart_plots_forecast_against_anchor(self, db, dashboard_config) -> None:
        revisions = data.revision_history(db, dashboard_config, horizon_days=365)
        figure = charts.revision_chart(revisions, horizon_days=365)
        assert len(figure.data) == 2
        assert "anchor" in figure.data[0].name

    def test_the_coverage_chart_draws_a_nominal_line_per_band(self) -> None:
        table = pd.DataFrame(
            [{"horizon_days": 1, "coverage_50": 0.48, "coverage_95": 0.99}]
        )
        figure = charts.coverage_chart(table)
        assert len(figure.data) == 2
        nominals = sorted(
            shape.y0 for shape in figure.layout.shapes if shape.type == "line"
        )
        assert nominals == pytest.approx([0.50, 0.95])

    def test_the_coverage_chart_says_so_when_nothing_is_scored(self) -> None:
        figure = charts.coverage_chart(pd.DataFrame())
        assert "No realized forecast" in figure.layout.annotations[0].text

    def test_the_provenance_chart_shows_the_weight_falling_to_zero(
        self, db, dashboard_config
    ) -> None:
        provenance = data.load_forecast_view(db, dashboard_config).provenance()
        figure = charts.provenance_chart(provenance)
        assert list(figure.data[0].y)[-1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# the pages themselves
# ---------------------------------------------------------------------------


class TestPages:
    def test_enabled_pages_come_from_config(self, dashboard_config) -> None:
        assert enabled_pages(dashboard_config) == [
            "dashboard",
            "prediction_log",
            "model_performance",
        ]

    @pytest.mark.parametrize(
        "page", ["dashboard", "prediction_log", "model_performance"]
    )
    def test_every_page_renders(self, page, dashboard_config, monkeypatch) -> None:
        from streamlit.testing.v1 import AppTest

        monkeypatch.setenv(CONFIG_ENV, str(dashboard_config.source_path))
        app = AppTest.from_file(APP_SCRIPT, default_timeout=90)
        app.session_state["page"] = page
        app.run()
        assert not app.exception, [str(item) for item in app.exception]

    def test_a_page_with_no_data_explains_itself(self, tmp_path, monkeypatch) -> None:
        """The state this project spends its first year in must not look broken."""
        from streamlit.testing.v1 import AppTest

        init_db(tmp_path / "btc_forecast.db")
        source = (REPO_ROOT / "config.yaml").read_text(encoding="utf-8")
        target = tmp_path / "config.yaml"
        target.write_text(
            source.replace("  data_dir: data\n", f"  data_dir: {tmp_path.as_posix()}\n"),
            encoding="utf-8",
        )
        monkeypatch.setenv(CONFIG_ENV, str(target))

        app = AppTest.from_file(APP_SCRIPT, default_timeout=90)
        app.session_state["page"] = "dashboard"
        app.run()
        assert not app.exception, [str(item) for item in app.exception]
        assert any("No forecast is stored yet" in item.value for item in app.warning)
