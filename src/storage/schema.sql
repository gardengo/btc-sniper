-- BTC Sniper SQLite schema.
--
-- Logical tables follow ARCHITECTURE.md section 3. Two deviations, both
-- documented in DATA_SPEC.md:
--   * `forecasts` is normalised into forecasts / forecast_points /
--     forecast_quantiles so the quantile list stays config-driven.
--   * `features` is stored long-format (one row per feature per day) so a new
--     feature_version can coexist with the old one without a schema migration.
--
-- All timestamps are UTC. `*_ms` columns are epoch milliseconds; `*_date` and
-- `*_at` columns are ISO-8601 strings.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

-- ---------------------------------------------------------------- market data

CREATE TABLE IF NOT EXISTS market_ohlcv (
    source          TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    timeframe       TEXT    NOT NULL,
    open_time_ms    INTEGER NOT NULL,
    close_time_ms   INTEGER NOT NULL,
    date_utc        TEXT    NOT NULL,
    open            REAL    NOT NULL,
    high            REAL    NOT NULL,
    low             REAL    NOT NULL,
    close           REAL    NOT NULL,
    volume          REAL    NOT NULL,
    quote_volume    REAL,
    trade_count     INTEGER,
    taker_buy_base  REAL,
    taker_buy_quote REAL,
    is_closed       INTEGER NOT NULL DEFAULT 1,
    ingested_at     TEXT    NOT NULL,
    PRIMARY KEY (source, symbol, timeframe, open_time_ms)
);

CREATE INDEX IF NOT EXISTS idx_market_ohlcv_date
    ON market_ohlcv (source, symbol, timeframe, date_utc);

CREATE TABLE IF NOT EXISTS realtime_price (
    source        TEXT    NOT NULL,
    symbol        TEXT    NOT NULL,
    event_time_ms INTEGER NOT NULL,
    price         REAL    NOT NULL,
    transport     TEXT    NOT NULL,
    received_at   TEXT    NOT NULL,
    PRIMARY KEY (source, symbol, event_time_ms)
);

CREATE INDEX IF NOT EXISTS idx_realtime_price_recent
    ON realtime_price (source, symbol, event_time_ms DESC);

-- -------------------------------------------------------------------- quality

CREATE TABLE IF NOT EXISTS data_quality_checks (
    check_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT    NOT NULL,
    checked_at   TEXT    NOT NULL,
    source       TEXT    NOT NULL,
    symbol       TEXT    NOT NULL,
    timeframe    TEXT    NOT NULL,
    check_name   TEXT    NOT NULL,
    severity     TEXT    NOT NULL CHECK (severity IN ('info', 'warning', 'error')),
    status       TEXT    NOT NULL CHECK (status IN ('pass', 'fail')),
    message      TEXT    NOT NULL,
    details      TEXT,
    rows_checked INTEGER NOT NULL DEFAULT 0,
    range_start  TEXT,
    range_end    TEXT
);

CREATE INDEX IF NOT EXISTS idx_data_quality_run
    ON data_quality_checks (run_id, checked_at);

-- ------------------------------------------------------------------- features

CREATE TABLE IF NOT EXISTS features (
    feature_version TEXT    NOT NULL,
    source          TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    timeframe       TEXT    NOT NULL,
    open_time_ms    INTEGER NOT NULL,
    date_utc        TEXT    NOT NULL,
    feature_name    TEXT    NOT NULL,
    value           REAL,
    computed_at     TEXT    NOT NULL,
    PRIMARY KEY (feature_version, source, symbol, timeframe, open_time_ms, feature_name)
);

CREATE INDEX IF NOT EXISTS idx_features_lookup
    ON features (feature_version, source, symbol, timeframe, date_utc);

-- --------------------------------------------------------------------- models

CREATE TABLE IF NOT EXISTS model_registry (
    model_id                TEXT PRIMARY KEY,
    model_version           TEXT NOT NULL UNIQUE,
    algorithm               TEXT NOT NULL,
    library_versions        TEXT,
    feature_version         TEXT NOT NULL,
    horizon_grid_version    TEXT NOT NULL,
    config_version          TEXT,
    code_commit             TEXT,
    training_window_strategy TEXT,
    training_start          TEXT,
    training_cutoff         TEXT NOT NULL,
    training_rows           INTEGER,
    validation_start        TEXT,
    validation_end          TEXT,
    test_start              TEXT,
    test_end                TEXT,
    hyperparameters         TEXT,
    random_seed             INTEGER,
    validation_metrics      TEXT,
    test_metrics            TEXT,
    status                  TEXT NOT NULL
        CHECK (status IN ('candidate', 'production', 'retired', 'rejected')),
    rejection_reason        TEXT,
    artifact_path           TEXT,
    created_at              TEXT NOT NULL,
    promoted_at             TEXT,
    retired_at              TEXT
);

CREATE INDEX IF NOT EXISTS idx_model_registry_status
    ON model_registry (status, created_at DESC);

CREATE TABLE IF NOT EXISTS model_runs (
    run_id         TEXT PRIMARY KEY,
    run_type       TEXT NOT NULL,
    model_id       TEXT REFERENCES model_registry (model_id),
    started_at     TEXT NOT NULL,
    finished_at    TEXT,
    status         TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    config_version TEXT,
    code_commit    TEXT,
    random_seed    INTEGER,
    dataset_cutoff TEXT,
    metrics        TEXT,
    notes          TEXT
);

-- ------------------------------------------------------------------ forecasts

CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id          TEXT PRIMARY KEY,
    run_id               TEXT,
    created_at           TEXT NOT NULL,
    source               TEXT NOT NULL,
    symbol               TEXT NOT NULL,
    timeframe            TEXT NOT NULL,
    forecast_origin_date TEXT NOT NULL,
    origin_open_time_ms  INTEGER NOT NULL,
    origin_close         REAL NOT NULL,
    current_price        REAL,
    model_id             TEXT REFERENCES model_registry (model_id),
    model_version        TEXT NOT NULL,
    feature_version      TEXT NOT NULL,
    horizon_grid_version TEXT NOT NULL,
    config_version       TEXT,
    code_commit          TEXT,
    UNIQUE (source, symbol, timeframe, forecast_origin_date, model_version)
);

CREATE INDEX IF NOT EXISTS idx_forecasts_origin
    ON forecasts (forecast_origin_date DESC);

-- `model_weight` / `blend_source` record what produced each point
-- (ARCHITECTURE.md section 2.4). The blend weights are frozen in config, so they
-- could be recomputed at read time -- but only under the config in force *now*,
-- which would silently relabel a forecast made under an earlier one. Provenance
-- that changes when you change the config is not provenance.
CREATE TABLE IF NOT EXISTS forecast_points (
    forecast_id          TEXT    NOT NULL
        REFERENCES forecasts (forecast_id) ON DELETE CASCADE,
    horizon_days         INTEGER NOT NULL,
    target_date          TEXT    NOT NULL,
    predicted_log_return REAL    NOT NULL,
    predicted_price      REAL    NOT NULL,
    direction_predicted  INTEGER,
    model_weight         REAL,
    blend_source         TEXT,
    PRIMARY KEY (forecast_id, horizon_days)
);

CREATE INDEX IF NOT EXISTS idx_forecast_points_target
    ON forecast_points (target_date);

CREATE TABLE IF NOT EXISTS forecast_quantiles (
    forecast_id          TEXT    NOT NULL
        REFERENCES forecasts (forecast_id) ON DELETE CASCADE,
    horizon_days         INTEGER NOT NULL,
    quantile_label       TEXT    NOT NULL,
    quantile             REAL    NOT NULL,
    predicted_log_return REAL    NOT NULL,
    predicted_price      REAL    NOT NULL,
    crossing_adjusted    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (forecast_id, horizon_days, quantile_label)
);

CREATE TABLE IF NOT EXISTS forecast_realizations (
    forecast_id         TEXT    NOT NULL
        REFERENCES forecasts (forecast_id) ON DELETE CASCADE,
    horizon_days        INTEGER NOT NULL,
    target_date         TEXT    NOT NULL,
    evaluation_status   TEXT    NOT NULL
        CHECK (evaluation_status IN ('pending', 'partially_evaluable', 'fully_evaluated')),
    actual_close        REAL,
    actual_log_return   REAL,
    absolute_error      REAL,
    percentage_error    REAL,
    log_return_error    REAL,
    direction_predicted INTEGER,
    direction_actual    INTEGER,
    direction_correct   INTEGER,
    in_interval_50      INTEGER,
    in_interval_80      INTEGER,
    in_interval_95      INTEGER,
    pinball_loss        REAL,
    regime              TEXT,
    updated_at          TEXT    NOT NULL,
    PRIMARY KEY (forecast_id, horizon_days)
);

CREATE INDEX IF NOT EXISTS idx_forecast_realizations_status
    ON forecast_realizations (evaluation_status, target_date);

-- ---------------------------------------------------------------- performance

-- Sentinel defaults ('all', -1) are used instead of NULL because SQLite treats
-- NULLs as distinct inside a UNIQUE constraint, which would break upserts.
CREATE TABLE IF NOT EXISTS performance_metrics (
    metric_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    computed_at   TEXT    NOT NULL,
    scope         TEXT    NOT NULL
        CHECK (scope IN ('validation', 'outer_test', 'production', 'baseline')),
    model_version TEXT    NOT NULL,
    run_id        TEXT    NOT NULL DEFAULT '',
    horizon_days  INTEGER NOT NULL DEFAULT -1,
    regime        TEXT    NOT NULL DEFAULT 'all',
    fold          TEXT    NOT NULL DEFAULT 'all',
    metric_name   TEXT    NOT NULL,
    metric_value  REAL,
    sample_size   INTEGER NOT NULL DEFAULT 0,
    period_start  TEXT,
    period_end    TEXT,
    UNIQUE (scope, model_version, run_id, horizon_days, regime, fold, metric_name)
);

CREATE INDEX IF NOT EXISTS idx_performance_lookup
    ON performance_metrics (scope, model_version, horizon_days, metric_name);
