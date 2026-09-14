# DATA_SPEC.md

## 1. Canonical market data

### Primary

- Exchange: Binance Spot
- Symbol: `BTCUSDT`
- Base interval: `1d`
- Timezone: UTC
- Primary fields: open, high, low, close, volume, quote volume, trade count, taker buy volume

Binance Spot REST Kline endpoint: `GET /api/v3/klines`. Klines are identified by open time, and the API supports `1d` among its supported intervals.

### Realtime

- Binance Spot WebSocket market stream
- Current price only for UI freshness
- Realtime ticks must not silently overwrite historical daily close used by the model

### Secondary

- Coinbase BTC-USD
- Purpose: cross-exchange data sanity check and optional robustness analysis
- Do not merge Coinbase OHLCV into canonical Binance training series by default

## 2. Time semantics

- All internal timestamps: UTC
- Daily candle boundary: Binance UTC daily candle by default
- Forecast origin: last fully closed daily candle
- `current_price` can be realtime and therefore differ from forecast origin price

## 3. Historical ingestion

The ingestion job must support:

- initial full backfill
- incremental update
- retry with exponential backoff
- deduplication by `(source, symbol, timeframe, open_time)`
- idempotent upsert

## 4. Realtime ingestion

Realtime process:

```text
WebSocket tick
 -> parse
 -> validate
 -> latest price cache/store
 -> Streamlit reads latest value
```

Failure handling:

- reconnect after disconnect
- heartbeat / connection health monitoring
- fallback to REST current price if WebSocket is temporarily unavailable

### 4.1 Implementation

| Concern | Decision |
| --- | --- |
| stream | `@miniTicker` (one summary per second). `@trade` is parsed too, but on BTCUSDT it emits thousands of messages per second - far more than a price display needs. |
| write side | `src/data/binance_stream.py` (async) |
| read side | `src/data/realtime.py` (sync, websocket-free so Streamlit can import it) |
| job | `python -m jobs.stream_realtime_price` |

**Throttled persistence.** The newest tick is always held in memory, but
SQLite is written at most once per `realtime.persist_interval_seconds`
(default 2s). The store is the only channel between the stream process and
Streamlit, so that interval - not the stream rate - is the real UI latency
floor. `realtime.retention_hours` (default 48) bounds table growth; at one row
every two seconds the table would otherwise grow by ~43k rows per day.

**Heartbeat by silence.** A TCP connection can stay open while the feed is
dead, so a read that returns nothing for `realtime.heartbeat_timeout_seconds`
tears the connection down and rebuilds it rather than waiting forever.
Reconnects use exponential backoff with jitter, capped at
`realtime.reconnect_max_seconds`.

**Plausibility gate.** A tick more than
`realtime.max_deviation_from_last_close_pct` (default 50%) away from the last
closed daily close is rejected and logged at ERROR level. A corrupted feed is
far more likely than such a move - the worst day on record, 2020-03-12, moved
39.6%. When the gate fires the price simply goes stale, which the dashboard
shows explicitly, so the failure is visible rather than silently wrong.

### 4.2 Current-price resolution order

`src.data.realtime.get_current_price()` resolves in this order:

1. newest stored tick, if younger than `realtime.max_price_age_seconds`
2. Binance REST ticker, when `realtime.rest_fallback` is enabled (the result is
   persisted with `transport = 'rest'`)
3. the stale tick, returned with `is_stale = True`

A merely stale price is returned rather than raised, because a last-known value
labelled as old is more useful on a dashboard than a blank. Only a total outage
with no stored history raises `RealtimePriceError`.

### 4.3 Isolation from model data

The realtime path writes to `realtime_price` and to nothing else. It never
writes `market_ohlcv`, so it cannot alter a closed candle, a feature or a
forecast anchor. `current_price` and the forecast anchor (`origin_close`) are
expected to differ; section 2 already states this, and the dashboard shows both.
This isolation is asserted by `tests/test_realtime.py::TestModelDataIsolation`.

## 5. Data leakage rules

- Never use today's incomplete daily candle as a closed-day feature.
- Do not forward-fill future values.
- Do not use revised external data unless point-in-time versioning is available.
- Features derived from a future target date are prohibited from the feature matrix.

## 6. Feature time alignment

For forecast origin `t`, every feature must be computable from data timestamp `<= t`.

Example:

```text
rolling_30d_mean[t] = mean(close[t-29:t])
```

not:

```text
mean(close[t-29:t+1])
```

## 7. Data quality rules

Fail the daily pipeline if any of the following is true:

- duplicate primary key
- non-monotonic candle ordering
- missing close/open/high/low
- invalid OHLC relationship
- negative volume
- excessive unexpected missing-data ratio

Warnings may be emitted for large price jumps, but the pipeline must not blindly delete legitimate crash/pump events.

## 8. Physical storage schema

Logical tables from `ARCHITECTURE.md` section 3 map onto the SQLite schema in
`src/storage/schema.sql` as follows. Two logical tables are normalised; both
deviations are deliberate.

| Logical table | Physical table(s) | Note |
| --- | --- | --- |
| market_ohlcv | `market_ohlcv` | PK `(source, symbol, timeframe, open_time_ms)` |
| realtime_price | `realtime_price` | UI freshness only, never a model input |
| features | `features` | **long format**: one row per (version, day, feature) |
| forecasts | `forecasts`, `forecast_points`, `forecast_quantiles` | **normalised** |
| forecast_realizations | `forecast_realizations` | |
| model_runs | `model_runs` | |
| model_registry | `model_registry` | |
| performance_metrics | `performance_metrics` | |
| data_quality_checks | `data_quality_checks` | |

**Why `features` is long-format.** A new `feature_version` can be written
alongside the previous one without a schema migration, and a feature that is not
yet computable stores as SQL NULL rather than disappearing from the row.

**Why `forecasts` is split three ways.** The quantile list lives in
`config.yaml`. Fixed columns (`q02_5`, `q10`, ...) would hard-code it into the
schema, so quantiles are rows in `forecast_quantiles` keyed by
`quantile_label`, and per-horizon median values live in `forecast_points`.

Column conventions:

- `*_ms` columns are epoch milliseconds (UTC).
- `*_date` and `*_at` columns are ISO-8601 UTC strings.
- `performance_metrics` uses the sentinel values `'all'` and `-1` instead of
  NULL in its uniqueness key, because SQLite treats NULLs as distinct inside a
  UNIQUE constraint and upserts would silently duplicate.

## 9. Configuration sections owned by the data layer

`config.yaml` carries these data-layer sections; code must not hard-code any
value that appears here.

| Section | Purpose |
| --- | --- |
| `paths` | data / logs / reports / model artefact directories |
| `logging` | level, log file, console toggle |
| `ingestion.binance` | base URL, endpoints, page size, history start, retry/backoff, rate limit |
| `ingestion.coinbase` | base URL, endpoint, granularity, page size, retry/backoff |
| `data_quality` | thresholds for every check in section 7 |
| `features.params` | window lengths for each feature group |
| `features.regime` | regime labelling thresholds |

## 10. Data quality severities

Section 7 lists what must fail the pipeline. The implementation records every
check into `data_quality_checks` with an explicit severity:

- `error` + `fail` -> **blocking**. The daily pipeline stops and no forecast is
  generated (`OPERATING_SPEC.md` section 9). Jobs exit with code 1.
- `warning` + `fail` -> recorded and surfaced, never blocking. Large price
  jumps and cross-exchange divergence live here, so a legitimate crash or pump
  candle is reported rather than deleted.
- `info` + `pass` -> the check ran and passed.

Cross-exchange disagreement is deliberately non-blocking: Binance BTCUSDT and
Coinbase BTC-USD are different instruments, so a small spread is normal and is
not one of the failure conditions listed in section 7.
