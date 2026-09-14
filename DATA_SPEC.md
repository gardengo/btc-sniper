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
