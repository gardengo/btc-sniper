# TASKS.md

## Phase 0. Project bootstrap

- [ ] Python package structure
- [ ] dependency management
- [ ] config loading
- [ ] logging
- [ ] SQLite initialization
- [ ] `.env` support for optional credentials

## Phase 1. Data ingestion

- [ ] Binance historical 1D BTCUSDT loader
- [ ] incremental upsert
- [ ] realtime Binance WebSocket price consumer
- [ ] REST fallback current-price loader
- [ ] Coinbase BTC-USD cross-check loader
- [ ] data quality validation

## Phase 2. Feature pipeline

- [ ] return features
- [ ] trend features
- [ ] volatility features
- [ ] volume features
- [ ] regime labels
- [ ] leakage unit tests
- [ ] feature versioning

## Phase 3. Baselines

- [ ] no-change baseline
- [ ] drift baseline
- [ ] rolling-return baseline
- [ ] baseline evaluator

## Phase 4. ML model

- [ ] horizon grid generation
- [ ] target generation
- [ ] train/validation split engine
- [ ] expanding/rolling window engine
- [ ] LightGBM quantile model
- [ ] XGBoost alternative where supported
- [ ] model serialization
- [ ] model registry

## Phase 5. Validation

- [ ] Walk-Forward CV
- [ ] outer/final test isolation
- [ ] regime tagging
- [ ] cycle coverage report
- [ ] point metrics
- [ ] directional metrics
- [ ] quantile metrics
- [ ] interval coverage
- [ ] training window comparison

## Phase 6. Forecast generation

- [ ] daily production forecast job
- [ ] quantile predictions
- [ ] price conversion from log return
- [ ] PCHIP visualization interpolation
- [ ] forecast persistence

## Phase 7. Realization and monitoring

- [ ] target-date resolver
- [ ] actual price join
- [ ] forecast realization table
- [ ] production metric aggregation
- [ ] performance drift detection

## Phase 8. Weekly model review

- [ ] weekly review job
- [ ] retraining trigger rules
- [ ] candidate model training
- [ ] incumbent vs candidate comparison
- [ ] promotion gate
- [ ] rejection logging

## Phase 9. Streamlit

- [ ] Dashboard
- [ ] Prediction Log
- [ ] Model Performance
- [ ] historical/future combined chart
- [ ] interval bands
- [ ] current model summary
- [ ] forecast revision chart

## Phase 10. Testing

- [ ] data tests
- [ ] leakage tests
- [ ] target alignment tests
- [ ] forecast horizon tests
- [ ] interval ordering tests
- [ ] model reproducibility tests
- [ ] promotion gate tests
- [ ] Streamlit smoke test

## Phase 11. Containerization

- [ ] Dockerfile
- [ ] docker-compose optional
- [ ] non-root runtime
- [ ] persistent DB volume
- [ ] healthcheck
- [ ] scheduled job execution strategy

## Definition of Done

- full historical backfill works
- daily forecast can be reproduced from a fixed cutoff
- no future leakage tests fail
- independent outer test is protected
- production model is versioned
- weekly candidate cannot overwrite production without promotion gate
- Streamlit shows historical actual + future forecast interval graph
