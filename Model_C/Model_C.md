Model C is a probabilistic BTC spot model that generates minutely YES-probability forecasts for the hourly Kalshi BTC contracts defined in `Data_Sourcing/Settlement_Outcomes/kalshi_btc_atm_settlements.csv`.

Pipeline

1. Use the settlement CSV as the contract manifest, but restrict the event universe to the hourly Kalshi pricing files so Model C does not generate past the Model K horizon.
2. Download matching Binance `1m` BTC spot data for the settlement window plus an extended history buffer that covers both the live state window and the regime-selection lookback.
3. Work in log returns using minute-open prices.
4. Classify each completed Binance hour using the same trailing real-time volatility rules already used in the volatility decomposition scripts: realized volatility is measured from 1-minute candles and compared with a trailing 72-hour rolling window.
5. For each refit, use a trailing historical window of completed hours and keep only the minutes that belong to the selected volatility regime, with `high_volatility` as the default training regime.
6. Fit the Student-t GARCH(1,1) parameters by maximum likelihood on those regime-filtered return blocks, while still using a recent contiguous return window to initialize the live conditional variance state.
7. Update the conditional variance each minute and simulate the remaining path to settlement.
8. Use Monte Carlo terminal prices to estimate YES probabilities for `ATM`, `OTM+1`, and `OTM-1`.
9. Write one hourly CSV per event under `Model_C_Output_Raw_Data` using the same raw layout as `Data_Sourcing/Kalshi_Pricing_Fetch/hourly_events_price_data`.

Script

`Model_C.py`

Default output directory

`Model_C/Model_C_Output_Raw_Data`

Notes

- The settlement CSV provides the contract tickers and strikes, while `Data_Sourcing/Kalshi_Pricing_Fetch/hourly_events_price_data` defines which hourly events are allowed to generate.
- The Binance minute download path now follows the same plain `requests` pagination pattern used in `Data_Sourcing/Kalshi_Pricing_Fetch/Kalshi_Contract_Price_Fetch.py`.
- The downloader uses the primary Binance spot API at `api.binance.com`.
- The default implementation re-fits parameters every 60 forecast minutes, uses a trailing 90-day regime-selection window, trains on the `high_volatility` segment defined by the existing 72-hour rolling percentile rules, and simulates `10,000` Monte Carlo terminal paths per forecast timestamp.
- Model C separates parameter estimation from state initialization: regime-filtered returns drive the MLE fit, while the contiguous `--history-minutes` window drives the live conditional-variance seed used for simulation.
- Increase `--mc-trials` or reduce `--refit-every-minutes` if you want a heavier run.
