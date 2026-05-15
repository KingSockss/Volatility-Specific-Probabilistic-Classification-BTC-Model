Model E is a probabilistic BTC spot model that generates minutely YES-probability forecasts for the hourly Kalshi BTC contracts defined in `Data_Sourcing/Settlement_Outcomes/kalshi_btc_atm_settlements.csv`.

**NOTE** Model E is the patch where we added isotonic regression layer.

Pipeline

1. Use the settlement CSV as the contract manifest, but restrict the event universe to the hourly Kalshi pricing files so Model E does not generate past the Model K horizon.
2. Download matching Binance `1m` BTC spot data for the settlement window plus an extended history buffer that covers both the live state window and the regime-selection lookback.
3. Work in log returns using minute-open prices.
4. Classify each completed Binance hour using the same trailing real-time volatility rules already used in the volatility decomposition scripts: realized volatility is measured from 1-minute candles and compared with a trailing 72-hour rolling window.
5. For each refit, use a trailing historical window of completed hours and keep only the minutes that belong to the selected volatility regime, with `high_volatility` as the default training regime.
6. Cap the parameter-estimation and isotonic-calibration sample before the first Kalshi backtest forecast hour, so neither layer trains on the settlement/pricing range being scored.
7. Fit the Student-t GARCH(1,1) parameters by maximum likelihood on those regime-filtered return blocks, while still using a recent contiguous return window to initialize the live conditional variance state.
8. Train an isotonic regression layer on synthetic ATM, OTM+1, and OTM-1 historical contract examples drawn from the same capped regime-filtered minutes used for the GARCH-T MLE fit.
9. Update the conditional variance each minute, simulate the remaining path to settlement, and pass each raw Monte Carlo YES probability through the isotonic layer.
10. Write one hourly CSV per event under `Model_E_Output_Raw_Data` using the same raw layout as `Data_Sourcing/Kalshi_Pricing_Fetch/hourly_events_price_data`.

Script

`Model_E.py`

Default output directory

`Model_E/Model_E_Output_Raw_Data`

Notes

- The settlement CSV provides the contract tickers and strikes, while `Data_Sourcing/Kalshi_Pricing_Fetch/hourly_events_price_data` defines which hourly events are allowed to generate.
- The Binance minute download path now follows the same plain `requests` pagination pattern used in `Data_Sourcing/Kalshi_Pricing_Fetch/Kalshi_Contract_Price_Fetch.py`.
- The downloader uses the primary Binance spot API at `api.binance.com`.
- The default implementation re-fits parameters and the isotonic layer every 60 forecast minutes, uses a trailing 90-day regime-selection window, trains on the `high_volatility` segment defined by the existing 72-hour rolling percentile rules, and simulates `10,000` Monte Carlo terminal paths per forecast timestamp.
- Model E separates parameter estimation from state initialization: capped regime-filtered returns outside the Kalshi backtest range drive the MLE and isotonic fits, while the contiguous `--history-minutes` window drives the live conditional-variance seed used for simulation.
- Isotonic training defaults to `1,500` historical minutes per refit, `1,000` Monte Carlo paths per synthetic calibration example, and Kalshi-style synthetic strikes spaced by `250` with a `-0.01` offset.
- Increase `--mc-trials`, increase `--isotonic-mc-trials`, or reduce `--refit-every-minutes` if you want a heavier run.
