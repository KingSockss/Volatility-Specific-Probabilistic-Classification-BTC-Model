Model H is a probabilistic BTC spot model that generates minutely YES-probability forecasts for the hourly Kalshi BTC contracts defined in `Data_Sourcing/Settlement_Outcomes/kalshi_btc_atm_settlements.csv`.

Pipeline

1. Use the settlement CSV as the contract manifest, but restrict the event universe to the hourly Kalshi pricing files so Model H does not generate past the Model K horizon.
2. Download matching Binance `1m` BTC spot data for the settlement window plus an extended history buffer that covers both the live state window and the regime-selection lookback.
3. Work in log returns using minute-open prices.
4. Classify each completed Binance hour using the same trailing real-time volatility rules already used in the volatility decomposition scripts: realized volatility is measured from 1-minute candles and compared with a trailing 72-hour rolling window.
5. For each refit, use a trailing historical window of completed hours and keep only the minutes that belong to the selected volatility regime, with `high_volatility_extreme` as the default training regime.
6. Cap the parameter-estimation and isotonic-calibration sample before the first Kalshi backtest forecast hour, so neither layer trains on the settlement/pricing range being scored.
7. Fit the Student-t GARCH(1,1) parameters by maximum likelihood on those regime-filtered return blocks, while still using a recent contiguous return window to initialize the live conditional variance state.
8. Train an isotonic regression layer on synthetic ATM, OTM+1, and OTM-1 historical contract examples drawn from the same capped regime-filtered minutes used for the GARCH-T MLE fit.
9. Update the conditional variance each minute, simulate the remaining path to settlement, and pass each raw Monte Carlo YES probability through the isotonic layer.
10. Write one hourly CSV per event under `Model_H_Output_Raw_Data`, preserving the scored `*_price` columns while also adding `*_raw_price` and `*_calibrated_price` columns for ATM, OTM+1, and OTM-1.
11. Write whole-sample and high-volatility-extreme isotonic diagnostics under `Model_H_Isotonic_Diagnostics_Outputs`.

Scripts

`Model_H.py`

`Model_H_Isotonic_Diagnostics_Dashboard.py`

Default output directory

`Model_H/Model_H_Output_Raw_Data`

Default isotonic diagnostics directory

`Model_H/Model_H_Isotonic_Diagnostics_Outputs`

Notes

- The settlement CSV provides the contract tickers and strikes, while `Data_Sourcing/Kalshi_Pricing_Fetch/hourly_events_price_data` defines which hourly events are allowed to generate.
- The Binance minute download path now follows the same plain `requests` pagination pattern used in `Data_Sourcing/Kalshi_Pricing_Fetch/Kalshi_Contract_Price_Fetch.py`.
- The downloader uses the primary Binance spot API at `api.binance.com`.
- The default implementation re-fits parameters and the isotonic layer every 60 forecast minutes, uses a trailing 90-day regime-selection window, trains on the `high_volatility_extreme` segment defined by the existing 72-hour rolling percentile rules, and simulates `10,000` Monte Carlo terminal paths per forecast timestamp.
- Model H separates parameter estimation from state initialization: capped regime-filtered returns outside the Kalshi backtest range drive the MLE and isotonic fits, while the contiguous `--history-minutes` window drives the live conditional-variance seed used for simulation.
- Isotonic training defaults to `1,500` historical minutes per refit, `1,000` Monte Carlo paths per synthetic calibration example, and Kalshi-style synthetic strikes spaced by `250` with a `-0.01` offset.
- `forecast_probability_diagnostics.csv` contains long-form pre/post isotonic probabilities for the whole scored forecast set; `forecast_probability_diagnostics_high_volatility_extreme.csv` is the same schema filtered to the high-volatility-extreme forecast hours.
- `isotonic_training_diagnostics.csv` contains `raw_probability`, `calibrated_probability`, `outcome`, `contract_label`, `minute_price_index`, and `refit_id` for the synthetic calibration examples; `isotonic_training_diagnostics_high_volatility_extreme.csv` is the high-volatility-extreme subset.
- `isotonic_curve_points.csv` and `isotonic_refit_summary.csv` describe the learned isotonic mapping and fit sample for each refit.
- `Model_H_Isotonic_Diagnostics_Dashboard.py` builds the overall dashboard at `Model_H_Isotonic_Diagnostics_Outputs/index.html`, a dedicated high-volatility-extreme dashboard at `Model_H_Isotonic_Diagnostics_Outputs/high_volatility_extreme/index.html`, and whole-set plus high-volatility-extreme CSV summaries comparing raw versus calibrated probabilities, Brier/log-loss deltas, contract buckets, minute buckets, volatility buckets, training-vs-backtest calibration, and the latest isotonic curve.
- Increase `--mc-trials`, increase `--isotonic-mc-trials`, or reduce `--refit-every-minutes` if you want a heavier run.
