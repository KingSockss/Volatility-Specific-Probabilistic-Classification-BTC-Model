Model A is a probabilistic BTC spot model that generates minutely YES-probability forecasts for the hourly Kalshi BTC contracts defined in `Data_Sourcing/Settlement_Outcomes/kalshi_btc_atm_settlements.csv`.

Pipeline

1. Use the settlement CSV as the contract manifest, but restrict the event universe to the hourly Kalshi pricing files so Model A does not generate past the Model K horizon.
2. Download matching Binance `1m` BTC spot data for the settlement window plus an extra history buffer.
3. Work in log returns using minute-open prices.
4. Fit a Student-t GARCH(1,1) return process by maximum likelihood on a rolling history window.
5. Update the conditional variance each minute and simulate the remaining path to settlement.
6. Use Monte Carlo terminal prices to estimate YES probabilities for `ATM`, `OTM+1`, and `OTM-1`.
7. Write one hourly CSV per event under `Model_A_Output_Raw_Data` using the same raw layout as `Data_Sourcing/Kalshi_Pricing_Fetch/hourly_events_price_data`.

Script

`Model_A.py`

Default output directory

`Model_A/Model_A_Output_Raw_Data`

Notes

- The settlement CSV provides the contract tickers and strikes, while `Data_Sourcing/Kalshi_Pricing_Fetch/hourly_events_price_data` defines which hourly events are allowed to generate.
- The Binance minute download path now follows the same plain `requests` pagination pattern used in `Data_Sourcing/Kalshi_Pricing_Fetch/Kalshi_Contract_Price_Fetch.py`.
- The downloader uses the primary Binance spot API at `api.binance.com`.
- The default implementation re-fits parameters every 60 forecast minutes and updates volatility recursively between refits.
- Increase `--mc-trials` or reduce `--refit-every-minutes` if you want a heavier run.
