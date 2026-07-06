Model O is a hybrid of Model R and the Model M shock-state setup.

Pipeline

1. Use Model R raw outputs as `model_a` / `p_normal`. Model R was sanity checked against Model A and only changes naming plus the default Monte Carlo path count from 500 to 20,000.
2. Use the full Model H fair-value raw outputs as `model_b` / `p_shock`, because Model M itself is a hard gate over that fair-value source.
3. Clone `Model_M/Additional_Data/all_test_predictions.csv` into `Model_O/Additional_Data/all_test_predictions.csv`.
4. Read `q_shock` from the cloned CSV using `scenario = drop_stale`, `model = logistic_regression`, and `target_minute` by default.
5. Write linear hybrid probabilities as `p_final = (1 - q_shock) * p_normal + q_shock * p_shock`.
6. Also write the lambda=2 exponential hybrid as a downstream strategy probability:
   - `w(q) = (exp(2 * q_shock) - 1) / (exp(2) - 1)`
   - `p_exponential_hybrid = (1 - w(q)) * p_normal + w(q) * p_shock`
   - The default Model O `*_price` output remains the linear hybrid for continuity.
7. Evaluate Model O with the same official Kalshi outcome, calibration, Brier, log-loss, sharpness, and time-bucket outputs used by the other models.
8. Compare Model O to Model K for profitability in both favorable directions:
   - BUY_YES when `p_model_o > p_model_k`
   - BUY_NO when `1 - p_model_o > 1 - p_model_k`
9. Run fee-aware profitability variants across linear Model O, exponential Model O lambda=2, Model A Normal, and Model B Shock:
   - gross-edge signal with fee-aware PnL
   - after-fee edge greater than 0
   - after-fee edge greater than 0.01
   - after-fee edge greater than 0.02
10. Run the same fee-aware variants in a separate maker-fee output folder when modeling maker-order execution.
11. Run trade diagnostics to store a fee-aware trade ledger, predicted-edge bucket calibration, segment summaries, and pairwise feature-combination matrices.
    - Current historical Model K price files only persist the configured `yes_mid` price, so the diagnostics ledger marks `market_price_is_executable = False` and keeps spread/liquidity fields as unavailable until bid/ask fields are persisted upstream.
12. Inside trade diagnostics, compute a causal HAR-RV filter:
    - Features are `rv_lag_1h`, `rv_mean_6h`, `rv_mean_24h`, and `rv_mean_72h`.
    - All HAR-RV features use shifted prior hourly realized volatility only.
    - The default activation threshold is `q75`: `har_rv_forecast >= historic_rv_q75`.
    - The diagnostics output also writes a q50-q95 threshold sweep in 5-point increments.
    - Post-hoc volatility labels remain attribution fields only and are not used for activation.
13. Review `LEAKAGE_AUDIT.md` before promoting Model O outputs into a production-facing backtest.

Scripts

`Model_O.py`

`Model_O_Eval.py`

`Model_O_Stats_Dashboard.py`

`Model_O_Profitability.py`

`Model_O_Fee_Aware_Profitability.py`

`Model_O_Trade_Diagnostics.py`

Default output directories

`Model_O/Model_O_Output_Raw_Data`

`Model_O/model_O_Evals_Outputs`

`Model_O/Model_O_Stats_Dashboard_Outputs`

`Model_O/Model_O_Profitability_Outputs`

`Model_O/Model_O_Fee_Aware_Profitability_Outputs`

`Model_O/Model_O_Maker_Fee_Profitability_Outputs`

`Model_O/Model_O_Trade_Diagnostics_Outputs`

Key HAR-RV diagnostics outputs

`Model_O/Model_O_Trade_Diagnostics_Outputs/har_rv_active_trade_ledger.csv`

`Model_O/Model_O_Trade_Diagnostics_Outputs/har_rv_filter_strategy_summary.csv`

`Model_O/Model_O_Trade_Diagnostics_Outputs/har_rv_filter_pnl_timeseries.csv`

`Model_O/Model_O_Trade_Diagnostics_Outputs/har_rv_hourly_forecasts.csv`

`Model_O/Model_O_Trade_Diagnostics_Outputs/har_rv_quantile_sweep_matrix.csv`
