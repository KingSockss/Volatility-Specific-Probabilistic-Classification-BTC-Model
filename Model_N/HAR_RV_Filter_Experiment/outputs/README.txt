HAR-RV q75 Filter Experiment

Purpose:
- Test a naive causal HAR-RV high-volatility activation filter on Model N trade diagnostics.

HAR-RV setup:
- Target = next/current forecast-hour realized volatility from the hourly volatility table.
- Features use only prior realized volatility:
  - rv_lag_1h
  - rv_mean_6h
  - rv_mean_24h
  - rv_mean_72h
- Expanding OLS is refit each hour using prior complete observations only.
- min_train_hours = 168

Activation rule:
- Compute historical q75 from prior realized volatility only.
- Activate trading when har_rv_forecast >= historic_rv_q75.
- min_q75_history = 72

Primary dashboard strategy:
- model_n_exp_lambda_2

Outputs:
- har_rv_hourly_forecasts.csv
- har_rv_filtered_trade_ledger.csv
- har_rv_filtered_trade_ledger_active_only.csv
- har_rv_strategy_summary.csv
- har_rv_segment_summary.csv
- har_rv_pnl_timeseries.csv
- har_rv_active_vs_posthoc_high_vol.csv
- har_rv_quantile_sweep_strategy_summary.csv
- har_rv_quantile_sweep_matrix.csv
- har_rv_filter_dashboard.html
