Model O Trade Diagnostics

Signal rule:
- Trade all favorable gross-edge rows.
- BUY_YES when selected strategy YES probability minus selected YES entry price exceeds 0.0.
- BUY_NO when selected strategy NO probability minus selected NO entry price exceeds 0.0.

Fee model:
- Taker fee = fee_rate * selected_side_price * (1 - selected_side_price)
- fee_rate = 0.07
- Cent round-up enabled = True

Execution price note:
- If selected-side ask columns are available, the script uses those as executable entry prices.
- The current Model_K/Model_O evaluated raw files only contain a single p_kalshi price from the historical price file.
- Those source files were produced with KALSHI_PRICE_FIELD = yes_mid, so the dashboard marks market_price_is_executable = False and uses the stored price as a proxy.
- Spread/liquidity fields are retained but marked unavailable unless future raw files include bid/ask/liquidity columns.

Primary dashboard strategy:
- model_o_exp_lambda_2

Combination matrix:
- Pairwise feature combinations are ranked with minimum trade count = 100.
- This avoids sparse multi-way segment overfitting while still showing which feature combinations have persistent realized net PnL.

HAR-RV filter:
- Forecasts next/current hour realized volatility from prior hourly realized volatility only.
- Features: rv_lag_1h, rv_mean_6h, rv_mean_24h, rv_mean_72h.
- Activation uses har_rv_forecast >= selected historical realized-volatility quantile.
- Threshold quantiles are computed from prior hours only.

Outputs:
- trade_diagnostics_ledger.csv
- har_rv_active_trade_ledger.csv
- trade_field_matrix.csv
- strategy_summary.csv
- har_rv_filter_strategy_summary.csv
- pnl_timeseries.csv
- har_rv_filter_pnl_timeseries.csv
- segment_summary.csv
- predicted_net_edge_bucket_summary.csv
- feature_pair_matrix.csv
- top_feature_combinations.csv
- bottom_feature_combinations.csv
- har_rv_hourly_forecasts.csv
- har_rv_quantile_sweep_strategy_summary.csv
- har_rv_quantile_sweep_matrix.csv
- diagnostics_dashboard.html
