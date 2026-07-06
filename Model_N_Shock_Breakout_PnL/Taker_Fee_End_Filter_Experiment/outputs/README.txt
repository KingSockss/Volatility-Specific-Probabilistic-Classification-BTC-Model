Model N Taker-Fee End Filter Experiment

Purpose:
- Test whether trading only the lower-fee ends remains profitable after Kalshi taker fees.
- Strategies are limited to Model A Normal and Model N exponential hybrid lambda=2.

Signal rule:
- First apply the scenario's Model K/Kalshi YES probability filter.
- BUY_YES when p_strategy - p_model_k > 0.0.
- BUY_NO when p_model_k - p_strategy > 0.0.
- This experiment does not require after-fee edge to be positive for entry; it measures the after-fee PnL of the filtered trade set.

Taker fee model:
- Entry fee per one $1 payout contract = fee_rate * selected_side_price * (1 - selected_side_price)
- fee_rate = 0.07
- Cent round-up enabled = True
- BUY_YES selected_side_price = p_model_k.
- BUY_NO selected_side_price = 1 - p_model_k.
- No exit or settlement fees are modeled.

Scenarios:
- baseline_all
- p_kalshi_lt_0_25
- p_kalshi_gt_0_75
- p_kalshi_outside_0_26_0_75
- p_kalshi_lt_0_10
- p_kalshi_gt_0_90
- p_kalshi_outside_0_10_0_90

Outputs:
- taker_fee_end_filter_strategy_summary.csv
- taker_fee_end_filter_strategy_side_summary.csv
- taker_fee_end_filter_pnl_timeseries.csv
- taker_fee_end_filter_trade_minutes.csv
- taker_fee_end_filter_dashboard.html
- one folder per scenario with per-scenario CSVs and dashboard.html
