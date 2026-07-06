Model B Shock Breakout PnL Experiment

This folder is standalone and can be removed without changing the main Model N scripts or outputs.

Rules:
- q_shock > 0.05
- q_shock > 0.10
- q_shock > 0.15
- q_shock > 0.20
- q_shock > 0.25
- Use p_shock as the strategy probability
- Compare against Model K as the entry price
- BUY_YES when p_shock - p_model_k > 0.0
- BUY_NO when (1 - p_shock) - (1 - p_model_k) > 0.0

Root comparison outputs:
- breakout_shock_threshold_comparison.csv
- breakout_shock_threshold_segment_summary.csv
- breakout_shock_threshold_pnl_timeseries.csv
- breakout_shock_threshold_trade_minutes.csv
- breakout_shock_threshold_comparison_dashboard.html

Per-threshold output folders:
- q_gt_0_05/
- q_gt_0_10/
- q_gt_0_15/
- q_gt_0_20/
- q_gt_0_25/

Compatibility outputs for q_shock > 0.20 are still written in this root outputs folder.
