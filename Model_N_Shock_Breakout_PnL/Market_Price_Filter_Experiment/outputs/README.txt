Model N Market Price Filter Experiment

This folder is standalone and can be removed without changing the main Model N scripts or outputs.

Market probability:
- p(kalshi) is modeled as Model K's YES probability column, p_model_k.

Strategies:
- Model N Linear: p_model_n
- Model N Exp lambda=2: p_exponential_hybrid
- Model A Normal: p_normal
- Model B Shock: p_shock

Trade rule:
- BUY_YES when p_strategy - p_model_k > 0.0
- BUY_NO when (1 - p_strategy) - (1 - p_model_k) > 0.0
- Fees are not applied in this experiment.

Scenarios:
- Baseline: all Model K overlap rows
- p_model_k < 0.25
- p_model_k > 0.75
- p_model_k < 0.25 OR > 0.75
- p_model_k < 0.15
- p_model_k > 0.85
- p_model_k < 0.15 OR > 0.85

Outputs:
- market_price_filter_summary.csv
- market_price_filter_side_summary.csv
- market_price_filter_pnl_timeseries.csv
- market_price_filter_trade_minutes.csv
- market_price_filter_dashboard.html
- scenario folders with per-scenario summaries, PnL time series, trade minutes, and dashboards.
