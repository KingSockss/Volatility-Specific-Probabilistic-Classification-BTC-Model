Exponential Hybrid Lambda Sweep

This folder is standalone and can be removed without changing the main Model N scripts or outputs.

Formula:
- w(q) = (exp(lambda * q) - 1) / (exp(lambda) - 1)
- p_final = (1 - w(q)) * p_normal + w(q) * p_shock

Lambda values:
- lambda = 1
- lambda = 1.5
- lambda = 2
- lambda = 2.5
- lambda = 3

PnL rule:
- Compare the strategy probability against Model K as the entry price
- BUY_YES when p_strategy - p_model_k > 0.0
- BUY_NO when (1 - p_strategy) - (1 - p_model_k) > 0.0

Root sweep outputs:
- exponential_hybrid_lambda_comparison.csv
- exponential_hybrid_lambda_eval_summary.csv
- exponential_hybrid_lambda_strategy_summary.csv
- exponential_hybrid_lambda_pnl_timeseries.csv
- exponential_hybrid_lambda_comparison_dashboard.html

Per-lambda folders contain dashboard/summary/time-series files for each lambda.
The lambda=2 compatibility outputs are also written in this root outputs folder.
