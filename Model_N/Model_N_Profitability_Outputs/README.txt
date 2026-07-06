Model N Profitability Analysis

Signal rules:
BUY_YES when p_model_n - p_model_k > 0.0.
BUY_NO when (1 - p_model_n) - (1 - p_model_k) > 0.0.

Trade assumption:
Each signal buys one $1 payout contract in the favorable direction. Model K probability is treated as the entry price.

Metrics:
- expected_value_per_contract = selected-side Model N probability - selected-side Model K price
- realized_pnl_per_contract = selected-side official payout - selected-side Model K price
- win = 1 when the selected side pays out, else 0
- total_realized_pnl = sum realized_pnl_per_contract

Additional strategy outputs:
- all_strategy_trade_minutes.csv includes linear Model N, exponential Model N lambda=2, model_a (p_normal), and model_b (p_shock).
- strategy_pnl_timeseries.csv aggregates realized PnL by forecast timestamp and strategy.
- pnl_dashboard.html overlays cumulative realized PnL for the strategy variants.
