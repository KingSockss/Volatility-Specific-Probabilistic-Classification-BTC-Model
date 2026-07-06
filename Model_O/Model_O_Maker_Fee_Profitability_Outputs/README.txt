Model O Maker Fee Profitability

Fee model:
- Label = Maker Fee
- Entry fee per one $1 payout contract = fee_rate * selected_price * (1 - selected_price)
- fee_rate = 0.0175
- Cent round-up enabled = True
- No exit or settlement fee is modeled.

Scenarios:
- gross_edge_fee_pnl: old signal rule, model probability greater than market selected-side price; PnL subtracts fees.
- after_fee_edge_0: model probability greater than market selected-side price plus fee.
- after_fee_edge_001: after-fee edge must exceed 0.01.
- after_fee_edge_002: after-fee edge must exceed 0.02.

Strategies:
- Linear Model O
- Exponential Model O lambda=2
- Model A Normal
- Model B Shock

Outputs:
- fee_aware_strategy_summary.csv
- fee_aware_strategy_side_summary.csv
- fee_aware_pnl_timeseries.csv
- fee_aware_dashboard.html
- scenario folders with per-scenario trade minutes, summaries, time series, and dashboards.
