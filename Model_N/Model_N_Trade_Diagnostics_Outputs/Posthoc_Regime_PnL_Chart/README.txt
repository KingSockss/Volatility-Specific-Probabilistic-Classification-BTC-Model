Model N Post-Hoc Regime BTC/PnL Chart

Strategy:
- model_n_exp_lambda_2: Model N Exp Hybrid lambda=2

Inputs:
- BTC price: Binance 1m close from Model_K_Volatility_Decomposition_RT_outputs/binance_1m_klines.csv
- PnL: after-fee Model N diagnostics pnl_timeseries.csv
- Regimes: post-hoc primary_volatility_band from hourly_market_volatility_segments.csv

Important:
- These volatility regimes are post-hoc labels and are not causal live trading signals.
- The chart is for visual diagnostics only.

Outputs:
- posthoc_regime_btc_pnl_dashboard.html
- posthoc_regime_btc_pnl_chart_timeseries.csv
- posthoc_regime_segments.csv
- posthoc_regime_pnl_summary.csv
