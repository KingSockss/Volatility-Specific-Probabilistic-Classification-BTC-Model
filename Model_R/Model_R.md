Model R is a probabilistic BTC spot model that matches Model A's GARCH(1,1) + Student-t Monte Carlo structure, but raises the default simulation depth to `20,000` terminal paths per forecast timestamp.

Pipeline

1. Use the settlement CSV as the hourly Kalshi contract manifest.
2. Restrict generated events to the hourly Kalshi pricing files so the model stays on the same scored event universe as Model K.
3. Download matching Binance `1m` BTC spot data for the settlement window plus the live history buffer.
4. Work in log returns using minute-open prices.
5. Fit Student-t GARCH(1,1) parameters on the rolling history window.
6. Update conditional variance each minute and simulate the remaining path to settlement.
7. Use Monte Carlo terminal prices to estimate YES probabilities for `ATM`, `OTM+1`, and `OTM-1`.
8. Write one hourly CSV per event under `Model_R_Output_Raw_Data`.

Scripts

`Model_R.py`

`Model_R_Eval.py`

`Model_R_Trade_Filter.py`

Default output directory

`Model_R/Model_R_Output_Raw_Data`

Default evaluation directory

`Model_R/model_R_Evals_Outputs`

Default trade-filter directory

`Model_R/Model_R_Trade_Filter_Outputs`

Notes

- Model R intentionally keeps Model A's modeling assumptions except for the default `--mc-trials` value.
- Model R defaults to `20,000` Monte Carlo terminal paths per forecast timestamp.
- `Model_R_Trade_Filter.py` compares Model R scored probabilities against Model K probabilities on the overlapping scored-row universe, keeps rows where `p_model_r > p_model_k`, and reports buy-YES trade performance assuming one unit-payout contract is bought for every qualifying minute.
