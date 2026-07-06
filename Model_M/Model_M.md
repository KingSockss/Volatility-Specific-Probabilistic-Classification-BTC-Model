Model M is the Model H GARCH-t Monte Carlo plus isotonic calibration pipeline evaluated only on minutes admitted by the Model M logistic-regression gate.

Pipeline

1. Start with the existing Model H raw forecast outputs, preserving the Student-t GARCH(1,1) Monte Carlo probabilities, the high-volatility-extreme isotonic layer, contract tickers, strikes, refit ids, and raw/calibrated probability columns.
2. Read `Model_M/Additional_Data/all_test_predictions.csv`.
3. Keep only gate rows with `scenario = drop_stale` and `model = logistic_regression`.
4. Use `target_minute` as the gated trading candle timestamp. This is the candle predicted by the logistic row.
5. Build three threshold-specific raw forecast folders using `y_prob > 0.10`, `y_prob > 0.15`, and `y_prob > 0.20`.
6. Evaluate each threshold folder with the cloned Model H scoring, time-bucket, isotonic diagnostic, stats dashboard, volatility decomposition, and volatility comparison dashboard scripts.

Default gate outputs

`Model_M/Model_M_Output_Raw_Data/p_gt_0_10`

`Model_M/Model_M_Output_Raw_Data/p_gt_0_15`

`Model_M/Model_M_Output_Raw_Data/p_gt_0_20`

Gate metadata

`Model_M/Model_M_Output_Raw_Data/gate_output_summary.csv`

`Model_M/Model_M_Output_Raw_Data/_gate_metadata/*_gate_selected_minutes.csv`

`Model_M/Model_M_Output_Raw_Data/_gate_metadata/*_gate_unmatched_minutes.csv`

Notes

- The generated threshold raw folders intentionally contain only Model H-shaped per-event forecast CSVs so `Model_M_Eval.py` can scan them the same way `Model_H_Eval.py` scans `Model_H_Output_Raw_Data`.
- Gate rows beyond the currently available Model H/Kalshi raw forecast universe are retained in the gate metadata and counted as unmatched gate minutes.
- The scoring outputs are threshold-scoped under `model_M_Evals_Outputs/<gate>`, `Model_M_Isotonic_Diagnostics_Outputs/<gate>`, `Model_M_Stats_Dashboard_Outputs/<gate>`, `Model_M_Volatility_Decomposition_RT_outputs/<gate>`, and `Model_M_Volatility_Dashboards_outputs/<gate>`.
