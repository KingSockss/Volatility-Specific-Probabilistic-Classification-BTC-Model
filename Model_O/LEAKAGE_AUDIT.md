Model O Leakage Audit

Scope

- Audited the Model O scripts created from Model N:
  - `Model_O.py`
  - `Model_O_Eval.py`
  - `Model_O_Profitability.py`
  - `Model_O_Fee_Aware_Profitability.py`
  - `Model_O_Stats_Dashboard.py`
  - `Model_O_Trade_Diagnostics.py`
- Focused on whether trade signals or HAR-RV regime activation use information unavailable at signal time.

Summary

- No direct outcome leakage was found in the Model O trade signal rule. Signals are based on model probabilities, Model K market/proxy prices, q_shock, and the causal HAR-RV activation flag.
- The integrated HAR-RV filter in `Model_O_Trade_Diagnostics.py` shifts realized volatility by one hour before computing all HAR features and historical quantile thresholds.
- Official Kalshi outcomes and realized PnL are used only after signal construction for scoring/evaluation.

No-Leakage Checks

1. Model O hybrid probability construction
   - `Model_O.py` builds `p_final` from `p_normal`, `p_shock`, and `q_shock`.
   - It does not read official outcomes or realized PnL.
   - Contract alignment is validated between normal and shock source files before writing outputs.

2. Evaluation labels
   - `Model_O_Eval.py` attaches official Kalshi outcomes after forecasts are loaded.
   - `p_reality`, `official_result`, and resolution mismatch diagnostics are not fed back into model probabilities.

3. Trade construction
   - `Model_O_Trade_Diagnostics.py` constructs BUY_YES / BUY_NO signals from selected strategy probability versus Model K selected-side price/proxy.
   - `p_reality` is used after the trade is selected to compute win/loss and PnL.

4. HAR-RV filter
   - HAR inputs are:
     - `rv_lag_1h`
     - `rv_mean_6h`
     - `rv_mean_24h`
     - `rv_mean_72h`
   - Each feature is computed from `realized_volatility.shift(1)`, so the current forecast hour's realized volatility is not used.
   - Historical quantile thresholds are also computed from `realized_volatility.shift(1)`.
   - Expanding HAR regressions train on rows strictly before the prediction row.

Leakage / Bias Risks To Keep Flagged

1. Upstream q_shock CSV provenance
   - `Model_O/Additional_Data/all_test_predictions.csv` is cloned from the Model M additional data.
   - Model O assumes the `y_prob` values were generated out-of-sample with features available before `target_minute`.
   - This script cannot prove that from the CSV alone. If Model M's gate training/prediction process used future volatility labels or non-walk-forward validation, q_shock can leak.

2. HAR-RV hour-boundary availability
   - The HAR-RV filter uses the prior hour's completed realized volatility.
   - That is causal once the prior hour has closed, but live trading at exactly the first minute of a new hour may need a data-latency buffer.
   - If the prior hour's final minute data is not available immediately, enforce a small delay before allowing the HAR-RV active flag.

3. Post-hoc volatility regime labels
   - `volatility_regime` / `primary_volatility_band` are retained as attribution fields.
   - They are based on realized volatility for the same hour and should not be used as a live activation rule.
   - Any dashboard segment showing high-volatility performance is diagnostic unless it uses the HAR-RV active flag instead.

4. Threshold selection bias
   - The q50-q95 HAR-RV sweep is research output.
   - Choosing q65 or any other threshold because it performed best on the same historical dataset is selection bias unless validated on a separate holdout or walk-forward split.
   - The default Model O diagnostics activation remains q75 because that was the pre-specified high-volatility filter.

5. Upstream normal/shock source audit
   - Model O inherits `p_normal` from Model R and `p_shock` from Model H.
   - This audit did not re-audit Model R or Model H feature engineering for time leakage.
   - If Model O is promoted, audit those upstream source pipelines separately.

6. Execution price realism
   - Current historical Model K raw values mostly provide a stored price proxy, not guaranteed executable bid/ask.
   - This is not label leakage, but it can overstate tradeability. The diagnostics ledger marks `market_price_is_executable` accordingly.

Recommended Promotion Gate

- Run a walk-forward or strict holdout validation for:
  - q_shock generation
  - HAR-RV threshold choice
  - Model O trade diagnostics after fees
- Keep post-hoc volatility regime labels out of any live signal path.
- Add a live data availability test for the prior-hour realized volatility feed before using HAR-RV activation at hour boundaries.
