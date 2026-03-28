# Mean Reversion Strategy

## Concept
Trade price deviations from a central tendency, expecting prices to revert to the mean.

## Key Ideas
- Estimate fair value using rolling mean, EMA, or VWAP
- Buy when price drops significantly below fair value
- Sell when price rises significantly above fair value
- Works best for products with stable underlying value

## Applicable Products
- **TOMATOES** (Round 0): Price fluctuates ~4980-5015, potential mean reversion candidate
- Products that show bounded/range-bound behavior

## Implementation Checklist
- [ ] Fair value estimator (rolling window, EMA)
- [ ] Deviation threshold calibration
- [ ] Position sizing based on deviation magnitude
- [ ] Position limit awareness
- [ ] Backtest on historical data to calibrate parameters
