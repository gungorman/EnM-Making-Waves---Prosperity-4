# Machine Learning Models

## Concept
Train predictive models on historical data to forecast short-term price movements.

## Key Ideas
- **Features**: mid-price changes, spread, volume, order imbalance, trade flow, position
- **Targets**: Next-step price change, direction, or optimal action
- **Models**: Linear regression, Ridge/Lasso, Random Forest, Gradient Boosting, LSTM
- **Caution**: Overfitting is the main risk with limited data

## Potential Approaches
1. **Linear regression** on order book features -> predict price move
2. **Classification** (up/down/flat) using gradient boosting
3. **Reinforcement learning** for optimal market making policy
4. **Feature engineering** from raw order book snapshots

## Implementation Checklist
- [ ] Feature engineering pipeline from price/trade CSVs
- [ ] Train/test split respecting time ordering
- [ ] Model training and cross-validation
- [ ] Feature importance analysis
- [ ] Integration into Trader class (model must be serializable or embedded)

## Notes
- Submissions cannot import external packages beyond standard library + numpy
- Model weights must be hardcoded or serialized in traderData string
- Keep models simple due to submission constraints
