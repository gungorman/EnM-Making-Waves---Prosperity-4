# Order Book Analysis & Imbalance

## Concept
Analyze the shape and dynamics of the order book to predict short-term price movements.

## Key Ideas
- **Volume imbalance**: Ratio of bid volume to ask volume predicts direction
- **Depth analysis**: Total liquidity at different price levels
- **Book pressure**: Large orders on one side may push price
- **Microstructure signals**: Queue position, fill rates, cancellation patterns

## Applicable Products
- All products (order book data is always available)
- Most useful for products with varying book shapes

## Key Metrics
- `bid_vol / (bid_vol + ask_vol)` — values > 0.5 suggest buying pressure
- Weighted mid-price: accounts for volume at best bid/ask
- Book depth ratio across multiple levels

## Implementation Checklist
- [ ] Calculate volume imbalance at each timestamp
- [ ] Correlate imbalance with future price changes
- [ ] Weighted mid-price estimator
- [ ] Multi-level depth analysis
- [ ] Integrate as feature into trading signals
