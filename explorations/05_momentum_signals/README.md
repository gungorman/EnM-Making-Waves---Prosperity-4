# Momentum / Trend Following

## Concept
Identify short-term trends in price or order flow and trade in their direction.

## Key Ideas
- **Price momentum**: Detect trends using moving average crossovers, breakouts
- **Order flow momentum**: Track aggressive buying/selling (market orders hitting the book)
- **Trade imbalance**: Net buyer-initiated vs seller-initiated trades
- Short time horizons (this is tick-level data, not daily)

## Applicable Products
- Products with trending behavior or regime changes
- Products where informed traders move the price (detectable in trade data)

## Implementation Checklist
- [ ] Trend detection (MA crossover, linear regression slope)
- [ ] Order flow analysis (trade direction classification)
- [ ] Signal strength filtering
- [ ] Quick entry/exit to capture short moves
- [ ] Combine with mean reversion for hybrid approach
