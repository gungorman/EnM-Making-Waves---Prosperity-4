# Statistical Arbitrage

## Concept
Exploit statistical relationships between products to find mispricings.

## Key Ideas
- **Pairs trading**: Find correlated products, trade the spread when it diverges
- **Basket/ETF arbitrage**: If a basket product exists (like PICNIC_BASKET in P3), trade deviations between basket price and sum of components
- **Cross-product signals**: Use one product's price movement to predict another

## When This Applies
- Multi-product rounds (Round 2+ typically introduces baskets)
- Products with fundamental relationships (e.g., ingredient products vs finished goods)

## Implementation Checklist
- [ ] Correlation analysis between products
- [ ] Spread calculation and z-score monitoring
- [ ] Entry/exit thresholds
- [ ] Hedge ratio estimation
- [ ] Risk management for spread blowouts
