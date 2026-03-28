# Market Making Strategy

## Concept
Quote both bid and ask prices around estimated fair value, profiting from the bid-ask spread.

## Key Ideas
- Estimate fair value (e.g., mid-price, VWAP, or model-based)
- Place symmetric buy/sell orders around fair value
- Manage inventory risk: skew quotes when position builds up
- Wider spreads = more profit per trade but fewer fills
- Tighter spreads = more fills but smaller edge

## Applicable Products
- **EMERALDS** (Round 0): Very stable price ~10,000, tight natural spread ~16. Classic MM target.
- Any product with mean-reverting behavior and sufficient liquidity

## Implementation Checklist
- [ ] Fair value estimator
- [ ] Spread calculation (fixed vs dynamic)
- [ ] Position-aware quote skewing
- [ ] Position limit management
- [ ] Order sizing logic

## References
- Avellaneda-Stoikov market making model
- Guéant-Lehalle-Fernandez-Tapia optimal MM
