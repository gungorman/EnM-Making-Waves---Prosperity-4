# NLP & Sentiment Analysis

## Concept
Extract trading signals from text data if the competition provides news, rumors, or trader messages.

## Key Ideas
- Previous Prosperity editions included "trader observations" or news items
- Sentiment analysis on provided text to gauge market direction
- Keyword detection for product-specific signals
- Pattern matching on structured text data

## When This Applies
- If observations contain text data (check `state.observations`)
- If manual trading rounds involve news/information interpretation
- Trader ID analysis (different bots may have predictable patterns)

## Implementation Checklist
- [ ] Identify if/when text data is available in the competition
- [ ] Build keyword/sentiment dictionaries for products
- [ ] Map sentiment scores to trading signals
- [ ] Combine with quantitative signals for hybrid approach

## Notes
- In P3, trader IDs were revealed in later rounds, allowing behavior-based strategies
- NLP may be more relevant for manual trading than algorithmic
