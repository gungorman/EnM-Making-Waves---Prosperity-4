# EnM Making Waves - IMC Prosperity 4

Algorithmic trading challenge repository for IMC Prosperity 4 (April 14-30, 2026).

## Repository Structure

```
submissions/       # Algorithm scripts submitted to the platform
data/              # Competition data organized by round
explorations/      # Strategy research & analysis
backtesting/       # Local backtesting utilities
utils/             # Shared code (datamodel, helpers)
```

## Quick Start

```bash
pip install pandas matplotlib numpy

# Run EDA on tutorial data
python explorations/01_eda_tutorial_data/eda_round0.py
```

## Explorations

| # | Strategy | Status |
|---|----------|--------|
| 01 | EDA - Tutorial Data | Ready |
| 02 | Market Making | Planning |
| 03 | Statistical Arbitrage | Planning |
| 04 | Mean Reversion | Planning |
| 05 | Momentum Signals | Planning |
| 06 | ML Models | Planning |
| 07 | NLP & Sentiment | Planning |
| 08 | Order Book Analysis | Planning |

## Tutorial Products (Round 0)

- **EMERALDS**: Stable ~10,000, tight spread. Market making candidate.
- **TOMATOES**: Volatile ~4980-5015, wider spread. Mean reversion candidate.
