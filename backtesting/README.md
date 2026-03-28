# Backtesting

## Overview
Local backtesting setup for testing trading algorithms against historical data before submission.

## Options

### 1. prosperity3bt (PyPI)
Community backtester from previous editions. Check if updated for Prosperity 4:
```bash
pip install prosperity3bt
```

### 2. Custom Backtester
Build a simple backtester that:
1. Loads price/trade CSV data
2. Constructs `TradingState` objects per timestamp
3. Calls `Trader.run()` each iteration
4. Simulates order matching against the book
5. Tracks PnL and position

### 3. Official Platform
The competition website provides a submission sandbox for testing. Use for final validation.

## Usage
```bash
# Install dependencies
pip install pandas matplotlib numpy

# Run EDA first
python explorations/01_eda_tutorial_data/eda_round0.py

# Test a submission locally (once backtester is set up)
python backtesting/run_backtest.py submissions/TEST0001.py
```
