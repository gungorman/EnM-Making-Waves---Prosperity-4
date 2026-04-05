"""
Data loader for IMC Prosperity 4 CSV data.
Works with local paths, Google Colab (Drive mount), and Kaggle datasets.
"""

import os
import pandas as pd


def detect_data_dir():
    """Auto-detect data directory based on environment."""
    candidates = [
        # Local (relative to repo root)
        os.path.join(os.path.dirname(__file__), "..", "..", "data"),
        # Google Colab with Drive mount
        "/content/drive/MyDrive/prosperity4/data",
        "/content/data",
        # Kaggle
        "/kaggle/input/prosperity4/data",
        "/kaggle/input/data",
    ]

    for path in candidates:
        if os.path.isdir(path):
            return os.path.abspath(path)

    raise FileNotFoundError(
        "Could not find data directory. Searched:\n"
        + "\n".join(f"  - {p}" for p in candidates)
        + "\nProvide the path explicitly via data_dir parameter."
    )


def load_prices(data_dir=None, round_name="round_0_tutorial"):
    """Load all price CSV files for a given round.

    Args:
        data_dir: Path to data/ folder. Auto-detected if None.
        round_name: Subfolder name (e.g. "round_0_tutorial").

    Returns:
        pd.DataFrame with columns: day, timestamp, product, bid/ask prices/volumes,
        mid_price, profit_and_loss.
    """
    if data_dir is None:
        data_dir = detect_data_dir()

    round_dir = os.path.join(data_dir, round_name)
    if not os.path.isdir(round_dir):
        raise FileNotFoundError(f"Round directory not found: {round_dir}")

    dfs = []
    for f in sorted(os.listdir(round_dir)):
        if f.startswith("prices_"):
            df = pd.read_csv(os.path.join(round_dir, f), sep=";")
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No price files found in {round_dir}")

    return pd.concat(dfs, ignore_index=True)


def load_trades(data_dir=None, round_name="round_0_tutorial"):
    """Load all trade CSV files for a given round.

    Args:
        data_dir: Path to data/ folder. Auto-detected if None.
        round_name: Subfolder name (e.g. "round_0_tutorial").

    Returns:
        pd.DataFrame with columns: timestamp, buyer, seller, symbol, currency,
        price, quantity.
    """
    if data_dir is None:
        data_dir = detect_data_dir()

    round_dir = os.path.join(data_dir, round_name)
    if not os.path.isdir(round_dir):
        raise FileNotFoundError(f"Round directory not found: {round_dir}")

    dfs = []
    for f in sorted(os.listdir(round_dir)):
        if f.startswith("trades_"):
            df = pd.read_csv(os.path.join(round_dir, f), sep=";")
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No trade files found in {round_dir}")

    return pd.concat(dfs, ignore_index=True)


def load_day_data(prices_df, trades_df, day, product=None):
    """Extract data for a specific day (and optionally product).

    Args:
        prices_df: Full prices DataFrame.
        trades_df: Full trades DataFrame.
        day: Day number (e.g. -2, -1).
        product: Optional product filter (e.g. "TOMATOES").

    Returns:
        (day_prices, day_trades) DataFrames sorted by timestamp.
    """
    day_prices = prices_df[prices_df["day"] == day].copy()
    day_trades = trades_df.copy()

    if product:
        day_prices = day_prices[day_prices["product"] == product]
        day_trades = day_trades[day_trades["symbol"] == product]

    day_prices = day_prices.sort_values("timestamp").reset_index(drop=True)
    day_trades = day_trades.sort_values("timestamp").reset_index(drop=True)

    return day_prices, day_trades


def get_available_rounds(data_dir=None):
    """List available round directories."""
    if data_dir is None:
        data_dir = detect_data_dir()

    return [d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("round")]
