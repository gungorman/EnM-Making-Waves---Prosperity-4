"""
EDA - Round 0 Tutorial Data (TOMATOES & EMERALDS)
Run this script to generate basic analysis plots and statistics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "round_0_tutorial")


def load_prices():
    dfs = []
    for f in sorted(os.listdir(DATA_DIR)):
        if f.startswith("prices_"):
            df = pd.read_csv(os.path.join(DATA_DIR, f), sep=";")
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_trades():
    dfs = []
    for f in sorted(os.listdir(DATA_DIR)):
        if f.startswith("trades_"):
            df = pd.read_csv(os.path.join(DATA_DIR, f), sep=";")
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def analyze_prices(prices: pd.DataFrame):
    products = prices["product"].unique()
    print(f"Products: {products}")
    print(f"Total rows: {len(prices)}")
    print()

    for product in products:
        p = prices[prices["product"] == product]
        print(f"=== {product} ===")
        print(f"  Mid price: min={p['mid_price'].min()}, max={p['mid_price'].max()}, "
              f"mean={p['mid_price'].mean():.2f}, std={p['mid_price'].std():.2f}")
        spread = p["ask_price_1"] - p["bid_price_1"]
        print(f"  Spread (L1): min={spread.min()}, max={spread.max()}, mean={spread.mean():.2f}")
        print(f"  Timestamps: {p['timestamp'].min()} to {p['timestamp'].max()}")
        print()


def plot_mid_prices(prices: pd.DataFrame):
    products = prices["product"].unique()
    fig, axes = plt.subplots(len(products), 1, figsize=(14, 5 * len(products)), sharex=False)
    if len(products) == 1:
        axes = [axes]

    for ax, product in zip(axes, products):
        p = prices[prices["product"] == product].sort_values("timestamp")
        for day in p["day"].unique():
            day_data = p[p["day"] == day]
            ax.plot(day_data["timestamp"], day_data["mid_price"], label=f"Day {day}", alpha=0.8)
        ax.set_title(f"{product} - Mid Price")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Mid Price")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "mid_prices.png"), dpi=150)
    plt.show()


def plot_spreads(prices: pd.DataFrame):
    products = prices["product"].unique()
    fig, axes = plt.subplots(len(products), 1, figsize=(14, 5 * len(products)), sharex=False)
    if len(products) == 1:
        axes = [axes]

    for ax, product in zip(axes, products):
        p = prices[prices["product"] == product].sort_values("timestamp")
        spread = p["ask_price_1"] - p["bid_price_1"]
        for day in p["day"].unique():
            mask = p["day"] == day
            ax.plot(p.loc[mask, "timestamp"], spread[mask], label=f"Day {day}", alpha=0.8)
        ax.set_title(f"{product} - Bid-Ask Spread (L1)")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Spread")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "spreads.png"), dpi=150)
    plt.show()


def plot_order_book_depth(prices: pd.DataFrame):
    products = prices["product"].unique()
    fig, axes = plt.subplots(len(products), 1, figsize=(14, 5 * len(products)))
    if len(products) == 1:
        axes = [axes]

    for ax, product in zip(axes, products):
        p = prices[prices["product"] == product].sort_values("timestamp")
        bid_vol = p[["bid_volume_1", "bid_volume_2", "bid_volume_3"]].sum(axis=1)
        ask_vol = p[["ask_volume_1", "ask_volume_2", "ask_volume_3"]].sum(axis=1)
        imbalance = bid_vol / (bid_vol + ask_vol)
        ax.plot(p["timestamp"], imbalance, alpha=0.5, linewidth=0.5)
        ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.5)
        ax.set_title(f"{product} - Order Book Imbalance (bid_vol / total_vol)")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Imbalance")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "imbalance.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    print("Loading data...")
    prices = load_prices()
    trades = load_trades()

    print("\n--- Price Analysis ---")
    analyze_prices(prices)

    print(f"\n--- Trades Summary ---")
    print(f"Total trades: {len(trades)}")
    for sym in trades["symbol"].unique():
        t = trades[trades["symbol"] == sym]
        print(f"  {sym}: {len(t)} trades, avg price={t['price'].mean():.2f}, total qty={t['quantity'].sum()}")

    print("\nGenerating plots...")
    plot_mid_prices(prices)
    plot_spreads(prices)
    plot_order_book_depth(prices)
    print("Done! Plots saved to explorations/01_eda_tutorial_data/")
