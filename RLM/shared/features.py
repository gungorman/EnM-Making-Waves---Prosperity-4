"""
Feature engineering for RL trading models.
Computes 19 trading indicators from order book and trade data.

This module is used both during training (from CSV DataFrames) and at runtime
(from TradingState objects). The same logic produces identical features in both cases.
"""

import numpy as np
from collections import deque

from .config import NUM_FEATURES, POSITION_LIMITS, ENV_CONFIG


class FeatureComputer:
    """Computes and normalizes trading features for a single product.

    Maintains a rolling history buffer for momentum/volatility calculations.
    Can be serialized to/from a dict for persistence via traderData.
    """

    def __init__(self, product, history_length=None):
        self.product = product
        self.history_length = history_length or ENV_CONFIG["history_length"]
        self.position_limit = POSITION_LIMITS.get(product, 50)

        # Rolling history buffers
        self.mid_prices = deque(maxlen=self.history_length)
        self.returns = deque(maxlen=self.history_length)
        self.trade_buy_volumes = deque(maxlen=50)
        self.trade_sell_volumes = deque(maxlen=50)
        self.trade_counts = deque(maxlen=50)

        # Normalization params (set during training via fit())
        self.feature_means = np.zeros(NUM_FEATURES)
        self.feature_stds = np.ones(NUM_FEATURES)

    def compute(self, bid_prices, bid_volumes, ask_prices, ask_volumes,
                position, trades=None, entry_value=0.0):
        """Compute 19 features from current market state.

        Args:
            bid_prices: list of bid prices [best, 2nd, 3rd] (can have None/NaN)
            bid_volumes: list of bid volumes [best, 2nd, 3rd]
            ask_prices: list of ask prices [best, 2nd, 3rd]
            ask_volumes: list of ask volumes [best, 2nd, 3rd]
            position: current inventory (int)
            trades: list of (price, quantity) tuples for recent trades
            entry_value: cost basis of current position for PnL calc

        Returns:
            np.array of shape (19,) with raw (unnormalized) features.
        """
        # Clean up NaN values
        bid_p = [p for p in bid_prices if p is not None and not _isnan(p)]
        bid_v = [v for v, p in zip(bid_volumes, bid_prices)
                 if p is not None and not _isnan(p)]
        ask_p = [p for p in ask_prices if p is not None and not _isnan(p)]
        ask_v = [v for v, p in zip(ask_volumes, ask_prices)
                 if p is not None and not _isnan(p)]

        # Fallbacks if book is empty
        best_bid = bid_p[0] if bid_p else 0
        best_ask = ask_p[0] if ask_p else 0
        best_bid_vol = bid_v[0] if bid_v else 1
        best_ask_vol = ask_v[0] if ask_v else 1

        total_bid_vol = sum(bid_v) if bid_v else 1
        total_ask_vol = sum(ask_v) if ask_v else 1

        # --- Price features (4) ---
        mid_price = (best_bid + best_ask) / 2.0 if (best_bid and best_ask) else 0

        # Micro price: volume-weighted fair value
        if best_bid and best_ask:
            micro_price = (best_bid * best_ask_vol + best_ask * best_bid_vol) / \
                          (best_bid_vol + best_ask_vol)
        else:
            micro_price = mid_price

        spread = best_ask - best_bid if (best_bid and best_ask) else 0
        spread_bps = (spread / mid_price * 10000) if mid_price > 0 else 0

        # Update history
        self.mid_prices.append(mid_price)
        if len(self.mid_prices) >= 2:
            ret = (self.mid_prices[-1] - self.mid_prices[-2]) / self.mid_prices[-2] \
                if self.mid_prices[-2] != 0 else 0
            self.returns.append(ret)

        # Normalized mid price (deviation from rolling mean)
        if len(self.mid_prices) >= 20:
            rolling_mean = np.mean(list(self.mid_prices)[-20:])
            mid_price_norm = (mid_price - rolling_mean) / rolling_mean if rolling_mean != 0 else 0
        else:
            mid_price_norm = 0.0

        # Micro price deviation from mid
        micro_price_norm = (micro_price - mid_price) / spread if spread > 0 else 0.0

        # --- Order book features (5) ---
        imbalance_l1 = best_bid_vol / (best_bid_vol + best_ask_vol)
        imbalance_total = total_bid_vol / (total_bid_vol + total_ask_vol)
        depth_ratio = np.clip(total_bid_vol / total_ask_vol, 0.1, 10.0)
        book_pressure = (best_bid_vol - best_ask_vol) / (best_bid_vol + best_ask_vol)
        depth_concentration = bid_v[0] / total_bid_vol if (bid_v and total_bid_vol > 0) else 0.5

        # --- Momentum features (4) ---
        returns_list = list(self.returns)
        return_5 = sum(returns_list[-5:]) if len(returns_list) >= 5 else 0.0
        return_20 = sum(returns_list[-20:]) if len(returns_list) >= 20 else 0.0
        return_100 = sum(returns_list[-100:]) if len(returns_list) >= 100 else 0.0

        # Price acceleration: change in short-term momentum
        if len(returns_list) >= 10:
            recent_mom = sum(returns_list[-5:])
            prev_mom = sum(returns_list[-10:-5])
            price_acceleration = recent_mom - prev_mom
        else:
            price_acceleration = 0.0

        # --- Volatility features (2) ---
        if len(returns_list) >= 50:
            rolling_vol_50 = float(np.std(returns_list[-50:]))
        else:
            rolling_vol_50 = 0.0

        if len(returns_list) >= 200:
            rolling_vol_200 = float(np.std(returns_list[-200:]))
        else:
            rolling_vol_200 = rolling_vol_50  # fallback

        # --- Trade flow features (2) ---
        if trades:
            buy_vol = 0
            sell_vol = 0
            for trade_price, trade_qty in trades:
                # Lee-Ready classification: trade above mid = buy, below = sell
                if trade_price > mid_price:
                    buy_vol += abs(trade_qty)
                elif trade_price < mid_price:
                    sell_vol += abs(trade_qty)
                else:
                    # At mid: split evenly
                    buy_vol += abs(trade_qty) / 2
                    sell_vol += abs(trade_qty) / 2
            self.trade_buy_volumes.append(buy_vol)
            self.trade_sell_volumes.append(sell_vol)
            self.trade_counts.append(len(trades))
        else:
            self.trade_buy_volumes.append(0)
            self.trade_sell_volumes.append(0)
            self.trade_counts.append(0)

        total_buy = sum(self.trade_buy_volumes)
        total_sell = sum(self.trade_sell_volumes)
        total_flow = total_buy + total_sell
        trade_flow_imbalance = (total_buy - total_sell) / total_flow if total_flow > 0 else 0.0

        avg_trade_count = np.mean(list(self.trade_counts)) if self.trade_counts else 0
        trade_intensity = avg_trade_count / 5.0  # normalize: ~5 trades/tick is "normal"

        # --- Position features (2) ---
        position_normalized = position / self.position_limit if self.position_limit > 0 else 0
        if mid_price > 0 and position != 0:
            position_pnl = (position * mid_price - entry_value) / (mid_price * self.position_limit)
        else:
            position_pnl = 0.0

        features = np.array([
            mid_price_norm,
            micro_price_norm,
            spread,
            spread_bps,
            imbalance_l1,
            imbalance_total,
            depth_ratio,
            book_pressure,
            depth_concentration,
            return_5,
            return_20,
            return_100,
            price_acceleration,
            rolling_vol_50,
            rolling_vol_200,
            trade_flow_imbalance,
            trade_intensity,
            position_normalized,
            position_pnl,
        ], dtype=np.float32)

        return features

    def normalize(self, features):
        """Z-score normalize features using stored mean/std. Clip to +/- clip_val."""
        clip_val = ENV_CONFIG["clip_features"]
        normed = (features - self.feature_means) / (self.feature_stds + 1e-8)
        return np.clip(normed, -clip_val, clip_val).astype(np.float32)

    def reset(self):
        """Clear history buffers (call at start of new episode)."""
        self.mid_prices.clear()
        self.returns.clear()
        self.trade_buy_volumes.clear()
        self.trade_sell_volumes.clear()
        self.trade_counts.clear()

    def to_dict(self):
        """Serialize state for traderData persistence."""
        return {
            "mid_prices": list(self.mid_prices),
            "returns": list(self.returns),
            "trade_buy_volumes": list(self.trade_buy_volumes),
            "trade_sell_volumes": list(self.trade_sell_volumes),
            "trade_counts": list(self.trade_counts),
        }

    def from_dict(self, d):
        """Restore state from traderData."""
        self.mid_prices = deque(d.get("mid_prices", []), maxlen=self.history_length)
        self.returns = deque(d.get("returns", []), maxlen=self.history_length)
        self.trade_buy_volumes = deque(d.get("trade_buy_volumes", []), maxlen=50)
        self.trade_sell_volumes = deque(d.get("trade_sell_volumes", []), maxlen=50)
        self.trade_counts = deque(d.get("trade_counts", []), maxlen=50)


def compute_features_from_row(row, feature_computer, position=0, trades=None, entry_value=0.0):
    """Compute features from a prices DataFrame row.

    Args:
        row: pandas Series from prices CSV.
        feature_computer: FeatureComputer instance for this product.
        position: current inventory.
        trades: list of (price, qty) tuples.
        entry_value: cost basis.

    Returns:
        np.array of shape (19,).
    """
    bid_prices = [
        row.get("bid_price_1", None),
        row.get("bid_price_2", None),
        row.get("bid_price_3", None),
    ]
    bid_volumes = [
        row.get("bid_volume_1", 0),
        row.get("bid_volume_2", 0),
        row.get("bid_volume_3", 0),
    ]
    ask_prices = [
        row.get("ask_price_1", None),
        row.get("ask_price_2", None),
        row.get("ask_price_3", None),
    ]
    ask_volumes = [
        row.get("ask_volume_1", 0),
        row.get("ask_volume_2", 0),
        row.get("ask_volume_3", 0),
    ]

    return feature_computer.compute(
        bid_prices, bid_volumes, ask_prices, ask_volumes,
        position, trades, entry_value
    )


def fit_normalizer(feature_matrix):
    """Compute mean/std from a matrix of raw features.

    Args:
        feature_matrix: np.array of shape (N, 19).

    Returns:
        (means, stds) each of shape (19,).
    """
    means = np.mean(feature_matrix, axis=0).astype(np.float32)
    stds = np.std(feature_matrix, axis=0).astype(np.float32)
    stds[stds < 1e-8] = 1.0  # avoid division by zero for constant features
    return means, stds


def _isnan(x):
    """Check if a value is NaN (works for float and numpy)."""
    try:
        return np.isnan(x)
    except (TypeError, ValueError):
        return False
