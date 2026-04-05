"""
RLM Configuration - Hyperparameters, feature definitions, and action space.
Central config used by all model variants.
"""

import numpy as np

# =============================================================================
# Products
# =============================================================================
PRODUCTS = ["EMERALDS", "TOMATOES"]
POSITION_LIMITS = {
    "EMERALDS": 50,
    "TOMATOES": 50,
}

# =============================================================================
# Feature definitions (19 per product)
# =============================================================================
FEATURE_NAMES = [
    # Price features (4)
    "mid_price_norm",       # normalized mid price (return from rolling mean)
    "micro_price_norm",     # volume-weighted fair value deviation from mid
    "spread",               # bid-ask spread in ticks
    "spread_bps",           # spread in basis points

    # Order book features (5)
    "imbalance_l1",         # bid_vol_1 / (bid_vol_1 + ask_vol_1)
    "imbalance_total",      # sum(bid_vols) / (sum(bid_vols) + sum(ask_vols))
    "depth_ratio",          # sum(bid_vols) / sum(ask_vols), clipped
    "book_pressure",        # (bid_vol_1 - ask_vol_1) / (bid_vol_1 + ask_vol_1)
    "depth_concentration",  # bid_vol_1 / sum(bid_vols)

    # Momentum features (4)
    "return_5",             # price return over 5 ticks
    "return_20",            # price return over 20 ticks
    "return_100",           # price return over 100 ticks
    "price_acceleration",   # return_5 - lagged return_5

    # Volatility features (2)
    "rolling_vol_50",       # std of returns over 50 ticks
    "rolling_vol_200",      # std of returns over 200 ticks

    # Trade flow features (2)
    "trade_flow_imbalance", # (buy_vol - sell_vol) / total_vol
    "trade_intensity",      # trades in last N ticks, normalized

    # Position features (2)
    "position_normalized",  # current_pos / position_limit
    "position_pnl",         # unrealized PnL, normalized
]

NUM_FEATURES = len(FEATURE_NAMES)  # 19
NUM_FEATURES_TOTAL = NUM_FEATURES * len(PRODUCTS)  # 38 for 2 products

# =============================================================================
# Action space (9 discrete actions per product)
# =============================================================================
# Each action is (order_type, quantity_fraction)
# quantity_fraction is relative to position limit
ACTION_NAMES = [
    "hold",              # 0: do nothing
    "market_buy_small",  # 1: buy 5 units at best ask
    "market_sell_small", # 2: sell 5 units at best bid
    "passive_buy",       # 3: limit buy at best bid (earn spread)
    "passive_sell",      # 4: limit sell at best ask (earn spread)
    "deep_buy",          # 5: limit buy at best_bid - 1
    "deep_sell",         # 6: limit sell at best_ask + 1
    "market_buy_large",  # 7: buy 15 units at best ask
    "market_sell_large", # 8: sell 15 units at best bid
]

NUM_ACTIONS = len(ACTION_NAMES)  # 9

# Action parameters: (price_offset_from_best, quantity)
# price_offset: 0 = at best, -1 = one tick worse (deeper), "cross" = cross spread
ACTION_PARAMS = {
    0: {"type": "hold",    "qty": 0},
    1: {"type": "cross",   "side": "buy",  "qty": 5},
    2: {"type": "cross",   "side": "sell", "qty": 5},
    3: {"type": "passive", "side": "buy",  "qty": 10, "offset": 0},
    4: {"type": "passive", "side": "sell", "qty": 10, "offset": 0},
    5: {"type": "passive", "side": "buy",  "qty": 10, "offset": 1},
    6: {"type": "passive", "side": "sell", "qty": 10, "offset": 1},
    7: {"type": "cross",   "side": "buy",  "qty": 15},
    8: {"type": "cross",   "side": "sell", "qty": 15},
}

# =============================================================================
# DQN Hyperparameters (shared defaults, models can override)
# =============================================================================
DQN_CONFIG = {
    "learning_rate": 1e-3,
    "buffer_size": 20_000,
    "learning_starts": 500,
    "batch_size": 64,
    "gamma": 0.99,
    "exploration_fraction": 0.3,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "target_update_interval": 500,
    "train_freq": 4,
    "gradient_steps": 1,
    "total_timesteps": 100_000,  # ~10 episodes of 10k steps
}

# Network architecture
NETWORK_CONFIG = {
    "hidden_sizes": [64, 64],
    "activation": "relu",
}

# =============================================================================
# Environment config
# =============================================================================
ENV_CONFIG = {
    "reward_inventory_penalty": 0.02,   # lambda for position^2 penalty (per step)
    "reward_terminal_penalty": 0.5,     # penalty per unit of open position at episode end
    "reward_scale": 0.001,              # scale raw PnL to keep rewards in reasonable range
    "history_length": 200,              # ticks of price history to maintain
    "normalize_features": True,
    "clip_features": 3.0,              # clip z-scores at +/- this value
}

# =============================================================================
# Data augmentation
# =============================================================================
AUGMENTATION_CONFIG = {
    "price_noise_std": 0.5,     # gaussian noise added to prices
    "volume_scale_range": (0.7, 1.3),  # random volume scaling
    "use_random_windows": True,
    "window_size": 2000,        # random sub-window length
}

# =============================================================================
# Training
# =============================================================================
TRAIN_CONFIG = {
    "train_days": [-2],         # days to train on
    "eval_days": [-1],          # days to evaluate on
    "n_eval_episodes": 5,
    "seed": 42,
    "device": "auto",           # "auto" picks GPU if available, else CPU
    "verbose": 1,
}
