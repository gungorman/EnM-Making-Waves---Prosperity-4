"""
Trading environment for RL training.
Wraps historical CSV data into a Gymnasium-compatible environment.
FinRL-inspired modular design.

Performance: pre-indexes DataFrames at episode start so step() uses
O(1) numpy lookups instead of DataFrame filtering.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .config import (
    PRODUCTS, NUM_FEATURES, NUM_ACTIONS, ACTION_PARAMS,
    ENV_CONFIG, AUGMENTATION_CONFIG, POSITION_LIMITS,
)
from .features import FeatureComputer, compute_features_from_row
from .data_loader import load_prices, load_trades, load_day_data


class TradingEnv(gym.Env):
    """Gymnasium environment for order-book trading simulation.

    One episode = one trading day. The agent steps through historical timestamps
    and places orders based on the current order book state.

    Supports multi-product trading (actions are flattened across products).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        prices_df,
        trades_df,
        products=None,
        day=None,
        augment=False,
        seed=None,
    ):
        super().__init__()

        self.prices_df = prices_df
        self.trades_df = trades_df
        self.products = products or PRODUCTS
        self.day = day
        self.augment = augment
        self.rng = np.random.RandomState(seed)

        self.num_products = len(self.products)

        # Action space: flattened discrete actions across products
        self.action_space = spaces.Discrete(NUM_ACTIONS ** self.num_products)

        # Observation space: features per product, concatenated
        obs_dim = NUM_FEATURES * self.num_products
        self.observation_space = spaces.Box(
            low=-ENV_CONFIG["clip_features"],
            high=ENV_CONFIG["clip_features"],
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Feature computers (one per product)
        self.feature_computers = {
            product: FeatureComputer(product)
            for product in self.products
        }

        # Episode state
        self.current_step = 0
        self.positions = {}
        self.entry_values = {}
        self.pnl = 0.0
        self.prev_pnl = 0.0
        self.day_prices = {}
        self.day_trades = {}
        self.timestamps = []

        # Pre-indexed data (built in reset, used in step for speed)
        self._price_rows = {}   # {product: list of row dicts}
        self._trade_index = {}  # {product: {timestamp: [(price, qty), ...]}}

    def reset(self, seed=None, options=None):
        """Reset environment to start of a trading day."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        # Pick a day
        if self.day is not None:
            day = self.day
        else:
            available_days = sorted(self.prices_df["day"].unique())
            day = self.rng.choice(available_days)

        # Load data for each product
        self.day_prices = {}
        self.day_trades = {}
        self.timestamps = None

        for product in self.products:
            dp, dt = load_day_data(self.prices_df, self.trades_df, day, product)
            self.day_prices[product] = dp
            self.day_trades[product] = dt

            if self.timestamps is None:
                self.timestamps = dp["timestamp"].values
            else:
                self.timestamps = np.intersect1d(self.timestamps, dp["timestamp"].values)

        # Apply augmentation
        if self.augment:
            self._apply_augmentation()

        # === Pre-index for fast step() lookups ===
        self._build_index()

        # Reset state
        self.current_step = 0
        self.positions = {p: 0 for p in self.products}
        self.entry_values = {p: 0.0 for p in self.products}
        self.pnl = 0.0
        self.prev_pnl = 0.0

        for fc in self.feature_computers.values():
            fc.reset()

        obs = self._get_observation()
        info = {"day": day, "total_steps": len(self.timestamps)}

        return obs, info

    def _build_index(self):
        """Pre-index price rows and trades by timestamp for O(1) lookups."""
        ts_set = set(self.timestamps)

        for product in self.products:
            # Index price rows: filter to valid timestamps, store as list of dicts
            df = self.day_prices[product]
            df_filtered = df[df["timestamp"].isin(ts_set)].sort_values("timestamp")
            self._price_rows[product] = df_filtered.to_dict("records")

            # Index trades: group by timestamp
            dt = self.day_trades[product]
            trade_idx = {}
            if len(dt) > 0:
                dt_filtered = dt[dt["timestamp"].isin(ts_set)]
                for ts, group in dt_filtered.groupby("timestamp"):
                    trade_idx[ts] = list(zip(group["price"].values, group["quantity"].values))
            self._trade_index[product] = trade_idx

    def step(self, action):
        """Execute one timestep."""
        # Decode flattened action into per-product actions
        product_actions = self._decode_action(action)

        # Execute actions for each product
        step_pnl = 0.0
        for product, act_id in zip(self.products, product_actions):
            pnl_delta = self._execute_action(product, act_id)
            step_pnl += pnl_delta

        # Compute mark-to-market PnL
        mtm_pnl = 0.0
        for product in self.products:
            row = self._get_price_row(product)
            if row is not None:
                mid = row["mid_price"]
                mtm_pnl += self.positions[product] * mid - self.entry_values[product]

        self.pnl = step_pnl + mtm_pnl

        # Advance step
        self.current_step += 1
        terminated = self.current_step >= len(self.timestamps)
        truncated = False

        # Reward: scaled PnL change minus inventory penalty
        scale = ENV_CONFIG.get("reward_scale", 0.001)
        reward = (self.pnl - self.prev_pnl) * scale

        # Quadratic inventory penalty (discourages holding large positions)
        for product in self.products:
            pos_ratio = self.positions[product] / POSITION_LIMITS.get(product, 50)
            penalty = ENV_CONFIG["reward_inventory_penalty"] * pos_ratio ** 2
            reward -= penalty

        # Terminal penalty: punish open positions at end of day
        if terminated:
            for product in self.products:
                terminal_cost = ENV_CONFIG.get("reward_terminal_penalty", 0.5) * abs(self.positions[product])
                reward -= terminal_cost

        self.prev_pnl = self.pnl

        obs = self._get_observation() if not terminated else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )

        info = {
            "pnl": self.pnl,
            "positions": dict(self.positions),
            "step": self.current_step,
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        """Compute feature vector for current timestep (uses pre-indexed data)."""
        all_features = []

        for product in self.products:
            row = self._get_price_row(product)
            if row is None:
                all_features.append(np.zeros(NUM_FEATURES, dtype=np.float32))
                continue

            # Get trades at this timestamp from pre-built index
            ts = self.timestamps[self.current_step]
            trades = self._trade_index[product].get(ts, None)

            fc = self.feature_computers[product]
            features = compute_features_from_row(
                row, fc,
                position=self.positions[product],
                trades=trades,
                entry_value=self.entry_values[product],
            )

            if ENV_CONFIG["normalize_features"]:
                features = fc.normalize(features)

            all_features.append(features)

        return np.concatenate(all_features).astype(np.float32)

    def _get_price_row(self, product):
        """Get the price row for a product at current step (O(1) lookup)."""
        if self.current_step >= len(self._price_rows.get(product, [])):
            return None
        return self._price_rows[product][self.current_step]

    def _decode_action(self, action):
        """Decode flattened action int into per-product action list."""
        actions = []
        remaining = action
        for _ in range(self.num_products):
            actions.append(remaining % NUM_ACTIONS)
            remaining //= NUM_ACTIONS
        return actions

    def _execute_action(self, product, action_id):
        """Execute a single action for one product. Returns realized PnL."""
        params = ACTION_PARAMS[action_id]
        if params["type"] == "hold":
            return 0.0

        row = self._get_price_row(product)
        if row is None:
            return 0.0

        best_bid = row.get("bid_price_1", 0)
        best_ask = row.get("ask_price_1", 0)
        if best_bid == 0 or best_ask == 0:
            return 0.0

        pos_limit = POSITION_LIMITS.get(product, 50)
        side = params["side"]
        qty = params["qty"]

        if params["type"] == "cross":
            # Market order: crosses the spread
            if side == "buy":
                price = best_ask
                max_buy = pos_limit - self.positions[product]
                qty = min(qty, max(0, max_buy))
                if qty <= 0:
                    return 0.0
                self.positions[product] += qty
                self.entry_values[product] += qty * price
                return 0.0
            else:
                price = best_bid
                max_sell = pos_limit + self.positions[product]
                qty = min(qty, max(0, max_sell))
                if qty <= 0:
                    return 0.0
                self.positions[product] -= qty
                self.entry_values[product] -= qty * price
                return 0.0

        elif params["type"] == "passive":
            # Limit order with simplified fill model
            offset = params.get("offset", 0)
            fill_prob = 0.5 if offset == 0 else 0.3

            if self.rng.random() > fill_prob:
                return 0.0  # not filled

            if side == "buy":
                price = best_bid - offset
                max_buy = pos_limit - self.positions[product]
                qty = min(qty, max(0, max_buy))
                if qty <= 0:
                    return 0.0
                self.positions[product] += qty
                self.entry_values[product] += qty * price
            else:
                price = best_ask + offset
                max_sell = pos_limit + self.positions[product]
                qty = min(qty, max(0, max_sell))
                if qty <= 0:
                    return 0.0
                self.positions[product] -= qty
                self.entry_values[product] -= qty * price

            return 0.0

        return 0.0

    def _apply_augmentation(self):
        """Apply data augmentation to the loaded day data."""
        cfg = AUGMENTATION_CONFIG
        for product in self.products:
            df = self.day_prices[product]

            # Add Gaussian noise to prices
            noise_std = cfg["price_noise_std"]
            for col in ["bid_price_1", "bid_price_2", "bid_price_3",
                        "ask_price_1", "ask_price_2", "ask_price_3", "mid_price"]:
                if col in df.columns:
                    noise = self.rng.normal(0, noise_std, len(df))
                    df[col] = df[col] + noise

            # Scale volumes
            lo, hi = cfg["volume_scale_range"]
            scale = self.rng.uniform(lo, hi)
            for col in ["bid_volume_1", "bid_volume_2", "bid_volume_3",
                        "ask_volume_1", "ask_volume_2", "ask_volume_3"]:
                if col in df.columns:
                    df[col] = (df[col].fillna(0) * scale).astype(int).clip(lower=0)

            self.day_prices[product] = df

            # Random sub-window
            if cfg["use_random_windows"] and len(self.timestamps) > cfg["window_size"]:
                max_start = len(self.timestamps) - cfg["window_size"]
                start = self.rng.randint(0, max_start)
                self.timestamps = self.timestamps[start:start + cfg["window_size"]]
