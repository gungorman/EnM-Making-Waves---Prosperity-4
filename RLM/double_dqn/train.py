"""
Double DQN training script for IMC Prosperity 4.

Uses stable-baselines3 DQN (which implements Double DQN by default).
Works on both CPU and GPU (auto-detected).

Usage:
    python -m RLM.double_dqn.train
    python -m RLM.double_dqn.train --total-timesteps 200000 --seed 42
"""

import argparse
import os
import sys
import numpy as np

# Add project root to path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

from RLM.shared.config import (
    DQN_CONFIG, NETWORK_CONFIG, TRAIN_CONFIG,
    PRODUCTS, NUM_FEATURES, NUM_ACTIONS, ENV_CONFIG,
)
from RLM.shared.data_loader import load_prices, load_trades
from RLM.shared.env import TradingEnv
from RLM.shared.features import FeatureComputer, compute_features_from_row, fit_normalizer


class FitNormalizerCallback(BaseCallback):
    """Fits feature normalizer on the first episode's data, then updates
    the environment's feature computers with the learned mean/std."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._fitted = False

    def _on_step(self):
        return True


def compute_normalization_params(prices_df, trades_df, products, days):
    """Pre-compute normalization parameters by running through the data."""
    all_features = {p: [] for p in products}

    for day in days:
        for product in products:
            fc = FeatureComputer(product)
            day_prices = prices_df[(prices_df["day"] == day) & (prices_df["product"] == product)]
            day_prices = day_prices.sort_values("timestamp").reset_index(drop=True)

            day_trades = trades_df[trades_df["symbol"] == product].sort_values("timestamp")

            for _, row in day_prices.iterrows():
                ts = row["timestamp"]
                ts_trades = day_trades[day_trades["timestamp"] == ts]
                trades = list(zip(ts_trades["price"], ts_trades["quantity"])) if len(ts_trades) > 0 else None

                features = compute_features_from_row(row, fc, position=0, trades=trades)
                all_features[product].append(features)

    # Combine all products' features and fit
    combined = np.vstack([np.array(v) for v in all_features.values()])
    means, stds = fit_normalizer(combined)

    return means, stds


def train(args):
    """Run Double DQN training."""
    print("=" * 60)
    print("Double DQN Training - IMC Prosperity 4")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    prices_df = load_prices()
    trades_df = load_trades()
    print(f"  Prices: {len(prices_df)} rows")
    print(f"  Trades: {len(trades_df)} rows")
    print(f"  Products: {PRODUCTS}")

    # Compute normalization parameters
    print("\nComputing normalization parameters...")
    feat_means, feat_stds = compute_normalization_params(
        prices_df, trades_df, PRODUCTS, TRAIN_CONFIG["train_days"]
    )
    print(f"  Feature means: {feat_means[:5]}...")
    print(f"  Feature stds:  {feat_stds[:5]}...")

    # Create training environment
    print("\nCreating environments...")
    train_env = TradingEnv(
        prices_df=prices_df,
        trades_df=trades_df,
        products=PRODUCTS,
        day=TRAIN_CONFIG["train_days"][0],
        augment=True,
        seed=args.seed,
    )

    # Set normalization params on the environment's feature computers
    for product in PRODUCTS:
        train_env.feature_computers[product].feature_means = feat_means
        train_env.feature_computers[product].feature_stds = feat_stds

    # Create eval environment (no augmentation)
    eval_env = TradingEnv(
        prices_df=prices_df,
        trades_df=trades_df,
        products=PRODUCTS,
        day=TRAIN_CONFIG["eval_days"][0],
        augment=False,
        seed=args.seed + 1,
    )
    for product in PRODUCTS:
        eval_env.feature_computers[product].feature_means = feat_means
        eval_env.feature_computers[product].feature_stds = feat_stds

    # Model output directory
    model_dir = os.path.join(os.path.dirname(__file__), "policy_weights")
    os.makedirs(model_dir, exist_ok=True)

    # Setup eval callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=model_dir,
        eval_freq=max(args.total_timesteps // 20, 1000),
        n_eval_episodes=TRAIN_CONFIG["n_eval_episodes"],
        deterministic=True,
        verbose=1,
    )

    # Network architecture
    policy_kwargs = dict(
        net_arch=NETWORK_CONFIG["hidden_sizes"],
    )

    # Create DQN model (Double DQN is default in SB3)
    print("\nInitializing Double DQN...")
    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=args.lr or DQN_CONFIG["learning_rate"],
        buffer_size=DQN_CONFIG["buffer_size"],
        learning_starts=DQN_CONFIG["learning_starts"],
        batch_size=DQN_CONFIG["batch_size"],
        gamma=DQN_CONFIG["gamma"],
        exploration_fraction=DQN_CONFIG["exploration_fraction"],
        exploration_initial_eps=DQN_CONFIG["exploration_initial_eps"],
        exploration_final_eps=DQN_CONFIG["exploration_final_eps"],
        target_update_interval=DQN_CONFIG["target_update_interval"],
        train_freq=DQN_CONFIG["train_freq"],
        gradient_steps=DQN_CONFIG["gradient_steps"],
        policy_kwargs=policy_kwargs,
        device=TRAIN_CONFIG["device"],
        verbose=TRAIN_CONFIG["verbose"],
        seed=args.seed,
    )

    print(f"\n  Device: {model.device}")
    print(f"  Network: {NETWORK_CONFIG['hidden_sizes']}")
    print(f"  Total timesteps: {args.total_timesteps}")
    print(f"  Seed: {args.seed}")

    # Train
    print("\nTraining...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}")

    # Save normalization params alongside weights
    np.savez(
        os.path.join(model_dir, "norm_params.npz"),
        feature_means=feat_means,
        feature_stds=feat_stds,
    )
    print(f"Normalization params saved to: {model_dir}/norm_params.npz")

    print("\nTraining complete!")
    return model, feat_means, feat_stds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Double DQN")
    parser.add_argument("--total-timesteps", type=int, default=DQN_CONFIG["total_timesteps"])
    parser.add_argument("--seed", type=int, default=TRAIN_CONFIG["seed"])
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    train(args)
