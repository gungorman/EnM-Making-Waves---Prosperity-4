"""
Ensemble DQN training for IMC Prosperity 4.

Trains N independent Double DQN agents with different random seeds.
At inference, Q-values are averaged across all members for more stable decisions.
This is the FinRL signature approach for handling noisy financial data.

Usage:
    python -m RLM.ensemble_dqn.train
    python -m RLM.ensemble_dqn.train --n-members 5 --total-timesteps 100000
"""

import argparse
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

from RLM.shared.config import (
    DQN_CONFIG, NETWORK_CONFIG, TRAIN_CONFIG, PRODUCTS, ENV_CONFIG,
)
from RLM.shared.data_loader import load_prices, load_trades
from RLM.shared.env import TradingEnv
from RLM.double_dqn.train import compute_normalization_params


def train(args):
    """Train an ensemble of DQN agents."""
    print("=" * 60)
    print(f"Ensemble DQN Training ({args.n_members} members)")
    print("=" * 60)

    prices_df = load_prices()
    trades_df = load_trades()

    print("\nComputing normalization parameters...")
    feat_means, feat_stds = compute_normalization_params(
        prices_df, trades_df, PRODUCTS, TRAIN_CONFIG["train_days"]
    )

    model_dir = os.path.join(os.path.dirname(__file__), "policy_weights")
    os.makedirs(model_dir, exist_ok=True)

    models = []

    for i in range(args.n_members):
        seed = args.seed + i * 100
        print(f"\n{'='*40}")
        print(f"Training member {i + 1}/{args.n_members} (seed={seed})")
        print(f"{'='*40}")

        train_env = TradingEnv(
            prices_df=prices_df, trades_df=trades_df,
            products=PRODUCTS, day=TRAIN_CONFIG["train_days"][0],
            augment=True, seed=seed,
        )
        eval_env = TradingEnv(
            prices_df=prices_df, trades_df=trades_df,
            products=PRODUCTS, day=TRAIN_CONFIG["eval_days"][0],
            augment=False, seed=seed + 1,
        )

        for product in PRODUCTS:
            train_env.feature_computers[product].feature_means = feat_means
            train_env.feature_computers[product].feature_stds = feat_stds
            eval_env.feature_computers[product].feature_means = feat_means
            eval_env.feature_computers[product].feature_stds = feat_stds

        member_dir = os.path.join(model_dir, f"member_{i}")
        os.makedirs(member_dir, exist_ok=True)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=member_dir,
            log_path=member_dir,
            eval_freq=max(args.total_timesteps // 20, 1000),
            n_eval_episodes=TRAIN_CONFIG["n_eval_episodes"],
            deterministic=True,
            verbose=1,
        )

        policy_kwargs = dict(net_arch=NETWORK_CONFIG["hidden_sizes"])

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
            seed=seed,
        )

        model.learn(
            total_timesteps=args.total_timesteps,
            callback=eval_callback,
            progress_bar=True,
        )

        final_path = os.path.join(member_dir, "final_model")
        model.save(final_path)
        models.append(model)
        print(f"  Member {i + 1} saved to: {final_path}")

    # Save normalization params
    np.savez(
        os.path.join(model_dir, "norm_params.npz"),
        feature_means=feat_means,
        feature_stds=feat_stds,
    )

    print(f"\n{'='*60}")
    print(f"Ensemble training complete! {args.n_members} members trained.")
    print(f"Models saved in: {model_dir}")
    return models, feat_means, feat_stds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Ensemble DQN")
    parser.add_argument("--n-members", type=int, default=3)
    parser.add_argument("--total-timesteps", type=int, default=DQN_CONFIG["total_timesteps"])
    parser.add_argument("--seed", type=int, default=TRAIN_CONFIG["seed"])
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    train(args)
