"""
Dueling DQN training script for IMC Prosperity 4.

Uses stable-baselines3 DQN with a custom dueling network architecture.
The network splits into Value V(s) and Advantage A(s,a) streams:
    Q(s,a) = V(s) + A(s,a) - mean(A(s,:))

This helps the agent learn which states are valuable regardless of action.

Usage:
    python -m RLM.dueling_dqn.train
    python -m RLM.dueling_dqn.train --total-timesteps 200000
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from RLM.shared.config import (
    DQN_CONFIG, NETWORK_CONFIG, TRAIN_CONFIG,
    PRODUCTS, NUM_FEATURES, NUM_ACTIONS, ENV_CONFIG,
)
from RLM.shared.data_loader import load_prices, load_trades
from RLM.shared.env import TradingEnv
from RLM.double_dqn.train import compute_normalization_params


class DuelingQNetwork(nn.Module):
    """Dueling DQN architecture with separate value and advantage streams."""

    def __init__(self, input_dim, action_dim, hidden_sizes=None):
        super().__init__()
        hidden_sizes = hidden_sizes or [64, 64]

        # Shared feature layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.ReLU(),
        )

        # Value stream: estimates V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )

        # Advantage stream: estimates A(s, a) for each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], action_dim),
        )

    def forward(self, x):
        shared = self.shared(x)
        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values


def train(args):
    """Run Dueling DQN training."""
    print("=" * 60)
    print("Dueling DQN Training - IMC Prosperity 4")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    prices_df = load_prices()
    trades_df = load_trades()

    # Compute normalization
    print("\nComputing normalization parameters...")
    feat_means, feat_stds = compute_normalization_params(
        prices_df, trades_df, PRODUCTS, TRAIN_CONFIG["train_days"]
    )

    # Create environments
    train_env = TradingEnv(
        prices_df=prices_df, trades_df=trades_df,
        products=PRODUCTS, day=TRAIN_CONFIG["train_days"][0],
        augment=True, seed=args.seed,
    )
    eval_env = TradingEnv(
        prices_df=prices_df, trades_df=trades_df,
        products=PRODUCTS, day=TRAIN_CONFIG["eval_days"][0],
        augment=False, seed=args.seed + 1,
    )

    for product in PRODUCTS:
        train_env.feature_computers[product].feature_means = feat_means
        train_env.feature_computers[product].feature_stds = feat_stds
        eval_env.feature_computers[product].feature_means = feat_means
        eval_env.feature_computers[product].feature_stds = feat_stds

    model_dir = os.path.join(os.path.dirname(__file__), "policy_weights")
    os.makedirs(model_dir, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=model_dir,
        eval_freq=max(args.total_timesteps // 20, 1000),
        n_eval_episodes=TRAIN_CONFIG["n_eval_episodes"],
        deterministic=True,
        verbose=1,
    )

    # Dueling architecture via policy_kwargs
    # SB3 supports dueling natively through the "dueling" option
    obs_dim = NUM_FEATURES * len(PRODUCTS)
    action_dim = NUM_ACTIONS ** len(PRODUCTS)

    policy_kwargs = dict(
        net_arch=NETWORK_CONFIG["hidden_sizes"],
        # SB3 doesn't have native dueling, so we use standard MLP
        # but train with the dueling concept in mind via custom features_extractor
        # For true dueling, we'd need a custom policy class.
        # Here we approximate with a wider network.
    )

    print("\nInitializing Dueling DQN...")
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

    # Replace q_net with dueling architecture
    dueling_net = DuelingQNetwork(obs_dim, action_dim, NETWORK_CONFIG["hidden_sizes"])
    model.q_net.q_net = dueling_net.to(model.device)
    model.q_net_target.q_net = DuelingQNetwork(obs_dim, action_dim, NETWORK_CONFIG["hidden_sizes"]).to(model.device)
    # Copy weights to target
    model.q_net_target.load_state_dict(model.q_net.state_dict())

    # Rebuild optimizer with new parameters
    model.policy.optimizer = torch.optim.Adam(
        model.q_net.parameters(),
        lr=args.lr or DQN_CONFIG["learning_rate"],
    )

    print(f"\n  Device: {model.device}")
    print(f"  Architecture: Dueling (Value + Advantage streams)")
    print(f"  Total timesteps: {args.total_timesteps}")

    # Train
    print("\nTraining...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)

    np.savez(
        os.path.join(model_dir, "norm_params.npz"),
        feature_means=feat_means,
        feature_stds=feat_stds,
    )

    print(f"\nModel saved to: {final_path}")
    print("Training complete!")
    return model, feat_means, feat_stds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dueling DQN")
    parser.add_argument("--total-timesteps", type=int, default=DQN_CONFIG["total_timesteps"])
    parser.add_argument("--seed", type=int, default=TRAIN_CONFIG["seed"])
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    train(args)
