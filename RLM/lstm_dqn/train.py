"""
LSTM DQN training for IMC Prosperity 4.

Uses a custom LSTM-based Q-network that captures temporal patterns
across timesteps without needing hand-crafted momentum features.

This model maintains hidden state across timesteps within an episode,
making it naturally suited for sequential trading decisions.

Note: This uses a custom PyTorch training loop rather than SB3,
since SB3's DQN doesn't natively support recurrent networks.
For recurrent policies with SB3, see sb3-contrib's RecurrentPPO.

Usage:
    python -m RLM.lstm_dqn.train
    python -m RLM.lstm_dqn.train --total-timesteps 100000 --hidden-size 64
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from RLM.shared.config import (
    DQN_CONFIG, TRAIN_CONFIG, PRODUCTS, NUM_FEATURES, NUM_ACTIONS, ENV_CONFIG,
)
from RLM.shared.data_loader import load_prices, load_trades
from RLM.shared.env import TradingEnv
from RLM.double_dqn.train import compute_normalization_params


class LSTMQNetwork(nn.Module):
    """LSTM-based Q-network for sequential decision making."""

    def __init__(self, input_dim, action_dim, hidden_size=64, head_hidden=64):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, action_dim),
        )

    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch, seq_len, input_dim) or (batch, input_dim)
            hidden: (h, c) tuple or None
        Returns:
            q_values: (batch, action_dim)
            hidden: (h, c) tuple
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)

        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)

        # Use last timestep output
        last_out = lstm_out[:, -1, :]
        q_values = self.head(last_out)
        return q_values, hidden

    def init_hidden(self, batch_size=1, device="cpu"):
        h = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return (h, c)


class SequenceReplayBuffer:
    """Replay buffer that stores sequences of transitions for LSTM training."""

    def __init__(self, capacity, seq_len=10):
        self.buffer = deque(maxlen=capacity)
        self.seq_len = seq_len
        self.current_episode = []

    def add(self, state, action, reward, next_state, done):
        self.current_episode.append((state, action, reward, next_state, done))
        if done:
            # Store overlapping sequences from the episode
            ep = self.current_episode
            for i in range(0, len(ep), self.seq_len // 2):
                seq = ep[i:i + self.seq_len]
                if len(seq) >= 2:
                    self.buffer.append(seq)
            self.current_episode = []

    def sample(self, batch_size):
        sequences = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return sequences

    def __len__(self):
        return len(self.buffer)


def train(args):
    """Train LSTM DQN with custom training loop."""
    print("=" * 60)
    print("LSTM DQN Training - IMC Prosperity 4")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"  Device: {device}")

    prices_df = load_prices()
    trades_df = load_trades()

    print("\nComputing normalization parameters...")
    feat_means, feat_stds = compute_normalization_params(
        prices_df, trades_df, PRODUCTS, TRAIN_CONFIG["train_days"]
    )

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

    obs_dim = NUM_FEATURES * len(PRODUCTS)
    action_dim = NUM_ACTIONS ** len(PRODUCTS)

    # Create networks
    q_net = LSTMQNetwork(obs_dim, action_dim, args.hidden_size).to(device)
    target_net = LSTMQNetwork(obs_dim, action_dim, args.hidden_size).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=args.lr or DQN_CONFIG["learning_rate"])
    replay_buffer = SequenceReplayBuffer(DQN_CONFIG["buffer_size"], seq_len=args.seq_len)

    model_dir = os.path.join(os.path.dirname(__file__), "policy_weights")
    os.makedirs(model_dir, exist_ok=True)

    # Training loop
    total_steps = 0
    best_eval_pnl = -float("inf")
    epsilon = DQN_CONFIG["exploration_initial_eps"]
    eps_decay = (DQN_CONFIG["exploration_initial_eps"] - DQN_CONFIG["exploration_final_eps"]) / \
                (args.total_timesteps * DQN_CONFIG["exploration_fraction"])

    print(f"\n  Hidden size: {args.hidden_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Total timesteps: {args.total_timesteps}")
    print("\nTraining...")

    episode = 0
    while total_steps < args.total_timesteps:
        obs, info = train_env.reset()
        hidden = q_net.init_hidden(1, device)
        done = False
        ep_reward = 0

        while not done and total_steps < args.total_timesteps:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = train_env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    q_values, hidden = q_net(obs_t, hidden)
                    action = q_values.argmax(dim=-1).item()

            next_obs, reward, terminated, truncated, info = train_env.step(action)
            done = terminated or truncated
            replay_buffer.add(obs, action, reward, next_obs, done)

            obs = next_obs
            ep_reward += reward
            total_steps += 1
            epsilon = max(DQN_CONFIG["exploration_final_eps"], epsilon - eps_decay)

            # Training step
            if len(replay_buffer) >= DQN_CONFIG["batch_size"] and total_steps % DQN_CONFIG["train_freq"] == 0:
                sequences = replay_buffer.sample(DQN_CONFIG["batch_size"])

                loss_total = 0
                for seq in sequences:
                    states = torch.FloatTensor([s[0] for s in seq]).unsqueeze(0).to(device)
                    actions = torch.LongTensor([s[1] for s in seq]).to(device)
                    rewards = torch.FloatTensor([s[2] for s in seq]).to(device)
                    next_states = torch.FloatTensor([s[3] for s in seq]).unsqueeze(0).to(device)
                    dones = torch.FloatTensor([float(s[4]) for s in seq]).to(device)

                    h = q_net.init_hidden(1, device)
                    q_values, _ = q_net(states, h)
                    q_values = q_values.squeeze(0)

                    # Only use last transition for loss
                    q_value = q_values[actions[-1]]

                    with torch.no_grad():
                        h_t = target_net.init_hidden(1, device)
                        next_q, _ = target_net(next_states, h_t)
                        next_q = next_q.squeeze(0)
                        target = rewards[-1] + DQN_CONFIG["gamma"] * next_q.max() * (1 - dones[-1])

                    loss_total += (q_value - target) ** 2

                loss = loss_total / len(sequences)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimizer.step()

            # Update target network
            if total_steps % DQN_CONFIG["target_update_interval"] == 0:
                target_net.load_state_dict(q_net.state_dict())

        episode += 1
        if episode % 5 == 0:
            print(f"  Episode {episode}: steps={total_steps}, reward={ep_reward:.2f}, eps={epsilon:.3f}")

        # Periodic evaluation
        if episode % 10 == 0:
            eval_pnl = _evaluate_episode(eval_env, q_net, device)
            print(f"  [EVAL] PnL={eval_pnl:.2f}")

            if eval_pnl > best_eval_pnl:
                best_eval_pnl = eval_pnl
                torch.save(q_net.state_dict(), os.path.join(model_dir, "best_model.pt"))
                print(f"  [EVAL] New best! Saved.")

    # Save final model
    torch.save(q_net.state_dict(), os.path.join(model_dir, "final_model.pt"))

    # Save metadata
    np.savez(
        os.path.join(model_dir, "norm_params.npz"),
        feature_means=feat_means,
        feature_stds=feat_stds,
    )
    np.savez(
        os.path.join(model_dir, "model_config.npz"),
        hidden_size=np.array(args.hidden_size),
        obs_dim=np.array(obs_dim),
        action_dim=np.array(action_dim),
    )

    print(f"\nTraining complete! Best eval PnL: {best_eval_pnl:.2f}")
    print(f"Models saved in: {model_dir}")


def _evaluate_episode(env, q_net, device):
    """Run one evaluation episode."""
    obs, _ = env.reset()
    hidden = q_net.init_hidden(1, device)
    done = False
    total_reward = 0

    while not done:
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            q_values, hidden = q_net(obs_t, hidden)
            action = q_values.argmax(dim=-1).item()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    return info.get("pnl", total_reward)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM DQN")
    parser.add_argument("--total-timesteps", type=int, default=DQN_CONFIG["total_timesteps"])
    parser.add_argument("--seed", type=int, default=TRAIN_CONFIG["seed"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    train(args)
