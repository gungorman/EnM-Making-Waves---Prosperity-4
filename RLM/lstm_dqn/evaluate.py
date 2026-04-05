"""
Evaluate a trained LSTM DQN model.

Usage:
    python -m RLM.lstm_dqn.evaluate
"""

import argparse
import os
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from RLM.shared.config import PRODUCTS, TRAIN_CONFIG, NUM_FEATURES, NUM_ACTIONS
from RLM.shared.data_loader import load_prices, load_trades
from RLM.shared.env import TradingEnv
from RLM.lstm_dqn.train import LSTMQNetwork


def evaluate(args):
    """Evaluate LSTM DQN on held-out data."""
    print("=" * 60)
    print("LSTM DQN Evaluation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prices_df = load_prices()
    trades_df = load_trades()

    # Load model config
    weights_dir = args.model_dir
    config_path = os.path.join(weights_dir, "model_config.npz")
    if os.path.exists(config_path):
        cfg = np.load(config_path)
        hidden_size = int(cfg["hidden_size"])
        obs_dim = int(cfg["obs_dim"])
        action_dim = int(cfg["action_dim"])
    else:
        hidden_size = 64
        obs_dim = NUM_FEATURES * len(PRODUCTS)
        action_dim = NUM_ACTIONS ** len(PRODUCTS)

    # Load normalization
    norm_path = os.path.join(weights_dir, "norm_params.npz")
    if os.path.exists(norm_path):
        norm_data = np.load(norm_path)
        feat_means = norm_data["feature_means"]
        feat_stds = norm_data["feature_stds"]
    else:
        feat_means, feat_stds = np.zeros(19), np.ones(19)

    # Load model
    model_path = os.path.join(weights_dir, "best_model.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(weights_dir, "final_model.pt")

    print(f"  Loading model from: {model_path}")
    q_net = LSTMQNetwork(obs_dim, action_dim, hidden_size).to(device)
    q_net.load_state_dict(torch.load(model_path, map_location=device))
    q_net.eval()

    for day in args.eval_days:
        print(f"\n--- Evaluating on Day {day} ---")

        env = TradingEnv(
            prices_df=prices_df, trades_df=trades_df,
            products=PRODUCTS, day=day, augment=False,
        )
        for product in PRODUCTS:
            env.feature_computers[product].feature_means = feat_means
            env.feature_computers[product].feature_stds = feat_stds

        episode_pnls = []

        for ep in range(args.n_episodes):
            obs, info = env.reset()
            hidden = q_net.init_hidden(1, device)
            done = False

            while not done:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    q_values, hidden = q_net(obs_t, hidden)
                    action = q_values.argmax(dim=-1).item()

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            pnl = info.get("pnl", 0)
            episode_pnls.append(pnl)
            print(f"  Episode {ep + 1}: PnL={pnl:.2f}")

        pnls = np.array(episode_pnls)
        print(f"\n  Mean PnL: {pnls.mean():.2f}, Std: {pnls.std():.2f}")


if __name__ == "__main__":
    model_dir = os.path.join(os.path.dirname(__file__), "policy_weights")

    parser = argparse.ArgumentParser(description="Evaluate LSTM DQN")
    parser.add_argument("--model-dir", default=model_dir)
    parser.add_argument("--eval-days", type=int, nargs="+", default=TRAIN_CONFIG["eval_days"])
    parser.add_argument("--n-episodes", type=int, default=10)
    args = parser.parse_args()

    evaluate(args)
