"""
Evaluate a trained Double DQN model on held-out data.

Usage:
    python -m RLM.double_dqn.evaluate
    python -m RLM.double_dqn.evaluate --model-path RLM/double_dqn/policy_weights/best_model
"""

import argparse
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import DQN

from RLM.shared.config import PRODUCTS, TRAIN_CONFIG
from RLM.shared.data_loader import load_prices, load_trades
from RLM.shared.env import TradingEnv


def evaluate(args):
    """Evaluate trained model on evaluation days."""
    print("=" * 60)
    print("Double DQN Evaluation")
    print("=" * 60)

    # Load data
    prices_df = load_prices()
    trades_df = load_trades()

    # Load normalization params
    weights_dir = os.path.dirname(args.model_path)
    norm_path = os.path.join(weights_dir, "norm_params.npz")
    if os.path.exists(norm_path):
        norm_data = np.load(norm_path)
        feat_means = norm_data["feature_means"]
        feat_stds = norm_data["feature_stds"]
    else:
        print("WARNING: norm_params.npz not found, using defaults")
        feat_means = np.zeros(19)
        feat_stds = np.ones(19)

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model = DQN.load(args.model_path)

    # Evaluate on each eval day
    for day in args.eval_days:
        print(f"\n--- Evaluating on Day {day} ---")

        env = TradingEnv(
            prices_df=prices_df,
            trades_df=trades_df,
            products=PRODUCTS,
            day=day,
            augment=False,
        )
        for product in PRODUCTS:
            env.feature_computers[product].feature_means = feat_means
            env.feature_computers[product].feature_stds = feat_stds

        episode_pnls = []
        episode_trades = []

        for ep in range(args.n_episodes):
            obs, info = env.reset()
            total_reward = 0
            done = False
            trade_count = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated

                # Count non-hold actions
                if action != 0:
                    trade_count += 1

            episode_pnls.append(info.get("pnl", total_reward))
            episode_trades.append(trade_count)

            print(f"  Episode {ep + 1}: PnL={info.get('pnl', total_reward):.2f}, "
                  f"Reward={total_reward:.2f}, Trades={trade_count}, "
                  f"Final positions={info.get('positions', {})}")

        pnls = np.array(episode_pnls)
        print(f"\n  Summary (Day {day}):")
        print(f"    Mean PnL:    {pnls.mean():.2f}")
        print(f"    Std PnL:     {pnls.std():.2f}")
        print(f"    Min PnL:     {pnls.min():.2f}")
        print(f"    Max PnL:     {pnls.max():.2f}")
        if pnls.std() > 0:
            sharpe = pnls.mean() / pnls.std()
            print(f"    Sharpe:      {sharpe:.2f}")
        print(f"    Avg trades:  {np.mean(episode_trades):.0f}")


if __name__ == "__main__":
    model_dir = os.path.join(os.path.dirname(__file__), "policy_weights")

    parser = argparse.ArgumentParser(description="Evaluate Double DQN")
    parser.add_argument("--model-path", default=os.path.join(model_dir, "best_model"),
                        help="Path to saved SB3 model (without .zip)")
    parser.add_argument("--eval-days", type=int, nargs="+", default=TRAIN_CONFIG["eval_days"])
    parser.add_argument("--n-episodes", type=int, default=10)
    args = parser.parse_args()

    evaluate(args)
