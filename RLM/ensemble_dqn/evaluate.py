"""
Evaluate an Ensemble DQN (averages Q-values across members).

Usage:
    python -m RLM.ensemble_dqn.evaluate
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


class EnsemblePredictor:
    """Wraps multiple DQN models for ensemble prediction."""

    def __init__(self, models):
        self.models = models

    def predict(self, obs, deterministic=True):
        all_q = []
        for model in self.models:
            # Get Q-values from each model
            obs_tensor = model.q_net.obs_to_tensor(obs.reshape(1, -1))[0]
            q_values = model.q_net(obs_tensor).detach().cpu().numpy()[0]
            all_q.append(q_values)

        avg_q = np.mean(all_q, axis=0)
        action = np.argmax(avg_q)
        return action, None


def evaluate(args):
    """Evaluate ensemble on held-out data."""
    print("=" * 60)
    print("Ensemble DQN Evaluation")
    print("=" * 60)

    prices_df = load_prices()
    trades_df = load_trades()

    # Load normalization params
    norm_path = os.path.join(args.model_dir, "norm_params.npz")
    if os.path.exists(norm_path):
        norm_data = np.load(norm_path)
        feat_means = norm_data["feature_means"]
        feat_stds = norm_data["feature_stds"]
    else:
        feat_means, feat_stds = np.zeros(19), np.ones(19)

    # Load all ensemble members
    models = []
    member_idx = 0
    while True:
        member_dir = os.path.join(args.model_dir, f"member_{member_idx}")
        model_path = os.path.join(member_dir, "best_model")
        if not os.path.exists(model_path + ".zip"):
            model_path = os.path.join(member_dir, "final_model")
        if not os.path.exists(model_path + ".zip"):
            break
        print(f"  Loading member {member_idx}: {model_path}")
        models.append(DQN.load(model_path))
        member_idx += 1

    if not models:
        print("ERROR: No ensemble members found!")
        return

    print(f"\nLoaded {len(models)} ensemble members")
    ensemble = EnsemblePredictor(models)

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
            total_reward = 0
            done = False

            while not done:
                action, _ = ensemble.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated

            episode_pnls.append(info.get("pnl", total_reward))
            print(f"  Episode {ep + 1}: PnL={info.get('pnl', total_reward):.2f}")

        pnls = np.array(episode_pnls)
        print(f"\n  Mean PnL: {pnls.mean():.2f}, Std: {pnls.std():.2f}")


if __name__ == "__main__":
    model_dir = os.path.join(os.path.dirname(__file__), "policy_weights")

    parser = argparse.ArgumentParser(description="Evaluate Ensemble DQN")
    parser.add_argument("--model-dir", default=model_dir)
    parser.add_argument("--eval-days", type=int, nargs="+", default=TRAIN_CONFIG["eval_days"])
    parser.add_argument("--n-episodes", type=int, default=10)
    args = parser.parse_args()

    evaluate(args)
