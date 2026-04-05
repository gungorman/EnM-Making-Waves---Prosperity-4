"""
Evaluate a trained Dueling DQN model.

Usage:
    python -m RLM.dueling_dqn.evaluate
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Reuse the Double DQN evaluator (same evaluation logic)
from RLM.double_dqn.evaluate import evaluate
from RLM.shared.config import TRAIN_CONFIG

if __name__ == "__main__":
    model_dir = os.path.join(os.path.dirname(__file__), "policy_weights")

    parser = argparse.ArgumentParser(description="Evaluate Dueling DQN")
    parser.add_argument("--model-path", default=os.path.join(model_dir, "best_model"))
    parser.add_argument("--eval-days", type=int, nargs="+", default=TRAIN_CONFIG["eval_days"])
    parser.add_argument("--n-episodes", type=int, default=10)
    args = parser.parse_args()

    evaluate(args)
