"""
Export trained LSTM DQN to numpy .npz format.

Extracts LSTM gate weights and MLP head weights, using the naming
convention expected by NumpyLSTM in shared/numpy_policy.py.

Usage:
    python -m RLM.lstm_dqn.export_policy
"""

import argparse
import os
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from RLM.shared.config import NUM_FEATURES, PRODUCTS, NUM_ACTIONS
from RLM.lstm_dqn.train import LSTMQNetwork


def export_policy(args):
    """Export LSTM DQN to numpy .npz with gate-level weight naming."""
    print("=" * 60)
    print("Exporting LSTM DQN to numpy")
    print("=" * 60)

    weights_dir = args.model_dir

    # Load model config
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

    # Load model
    model_path = os.path.join(weights_dir, "best_model.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(weights_dir, "final_model.pt")

    print(f"  Loading model from: {model_path}")
    q_net = LSTMQNetwork(obs_dim, action_dim, hidden_size)
    q_net.load_state_dict(torch.load(model_path, map_location="cpu"))

    weights = {}
    weights["lstm_hidden_size"] = np.array(hidden_size)

    # Extract LSTM weights
    # PyTorch LSTM packs all 4 gates into single weight matrices
    # weight_ih_l0: (4*hidden, input) -- gates: i, f, g, o
    # weight_hh_l0: (4*hidden, hidden)
    # bias_ih_l0: (4*hidden,)
    # bias_hh_l0: (4*hidden,)
    lstm = q_net.lstm

    W_ih = lstm.weight_ih_l0.detach().cpu().numpy()
    W_hh = lstm.weight_hh_l0.detach().cpu().numpy()
    b_ih = lstm.bias_ih_l0.detach().cpu().numpy()
    b_hh = lstm.bias_hh_l0.detach().cpu().numpy()

    H = hidden_size
    gate_names = ["i", "f", "g", "o"]  # input, forget, cell, output

    for idx, gate in enumerate(gate_names):
        weights[f"lstm_W_i{gate}"] = W_ih[idx * H:(idx + 1) * H]
        weights[f"lstm_W_h{gate}"] = W_hh[idx * H:(idx + 1) * H]
        weights[f"lstm_b_{gate}"] = b_ih[idx * H:(idx + 1) * H] + b_hh[idx * H:(idx + 1) * H]

    # Extract MLP head weights
    head_idx = 0
    for name, param in q_net.head.named_parameters():
        tensor = param.detach().cpu().numpy()
        if "weight" in name:
            weights[f"head_W{head_idx}"] = tensor
        elif "bias" in name:
            weights[f"head_B{head_idx}"] = tensor
            head_idx += 1

    # Normalization params
    norm_path = os.path.join(weights_dir, "norm_params.npz")
    if os.path.exists(norm_path):
        norm_data = np.load(norm_path)
        weights["feature_means"] = norm_data["feature_means"]
        weights["feature_stds"] = norm_data["feature_stds"]

    output_path = args.output or os.path.join(weights_dir, "exported_weights.npz")
    np.savez(output_path, **weights)

    file_size = os.path.getsize(output_path)
    print(f"\n  Exported {len(weights)} arrays")
    print(f"  File size: {file_size / 1024:.1f} KB")
    print(f"  Saved to: {output_path}")

    # Verify
    from RLM.shared.numpy_policy import NumpyLSTM

    numpy_model = NumpyLSTM(weights_path=output_path)
    test_input = np.random.randn(obs_dim).astype(np.float32)
    action, q_values = numpy_model.predict(test_input, normalize=False)
    print(f"  Verification: action={action}, Q-values shape={q_values.shape}")

    print("\nExport complete!")
    return output_path


if __name__ == "__main__":
    model_dir = os.path.join(os.path.dirname(__file__), "policy_weights")

    parser = argparse.ArgumentParser(description="Export LSTM DQN")
    parser.add_argument("--model-dir", default=model_dir)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    export_policy(args)
