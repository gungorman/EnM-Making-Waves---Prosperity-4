"""
Export trained Dueling DQN to numpy .npz format.

Maps the dueling architecture (shared + value + advantage streams)
to the naming convention expected by NumpyDuelingMLP.

Usage:
    python -m RLM.dueling_dqn.export_policy
"""

import argparse
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import DQN


def export_policy(args):
    """Export Dueling DQN weights with stream-specific naming."""
    print("=" * 60)
    print("Exporting Dueling DQN to numpy")
    print("=" * 60)

    print(f"\nLoading model from: {args.model_path}")
    model = DQN.load(args.model_path)

    q_net = model.q_net
    weights = {}

    # The dueling net has: shared, value_stream, advantage_stream
    dueling = q_net.q_net  # Our custom DuelingQNetwork

    # Shared layers
    shared_idx = 0
    for name, param in dueling.shared.named_parameters():
        tensor = param.detach().cpu().numpy()
        if "weight" in name:
            weights[f"shared_W{shared_idx}"] = tensor
        elif "bias" in name:
            weights[f"shared_B{shared_idx}"] = tensor
            shared_idx += 1

    # Value stream
    val_idx = 0
    for name, param in dueling.value_stream.named_parameters():
        tensor = param.detach().cpu().numpy()
        if "weight" in name:
            weights[f"value_W{val_idx}"] = tensor
        elif "bias" in name:
            weights[f"value_B{val_idx}"] = tensor
            val_idx += 1

    # Advantage stream
    adv_idx = 0
    for name, param in dueling.advantage_stream.named_parameters():
        tensor = param.detach().cpu().numpy()
        if "weight" in name:
            weights[f"advantage_W{adv_idx}"] = tensor
        elif "bias" in name:
            weights[f"advantage_B{adv_idx}"] = tensor
            adv_idx += 1

    # Normalization params
    weights_dir = os.path.dirname(args.model_path)
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
    from RLM.shared.numpy_policy import NumpyDuelingMLP
    from RLM.shared.config import NUM_FEATURES, PRODUCTS

    numpy_model = NumpyDuelingMLP(weights_path=output_path)
    test_input = np.random.randn(NUM_FEATURES * len(PRODUCTS)).astype(np.float32)
    action, q_values = numpy_model.predict(test_input, normalize=False)
    print(f"  Verification: action={action}, Q-values shape={q_values.shape}")

    print("\nExport complete!")
    return output_path


if __name__ == "__main__":
    model_dir = os.path.join(os.path.dirname(__file__), "policy_weights")

    parser = argparse.ArgumentParser(description="Export Dueling DQN")
    parser.add_argument("--model-path", default=os.path.join(model_dir, "best_model"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    export_policy(args)
