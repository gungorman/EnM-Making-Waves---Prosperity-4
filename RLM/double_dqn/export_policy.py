"""
Export trained Double DQN (SB3) model to numpy .npz format.

Extracts MLP weights from the PyTorch model and saves them as numpy arrays
compatible with shared/numpy_policy.py and build_submission.py.

Usage:
    python -m RLM.double_dqn.export_policy
    python -m RLM.double_dqn.export_policy --model-path RLM/double_dqn/policy_weights/best_model
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
    """Export SB3 DQN model weights to numpy .npz format."""
    print("=" * 60)
    print("Exporting Double DQN to numpy")
    print("=" * 60)

    # Load SB3 model
    print(f"\nLoading model from: {args.model_path}")
    model = DQN.load(args.model_path)

    # Extract Q-network weights
    q_net = model.q_net
    print(f"\nQ-Network architecture:")
    print(q_net)

    weights = {}

    # SB3 DQN uses q_net.q_net (the inner MLP)
    # Architecture: features_extractor -> q_net layers
    layer_idx = 0
    for name, param in q_net.named_parameters():
        tensor = param.detach().cpu().numpy()
        print(f"  {name}: {tensor.shape}")

        # Map SB3 parameter names to our convention
        # SB3 names: q_net.0.weight, q_net.0.bias, q_net.2.weight, etc.
        if "weight" in name:
            weights[f"W{layer_idx}"] = tensor
        elif "bias" in name:
            weights[f"B{layer_idx}"] = tensor
            layer_idx += 1

    # Load normalization params
    weights_dir = os.path.dirname(args.model_path)
    norm_path = os.path.join(weights_dir, "norm_params.npz")
    if os.path.exists(norm_path):
        norm_data = np.load(norm_path)
        weights["feature_means"] = norm_data["feature_means"]
        weights["feature_stds"] = norm_data["feature_stds"]
        print(f"\n  Normalization params loaded from: {norm_path}")
    else:
        print(f"\n  WARNING: {norm_path} not found, normalization params not included")

    # Save
    output_path = args.output or os.path.join(weights_dir, "exported_weights.npz")
    np.savez(output_path, **weights)

    # Report
    total_params = sum(v.size for v in weights.values()
                       if not v.shape == () and "feature" not in str(v.dtype))
    file_size = os.path.getsize(output_path)
    print(f"\n  Exported {len(weights)} arrays")
    print(f"  Total parameters: {total_params:,}")
    print(f"  File size: {file_size / 1024:.1f} KB")
    print(f"  Saved to: {output_path}")

    # Verify by loading and doing a forward pass
    print("\n  Verifying export...")
    from RLM.shared.numpy_policy import NumpyMLP
    from RLM.shared.config import NUM_FEATURES, PRODUCTS

    numpy_model = NumpyMLP(weights_path=output_path)
    test_input = np.random.randn(NUM_FEATURES * len(PRODUCTS)).astype(np.float32)

    action, q_values = numpy_model.predict(test_input, normalize=False)
    print(f"  Test forward pass: action={action}, Q-values shape={q_values.shape}")
    print(f"  Q-values: {q_values[:5]}...")

    print("\nExport complete!")
    return output_path


if __name__ == "__main__":
    model_dir = os.path.join(os.path.dirname(__file__), "policy_weights")

    parser = argparse.ArgumentParser(description="Export Double DQN to numpy")
    parser.add_argument("--model-path", default=os.path.join(model_dir, "best_model"))
    parser.add_argument("--output", default=None, help="Output .npz path")
    args = parser.parse_args()

    export_policy(args)
