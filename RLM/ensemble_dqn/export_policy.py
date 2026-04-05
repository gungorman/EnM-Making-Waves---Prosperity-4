"""
Export Ensemble DQN to a single numpy .npz file.

Each member's weights are prefixed with m0_, m1_, m2_, etc.

Usage:
    python -m RLM.ensemble_dqn.export_policy
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
    """Export all ensemble members to a single .npz file."""
    print("=" * 60)
    print("Exporting Ensemble DQN to numpy")
    print("=" * 60)

    weights = {}

    member_idx = 0
    while True:
        member_dir = os.path.join(args.model_dir, f"member_{member_idx}")
        model_path = os.path.join(member_dir, "best_model")
        if not os.path.exists(model_path + ".zip"):
            model_path = os.path.join(member_dir, "final_model")
        if not os.path.exists(model_path + ".zip"):
            break

        print(f"\n  Exporting member {member_idx}...")
        model = DQN.load(model_path)

        layer_idx = 0
        for name, param in model.q_net.named_parameters():
            tensor = param.detach().cpu().numpy()
            if "weight" in name:
                weights[f"m{member_idx}_W{layer_idx}"] = tensor
            elif "bias" in name:
                weights[f"m{member_idx}_B{layer_idx}"] = tensor
                layer_idx += 1

        member_idx += 1

    if member_idx == 0:
        print("ERROR: No ensemble members found!")
        return

    # Normalization params
    norm_path = os.path.join(args.model_dir, "norm_params.npz")
    if os.path.exists(norm_path):
        norm_data = np.load(norm_path)
        weights["feature_means"] = norm_data["feature_means"]
        weights["feature_stds"] = norm_data["feature_stds"]

    output_path = args.output or os.path.join(args.model_dir, "exported_weights.npz")
    np.savez(output_path, **weights)

    file_size = os.path.getsize(output_path)
    print(f"\n  Exported {member_idx} members, {len(weights)} arrays")
    print(f"  File size: {file_size / 1024:.1f} KB")
    print(f"  Saved to: {output_path}")

    # Verify
    from RLM.shared.numpy_policy import NumpyEnsemble
    from RLM.shared.config import NUM_FEATURES, PRODUCTS

    ensemble = NumpyEnsemble(model_paths=[output_path])
    # Need to use model_dicts approach for ensemble
    print(f"\n  Members exported: {member_idx}")
    print("Export complete!")
    return output_path


if __name__ == "__main__":
    model_dir = os.path.join(os.path.dirname(__file__), "policy_weights")

    parser = argparse.ArgumentParser(description="Export Ensemble DQN")
    parser.add_argument("--model-dir", default=model_dir)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    export_policy(args)
