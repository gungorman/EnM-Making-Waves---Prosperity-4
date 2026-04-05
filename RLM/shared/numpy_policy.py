"""
Pure-numpy inference for DQN policies.
Used in the final submission where only stdlib + numpy are allowed.

Supports:
- Standard MLP (Double DQN, Vanilla DQN)
- Dueling MLP (Dueling DQN)
- Ensemble of multiple MLPs (Ensemble DQN)
- LSTM + MLP (LSTM DQN)
"""

import numpy as np


def relu(x):
    return np.maximum(0, x)


class NumpyMLP:
    """Standard MLP forward pass using numpy.

    Weights are loaded from a .npz file exported from PyTorch/SB3.
    """

    def __init__(self, weights_path=None, weights_dict=None):
        """
        Args:
            weights_path: Path to .npz file with weight arrays.
            weights_dict: Dict of weight arrays (alternative to file).
        """
        if weights_path is not None:
            data = np.load(weights_path)
            self.weights = {k: data[k] for k in data.files}
        elif weights_dict is not None:
            self.weights = weights_dict
        else:
            self.weights = {}

        self.feature_means = self.weights.get("feature_means", np.zeros(1))
        self.feature_stds = self.weights.get("feature_stds", np.ones(1))

    def normalize(self, features, clip=3.0):
        """Z-score normalize features."""
        normed = (features - self.feature_means) / (self.feature_stds + 1e-8)
        return np.clip(normed, -clip, clip)

    def forward(self, features):
        """Forward pass through MLP layers.

        Expects weights named: W0, B0, W1, B1, ..., W_out, B_out
        """
        x = features.astype(np.float32)

        # Find all layer indices
        layer_idx = 0
        while f"W{layer_idx}" in self.weights:
            W = self.weights[f"W{layer_idx}"]
            B = self.weights[f"B{layer_idx}"]
            x = x @ W.T + B
            # Apply ReLU to all hidden layers (not the output)
            if f"W{layer_idx + 1}" in self.weights:
                x = relu(x)
            layer_idx += 1

        return x

    def predict(self, features, normalize=True):
        """Get action from features.

        Args:
            features: np.array of raw features.
            normalize: Whether to z-score normalize first.

        Returns:
            action: int, the argmax action.
            q_values: np.array of Q-values for all actions.
        """
        if normalize:
            features = self.normalize(features)

        q_values = self.forward(features)
        action = int(np.argmax(q_values))
        return action, q_values


class NumpyDuelingMLP:
    """Dueling DQN forward pass: splits into value and advantage streams.

    Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    """

    def __init__(self, weights_path=None, weights_dict=None):
        if weights_path is not None:
            data = np.load(weights_path)
            self.weights = {k: data[k] for k in data.files}
        elif weights_dict is not None:
            self.weights = weights_dict
        else:
            self.weights = {}

        self.feature_means = self.weights.get("feature_means", np.zeros(1))
        self.feature_stds = self.weights.get("feature_stds", np.ones(1))

    def normalize(self, features, clip=3.0):
        normed = (features - self.feature_means) / (self.feature_stds + 1e-8)
        return np.clip(normed, -clip, clip)

    def forward(self, features):
        x = features.astype(np.float32)

        # Shared layers
        layer_idx = 0
        while f"shared_W{layer_idx}" in self.weights:
            W = self.weights[f"shared_W{layer_idx}"]
            B = self.weights[f"shared_B{layer_idx}"]
            x = relu(x @ W.T + B)
            layer_idx += 1

        # Value stream
        v = x
        v_idx = 0
        while f"value_W{v_idx}" in self.weights:
            W = self.weights[f"value_W{v_idx}"]
            B = self.weights[f"value_B{v_idx}"]
            v = v @ W.T + B
            if f"value_W{v_idx + 1}" in self.weights:
                v = relu(v)
            v_idx += 1

        # Advantage stream
        a = x
        a_idx = 0
        while f"advantage_W{a_idx}" in self.weights:
            W = self.weights[f"advantage_W{a_idx}"]
            B = self.weights[f"advantage_B{a_idx}"]
            a = a @ W.T + B
            if f"advantage_W{a_idx + 1}" in self.weights:
                a = relu(a)
            a_idx += 1

        # Combine: Q = V + A - mean(A)
        q_values = v + a - np.mean(a, axis=-1, keepdims=True)
        return q_values

    def predict(self, features, normalize=True):
        if normalize:
            features = self.normalize(features)
        q_values = self.forward(features)
        action = int(np.argmax(q_values))
        return action, q_values


class NumpyEnsemble:
    """Ensemble of multiple MLP policies. Averages Q-values for voting."""

    def __init__(self, model_paths=None, model_dicts=None, model_class=NumpyMLP):
        """
        Args:
            model_paths: List of .npz file paths.
            model_dicts: List of weight dicts.
            model_class: NumpyMLP or NumpyDuelingMLP.
        """
        self.models = []
        if model_paths:
            for path in model_paths:
                self.models.append(model_class(weights_path=path))
        elif model_dicts:
            for d in model_dicts:
                self.models.append(model_class(weights_dict=d))

    def predict(self, features, normalize=True):
        """Average Q-values across ensemble members."""
        all_q = []
        for model in self.models:
            _, q_values = model.predict(features, normalize=normalize)
            all_q.append(q_values)

        avg_q = np.mean(all_q, axis=0)
        action = int(np.argmax(avg_q))
        return action, avg_q


class NumpyLSTM:
    """LSTM + MLP forward pass in pure numpy.

    Maintains hidden state across timesteps for sequential inference.
    """

    def __init__(self, weights_path=None, weights_dict=None):
        if weights_path is not None:
            data = np.load(weights_path)
            self.weights = {k: data[k] for k in data.files}
        elif weights_dict is not None:
            self.weights = weights_dict
        else:
            self.weights = {}

        self.feature_means = self.weights.get("feature_means", np.zeros(1))
        self.feature_stds = self.weights.get("feature_stds", np.ones(1))

        # LSTM state
        hidden_size = self.weights.get("lstm_hidden_size", np.array(64)).item()
        self.hidden_size = int(hidden_size)
        self.h = np.zeros(self.hidden_size, dtype=np.float32)
        self.c = np.zeros(self.hidden_size, dtype=np.float32)

    def normalize(self, features, clip=3.0):
        normed = (features - self.feature_means) / (self.feature_stds + 1e-8)
        return np.clip(normed, -clip, clip)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

    def _lstm_step(self, x):
        """Single LSTM timestep."""
        W_ii = self.weights["lstm_W_ii"]
        W_hi = self.weights["lstm_W_hi"]
        b_i = self.weights["lstm_b_i"]

        W_if = self.weights["lstm_W_if"]
        W_hf = self.weights["lstm_W_hf"]
        b_f = self.weights["lstm_b_f"]

        W_ig = self.weights["lstm_W_ig"]
        W_hg = self.weights["lstm_W_hg"]
        b_g = self.weights["lstm_b_g"]

        W_io = self.weights["lstm_W_io"]
        W_ho = self.weights["lstm_W_ho"]
        b_o = self.weights["lstm_b_o"]

        i = self._sigmoid(x @ W_ii.T + self.h @ W_hi.T + b_i)
        f = self._sigmoid(x @ W_if.T + self.h @ W_hf.T + b_f)
        g = np.tanh(x @ W_ig.T + self.h @ W_hg.T + b_g)
        o = self._sigmoid(x @ W_io.T + self.h @ W_ho.T + b_o)

        self.c = f * self.c + i * g
        self.h = o * np.tanh(self.c)

        return self.h

    def forward(self, features):
        x = features.astype(np.float32)

        # LSTM layer
        h = self._lstm_step(x)

        # MLP head
        layer_idx = 0
        while f"head_W{layer_idx}" in self.weights:
            W = self.weights[f"head_W{layer_idx}"]
            B = self.weights[f"head_B{layer_idx}"]
            h = h @ W.T + B
            if f"head_W{layer_idx + 1}" in self.weights:
                h = relu(h)
            layer_idx += 1

        return h

    def predict(self, features, normalize=True):
        if normalize:
            features = self.normalize(features)
        q_values = self.forward(features)
        action = int(np.argmax(q_values))
        return action, q_values

    def reset_state(self):
        """Reset LSTM hidden state (call at start of new episode)."""
        self.h = np.zeros(self.hidden_size, dtype=np.float32)
        self.c = np.zeros(self.hidden_size, dtype=np.float32)
