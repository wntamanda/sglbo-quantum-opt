from __future__ import annotations
from typing import Dict, Any
import numpy as np
from sklearn import datasets, preprocessing, model_selection
from qcbo.models.pl_qnn import build_pl_qnode, make_pl_predict_fn

def _load_split(seed: int = 42, test_size: float = 0.2):
    X, y = datasets.load_iris(return_X_y=True)
    y = (y == 0).astype(int)
    X = preprocessing.StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    return X_train, X_test, y_train, y_test

def build_problem_pl(*, batch_size=16, noise_tag="noiseless", seed=42, ansatz_reps=1):
    X_train, X_test, y_train, y_test = _load_split(seed=seed)
    n_qubits = X_train.shape[1]

    qnode, n_params, p_ro = build_pl_qnode(n_qubits, ansatz_reps, noise_tag=noise_tag)

    predict_raw = make_pl_predict_fn(qnode)

    # wrap with readout flip (p' = (1-ε)p + ε(1-p))
    def predict_fn(theta, X, shots=None):
        probs = predict_raw(theta, X, shots=shots)
        if p_ro and p_ro > 0.0:
            probs = (1 - p_ro) * probs + p_ro * (1 - probs)
        return probs

    # backend-neutral cost that uses predict_fn and respects shots
    rng = np.random.default_rng(seed)
    n = len(X_train)
    def cost_fn(theta, shots, return_var=False, eps=1e-9):
        idx = rng.choice(n, batch_size, replace=False)
        Xb, yb = X_train[idx], y_train[idx]
        probs = np.clip(predict_fn(theta, Xb, shots=shots), eps, 1 - eps)
        loss = -np.mean(yb * np.log(probs) + (1 - yb) * np.log(1 - probs))
        if return_var:
            var = float(np.mean(probs * (1 - probs) / max(1, shots)))
            return float(loss), var
        return float(loss)

    # placeholders
    qnn = None
    estimator = None
    set_eval_shots_fn = lambda shots: None

    model_info: Dict[str, Any] = dict(
        num_qubits=n_qubits, n_params=n_params, backend="pennylane", ansatz_reps=ansatz_reps
    )
    meta: Dict[str, Any] = dict(
        dataset="iris", num_qubits=n_qubits, ansatz_reps=ansatz_reps,
        batch_size=batch_size, noise_tag=noise_tag, seed=seed, backend="pennylane"
    )
    splits = (X_train, y_train, X_test, y_test)
    return qnn, estimator, cost_fn, splits, meta, model_info, predict_fn, set_eval_shots_fn
