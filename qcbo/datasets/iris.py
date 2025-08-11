from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any

from sklearn import datasets, preprocessing, model_selection

from qcbo.noise.aer_noise import estimators_by_level
from qcbo.models.qnn import build_qnn, make_cost_fn


def _load_split(seed: int = 42, test_size: float = 0.2):
    X, y = datasets.load_iris(return_X_y=True)
    y = (y == 0).astype(int)  # class 0 vs rest
    X = preprocessing.StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    return X_train, X_test, y_train, y_test


def build_problem(
    *,
    batch_size: int = 16,
    noise_tag: str = "noiseless",
    seed: int = 42,
    ansatz_reps: int = 1,
):
    """
    Returns:
      qnn, estimator, cost_fn, (X_train, y_train, X_test, y_test), meta, model_info
    where model_info is the dict returned by your build_qnn (contains ansatz, feature_map, etc.)
    """
    X_train, X_test, y_train, y_test = _load_split(seed=seed)
    num_qubits = X_train.shape[1]

    # Estimator from Aer noise helper
    est_map = estimators_by_level(num_qubits)
    if noise_tag not in est_map:
        raise ValueError(f"Unknown noise tag '{noise_tag}'. Expected one of {list(est_map)}.")
    estimator = est_map[noise_tag]

    # QNN from qnn helper (returns a dict with qnn, ansatz, feature_map, qc, estimator, num_qubits)
    model_info = build_qnn(X_train, ansatz_reps=ansatz_reps, estimator=estimator)
    qnn = model_info["qnn"]

    # Cost function from qnn helper
    cost_fn = make_cost_fn(qnn, estimator, X_train, y_train, batch_size=batch_size)

    meta: Dict[str, Any] = dict(
        dataset="iris",
        num_qubits=num_qubits,
        ansatz_reps=ansatz_reps,
        batch_size=batch_size,
        noise_tag=noise_tag,
        seed=seed,
    )
    splits = (X_train, y_train, X_test, y_test)
    return qnn, estimator, cost_fn, splits, meta, model_info
