import numpy as np
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def build_qnn(X_train, ansatz_reps=1, estimator=None):
    num_qubits = X_train.shape[1]
    feature_map = ZZFeatureMap(num_qubits, reps=ansatz_reps)
    ansatz = RealAmplitudes(num_qubits, reps=ansatz_reps, entanglement="full")
    qc = feature_map.compose(ansatz)
    if estimator is None:
        estimator = Estimator(options={"shots": 1024})
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return dict(num_qubits=num_qubits, feature_map=feature_map, ansatz=ansatz, qc=qc, qnn=qnn, estimator=estimator)

def make_cost_fn(qnn, estimator, X_train, y_train, batch_size=16):
    rng = np.random.default_rng(0)
    n = len(X_train)
    def cost_fn(theta, shots, return_var=False, eps=1e-9):
        idx = rng.choice(n, batch_size, replace=False)
        Xb, yb = X_train[idx], y_train[idx]
        if hasattr(estimator, "set_options"):
            estimator.set_options(shots=shots)
        raw = qnn.forward(Xb, theta).flatten()
        probs = np.clip((raw + 1.0) * 0.5, eps, 1 - eps)
        loss = -np.mean(yb * np.log(probs) + (1 - yb) * np.log(1 - probs))
        if return_var:
            var = np.mean(probs * (1 - probs) / shots)
            return float(loss), float(var)
        return float(loss)
    return cost_fn

def make_predict_fn(qnn):
    # probs in [0,1]
    return lambda theta, X, shots=None: (qnn.forward(X, theta).flatten() + 1.0) * 0.5

def make_set_eval_shots_fn(estimator):
    def _set(shots: int):
        if hasattr(estimator, "set_options"):
            estimator.set_options(shots=shots)
    return _set
