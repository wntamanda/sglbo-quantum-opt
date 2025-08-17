import numpy as np
import pennylane as qml

_LEVELS = {
    "noiseless": (0.0,   0.0,   0.0),
    "low":       (5e-4,  5e-3,  0.01),
    "medium":    (1e-3,  1e-2,  0.02),
    "high":      (3e-3,  3e-2,  0.05),
}

def build_pl_qnode(n_qubits: int, ansatz_reps: int, noise_tag: str = "noiseless"):
    p1q, p2q, p_ro = _LEVELS.get(noise_tag, _LEVELS["noiseless"])

    # use default.mixed when insert channels
    dev_name = "default.mixed" if (p1q > 0 or p2q > 0) else "default.qubit"
    dev = qml.device(dev_name, wires=n_qubits)

    wires = list(range(n_qubits))
    n_params = ansatz_reps * (2 * n_qubits)

    def _ent_layer():
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[wires[i], wires[i+1]])
            if p2q > 0:
                qml.DepolarizingChannel(p2q, wires=wires[i])
                qml.DepolarizingChannel(p2q, wires=wires[i+1])
        if n_qubits > 2:
            qml.CNOT(wires=[wires[-1], wires[0]])
            if p2q > 0:
                qml.DepolarizingChannel(p2q, wires=wires[-1])
                qml.DepolarizingChannel(p2q, wires=wires[0])

    @qml.qnode(dev)
    def qnode(theta, x):
        qml.templates.AngleEmbedding(x, wires=wires)
        if p1q > 0:
            for w in wires:
                qml.DepolarizingChannel(p1q, wires=w)
        idx = 0
        for _ in range(ansatz_reps):
            for w in wires:
                qml.RY(theta[idx], wires=w); idx += 1
                if p1q > 0:
                    qml.DepolarizingChannel(p1q, wires=w)
            _ent_layer()
            for w in wires:
                qml.RZ(theta[idx], wires=w); idx += 1
                if p1q > 0:
                    qml.DepolarizingChannel(p1q, wires=w)
        return qml.expval(qml.PauliZ(wires[0]))

    return qnode, n_params, p_ro

def make_pl_predict_fn(qnode):
    def predict_fn(theta, X, shots=None):
        vals = np.array([qnode(theta, x, shots=shots) for x in X], dtype=float)
        return (vals + 1.0) * 0.5
    return predict_fn
