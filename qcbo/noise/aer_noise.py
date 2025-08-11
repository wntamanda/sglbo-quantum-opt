from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qiskit.primitives import BackendEstimator, StatevectorEstimator
import warnings

def make_noise_model(num_qubits, p_1q=0.001, p_2q=0.01, p_ro=0.02):
    noise = NoiseModel()
    for g in ['rz','sx','x']:
        noise.add_all_qubit_quantum_error(depolarizing_error(p_1q, 1), [g])
    noise.add_all_qubit_quantum_error(depolarizing_error(p_2q, 2), ['cx'])
    ro = ReadoutError([[1-p_ro, p_ro], [p_ro, 1-p_ro]])
    for q in range(num_qubits):
        noise.add_readout_error(ro, [q])
    return noise

def estimators_by_level(num_qubits):
    warnings.filterwarnings("ignore", message=r".*BackendEstimator.*deprecated.*")
    warnings.filterwarnings("ignore", message=r".*StatevectorEstimator.*deprecated.*")
    levels = {
        "noiseless": None,
        "low":    make_noise_model(num_qubits, 0.0005, 0.005, 0.01),
        "medium": make_noise_model(num_qubits, 0.001,  0.01,  0.02),
        "high":   make_noise_model(num_qubits, 0.003,  0.03,  0.05),
    }
    ests = {}
    for tag, nm in levels.items():
        if nm is None:
            ests[tag] = StatevectorEstimator()
        else:
            simul = AerSimulator(noise_model=nm)
            simul.set_options(shots=1024)
            ests[tag] = BackendEstimator(backend=simul, options={"shots": 1024})
    return ests
