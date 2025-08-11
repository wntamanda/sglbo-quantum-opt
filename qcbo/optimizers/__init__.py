from .sglbo import run_sglbo
from .spsa_adapter import run_spsa
from .adam_adapter import run_adam
from .cobyla_adapter import run_cobyla

OPTIMIZERS = {
    "sglbo": run_sglbo,
    "spsa": run_spsa,
    "adam": run_adam,
    "cobyla": run_cobyla,
}

def get_runner(name: str):
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer '{name}'. Available: {list(OPTIMIZERS)}")
    return OPTIMIZERS[name]