def get_runner(name: str):
    if name == "sglbo":
        from .sglbo import run_sglbo
        return run_sglbo
    if name == "spsa":
        from .spsa_adapter import run_spsa
        return run_spsa
    if name == "adam":
        from .adam_adapter import run_adam
        return run_adam
    if name == "cobyla":
        from .cobyla_adapter import run_cobyla
        return run_cobyla
    raise ValueError(f"Unknown optimizer '{name}'. Available: ['sglbo','spsa','adam','cobyla']")