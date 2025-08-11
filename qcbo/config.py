PROFILES = {
    "FAST": {
        "ansatz_reps": 1,
        "budgets": [5_000, 10_000],
        "n_seeds": 3,
        "plot_step": 64,
        "sglbo": dict(
            kappa=0.06, alpha=0.3,
            shots_grad_min=16, shots_grad_max=128, shots_line=64, eta_bound=0.2
        ),
    },
    "FULL": {
        "ansatz_reps": 2,
        "budgets": [20_000, 50_000],
        "n_seeds": 5,
        "plot_step": 128,
        "sglbo": dict(
            kappa=0.05, alpha=0.3,
            shots_grad_min=32, shots_grad_max=256, shots_line=128, eta_bound=0.2
        ),
    },
}

NOISE_MULT = {"noiseless": 1.0, "low": 1.5, "medium": 2.0, "high": 3.0}


def sglbo_defaults(profile: dict, noise_tag: str = "noiseless") -> dict:
    """Return SGLBO defaults scaled by noise level."""
    base = dict(profile["sglbo"])
    m = NOISE_MULT.get(noise_tag, 1.0)
    for k in ("shots_grad_min", "shots_grad_max", "shots_line"):
        base[k] = int(round(base[k] * m))
    return base