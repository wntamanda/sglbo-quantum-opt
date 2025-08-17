from __future__ import annotations
from typing import Dict, Any, Tuple, List

from qcbo.config import PROFILES, NOISE_MULT, sglbo_defaults


def _n_params_from(model_info: dict, qnn) -> int:
    # Qiskit path: prefer ansatz object if present
    if "ansatz" in model_info:
        return len(model_info["ansatz"].parameters)
    n = getattr(qnn, "num_weights", None)
    if n is not None:
        return int(n)
    wp = getattr(qnn, "weight_params", None)
    if wp is not None:
        return len(wp)
    # PennyLane path: builder puts n_params directly
    if "n_params" in model_info:
        return int(model_info["n_params"])
    raise ValueError("Could not determine number of trainable parameters.")


def run_experiment(
    *,
    dataset: str = "iris",
    optimizers: List[str] = ("sglbo", "spsa"),
    profile: str = "FAST",
    noise: str = "noiseless",
    seed: int = 42,
    batch_size: int = 16,
    backend: str = "qiskit", #"pennylane"
) -> Tuple[Dict[str, Dict[int, list]], Dict[str, Any]]:
    """
    Run one experiment profile across the requested optimizers.
    Returns (results, meta).
    """
    prof = PROFILES[profile]

    if dataset != "iris":
        raise ValueError(f"Unknown dataset '{dataset}'")

    if backend == "qiskit":
        from qcbo.datasets.iris import build_problem
        qnn, estimator, cost_fn, splits, meta_ds, model_info, predict_fn, set_eval_shots_fn = build_problem(
            batch_size=batch_size, noise_tag=noise, seed=seed, ansatz_reps=prof.get("ansatz_reps", 1)
        )
    elif backend == "pennylane":
        from qcbo.datasets.iris_pl import build_problem_pl
        qnn, estimator, cost_fn, splits, meta_ds, model_info, predict_fn, set_eval_shots_fn = build_problem_pl(
            batch_size=batch_size, noise_tag=noise, seed=seed, ansatz_reps=prof.get("ansatz_reps", 1)
        )
    else:
        raise ValueError("backend must be 'qiskit' or 'pennylane'")

    X_train, y_train, X_test, y_test = splits
    n_params = _n_params_from(model_info, qnn)
    budgets = prof["budgets"]; n_seeds = prof["n_seeds"]

    results = {name: {b: [] for b in budgets} for name in optimizers}

    for b in budgets:
        for s in range(n_seeds):
            cur_seed = seed + s

            for name in optimizers:
                if name == "sglbo":
                    from qcbo.optimizers.sglbo import run_sglbo
                    defaults = sglbo_defaults(prof, noise_tag=noise)
                    tup = run_sglbo(
                        seed=cur_seed, budget_shots=b, n_params=n_params,
                        cost_fn=cost_fn, predict_fn=predict_fn,
                        set_eval_shots_fn=set_eval_shots_fn,
                        X_test=X_test, y_test=y_test,
                        **defaults, verbose=False,
                    )

                elif name == "spsa":
                    # backend-neutral SPSA call
                    from qcbo.optimizers.spsa_adapter import run_spsa
                    shots_per_eval = int(round(256 * NOISE_MULT.get(noise, 1.0)))
                    tup = run_spsa(
                        seed=cur_seed, budget_shots=b, n_params=n_params,
                        cost_fn=cost_fn, predict_fn=predict_fn,
                        set_eval_shots_fn=set_eval_shots_fn,
                        X_test=X_test, y_test=y_test,
                        shots_per_eval=shots_per_eval, verbose=False,
                    )

                else:
                    raise ValueError(f"Unknown optimizer '{name}'")

                results[name][b].append(tup)

    meta = dict(
        profile=PROFILES[profile], budgets=budgets, n_seeds=n_seeds, noise=noise,
        plot_step=PROFILES[profile]["plot_step"], dataset=dataset,
        data_meta=meta_ds, optimizers=optimizers, backend=backend,
    )
    return results, meta