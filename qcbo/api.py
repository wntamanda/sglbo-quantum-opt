from __future__ import annotations
from typing import Dict, Any, Tuple, List

from qcbo.config import PROFILES, NOISE_MULT, sglbo_defaults
from qcbo.datasets.iris import build_problem
from qcbo.optimizers.sglbo import run_sglbo
from qcbo.optimizers.spsa_adapter import run_spsa


def _n_params_from(model_info: dict, qnn) -> int:
    # Prefer the ansatz in model_info
    if "ansatz" in model_info:
        return len(model_info["ansatz"].parameters)
    # Fallbacks for different EstimatorQNN versions
    n = getattr(qnn, "num_weights", None)
    if n is not None:
        return int(n)
    wp = getattr(qnn, "weight_params", None)
    if wp is not None:
        return len(wp)
    raise ValueError("Could not determine number of trainable parameters.")


def run_experiment(
    *,
    dataset: str = "iris",
    optimizers: List[str] = ("sglbo", "spsa"),
    profile: str = "FAST",
    noise: str = "noiseless",
    seed: int = 42,
    batch_size: int = 16,
) -> Tuple[Dict[str, Dict[int, list]], Dict[str, Any]]:
    """
    Run one experiment profile across the requested optimizers.
    Returns (results, meta).
    """
    prof = PROFILES[profile]

    # Build dataset
    if dataset == "iris":
        qnn, estimator, cost_fn, (X_train, y_train, X_test, y_test), meta_ds, model_info = build_problem(
            batch_size=batch_size,
            noise_tag=noise,
            seed=seed,
            ansatz_reps=prof.get("ansatz_reps", 1),
        )
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")

    n_params = _n_params_from(model_info, qnn)
    budgets = prof["budgets"]
    n_seeds = prof["n_seeds"]

    results: Dict[str, Dict[int, list]] = {name: {b: [] for b in budgets} for name in optimizers}
    rng_seed_base = seed

    for b in budgets:
        for s in range(n_seeds):
            cur_seed = rng_seed_base + s

            for name in optimizers:
                if name == "sglbo":
                    defaults = sglbo_defaults(prof, noise_tag=noise)
                    tup = run_sglbo(
                        seed=cur_seed,
                        budget_shots=b,
                        n_params=n_params,
                        cost_fn=cost_fn,
                        qnn=qnn, estimator=estimator,
                        X_test=X_test, y_test=y_test,
                        **defaults,
                        verbose=False,
                    )
                elif name == "spsa":
                    shots_per_eval = int(round(256 * NOISE_MULT.get(noise, 1.0)))
                    tup = run_spsa(
                        seed=cur_seed,
                        budget_shots=b,
                        n_params=n_params,
                        cost_fn=cost_fn,
                        qnn=qnn, estimator=estimator,
                        X_test=X_test, y_test=y_test,
                        shots_per_eval=shots_per_eval,
                        verbose=False,
                    )
                else:
                    raise ValueError(f"Unknown optimizer '{name}'")

                results[name][b].append(tup)

    meta = dict(
        profile=prof,                 # plot_grid expects this
        budgets=budgets,
        n_seeds=n_seeds,
        noise=noise,
        plot_step=prof["plot_step"],
        dataset=dataset,
        data_meta=meta_ds,
        optimizers=optimizers,
    )
    return results, meta