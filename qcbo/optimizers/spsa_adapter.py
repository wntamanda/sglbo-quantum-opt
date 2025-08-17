# 17.08.2025 (fixes)
# strict budget compliance (clamp final shots)
# shared plateau helper with SGLBO
# optional log_every argument

from __future__ import annotations
import numpy as np

def _plateau(losses: list[float], window: int, eps: float) -> bool:
    """True if loss has stagnated (relative) for <window> steps."""
    if len(losses) <= window:
        return False
    a, b = losses[-1], losses[-1 - window]
    return abs(a - b) < eps * max(abs(b), 1e-12)

def run_spsa(
    *,
    seed: int,
    budget_shots: int,
    n_params: int,
    cost_fn,
    predict_fn,
    set_eval_shots_fn,
    X_test, y_test,
    shots_per_eval: int = 256,
    verbose: bool = False,
    log_every: int = 5,
    plateau_window: int = 25,
    plateau_eps: float = 3e-3,
    # SPSA hyper-params
    a0: float = 0.05,
    c0: float = 0.1,
    alpha: float = 0.602,
    gamma: float = 0.101,
):
    """
    Returns
        shots_hist, loss_hist, acc_hist, theta_final, stop_reason
    Shot budget counts 2*shots_per_eval per iteration (± perturb).
    """
    rng   = np.random.default_rng(seed)
    theta = rng.standard_normal(n_params)

    shots_used = 0
    shots_hist, loss_hist, acc_hist = [], [], []
    stop_reason = "unknown"
    k = 1

    # diagnostic logging helper (not budgeted)
    def _log(theta_now):
        set_eval_shots_fn(256)
        loss = float(cost_fn(theta_now, 256, False))
        probs = predict_fn(theta_now, X_test, shots=None).flatten()
        acc   = float(((probs > 0.5).astype(int) == y_test).mean())

        loss_hist.append(loss)
        acc_hist.append(acc)

        if verbose and (len(loss_hist) == 1 or len(loss_hist) % log_every == 0):
            print(f"SPSA iter {len(loss_hist)-1:3d} | "
                  f"shots {shots_used:7d} | loss {loss:6.3f} | acc {acc:5.3f}")

    # initial log at 0 shots
    shots_hist.append(shots_used)
    _log(theta)

    # optimisation loop
    while True:
        # remaining budget check (≥ two calls needed)
        budget_left = budget_shots - shots_used
        if budget_left < 2:
            stop_reason = "budget"
            break

        # clamp shots for the last possible iteration if needed
        cur_shots = min(shots_per_eval, budget_left // 2)

        # schedules
        ak = a0 / (k ** alpha)
        ck = c0 / (k ** gamma)

        # Rademacher perturbation
        delta = rng.choice([-1.0, 1.0], size=n_params)

        # two evaluations at +/- ck
        set_eval_shots_fn(cur_shots)
        f_plus  = float(cost_fn(theta + ck * delta, cur_shots, False))
        set_eval_shots_fn(cur_shots)
        f_minus = float(cost_fn(theta - ck * delta, cur_shots, False))

        # gradient estimate and update
        ghat  = (f_plus - f_minus) / (2.0 * ck) * delta
        theta = theta - ak * ghat

        # bookkeeping & logging
        shots_used += 2 * cur_shots
        shots_hist.append(shots_used)
        _log(theta)

        # plateau stopping
        if _plateau(loss_hist, plateau_window, plateau_eps):
            stop_reason = "plateau"
            break

        k += 1

    return (np.array(shots_hist),
            np.array(loss_hist),
            np.array(acc_hist),
            theta,
            stop_reason)
