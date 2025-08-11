import numpy as np
from qiskit_algorithms.optimizers import SPSA

def run_spsa(
    *,
    seed: int,
    budget_shots: int,
    n_params: int,
    cost_fn,
    qnn, estimator, X_test, y_test,
    shots_per_eval: int = 256,
    verbose: bool = False,
    plateau_window: int = 25,
    plateau_eps: float = 3e-3,
):
    """
    Pure SPSA runner. No globals. Returns (shots_hist, loss_hist, acc_hist, theta_final, stop_reason).
    """
    rng = np.random.default_rng(seed)
    theta0 = rng.standard_normal(n_params)

    spsa_shots_used = 0
    shots_hist, loss_hist, acc_hist = [], [], []
    theta_last = theta0.copy()
    stop_reason = "unknown"

    def spsa_objective(theta):
        nonlocal spsa_shots_used, theta_last, stop_reason
        theta_last = theta

        if hasattr(estimator, "set_options"):
            estimator.set_options(shots=shots_per_eval)
        loss = cost_fn(theta, shots_per_eval, False)

        spsa_shots_used += shots_per_eval
        if spsa_shots_used >= budget_shots:
            stop_reason = "budget"
            raise StopIteration

        # log once per iteration
        if len(shots_hist) == 0 or spsa_shots_used > shots_hist[-1]:
            shots_hist.append(spsa_shots_used)

            if hasattr(estimator, "set_options"):
                estimator.set_options(shots=256)
            loss_val = cost_fn(theta, 256, False)
            loss_hist.append(loss_val)

            probs = (qnn.forward(X_test, theta).flatten() + 1) * 0.5
            acc_val = ((probs > 0.5).astype(int) == y_test).mean()
            acc_hist.append(acc_val)

            if verbose:
                print(f"SPSA iter {len(shots_hist):3d} | shots {spsa_shots_used:7d} "
                      f"| loss {loss_val:6.3f} | acc {acc_val:5.3f}")

            flat = (
                len(loss_hist) > plateau_window and
                abs(loss_hist[-1] - loss_hist[-1 - plateau_window]) <
                plateau_eps * max(abs(loss_hist[-1 - plateau_window]), 1e-12)
            )
            if flat:
                stop_reason = "plateau"
                raise StopIteration

        return loss

    spsa = SPSA(maxiter=300, learning_rate=0.05, perturbation=0.1)

    try:
        res = spsa.minimize(fun=spsa_objective, x0=theta0)
        theta_final = res.x
        stop_reason = "maxiter"
    except StopIteration:
        theta_final = theta_last

    if verbose:
        print(f"SPSA stopped due to: {stop_reason} | final shots: {shots_hist[-1]}")

    return (np.array(shots_hist), np.array(loss_hist), np.array(acc_hist), theta_final, stop_reason)