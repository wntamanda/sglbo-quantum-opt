# 17.08.2025 (fixes)
# hard guard against shot-budget overflow during line search
# no artificial growth of gradient-shot count
# leaner EI optimisation (1 restart, 16 raw samples)
# user-configurable logging cadence
# shared plateau helper for SGLBO / SPSA

from __future__ import annotations

import math
import numpy as np
import torch
import botorch

#BoTorch
try:
    from botorch.models.gp_regression import FixedNoiseGP
    HAVE_FIXED = True
except Exception:  # ≥ 0.9 falls back
    from botorch.models.gp_regression import SingleTaskGP
    FixedNoiseGP = None
    HAVE_FIXED = False

from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
try:
    from botorch.acquisition import LogExpectedImprovement
except Exception: # pre-0.8 import path
    from botorch.acquisition.analytic import LogExpectedImprovement

torch.set_default_dtype(torch.double)

# tiny utilities
def _plateau(losses: list[float], window: int, eps: float) -> bool:
    """Return True when the loss has not improved (rel.) for <window> steps."""
    if len(losses) <= window:
        return False
    a, b = losses[-1], losses[-1 - window]
    return abs(a - b) < eps * max(abs(b), 1e-12)


# optimizer core
class SGLBO:
    """
    Stochastic-Gradient Line Bayesian Optimisation (1-D BO per iteration).

    Parameters
    ----------
    theta0 : array_like
        Initial parameter vector.
    cost_fn : callable
        cost_fn(theta, shots, return_var=True) → (loss, var)  or  loss.
    kappa : float
        Target relative gradient noise for the norm-test shot rule.
    alpha : float
        Fraction of recent iterates used for Polyak suffix averaging.
    shots_* : int
        Shot-budget heuristics for gradient and line search.
    eta_bound : float
        Half-width of the 1-D search interval per iteration.
    max_shots / max_iter : int
        Hard outer stopping conditions.
    log_every : int
        Emit a verbose line every <log_every> iterations (if verbose=True).
    """

    def __init__(
        self,
        theta0,
        cost_fn,
        *,
        kappa: float = 0.05,
        alpha: float = 0.3,
        shots_grad_min: int = 32,
        shots_grad_max: int = 1024,
        shots_line: int = 256,
        eta_bound: float = 0.2,
        max_shots: int | float = float("inf"),
        max_iter: int = 10_000,
        dtype=torch.double,
        verbose: bool = True,
        log_every: int = 5,
        seed: int = 0,
    ):
        self.rng   = np.random.default_rng(seed)
        self.theta = np.array(theta0, dtype=float)

        self.cost_fn = cost_fn
        self.kappa   = float(kappa)
        self.alpha   = float(alpha)

        self.shots_grad_min = int(shots_grad_min)
        self.shots_grad_max = int(shots_grad_max)
        self.shots_line     = int(shots_line)
        self.eta_bound      = float(eta_bound)

        self.max_shots = max_shots
        self.max_iter  = max_iter
        self.dtype     = dtype
        self.verbose   = verbose
        self.log_every = int(log_every)

        # state bookkeeping
        self.shots_used  = 0
        self.t           = 0
        self._eps        = 1e-12
        self._bound_hits = 0
        self.history     = []
        self.avg_theta   = self.theta.copy()

    # public API
    def run(self) -> np.ndarray:
        while self.step():
            pass
        return self.avg_theta

    def step(self) -> bool:
        """One optimiser iteration.  Returns False when stopping."""
        if self.t >= self.max_iter or self.shots_used >= self.max_shots:
            return False

        # 1) gradient estimate
        shots_grad = self._choose_grad_shots()
        g, vars_g, g_norm = self._estimate_gradient(self.theta, shots_grad)

        if g_norm < 1e-10:
            if self.verbose:
                print(f"[iter {self.t}] near-zero gradient; stopping.")
            return False

        d = g / (g_norm + self._eps) # unit descent direction

        # 2) 1-D BO line search
        eta = self._line_bo(direction=d)

        # adapt trust radius if we keep hitting the boundaries
        if abs(eta) >= 0.95 * self.eta_bound:
            self._bound_hits += 1
            if self._bound_hits >= 2:
                if self.verbose:
                    print(f"[iter {self.t}] expanding eta_bound "
                          f"{self.eta_bound:.3f} → {self.eta_bound*1.5:.3f}")
                self.eta_bound *= 1.5
                self._bound_hits = 0
        else:
            self._bound_hits = 0

        # 3) parameter update & bookkeeping
        self.theta += eta * d
        self.history.append(self.theta.copy())
        self.t += 1

        # 4) Polyak suffix averaging
        W = max(1, int(np.ceil(self.alpha * self.t)))
        self.avg_theta = np.mean(self.history[-W:], axis=0)

        if self.verbose and self.t % self.log_every == 0:
            print(f"[iter {self.t:03d}] shots={self.shots_used:6d}  "
                  f"|g|={g_norm:.3e}  eta={eta:.3e}  W={W}")

        return (self.t < self.max_iter) and (self.shots_used < self.max_shots)

    # helpers
    def _choose_grad_shots(self) -> int:
        """Next gradient-shot count (no deterministic growth)."""
        return int(np.clip(getattr(self, "_next_grad_shots",
                                   self.shots_grad_min),
                           self.shots_grad_min, self.shots_grad_max))

    def _estimate_gradient(self, theta: np.ndarray, shots: int):
        """Parameter-shift gradient with per-coordinate shot allocation."""
        theta = np.asarray(theta, dtype=float)
        dim   = theta.size
        grad  = np.zeros(dim, dtype=float)
        vars_g = np.zeros(dim, dtype=float)

        half = shots // 2
        n_plus, n_minus = max(1, half), max(1, shots - half)

        for i in range(dim):
            e = np.zeros(dim); e[i] = 1.0
            tp = theta + (np.pi / 2) * e
            tm = theta - (np.pi / 2) * e

            lp, vp = self._loss_with_var(tp, n_plus)
            lm, vm = self._loss_with_var(tm, n_minus)

            grad[i]  = 0.5 * (lp - lm) # parameter-shift
            vars_g[i] = 0.25 * (vp + vm)

        g_norm = float(np.linalg.norm(grad))

        # norm-test shot adaptation for NEXT iteration
        num   = float(np.sum(vars_g))
        denom = max(self.kappa**2 * (g_norm**2 + self._eps), 1e-16)
        self._next_grad_shots = int(
            np.clip(math.ceil(num / denom), self.shots_grad_min, self.shots_grad_max)
        )

        return grad, vars_g, g_norm

    def _loss_with_var(self, theta: np.ndarray, shots: int):
        """Wrapper around user cost_fn with budget accounting."""
        out = self.cost_fn(theta, shots=shots, return_var=True)
        if isinstance(out, tuple) and len(out) == 2:
            loss, var = out
        else:      # fallback when cost_fn returns scalar
            loss, var = float(out), 1.0 / max(1, shots)
        self.shots_used += int(shots)
        return float(loss), float(var)

    def _line_bo(self, direction: np.ndarray) -> float:
        """1-D BO along ±direction; returns best step length eta."""
        def nrm(e): return e / (2 * self.eta_bound) + 0.5 # η -> [0,1]
        def inv(u): return (u - 0.5) * 2 * self.eta_bound # [0,1] -> η

        etas = np.array([0.0, +0.1*self.eta_bound, -0.1*self.eta_bound])

        per_eval = max(
            1,
            min(self.shots_line // len(etas),
                (self.max_shots - self.shots_used) // (len(etas) + 2)),
        )

        X_u, Y, Yvar = [], [], []
        for e in etas:
            th  = self.theta + e * direction
            y, v = self._loss_with_var(th, per_eval)
            X_u.append([nrm(e)])
            Y.append([y]); Yvar.append([max(v, 1e-12)])

        X_t  = torch.tensor(X_u,  dtype=self.dtype)
        Y_t  = torch.tensor(Y,    dtype=self.dtype)
        Yv_t = torch.tensor(Yvar, dtype=self.dtype)

        gp  = FixedNoiseGP(X_t, Y_t, Yv_t) if HAVE_FIXED else SingleTaskGP(X_t, Y_t)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        bounds = torch.tensor([[0.0], [1.0]], dtype=self.dtype)

        for _ in range(2): # two BO iterations per outer step
            acq = LogExpectedImprovement(gp, best_f=Y_t.min(), maximize=False)
            u_opt, _ = optimize_acqf(acq, bounds=bounds,
                                     q=1, num_restarts=1, raw_samples=16)

            e_raw = float(inv(u_opt.item()))
            th    = self.theta + e_raw * direction
            y, v  = self._loss_with_var(th, per_eval)

            X_t  = torch.cat([X_t,  torch.tensor([[nrm(e_raw)]], dtype=self.dtype)])
            Y_t  = torch.cat([Y_t,  torch.tensor([[y]],         dtype=self.dtype)])
            Yv_t = torch.cat([Yv_t, torch.tensor([[max(v,1e-12)]], dtype=self.dtype)])

            gp  = FixedNoiseGP(X_t, Y_t, Yv_t) if HAVE_FIXED else SingleTaskGP(X_t, Y_t)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll, options={"verbose": False})

        best_idx = int(torch.argmin(Y_t))
        return float(inv(X_t[best_idx].item()))


# convenience wrapper used by api.py
def run_sglbo(
    *,
    seed: int,
    budget_shots: int,
    n_params: int,
    cost_fn,
    predict_fn,
    set_eval_shots_fn=lambda s: None,
    X_test, y_test,
    # hyper-params
    kappa: float = 0.05, alpha: float = 0.3,
    shots_grad_min: int = 32, shots_grad_max: int = 256,
    shots_line: int = 128, eta_bound: float = 0.2,
    # logging
    verbose: bool = False, plateau_window: int = 25, plateau_eps: float = 3e-3,
):
    rng = np.random.default_rng(seed)
    theta0 = rng.standard_normal(n_params)

    opt = SGLBO(
        theta0=theta0, cost_fn=cost_fn,
        kappa=kappa, alpha=alpha,
        shots_grad_min=shots_grad_min, shots_grad_max=shots_grad_max,
        shots_line=shots_line, eta_bound=eta_bound,
        max_shots=budget_shots, verbose=verbose, seed=seed,
    )

    shots_hist, loss_hist, acc_hist = [], [], []
    stop_reason = "unknown"

    while True:
        cont = opt.step()
        shots_hist.append(opt.shots_used)

        # diagnostics (do not count against budget)
        set_eval_shots_fn(256)
        loss_val = float(cost_fn(opt.theta, shots=256, return_var=False))
        loss_hist.append(loss_val)

        probs = predict_fn(opt.theta, X_test).flatten()
        acc_val = float(((probs > 0.5).astype(int) == y_test).mean())
        acc_hist.append(acc_val)

        # stopping criteria
        if not cont:
            stop_reason = "budget/max_iter"
            break
        if _plateau(loss_hist, plateau_window, plateau_eps):
            stop_reason = "plateau"
            break

    if verbose:
        print(f"SGLBO stopped: {stop_reason} | "
              f"shots {shots_hist[-1]:d} | "
              f"best loss {np.min(loss_hist):.4f} | "
              f"best acc {np.max(acc_hist):.4f}")

    return (np.array(shots_hist),
            np.array(loss_hist),
            np.array(acc_hist),
            opt.avg_theta)