import numpy as np
import torch
import botorch

try:
    print("BoTorch:", botorch.__version__)
except Exception:
    pass

# FixedNoiseGP vs SingleTaskGP fallback for BoTorch versions
try:
    from botorch.models.gp_regression import FixedNoiseGP
    HAVE_FIXED = True
except Exception:
    from botorch.models.gp_regression import SingleTaskGP
    FixedNoiseGP = None
    HAVE_FIXED = False

from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
try:
    from botorch.acquisition import LogExpectedImprovement
except Exception:
    from botorch.acquisition.analytic import LogExpectedImprovement

torch.set_default_dtype(torch.double)


class SGLBO:
    """
    Stochastic-Gradient Line Bayesian Optimization
    - Per-iteration 1D BO along gradient direction (fresh GP each line)
    - Uses FixedNoiseGP (if available) with per-eval variance from cost_fn
    - Simple norm-test style shot update for gradients
    - Suffix (Polyak) averaging over last alpha-fraction of iterates
    """

    def __init__(
        self,
        theta0,
        cost_fn,                     # callable(theta, shots, return_var=True) -> (loss, var) or loss
        *,
        kappa=0.05,                  # target relative grad noise
        alpha=0.3,                   # suffix averaging fraction
        shots_grad_min=32,
        shots_grad_max=1024,
        shots_line=256,              # total shots used across line-search evals
        eta_bound=0.2,               # step bound along the line
        max_shots=float("inf"),
        max_iter=10_000,
        dtype=torch.double,
        verbose=True,
        seed=0,
    ):
        self.rng = np.random.default_rng(seed)
        self.theta = np.array(theta0, dtype=float)
        self.cost_fn = cost_fn
        self.kappa = float(kappa)
        self.alpha = float(alpha)

        self.shots_grad_min = int(shots_grad_min)
        self.shots_grad_max = int(shots_grad_max)
        self.shots_line = int(shots_line)
        self.eta_bound = float(eta_bound)

        self.max_shots = max_shots
        self.max_iter = max_iter
        self.verbose = verbose
        self.dtype = dtype

        self.shots_used = 0
        self.t = 0
        self._eps = 1e-12
        self._bound_hits = 0

        # suffix averaging
        self.history = []
        self.avg_theta = self.theta.copy()

    # ---------- public API ----------
    def run(self):
        while self.step():
            pass
        return self.avg_theta

    def step(self):
        if self.t >= self.max_iter or self.shots_used >= self.max_shots:
            return False

        # 1) gradient estimate
        shots_grad = self._choose_grad_shots()
        g, g_vars, g_norm = self._estimate_gradient(self.theta, shots_grad)

        if g_norm < 1e-10:
            if self.verbose:
                print(f"[iter {self.t}] near-zero gradient; stopping.")
            return False

        d = g / (g_norm + self._eps)  # unit descent direction

        # 2) 1D BO along the line
        eta = self._line_bo(direction=d)

        if abs(eta) >= 0.95 * self.eta_bound:
            self._bound_hits += 1
            if self._bound_hits >= 2 and self.verbose:
                print(f"[iter {self.t}] expanding eta_bound {self.eta_bound:.3f} -> {self.eta_bound*1.5:.3f}")
            if self._bound_hits >= 2:
                self.eta_bound *= 1.5
                self._bound_hits = 0
        else:
            self._bound_hits = 0

        # 3) update iterate
        self.theta = self.theta + eta * d
        self.history.append(self.theta.copy())
        self.t += 1

        # 4) suffix averaging
        W = max(1, int(np.ceil(self.alpha * self.t)))
        self.avg_theta = np.mean(self.history[-W:], axis=0)

        if self.verbose and self.t % 5 == 0:
            print(f"[iter {self.t:03d}] shots_used={self.shots_used}  |g|={g_norm:.4e}  eta={eta:.4e}  W={W}")

        return (self.t < self.max_iter) and (self.shots_used < self.max_shots)

    # ---------- internals ----------
    def _choose_grad_shots(self):
        base = getattr(self, "_next_grad_shots", self.shots_grad_min)
        base *= (1.0 + 0.1 * (self.t // 10))
        return int(np.clip(base, self.shots_grad_min, self.shots_grad_max))

    def _estimate_gradient(self, theta, shots):
        theta = np.asarray(theta, dtype=float)
        dim = theta.size
        grad = np.zeros(dim, dtype=float)
        vars_g = np.zeros(dim, dtype=float)

        half = shots // 2
        n_plus = max(1, half)
        n_minus = max(1, shots - n_plus)

        for i in range(dim):
            e = np.zeros(dim); e[i] = 1.0
            tp = theta + (np.pi/2.0) * e
            tm = theta - (np.pi/2.0) * e

            lp, vp = self._loss_with_var(tp, n_plus)
            lm, vm = self._loss_with_var(tm, n_minus)

            gi = 0.5 * (lp - lm)
            grad[i] = gi
            vars_g[i] = 0.25 * (vp + vm)

        g_norm = float(np.linalg.norm(grad))

        # norm-test-ish adapt for next grad shots
        num = float(np.sum(vars_g))
        denom = max(self.kappa**2 * (g_norm**2 + self._eps), 1e-16)
        shots_next = int(np.ceil(num / denom))
        shots_next = int(np.clip(shots_next, self.shots_grad_min, self.shots_grad_max))
        self._next_grad_shots = shots_next

        return grad, vars_g, g_norm

    def _loss_with_var(self, theta, shots):
        out = self.cost_fn(theta, shots=shots, return_var=True)
        if isinstance(out, tuple) and len(out) == 2:
            loss, var = out
        else:
            loss = float(out)
            var = 1.0 / max(1, shots)
        self.shots_used += int(shots)
        return float(loss), float(var)

    def _line_bo(self, direction):
        def nrm(e):   # [-eta, +eta] -> [0,1]
            return e / (2*self.eta_bound) + 0.5
        def inv(u):
            return (u - 0.5) * 2 * self.eta_bound

        etas = np.array([0.0,  0.1*self.eta_bound, -0.1*self.eta_bound], dtype=float)
        per_eval = max(1, self.shots_line // len(etas))

        X_u, Y, Yvar = [], [], []
        for e in etas:
            th = self.theta + e * direction
            y, v = self._loss_with_var(th, per_eval)
            X_u.append([nrm(e)])
            Y.append([y]); Yvar.append([max(v, 1e-12)])

        X_t = torch.tensor(X_u, dtype=self.dtype)
        Y_t = torch.tensor(Y, dtype=self.dtype)
        Yv_t = torch.tensor(Yvar, dtype=self.dtype)

        gp = FixedNoiseGP(X_t, Y_t, Yv_t) if HAVE_FIXED else SingleTaskGP(X_t, Y_t)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        bounds = torch.tensor([[0.0], [1.0]], dtype=self.dtype)
        for _ in range(2):
            acq = LogExpectedImprovement(gp, best_f=Y_t.min(), maximize=False)
            u_opt, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=5, raw_samples=32)
            e_raw = float(inv(u_opt.item()))

            th = self.theta + e_raw * direction
            y, v = self._loss_with_var(th, per_eval)

            X_t = torch.cat([X_t, torch.tensor([[nrm(e_raw)]], dtype=self.dtype)])
            Y_t = torch.cat([Y_t, torch.tensor([[y]], dtype=self.dtype)])
            Yv_t = torch.cat([Yv_t, torch.tensor([[max(v, 1e-12)]], dtype=self.dtype)])

            gp = FixedNoiseGP(X_t, Y_t, Yv_t) if HAVE_FIXED else SingleTaskGP(X_t, Y_t)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

        best_idx = int(torch.argmin(Y_t))
        best_u = float(X_t[best_idx].item())
        eta_best = inv(best_u)

        if self.shots_used > self.max_shots and self.verbose:
            print(f"[warn] shot budget exceeded on line-search; used {self.shots_used} / {self.max_shots}")

        return float(eta_best)


def run_sglbo(
    *,
    seed: int,
    budget_shots: int,
    n_params: int,
    cost_fn,
    qnn, estimator, X_test, y_test,
    # SGLBO hyperparams
    kappa=0.05, alpha=0.3,
    shots_grad_min=32, shots_grad_max=256, shots_line=128, eta_bound=0.2,
    # logging
    verbose=False, plateau_window=25, plateau_eps=3e-3,
):
    """
    Pure runner. No globals. Returns (shots_hist, loss_hist, acc_hist, avg_theta).
    """
    rng = np.random.default_rng(seed)
    theta0 = rng.standard_normal(n_params)

    opt = SGLBO(
        theta0=theta0, cost_fn=cost_fn, kappa=kappa, alpha=alpha,
        shots_grad_min=shots_grad_min, shots_grad_max=shots_grad_max,
        shots_line=shots_line, eta_bound=eta_bound,
        max_shots=budget_shots, verbose=verbose, seed=seed,
    )

    shots_hist, loss_hist, acc_hist = [], [], []
    stop_reason = "unknown"

    while True:
        cont = opt.step()
        shots_hist.append(opt.shots_used)

        # diagnostics (not counted in shots_used)
        if hasattr(estimator, "set_options"):
            estimator.set_options(shots=256)
        loss_val = float(cost_fn(opt.theta, shots=256, return_var=False))
        loss_hist.append(loss_val)

        probs = (qnn.forward(X_test, opt.theta).flatten() + 1.0) * 0.5
        acc_val = float(((probs > 0.5).astype(int) == y_test).mean())
        acc_hist.append(acc_val)

        flat = (
            len(loss_hist) > plateau_window and
            abs(loss_hist[-1] - loss_hist[-1 - plateau_window]) <
            plateau_eps * max(abs(loss_hist[-1 - plateau_window]), 1e-12)
        )
        if not cont:
            stop_reason = "budget/max_iter"
            break
        if flat:
            stop_reason = "plateau"
            break

    if verbose:
        print(f"SGLBO stopped: {stop_reason} | final shots: {shots_hist[-1]} | "
              f"best loss: {np.min(loss_hist):.4f} | best acc: {np.max(acc_hist):.4f}")

    return (np.array(shots_hist), np.array(loss_hist), np.array(acc_hist), opt.avg_theta)
