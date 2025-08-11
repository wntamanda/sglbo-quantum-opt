import numpy as np, matplotlib.pyplot as plt

def _best_so_far(a): return np.minimum.accumulate(a)

def _agg(results_for_opt, b, grid):
    curves=[]
    for sh, lh, *_ in results_for_opt[b]:
        if len(sh)==0 or len(lh)==0: continue
        running = _best_so_far(lh)
        curves.append(np.interp(grid, sh, running, left=running[0], right=running[-1]))
    if not curves: return None, None, None
    curves = np.vstack(curves)
    return np.nanmedian(curves, axis=0), *np.nanpercentile(curves, [25, 75], axis=0)

def plot_grid(results, meta, normalized=False, title=None):
    budgets = meta["profile"]["budgets"]
    step    = meta["profile"]["plot_step"]
    title   = title or meta.get("title", "Results")

    for b in budgets:
        grid = np.linspace(step, b, num=max(2, b // step))
        if normalized:
            x = grid / b
        else:
            x = grid

        plt.figure()
        for opt_name, by_budget in results.items():
            med, q1, q3 = _agg(by_budget, b, grid)
            if med is None: continue
            plt.plot(x, med, label=opt_name.upper())
            plt.fill_between(x, q1, q3, alpha=0.25)
        plt.title(f"{title} – Budget {b:,} " + ("(normalized)" if normalized else ""))
        plt.xlabel("fraction of budget used" if normalized else "cumulative shots")
        plt.ylabel("best loss (BCE) so far")
        plt.legend()
        plt.show(); plt.close()