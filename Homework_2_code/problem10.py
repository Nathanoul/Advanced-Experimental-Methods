import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

s = 18.0
d = 5.0
P = 0.95

N_guess = 10
history = []
for it in range(1, 20):
    dof = N_guess - 1
    tval = stats.t.ppf(1 - (1 - P) / 2, dof)
    N_calc = (tval * s / d) ** 2
    N_new = int(np.ceil(N_calc))
    history.append((it, N_guess, dof, tval, N_calc, N_new))
    print(f"iter {it}: N_guess={N_guess}, dof={dof}, t={tval:.4f}, N_calc={N_calc:.3f} -> {N_new}")
    if N_new == N_guess:
        break
    N_guess = N_new
N_required = N_guess
print(f"N required = {N_required}")

z95 = stats.norm.ppf(0.975)
N_z = int(np.ceil((z95 * s / d) ** 2))
print(f"z=1.96 -> N = {N_z}  (vs iterated t-based N = {N_required})")

fig, ax = plt.subplots(figsize=(6.4, 3.8))
iters = [h[0] for h in history]
Ncalc = [h[4] for h in history]
Nused = [h[1] for h in history]
ax.plot(iters, Nused, 'o-', color='tab:blue', label='$N$ used to look up $t$')
ax.plot(iters, Ncalc, 's--', color='tab:red', label='$N$ calculated from $t$')
ax.axhline(N_z, color='tab:green', linestyle=':', label=f'$z=1.96$ estimate (N={N_z})')
ax.set_xlabel('iteration')
ax.set_ylabel('N')
ax.set_xticks(iters)
ax.set_title('Iterative estimate of required specimen count')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'p10_sample_size.png'), dpi=160)
print("saved figure p10_sample_size.png")
