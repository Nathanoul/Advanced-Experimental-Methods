import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

Pr = np.array([4.82, 4.91, 4.76, 4.88, 4.95, 4.79, 4.86, 4.90, 4.83, 4.99])
N = len(Pr)
dof = N - 1

mean = np.mean(Pr)
var = np.var(Pr, ddof=1)
std = np.std(Pr, ddof=1)
sxbar = std / np.sqrt(N)
print(f"N={N}, mean={mean:.4f}, var={var:.5f}, std={std:.4f}, s_xbar={sxbar:.4f}")

levels = [0.90, 0.95, 0.99]
print("\nt-based CI:")
CI_t = {}
for P in levels:
    tval = stats.t.ppf(1 - (1 - P) / 2, dof)
    ci = tval * sxbar
    CI_t[P] = (tval, ci)
    print(f"  P={P*100:.0f}%: t_({dof},{ (1+P)/2:.3f})={tval:.4f}, CI=+/-{ci:.4f} -> [{mean-ci:.4f}, {mean+ci:.4f}]")

print("\nz-based (normal) CI for comparison:")
CI_z = {}
for P in levels:
    zval = stats.norm.ppf(1 - (1 - P) / 2)
    ci = zval * sxbar
    CI_z[P] = (zval, ci)
    print(f"  P={P*100:.0f}%: z={zval:.4f}, CI=+/-{ci:.4f} -> [{mean-ci:.4f}, {mean+ci:.4f}]")

P_model = 4.70
t95, ci95 = CI_t[0.95]
lo, hi = mean - ci95, mean + ci95
print(f"\nmodel prediction {P_model} bar vs 95% CI [{lo:.4f},{hi:.4f}] -> "
      f"{'inside' if lo <= P_model <= hi else 'outside'}")

fig, ax = plt.subplots(figsize=(6.4, 3.8))
trial = np.arange(1, N + 1)
ax.plot(trial, Pr, 'o', color='tab:blue', zorder=5, label='measured $P_r$')
ax.axhline(mean, color='k', linewidth=1, label='sample mean')
colors = ['tab:green', 'tab:orange', 'tab:red']
for (P, c) in zip(levels, colors):
    _, ci = CI_t[P]
    ax.axhspan(mean - ci, mean + ci, color=c, alpha=0.15, label=f'{int(P*100)}% CI (t)')
ax.axhline(P_model, color='purple', linestyle='--', linewidth=1.3, label='model prediction 4.70 bar')
ax.set_xlabel('shot number')
ax.set_ylabel(r'reflected pressure $P_r$ [bar]')
ax.set_title('Shock-tube pressure: mean and nested confidence intervals')
ax.set_xticks(trial)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.42), ncol=3, fontsize=7.5)
fig.subplots_adjust(bottom=0.38)
fig.savefig(os.path.join(FIGDIR, 'p2_pressure.png'), dpi=160)
print("saved figure p2_pressure.png")
