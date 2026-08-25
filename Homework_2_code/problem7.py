import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

eta = np.array([78.2, 78.6, 78.4, 78.7, 78.5, 78.3, 78.6, 78.4, 76.9])
N = len(eta)
suspect_idx = 8
suspect = eta[suspect_idx]

mean_all = np.mean(eta)
std_all = np.std(eta, ddof=1)
z0_all = abs(suspect - mean_all) / std_all
print(f"With all N={N} points: mean={mean_all:.4f}, std={std_all:.4f}, z0={z0_all:.4f}")

print(f"3-sigma test: |z0|={z0_all:.4f} vs 3  -> "
      f"{'OUTLIER' if z0_all > 3 else 'not flagged'}")

P1_z0 = stats.norm.cdf(z0_all) - 0.5
lhs = 1 - 2 * P1_z0
rhs = 1 / (2 * N)
print(f"Chauvenet: P1(z0)={P1_z0:.5f}, LHS=1-2P1={lhs:.5f}, 1/(2N)={rhs:.5f} -> "
      f"{'OUTLIER (flag)' if lhs < rhs else 'not flagged'}")

eta_clean = np.delete(eta, suspect_idx)
mean_c = np.mean(eta_clean)
std_c = np.std(eta_clean, ddof=1)
z0_c = abs(suspect - mean_c) / std_c
print(f"\nWithout suspect point: N={len(eta_clean)}, mean={mean_c:.4f}, std={std_c:.4f}")
print(f"z0 (suspect vs reduced stats) = {z0_c:.4f}")
print(f"3-sigma (reduced stats): {'OUTLIER' if z0_c > 3 else 'not flagged'}")
P1_z0c = stats.norm.cdf(z0_c) - 0.5
lhs_c = 1 - 2 * P1_z0c
rhs_c = 1 / (2 * len(eta_clean))
print(f"Chauvenet (reduced stats): LHS={lhs_c:.5f}, 1/(2N)={rhs_c:.5f} -> "
      f"{'OUTLIER' if lhs_c < rhs_c else 'not flagged'}")

fig, ax = plt.subplots(figsize=(6.6, 3.8))
trial = np.arange(1, N + 1)
colors = ['tab:blue'] * N
colors[suspect_idx] = 'tab:red'
ax.scatter(trial, eta, c=colors, s=60, zorder=5)
ax.axhline(mean_all, color='k', linewidth=1, label='mean (all points)')
ax.axhspan(mean_all - 3 * std_all, mean_all + 3 * std_all, color='tab:orange', alpha=0.15,
           label=r'mean $\pm\,3\sigma$ (all points)')
ax.axhline(mean_c, color='green', linewidth=1, linestyle='--', label='mean (point 9 removed)')
ax.annotate('point 9\n(suspect)', (trial[suspect_idx], suspect), textcoords="offset points",
            xytext=(10, -25), fontsize=8, color='tab:red')
ax.set_xlabel('measurement number')
ax.set_ylabel(r'efficiency $\eta$ [%]')
ax.set_title('Pump efficiency measurements and outlier bounds')
ax.set_xticks(trial)
ax.legend(fontsize=7.5, loc='lower left')
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'p7_outlier.png'), dpi=160)
print("saved figure p7_outlier.png")
