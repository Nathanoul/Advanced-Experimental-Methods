import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

eps = np.array([512, 519, 508, 515, 521, 510, 517, 514, 520, 511], dtype=float)
N = len(eps)

mean = np.mean(eps)
std = np.std(eps, ddof=1)
print(f"N = {N}")
print(f"mean = {mean:.4f} uE")
print(f"std  = {std:.4f} uE")

dof = N - 1
t95 = stats.t.ppf(0.975, dof)
sxbar = std / np.sqrt(N)
CI95 = t95 * sxbar
print(f"dof = {dof}, t_(9,0.975) = {t95:.4f}")
print(f"s_xbar = {sxbar:.4f}")
print(f"95% CI = mean +/- {CI95:.4f}  -> [{mean-CI95:.4f}, {mean+CI95:.4f}]")

d = 2.0
N_guess = 10
for it in range(1, 20):
    dof_g = N_guess - 1
    t_g = stats.t.ppf(0.975, dof_g)
    N_new = (t_g * std / d) ** 2
    N_new_round = int(np.ceil(N_new))
    print(f"iter {it}: N_guess={N_guess}, dof={dof_g}, t={t_g:.4f}, N_calc={N_new:.3f} -> {N_new_round}")
    if N_new_round == N_guess:
        break
    N_guess = N_new_round
N_required = N_guess
print(f"N required = {N_required}")

fig, ax = plt.subplots(figsize=(6.4, 3.6))
trial = np.arange(1, N + 1)
ax.plot(trial, eps, 'o', color='tab:blue', label='measured strain')
ax.axhline(mean, color='k', linewidth=1, label='sample mean')
ax.axhspan(mean - CI95, mean + CI95, color='tab:orange', alpha=0.25, label='95% CI of the mean')
ax.set_xlabel('measurement number')
ax.set_ylabel(r'strain $\varepsilon$ [$\mu\varepsilon$]')
ax.set_title('Strain-gauge readings, sample mean, and 95% CI of the mean')
ax.set_xticks(trial)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=8)
fig.subplots_adjust(bottom=0.32)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'p1_strain.png'), dpi=160)
print("saved figure p1_strain.png")
