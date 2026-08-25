import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

Nf = np.array([1.21, 1.35, 1.18, 1.42, 1.29, 1.33, 1.26, 1.39]) * 1e6
N = len(Nf)
dof = N - 1

mean_Nf = np.mean(Nf)
std_Nf = np.std(Nf, ddof=1)
print(f"N={N}")
print(f"mean(Nf) = {mean_Nf:.4e}, std(Nf) = {std_Nf:.4e}")

lnNf = np.log(Nf)
mean_ln = np.mean(lnNf)
std_ln = np.std(lnNf, ddof=1)
print(f"mean(ln Nf) = {mean_ln:.5f}, std(ln Nf) = {std_ln:.5f}")

sxbar_ln = std_ln / np.sqrt(N)
t95 = stats.t.ppf(0.975, dof)
ci95_ln = t95 * sxbar_ln
lo_ln, hi_ln = mean_ln - ci95_ln, mean_ln + ci95_ln
print(f"t_({dof},0.975)={t95:.4f}, s_xbar(ln)={sxbar_ln:.5f}, 95% CI(ln)= [{lo_ln:.5f},{hi_ln:.5f}]")

lo_phys, hi_phys = np.exp(lo_ln), np.exp(hi_ln)
med_phys = np.exp(mean_ln)
print(f"back-transformed 95% CI = [{lo_phys:.4e}, {hi_phys:.4e}] cycles, median = {med_phys:.4e}")

fig1, ax1 = plt.subplots(figsize=(6.4, 3.8))
ax1.hist(Nf, bins=5, color='tab:blue', alpha=0.7, edgecolor='k')
ax1.axvline(mean_Nf, color='k', linewidth=1, label='mean')
ax1.set_xlabel(r'$N_f$ [cycles]')
ax1.set_ylabel('count')
ax1.set_title(r'Histogram of raw $N_f$')
ax1.legend(fontsize=8)
ax1.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
fig1.tight_layout()
fig1.savefig(os.path.join(FIGDIR, 'p3_fatigue_raw.png'), dpi=160)
print("saved figure p3_fatigue_raw.png")

fig2, ax2 = plt.subplots(figsize=(6.4, 3.8))
ax2.hist(lnNf, bins=5, color='tab:green', alpha=0.7, edgecolor='k')
ax2.axvline(mean_ln, color='k', linewidth=1, label='mean')
ax2.axvspan(lo_ln, hi_ln, color='tab:orange', alpha=0.25, label='95% CI of mean')
ax2.set_xlabel(r'$\ln(N_f)$')
ax2.set_ylabel('count')
ax2.set_title(r'Histogram of $\ln(N_f)$')
ax2.legend(fontsize=8)
fig2.tight_layout()
fig2.savefig(os.path.join(FIGDIR, 'p3_fatigue_log.png'), dpi=160)
print("saved figure p3_fatigue_log.png")
