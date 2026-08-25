import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

bin_labels = ['0.20-0.30', '0.30-0.40', '0.40-0.50', '0.50-0.60',
              '0.60-0.70', '0.70-0.80', '0.80-0.90', '0.90-1.00']
observed = np.array([4, 9, 17, 21, 16, 8, 4, 1], dtype=float)
expected = np.array([3.2, 8.6, 16.5, 21.7, 17.0, 8.9, 3.3, 0.8], dtype=float)
N = observed.sum()
print(f"N = {N}, sum expected = {expected.sum()}")

chi2_full = np.sum((observed - expected) ** 2 / expected)
K_full = len(observed)
dof_full = K_full - 1 - 2
chi2_crit_full = stats.chi2.ppf(0.95, dof_full)
print(f"\nFull 8-bin test: chi2 = {chi2_full:.4f}, dof = {dof_full}, "
      f"chi2_crit(0.95) = {chi2_crit_full:.4f} -> "
      f"{'REJECT' if chi2_full > chi2_crit_full else 'no reason to reject'}")

print("\nbins with expected < 5:", [bin_labels[i] for i in range(K_full) if expected[i] < 5])

obs_c = np.array([observed[0] + observed[1], observed[2], observed[3], observed[4],
                   observed[5] + observed[6] + observed[7]])
exp_c = np.array([expected[0] + expected[1], expected[2], expected[3], expected[4],
                   expected[5] + expected[6] + expected[7]])
labels_c = ['0.20-0.40', '0.40-0.50', '0.50-0.60', '0.60-0.70', '0.70-1.00']
print("\ncombined bins:", labels_c)
print("observed:", obs_c, " sum=", obs_c.sum())
print("expected:", exp_c, " sum=", exp_c.sum())

chi2_c = np.sum((obs_c - exp_c) ** 2 / exp_c)
K_c = len(obs_c)
dof_c = K_c - 1 - 2
chi2_crit_c = stats.chi2.ppf(0.95, dof_c)
print(f"\nCombined 5-bin test: chi2 = {chi2_c:.4f}, dof = {dof_c}, "
      f"chi2_crit(0.95) = {chi2_crit_c:.4f} -> "
      f"{'REJECT' if chi2_c > chi2_crit_c else 'no reason to reject'}")

fig1, ax1 = plt.subplots(figsize=(6.6, 4.0))
x = np.arange(K_full)
w = 0.38
ax1.bar(x - w/2, observed, width=w, label='observed', color='tab:blue')
ax1.bar(x + w/2, expected, width=w, label='expected (normal fit)', color='tab:orange')
ax1.set_xticks(x)
ax1.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)
ax1.set_ylabel('count')
ax1.set_title(f'8 raw bins ($\\chi^2$={chi2_full:.2f}, dof={dof_full})')
ax1.legend(fontsize=8)
fig1.tight_layout()
fig1.savefig(os.path.join(FIGDIR, 'p6_goodness_of_fit_raw.png'), dpi=160)
print("saved figure p6_goodness_of_fit_raw.png")

fig2, ax2 = plt.subplots(figsize=(6.6, 4.0))
x2 = np.arange(K_c)
ax2.bar(x2 - w/2, obs_c, width=w, label='observed', color='tab:blue')
ax2.bar(x2 + w/2, exp_c, width=w, label='expected (normal fit)', color='tab:orange')
ax2.set_xticks(x2)
ax2.set_xticklabels(labels_c, rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('count')
ax2.set_title(f'5 combined bins ($\\chi^2$={chi2_c:.2f}, dof={dof_c})')
ax2.legend(fontsize=8)
fig2.tight_layout()
fig2.savefig(os.path.join(FIGDIR, 'p6_goodness_of_fit_combined.png'), dpi=160)
print("saved figure p6_goodness_of_fit_combined.png")
