import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

N = 18
models = ['linear', 'quadratic', 'cubic']
p_params = {'linear': 2, 'quadratic': 3, 'cubic': 4}
chi2_val = {'linear': 96.5, 'quadratic': 31.4, 'cubic': 16.9}

dof = {}
chi2_red = {}
chi2_crit = {}
decision = {}
for mdl in models:
    dof[mdl] = N - p_params[mdl]
    chi2_red[mdl] = chi2_val[mdl] / dof[mdl]
    chi2_crit[mdl] = stats.chi2.ppf(0.95, dof[mdl])
    decision[mdl] = 'REJECT' if chi2_val[mdl] > chi2_crit[mdl] else 'accept'
    print(f"{mdl:10s}: dof={dof[mdl]:2d}, chi2={chi2_val[mdl]:.1f}, chi2/dof={chi2_red[mdl]:.3f}, "
          f"chi2_crit(0.95)={chi2_crit[mdl]:.3f} -> {decision[mdl]}")

d_chi2 = chi2_val['quadratic'] - chi2_val['cubic']
d_dof = dof['quadratic'] - dof['cubic']
d_crit = stats.chi2.ppf(0.95, d_dof)
print(f"\nNested test cubic vs quadratic: delta_chi2={d_chi2:.2f}, delta_dof={d_dof}, "
      f"chi2_crit(0.95,{d_dof})={d_crit:.3f} -> "
      f"{'cubic term justified (significant improvement)' if d_chi2 > d_crit else 'cubic term NOT justified'}")

d_chi2_lq = chi2_val['linear'] - chi2_val['quadratic']
d_dof_lq = dof['linear'] - dof['quadratic']
d_crit_lq = stats.chi2.ppf(0.95, d_dof_lq)
print(f"Nested test quadratic vs linear: delta_chi2={d_chi2_lq:.2f}, delta_dof={d_dof_lq}, "
      f"chi2_crit={d_crit_lq:.3f} -> "
      f"{'quadratic term justified' if d_chi2_lq > d_crit_lq else 'not justified'}")

fig1, ax1 = plt.subplots(figsize=(6.2, 3.9))
x = np.arange(len(models))
ax1.bar(x, [chi2_red[m] for m in models], color='tab:blue', alpha=0.8)
ax1.axhline(1.0, color='k', linestyle='--', linewidth=1, label=r'$\chi^2_\nu = 1$ (ideal)')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.set_ylabel(r'$\chi^2/\nu$')
ax1.set_title('Reduced chi-square by model')
ax1.legend(fontsize=8)
fig1.tight_layout()
fig1.savefig(os.path.join(FIGDIR, 'p9_model_selection_reduced.png'), dpi=160)
print("saved figure p9_model_selection_reduced.png")

fig2, ax2 = plt.subplots(figsize=(6.2, 3.9))
ax2.bar(x - 0.2, [chi2_val[m] for m in models], width=0.4, color='tab:blue', label=r'$\chi^2$')
ax2.bar(x + 0.2, [chi2_crit[m] for m in models], width=0.4, color='tab:red', label=r'$\chi^2_{crit}(0.95)$')
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.set_ylabel(r'$\chi^2$')
ax2.set_title('Fit statistic vs. 95% critical value')
ax2.legend(fontsize=8)
fig2.tight_layout()
fig2.savefig(os.path.join(FIGDIR, 'p9_model_selection_crit.png'), dpi=160)
print("saved figure p9_model_selection_crit.png")
