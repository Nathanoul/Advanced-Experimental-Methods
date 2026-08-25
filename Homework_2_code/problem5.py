import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

s = 7.5
N = 12
nu = N - 1

chi2_lo = stats.chi2.ppf(0.025, nu)
chi2_hi = stats.chi2.ppf(0.975, nu)
print(f"nu={nu}, chi2_0.025={chi2_lo:.4f}, chi2_0.975={chi2_hi:.4f}")

var_lo = nu * s ** 2 / chi2_hi
var_hi = nu * s ** 2 / chi2_lo
print(f"95% CI for variance: [{var_lo:.4f}, {var_hi:.4f}] MPa^2")

std_lo, std_hi = np.sqrt(var_lo), np.sqrt(var_hi)
print(f"95% CI for std: [{std_lo:.4f}, {std_hi:.4f}] MPa")
print(f"distance below s: {s-std_lo:.4f}, distance above s: {std_hi-s:.4f}")

N2 = 30
nu2 = N2 - 1
chi2_lo2 = stats.chi2.ppf(0.025, nu2)
chi2_hi2 = stats.chi2.ppf(0.975, nu2)
var_lo2 = nu2 * s ** 2 / chi2_hi2
var_hi2 = nu2 * s ** 2 / chi2_lo2
std_lo2, std_hi2 = np.sqrt(var_lo2), np.sqrt(var_hi2)
print(f"\nN=30: nu={nu2}, chi2_0.025={chi2_lo2:.4f}, chi2_0.975={chi2_hi2:.4f}")
print(f"95% CI for std (N=30): [{std_lo2:.4f}, {std_hi2:.4f}] MPa")
print(f"relative half-width N=12: {(std_hi-std_lo)/2/s*100:.1f}%  N=30: {(std_hi2-std_lo2)/2/s*100:.1f}%")

Ns = np.arange(4, 61)
lo_arr = np.zeros_like(Ns, dtype=float)
hi_arr = np.zeros_like(Ns, dtype=float)
for i, Nv in enumerate(Ns):
    nuv = Nv - 1
    cl = stats.chi2.ppf(0.025, nuv)
    ch = stats.chi2.ppf(0.975, nuv)
    lo_arr[i] = np.sqrt(nuv * s ** 2 / ch)
    hi_arr[i] = np.sqrt(nuv * s ** 2 / cl)

fig, ax = plt.subplots(figsize=(6.6, 4.0))
ax.plot(Ns, lo_arr, color='tab:blue', label='lower bound')
ax.plot(Ns, hi_arr, color='tab:red', label='upper bound')
ax.fill_between(Ns, lo_arr, hi_arr, color='tab:orange', alpha=0.2)
ax.axhline(s, color='k', linestyle=':', linewidth=1, label='sample $s=7.5$ MPa')
for Nv, lo, hi in [(12, std_lo, std_hi), (30, std_lo2, std_hi2)]:
    ax.plot([Nv, Nv], [lo, hi], 'ko', markersize=5)
    ax.annotate(f'N={Nv}', (Nv, hi), textcoords="offset points", xytext=(5, 5), fontsize=8)
ax.set_xlabel('number of specimens N')
ax.set_ylabel(r'true standard deviation $\sigma$ [MPa]')
ax.set_title('95% confidence interval for $\\sigma$ vs. sample size')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'p5_chi2_variance.png'), dpi=160)
print("saved figure p5_chi2_variance.png")
