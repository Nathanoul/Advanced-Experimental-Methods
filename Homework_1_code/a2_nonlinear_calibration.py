import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from common import FIGDIR

rng = np.random.default_rng(2)

a = 0.02
b = -2e-5
c = 0.10
Pmax = 200.0

def V_true(P):
    return a * P + b * P**2 + c

V0 = V_true(0.0)
Vmax = V_true(Pmax)
a_lin = (Vmax - V0) / Pmax
b_lin = V0

def V_lin(P):
    return a_lin * P + b_lin

def error(P):

    return b * P * (P - Pmax)

P = np.linspace(0, Pmax, 400)

print("a_lin =", a_lin, " (true a =", a, ")")
print("b_lin (intercept) =", b_lin, " (true c =", c, ")")
print("max |error| predicted at P=Pmax/2:", abs(b) * Pmax**2 / 4)
print("error at Pmax/2 numeric:", error(Pmax/2))

fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
axs[0].plot(P, V_true(P), label="true response $V=aP+bP^2+c$")
axs[0].plot(P, V_lin(P), "--", label="linear calibration (zero + full-scale)")
axs[0].scatter([0, Pmax], [V0, Vmax], color="k", zorder=5, label="calibration points")
axs[0].set_xlabel("P [kPa]")
axs[0].set_ylabel("V [V]")
axs[0].set_title("True sensor curve vs. linear calibration")
axs[0].legend()

axs[1].plot(P, error(P), color="C3")
axs[1].axhline(0, color="k", lw=0.8)
axs[1].set_xlabel("P [kPa]")
axs[1].set_ylabel("$V_{true}-V_{lin}$ [V]")
axs[1].set_title("Systematic error vs. pressure")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/a2_error_curve.png")
plt.close(fig)

P_cal = np.arange(0, Pmax + 1e-9, 20.0)
sigma_V = 0.03
N = P_cal.size

V_data = V_true(P_cal) + rng.normal(0, sigma_V, N)

coef1 = np.polyfit(P_cal, V_data, 1)
coef2 = np.polyfit(P_cal, V_data, 2)
fit1 = np.polyval(coef1, P_cal)
fit2 = np.polyval(coef2, P_cal)

dof1 = N - 2
dof2 = N - 3

chi2_1 = np.sum((V_data - fit1)**2) / sigma_V**2
chi2_2 = np.sum((V_data - fit2)**2) / sigma_V**2

crit1 = stats.chi2.ppf(0.95, dof1)
crit2 = stats.chi2.ppf(0.95, dof2)

print("\nLinear fit:   chi2 =", chi2_1, " dof =", dof1, " crit(95%) =", crit1,
      " ->", "REJECT" if chi2_1 > crit1 else "accept")
print("Quadratic fit: chi2 =", chi2_2, " dof =", dof2, " crit(95%) =", crit2,
      " ->", "REJECT" if chi2_2 > crit2 else "accept")

Pfine = np.linspace(0, Pmax, 300)
fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(P_cal, V_data, yerr=sigma_V, fmt="o", color="k", capsize=3, label="noisy calibration data")
ax.plot(Pfine, np.polyval(coef1, Pfine), "--", label=f"linear fit ($\\chi^2/\\nu$={chi2_1/dof1:.2f})")
ax.plot(Pfine, np.polyval(coef2, Pfine), "-", label=f"quadratic fit ($\\chi^2/\\nu$={chi2_2/dof2:.2f})")
ax.set_xlabel("P [kPa]")
ax.set_ylabel("V [V]")
ax.set_title("Goodness-of-fit test: linear vs. quadratic model")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIGDIR}/a2_chi2_test.png")
plt.close(fig)

n_rep = 6
V_rep = V_true(P_cal)[:, None] + rng.normal(0, sigma_V, (N, n_rep))
rep_mean = V_rep.mean(axis=1)
rep_std = V_rep.std(axis=1, ddof=1)

lin_of_means = np.polyfit(P_cal, rep_mean, 1)
resid = rep_mean - np.polyval(lin_of_means, P_cal)

print("\nMean repeat-measurement std (noise level):", rep_std.mean())
print("Std of the residual trend (model error level):", resid.std())

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.errorbar(P_cal, rep_mean, yerr=rep_std, fmt="o", capsize=3, color="C0",
            label="mean of 6 repeats $\\pm$ 1 std (repeatability / noise)")
ax.plot(P_cal, np.polyval(lin_of_means, P_cal), "--", color="C1", label="best linear fit through the means")
ax.set_xlabel("P [kPa]")
ax.set_ylabel("V [V]")
ax.set_title("Repeated trials: random scatter vs. systematic trend")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIGDIR}/a2_repeatability.png")
plt.close(fig)

print("\nDone: figures saved to", FIGDIR)
