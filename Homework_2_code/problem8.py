import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

t = np.array([0.12, 0.37, 0.62, 0.87, 1.12, 1.37])
x = np.array([14.8, 10.9, 8.1, 6.0, 4.5, 3.4])
N = len(t)

y = np.log(x)

m, b = np.polyfit(t, y, 1)
print(f"slope m = {m:.5f}, intercept b = {b:.5f}")

zeta_wn = -m
print(f"decay constant zeta*wn = {zeta_wn:.5f} 1/s")

A = np.exp(b)
print(f"A = {A:.5f} mm")

y_fit = m * t + b
resid_log = y - y_fit
x_fit = np.exp(y_fit)
resid_phys = x - x_fit
print("\ni   t     x     x_fit    resid(mm)   ln(x)    ln(x)_fit  resid(ln)")
for i in range(N):
    print(f"{i+1}  {t[i]:.2f}  {x[i]:5.2f}  {x_fit[i]:6.3f}  {resid_phys[i]:+7.4f}    "
          f"{y[i]:.4f}   {y_fit[i]:.4f}    {resid_log[i]:+.5f}")

rms_log = np.sqrt(np.mean(resid_log ** 2))
print(f"\nRMS residual (log domain) = {rms_log:.5f}")

fig1, ax1 = plt.subplots(figsize=(6.4, 3.9))
ax1.plot(t, y, 'o', color='tab:blue', label='data: $\\ln(x_i)$')
tt = np.linspace(0, 1.5, 100)
ax1.plot(tt, m * tt + b, color='tab:red', label='linear fit')
ax1.set_xlabel('$t$ [s]')
ax1.set_ylabel('$\\ln(x_{peak})$')
ax1.set_title('Linearized peak amplitudes and fit')
ax1.legend(fontsize=8)
fig1.tight_layout()
fig1.savefig(os.path.join(FIGDIR, 'p8_spring_damper_fit.png'), dpi=160)
print("saved figure p8_spring_damper_fit.png")

fig2, ax2 = plt.subplots(figsize=(6.4, 3.9))
ax2.axhline(0, color='k', linewidth=1)
ax2.plot(t, resid_log, 'o-', color='tab:purple')
ax2.set_xlabel('$t$ [s]')
ax2.set_ylabel('residual, $\\ln(x_i) - $ fit')
ax2.set_title('Fit residuals')
fig2.tight_layout()
fig2.savefig(os.path.join(FIGDIR, 'p8_spring_damper_resid.png'), dpi=160)
print("saved figure p8_spring_damper_resid.png")
