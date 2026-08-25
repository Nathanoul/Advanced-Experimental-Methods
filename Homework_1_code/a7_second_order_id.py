import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from common import FIGDIR

OS = 0.20
ts_spec = 0.4
delta = 0.05

def overshoot_eq(zeta):
    return np.exp(-zeta*np.pi/np.sqrt(1-zeta**2)) - OS

zeta = brentq(overshoot_eq, 1e-6, 0.999999)
print("zeta =", zeta)

wn_envelope = np.log(1.0/(delta*np.sqrt(1-zeta**2)))/(zeta*ts_spec)
print("omega_n (envelope estimate) =", wn_envelope, "rad/s")

def settling_time_numeric(wn_, zeta_, delta_=delta, t_end_factor=8.0):
    Theta_ = np.arccos(zeta_)
    wd_ = wn_*np.sqrt(1-zeta_**2)
    tt = np.linspace(0, t_end_factor/(zeta_*wn_), 20000)
    yy = 1 - (1/np.sqrt(1-zeta_**2))*np.exp(-zeta_*wn_*tt)*np.sin(wd_*tt+Theta_)
    outside_ = np.where(np.abs(yy-1) > delta_)[0]
    return tt[outside_[-1]] if outside_.size else 0.0

def settle_residual(wn_):
    return settling_time_numeric(wn_, zeta) - ts_spec

wn = brentq(settle_residual, 1.0, 100.0)
print("omega_n (exact numeric match to ts=0.4 s) =", wn, "rad/s   (=", wn/(2*np.pi), "Hz)")

wd = wn*np.sqrt(1-zeta**2)
print("damped frequency wd =", wd, "rad/s")

K = 1.0

a2c, a1c, a0c = 1/wn**2, 2*zeta/wn, 1.0
print("Canonical equation: (1/wn^2) y'' + (2 zeta/wn) y' + y = K*F(t)")
print(f"  1/wn^2 = {a2c:.6e},  2*zeta/wn = {a1c:.6e}")

Theta = np.arccos(zeta)
def step_response(t):
    return K*(1 - (1/np.sqrt(1-zeta**2))*np.exp(-zeta*wn*t)*np.sin(wd*t+Theta))

t = np.linspace(0, 1.2, 3000)
y = step_response(t)

y_max = y.max()
OS_num = y_max - K
t_peak_num = t[np.argmax(y)]
print("numeric peak value:", y_max, " -> overshoot =", OS_num*100, "%  (target 20%)")
print("numeric peak time:", t_peak_num, " vs pi/wd =", np.pi/wd)

outside = np.where(np.abs(y-K) > delta*K)[0]
t_settle_num = t[outside[-1]] if outside.size else 0.0
print("numeric settling time (+/-5%):", t_settle_num, " (target 0.4 s)")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t, y, color="C0", label="reconstructed step response")
ax.axhline(K, color="k", lw=0.8)
ax.axhline(K*(1+delta), color="gray", ls=":", lw=1)
ax.axhline(K*(1-delta), color="gray", ls=":", lw=1)
ax.axvline(ts_spec, color="C3", ls="--", lw=1, label=f"settling time spec = {ts_spec} s")
ax.plot(t_peak_num, y_max, "ko", ms=5)
ax.annotate(f"{OS_num*100:.1f}% overshoot", xy=(t_peak_num, y_max),
            xytext=(t_peak_num+0.15, y_max+0.02))
ax.set_xlabel("t [s]")
ax.set_ylabel("y(t)/KA")
ax.set_title(f"Reconstructed 2nd-order step response ($\\zeta$={zeta:.3f}, $\\omega_n$={wn:.2f} rad/s)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIGDIR}/a7_step_response.png")
plt.close(fig)

def M2(w, zeta_, wn_):
    r = w/wn_
    return 1.0/np.sqrt((1-r**2)**2 + (2*zeta_*r)**2)

def phase2(w, zeta_, wn_):
    r = w/wn_
    return -np.arctan2(2*zeta_*r, 1-r**2)

w_axis = np.linspace(0.01, 4*wn, 500)
fig, axs = plt.subplots(2, 1, figsize=(7, 5.6), sharex=True)
axs[0].plot(w_axis/wn, M2(w_axis, zeta, wn), color="C0")
axs[0].axvline(1.0, color="gray", ls=":", lw=1)
axs[0].set_ylabel("$M(\\omega)$")
axs[0].set_title(f"Frequency response of the reconstructed system ($\\zeta$={zeta:.3f})")

axs[1].plot(w_axis/wn, np.degrees(phase2(w_axis, zeta, wn)), color="C3")
axs[1].axvline(1.0, color="gray", ls=":", lw=1)
axs[1].set_xlabel("$\\omega/\\omega_n$")
axs[1].set_ylabel("phase [deg]")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/a7_frequency_response.png")
plt.close(fig)

w_ex = wn
M_ex = M2(w_ex, zeta, wn)
phi_ex = phase2(w_ex, zeta, wn)
print(f"\nExample sinusoidal input at w=wn={w_ex:.2f} rad/s: M={M_ex:.4f}, phase={np.degrees(phi_ex):.2f} deg")

tt = np.linspace(0, 4*2*np.pi/w_ex, 800)
x_in = np.sin(w_ex*tt)
y_out = M_ex*np.sin(w_ex*tt+phi_ex)

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(tt, x_in, "--", color="k", label="input, $\\omega=\\omega_n$")
ax.plot(tt, y_out, color="C2", label=f"predicted output (M={M_ex:.2f}, $\\phi$={np.degrees(phi_ex):.1f}$^\\circ$)")
ax.set_xlabel("t [s]")
ax.set_ylabel("amplitude")
ax.set_title("Predicted steady-state response to a sinusoidal input at $\\omega=\\omega_n$")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIGDIR}/a7_sine_response.png")
plt.close(fig)

print("\nDone: figures saved to", FIGDIR)
