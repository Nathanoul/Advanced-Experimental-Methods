import numpy as np
from scipy.optimize import brentq, minimize_scalar
import matplotlib.pyplot as plt
from common import FIGDIR

f_max = 10.0
w_max = 2*np.pi*f_max
wn_r = 6*w_max
print("w_max (10 Hz) =", w_max, "rad/s")
print("chosen recorder wn_r =", wn_r, "rad/s  (=", wn_r/(2*np.pi), "Hz)")

def M1(w, tau):
    return 1.0/np.sqrt(1+(w*tau)**2)

def phase1(w, tau):
    return -np.arctan(w*tau)

def M2(w, zeta_, wn_):
    r = w/wn_
    return 1.0/np.sqrt((1-r**2)**2 + (2*zeta_*r)**2)

def phase2(w, zeta_, wn_):
    r = w/wn_
    return -np.arctan2(2*zeta_*r, 1-r**2)

def M_total(w, tau, zeta_):
    return M1(w, tau)*M2(w, zeta_, wn_r)

def phase_total(w, tau, zeta_):
    return phase1(w, tau) + phase2(w, zeta_, wn_r)

def low_freq_constraint(tau_):
    return M1(w_max, tau_) - 0.98

tau_opt = brentq(low_freq_constraint, 1e-8, 1.0)
print("Chosen sensor tau (M1(w_max)=0.98):", tau_opt, "s")

w_band = np.linspace(1e-3, w_max, 300)

def worst_case_ripple_dB(zeta_):
    Mt = M_total(w_band, tau_opt, zeta_)
    return np.max(np.abs(-20*np.log10(Mt)))

zetas = np.linspace(0.3, 1.2, 200)
ripple = [worst_case_ripple_dB(z) for z in zetas]
res = minimize_scalar(worst_case_ripple_dB, bounds=(0.3, 1.2), method="bounded")
zeta_opt = res.x
print("Numerically optimal zeta (minimises worst-case passband ripple):", zeta_opt)
print("  worst-case ripple there:", res.fun, "dB")
print("  ripple at the course's ζ=0.707 reference point:", worst_case_ripple_dB(0.707), "dB")

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(zetas, ripple, color="C0")
ax.axvline(0.707, color="gray", ls=":", lw=1, label="$\\zeta$=0.707 (course reference)")
ax.axvline(zeta_opt, color="C3", ls="--", lw=1, label=f"numeric optimum $\\zeta$={zeta_opt:.3f}")
ax.set_xlabel("$\\zeta$ (recorder)")
ax.set_ylabel("worst-case passband ripple [dB]")
ax.set_title("Choosing the recorder damping ratio")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIGDIR}/a8_zeta_optimization.png")
plt.close(fig)

w_axis = np.linspace(1e-3, 3*w_max, 500)
Mt_opt = M_total(w_axis, tau_opt, zeta_opt)
phit_opt = phase_total(w_axis, tau_opt, zeta_opt)

fig, axs = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)
axs[0].plot(w_axis, Mt_opt, color="C0", label="combined $M_{total}(\\omega)$")
axs[0].plot(w_axis, M1(w_axis, tau_opt), "--", color="C1", label="sensor alone")
axs[0].plot(w_axis, M2(w_axis, zeta_opt, wn_r), "--", color="C2", label="recorder alone")
axs[0].axvline(w_max, color="gray", ls=":", lw=1, label="10 Hz band edge")
axs[0].axhline(0.98, color="k", ls=":", lw=0.7)
axs[0].set_ylabel("magnitude ratio")
axs[0].set_title(f"Combined system: $\\tau$={tau_opt*1e3:.2f} ms, recorder $\\zeta$={zeta_opt:.3f}")
axs[0].legend(fontsize=8)

axs[1].plot(w_axis, np.degrees(phit_opt), color="C3")
axs[1].axvline(w_max, color="gray", ls=":", lw=1)
axs[1].set_xlabel("$\\omega$ [rad/s]")
axs[1].set_ylabel("phase [deg]")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/a8_combined_response.png")
plt.close(fig)

amp_err = 1 - M_total(w_band, tau_opt, zeta_opt)
phase_err = phase_total(w_band, tau_opt, zeta_opt)

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
axs[0].plot(w_band/(2*np.pi), amp_err*100, color="C0")
axs[0].set_xlabel("f [Hz]")
axs[0].set_ylabel("amplitude error [%]")
axs[0].set_title("Amplitude error over the 0-10 Hz band")

axs[1].plot(w_band/(2*np.pi), np.degrees(phase_err), color="C3")
axs[1].set_xlabel("f [Hz]")
axs[1].set_ylabel("phase error [deg]")
axs[1].set_title("Phase error over the 0-10 Hz band")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/a8_band_errors.png")
plt.close(fig)

print("\nmax amplitude error in band:", np.max(np.abs(amp_err))*100, "%")
print("max phase error in band:", np.degrees(np.max(np.abs(phase_err))), "deg")

f_a, f_b = 2.0, 8.0
w_a, w_b = 2*np.pi*f_a, 2*np.pi*f_b
tt = np.linspace(0, 1.0, 4000)
x_sig = np.sin(w_a*tt) + 0.6*np.sin(w_b*tt)
y_sig = (M_total(w_a, tau_opt, zeta_opt)*np.sin(w_a*tt+phase_total(w_a, tau_opt, zeta_opt))
         + 0.6*M_total(w_b, tau_opt, zeta_opt)*np.sin(w_b*tt+phase_total(w_b, tau_opt, zeta_opt)))

fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.plot(tt, x_sig, "--", color="k", lw=1, label="input, 2 Hz + 8 Hz")
ax.plot(tt, y_sig, color="C0", lw=1.3, label="combined-system output")
ax.set_xlim(0, 0.6)
ax.set_xlabel("t [s]")
ax.set_ylabel("amplitude")
ax.set_title("Time-domain check on an in-band two-tone signal")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIGDIR}/a8_time_domain_check.png")
plt.close(fig)

print("\nDone: figures saved to", FIGDIR)
