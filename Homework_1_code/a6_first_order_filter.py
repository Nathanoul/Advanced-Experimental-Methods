import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from common import FIGDIR

w1, A1 = 2.0, 1.0
w2, A2 = 40.0, 1.0

def M(w, tau):
    return 1.0/np.sqrt(1+(w*tau)**2)

def phase(w, tau):
    return -np.arctan(w*tau)

tau_demo = 0.05
t = np.linspace(0, 3, 3000)
x = A1*np.sin(w1*t) + A2*np.sin(w2*t)
y_steady = (A1*M(w1, tau_demo)*np.sin(w1*t + phase(w1, tau_demo))
            + A2*M(w2, tau_demo)*np.sin(w2*t + phase(w2, tau_demo)))

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(t, x, color="gray", lw=1, label="input $x(t)=\\sin(2t)+\\sin(40t)$")
ax.plot(t, y_steady, color="C0", lw=1.3, label=f"steady output, $\\tau$={tau_demo} s")
ax.set_xlim(0, 2)
ax.set_xlabel("t [s]")
ax.set_ylabel("amplitude")
ax.set_title("First-order sensor acting as a low-pass filter")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIGDIR}/a6_timedomain.png")
plt.close(fig)

w_axis = np.logspace(-2, 3, 400)
fig, ax = plt.subplots(figsize=(6.5, 4))
for tau_ in [0.01, 0.025, 0.05, 0.1]:
    ax.semilogx(w_axis, M(w_axis, tau_), label=f"$\\tau$={tau_} s (cutoff={1/tau_:.0f} rad/s)")
ax.axhline(1/np.sqrt(2), color="k", ls=":", lw=1, label="$M=1/\\sqrt{2}$ (cutoff)")
ax.set_xlabel("$\\omega$ [rad/s]")
ax.set_ylabel("$M(\\omega)$")
ax.set_title("Magnitude ratio of a first-order system")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/a6_magnitude.png")
plt.close(fig)

def low_freq_constraint(tau_):
    return M(w1, tau_) - 0.98

def high_freq_constraint(tau_):
    return M(w2, tau_) - 0.10

tau_max_low = brentq(low_freq_constraint, 1e-6, 10)
tau_min_high = brentq(high_freq_constraint, 1e-6, 10)

print("Upper bound on tau from low-frequency (<2% error) requirement: tau <=", tau_max_low)
print("Lower bound on tau from high-frequency (>90% attenuation) requirement: tau >=", tau_min_high)

if tau_max_low >= tau_min_high:
    tau_design = 0.5*(tau_min_high + tau_max_low)
    print("Both requirements can be met; chosen design tau (midpoint):", tau_design)
else:
    print("The two requirements do NOT overlap (tau<=%.4f and tau>=%.4f cannot both hold):"
          " a single first-order stage cannot meet both specs at once." % (tau_max_low, tau_min_high))
    tau_design = tau_max_low
    print("Chosen tau = the low-frequency bound,", tau_design,
          "so the wanted 2 rad/s component is kept within 2% error.")

print("Check M(w1,tau_design) =", M(w1, tau_design), " (need >=0.98)")
print("Check M(w2,tau_design) =", M(w2, tau_design), " (need <=0.10 for >90% attenuation)")
print("Actual attenuation of the 40 rad/s component at this tau:",
      (1-M(w2, tau_design))*100, "%")

fig, ax = plt.subplots(figsize=(6.5, 4))
tau_axis = np.linspace(1e-4, 0.4, 400)
ax.plot(tau_axis, M(w1, tau_axis), label="$M(\\omega_1=2)$  (must stay $\\geq$0.98)")
ax.plot(tau_axis, M(w2, tau_axis), label="$M(\\omega_2=40)$  (must drop $\\leq$0.10)")
ax.axhline(0.98, color="C0", ls=":", lw=1)
ax.axhline(0.10, color="C1", ls=":", lw=1)

ax.axvspan(0, tau_max_low, color="C0", alpha=0.12, label="tau satisfies low-freq spec")

ax.axvspan(tau_min_high, tau_axis[-1], color="C1", alpha=0.12, label="tau satisfies high-freq spec")
ax.axvline(tau_design, color="k", ls="--", lw=1, label=f"chosen $\\tau$={tau_design:.4f} s")
ax.set_xlabel("$\\tau$ [s]")
ax.set_ylabel("magnitude ratio")
ax.set_ylim(0, 1.05)
ax.set_xlim(0, tau_axis[-1])
ax.set_title("The two requirements do not overlap: no single $\\tau$ meets both")
ax.legend(fontsize=7.5, loc="upper right")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/a6_tau_design.png")
plt.close(fig)

y_design = (A1*M(w1, tau_design)*np.sin(w1*t + phase(w1, tau_design))
            + A2*M(w2, tau_design)*np.sin(w2*t + phase(w2, tau_design)))
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(t, x, color="gray", lw=1, label="input $x(t)$")
ax.plot(t, y_design, color="C2", lw=1.3, label=f"output with designed $\\tau$={tau_design:.4f} s")
ax.set_xlim(0, 2)
ax.set_xlabel("t [s]")
ax.set_ylabel("amplitude")
ax.set_title("Response with the designed time constant")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIGDIR}/a6_designed_response.png")
plt.close(fig)

print("Done: figures saved to", FIGDIR)
