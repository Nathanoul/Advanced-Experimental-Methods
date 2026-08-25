import numpy as np
import matplotlib.pyplot as plt
from common import FIGDIR

tau = 0.3

def y_region2(t):
    return t - tau + tau*np.exp(-t/tau)

y2_at_2 = y_region2(2.0)
print("y(2^-) from region 2:", y2_at_2)

D = y2_at_2 - 2.0
print("D =", D)

def y_region3(t):
    return 2.0 + D*np.exp(-(t-2.0)/tau)

t1 = np.linspace(-0.5, 0, 50)
t2 = np.linspace(0, 2, 400)
t3 = np.linspace(2, 5, 400)

y1 = np.zeros_like(t1)
y2 = y_region2(t2)
y3 = y_region3(t3)

x1 = np.zeros_like(t1)
x2 = t2
x3 = 2.0*np.ones_like(t3)

t_all = np.concatenate([t1, t2, t3])
y_all = np.concatenate([y1, y2, y3])
x_all = np.concatenate([x1, x2, x3])

err_all = x_all - y_all

i_max = np.argmax(np.abs(err_all))
print("Max |dynamic error| =", abs(err_all[i_max]), " at t =", t_all[i_max])

print("Steady-state ramp-following error (t>>tau, region 2) = tau =", tau)

fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
axs[0].plot(t_all, x_all, "--", color="k", label="input $x(t)$")
axs[0].plot(t_all, y_all, color="C0", label="response $y(t)$, $\\tau$=0.3")
axs[0].axvline(0, color="gray", lw=0.6)
axs[0].axvline(2, color="gray", lw=0.6)
axs[0].set_ylabel("amplitude")
axs[0].set_title("First-order system response to a ramp-then-hold input")
axs[0].legend()

axs[1].plot(t_all, err_all, color="C3")
axs[1].axvline(0, color="gray", lw=0.6)
axs[1].axvline(2, color="gray", lw=0.6)
axs[1].axhline(0, color="k", lw=0.6)
axs[1].set_xlabel("t [s]")
axs[1].set_ylabel("$x(t)-y(t)$")
axs[1].set_title("Dynamic error")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/a5_response.png")
plt.close(fig)

print("Done: figure saved to", FIGDIR)
