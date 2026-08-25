import numpy as np
import matplotlib.pyplot as plt
from common import FIGDIR

rng = np.random.default_rng(3)

fs = 200.0
T = 2.0
N = int(fs * T)
t = np.arange(N) / fs

f_nyquist = fs / 2
df = 1.0 / T
print("Nyquist / max resolvable frequency:", f_nyquist, "Hz")
print("Frequency resolution df = 1/T:", df, "Hz")

def amp_spectrum(sig, fs, window=None):
    n = sig.size
    w = np.ones(n) if window is None else window
    xw = sig * w
    X = np.fft.rfft(xw)
    f = np.fft.rfftfreq(n, d=1 / fs)
    amp = np.abs(X) / np.sum(w)
    return f, amp

f1, f2 = 25.0, 26.0
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, Tseg, label in zip(axs, [0.5, 2.0],
                            ["T = 0.5 s  ->  df = 2.0 Hz (cannot resolve)",
                             "T = 2.0 s  ->  df = 0.5 Hz (resolves both tones)"]):
    Nseg = int(fs * Tseg)
    tseg = np.arange(Nseg) / fs
    xseg = np.sin(2 * np.pi * f1 * tseg) + np.sin(2 * np.pi * f2 * tseg)
    f, amp = amp_spectrum(xseg, fs)
    ax.plot(f, amp, "o-", ms=3)
    ax.axvline(f1, color="r", ls="--", lw=0.8)
    ax.axvline(f2, color="r", ls="--", lw=0.8)
    ax.set_xlim(20, 31)
    ax.set_xlabel("f [Hz]")
    ax.set_ylabel("amplitude")
    ax.set_title(label)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/a3_resolution_tradeoff.png")
plt.close(fig)

f0 = 27.3
A0 = 0.15
noise_std = 1.0
x = A0 * np.sin(2 * np.pi * f0 * t) + rng.normal(0, noise_std, N)
f_single, amp_single = amp_spectrum(x, fs)

M = 20
f_ref, _ = amp_spectrum(x, fs)
power_accum = np.zeros_like(f_ref)
for k in range(M):
    xk = A0 * np.sin(2 * np.pi * f0 * t) + rng.normal(0, noise_std, N)
    _, amp_k = amp_spectrum(xk, fs)
    power_accum += amp_k ** 2
power_avg = power_accum / M

fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
axs[0].plot(f_single, amp_single ** 2, color="C0")
axs[0].axvline(f0, color="r", ls="--", lw=1, label=f"true $f_0$={f0} Hz")
axs[0].set_xlim(0, 50)
axs[0].set_title("Single 2 s record")
axs[0].set_xlabel("f [Hz]"); axs[0].set_ylabel("power")
axs[0].legend()

axs[1].plot(f_ref, power_avg, color="C0")
axs[1].axvline(f0, color="r", ls="--", lw=1, label=f"true $f_0$={f0} Hz")
axs[1].set_xlim(0, 50)
axs[1].set_title(f"Average of M={M} independent records")
axs[1].set_xlabel("f [Hz]"); axs[1].set_ylabel("power")
axs[1].legend()
fig.tight_layout()
fig.savefig(f"{FIGDIR}/a3_averaging_detection.png")
plt.close(fig)

f_int = 10.0
f_noint = 10.3
win = np.hanning(N)

fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
x_int = np.sin(2 * np.pi * f_int * t)
f_, amp_ = amp_spectrum(x_int, fs)
axs[0].semilogy(f_, amp_ + 1e-6, "o-", ms=3)
axs[0].set_title(f"$f$={f_int} Hz, integer cycles in T")

x_noint = np.sin(2 * np.pi * f_noint * t)
f_, amp_ = amp_spectrum(x_noint, fs)
axs[1].semilogy(f_, amp_ + 1e-6, "o-", ms=3, color="C3")
axs[1].set_title(f"$f$={f_noint} Hz, no window (leaks)")

f_, amp_w = amp_spectrum(x_noint, fs, window=win)
axs[2].semilogy(f_, amp_w + 1e-6, "o-", ms=3, color="C2")
axs[2].set_title(f"$f$={f_noint} Hz, Hanning window")

for ax in axs:
    ax.set_xlim(0, 40)
    ax.set_ylim(1e-4, 1)
    ax.set_xlabel("f [Hz]")
    ax.set_ylabel("amplitude (log scale)")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/a3_leakage.png")
plt.close(fig)

print("Done: figures saved to", FIGDIR)
