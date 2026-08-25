import numpy as np
import matplotlib.pyplot as plt
from common import FIGDIR

fs = 200.0
f_true = [30.0, 90.0, 160.0]

def folded_observed_freq(f, fs):
    FN = 2 * f / fs
    FN_mod = FN % 2
    if FN_mod <= 1:
        FN_fold = FN_mod
    else:
        FN_fold = 2 - FN_mod
    return FN_fold * fs / 2, FN

print("fs =", fs, " Nyquist =", fs / 2)
for f in f_true:
    f_obs, FN = folded_observed_freq(f, fs)
    print(f"f_true={f:6.1f} Hz  FN={FN:5.2f}  -> observed at {f_obs:6.1f} Hz "
          f"{'(no aliasing)' if abs(f_obs-f) < 1e-9 else '(ALIASED)'}")

FN_axis = np.linspace(0, 3, 400)
fold = np.where(FN_axis <= 1, FN_axis,
        np.where(FN_axis <= 2, 2 - FN_axis, FN_axis - 2))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(FN_axis, fold, color="C0")
for f in f_true:
    _, FN = folded_observed_freq(f, fs)
    FN_mod = FN % 2
    fold_val = FN_mod if FN_mod <= 1 else 2 - FN_mod
    ax.plot(FN, fold_val, "o", ms=7, label=f"{f:.0f} Hz -> $F_N$={FN:.2f}")
ax.set_xlabel("$F_N = 2f/f_s$")
ax.set_ylabel("folded value")
ax.set_title("Folding diagram, $f_s$=200 Hz")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIGDIR}/a4_folding_diagram.png")
plt.close(fig)

T = 1.0
t_fine = np.arange(0, T, 1/20000.0)
x_fine = sum(np.sin(2*np.pi*f*t_fine) for f in f_true)

t_samp = np.arange(0, T, 1/fs)
x_samp = sum(np.sin(2*np.pi*f*t_samp) for f in f_true)

def amp_spectrum(sig, fs_):
    n = sig.size
    X = np.fft.rfft(sig)
    f = np.fft.rfftfreq(n, d=1/fs_)
    return f, np.abs(X)/n*2

f1, a1 = amp_spectrum(x_fine, 20000.0)
f2, a2 = amp_spectrum(x_samp, fs)

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
axs[0].plot(f1, a1, color="C0")
axs[0].set_xlim(0, 200)
axs[0].set_title("True spectrum (finely sampled reference)")
axs[0].set_xlabel("f [Hz]"); axs[0].set_ylabel("amplitude")
for f in f_true:
    axs[0].axvline(f, color="gray", ls=":", lw=0.8)

axs[1].plot(f2, a2, color="C3")
axs[1].set_xlim(0, fs/2)
axs[1].set_title(f"Spectrum measured at $f_s$={fs:.0f} Hz")
axs[1].set_xlabel("f [Hz]"); axs[1].set_ylabel("amplitude")
axs[1].axvline(30, color="gray", ls=":", lw=0.8)
axs[1].axvline(90, color="gray", ls=":", lw=0.8)
axs[1].axvline(40, color="gray", ls=":", lw=0.8)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/a4_spectrum_aliasing.png")
plt.close(fig)

print("\nDone: figures saved to", FIGDIR)
