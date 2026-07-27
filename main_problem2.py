import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline

FREQS = [40.0, 110.0, 260.0]
AMPS = [1.2, 0.5, 0.15]
T_TOTAL = 0.5


def analog_signal(t):
    t = np.asarray(t, dtype=float)
    s = np.zeros_like(t)
    for f, A in zip(FREQS, AMPS):
        s += A * np.sin(2 * np.pi * f * t)
    return s


def sample_signal(fs):
    N = int(round(fs * T_TOTAL))
    t = np.arange(N) / fs
    return t, analog_signal(t)


def quantize(x, bits, vmin=-2.0, vmax=2.0):
    lsb = (vmax - vmin) / (2 ** bits)
    xc = np.clip(x, vmin, vmax)
    return vmin + (np.floor((xc - vmin) / lsb) + 0.5) * lsb


def sinc_reconstruct(t_eval, t_samples, x_samples, fs):
    out = np.zeros_like(t_eval)
    for tn, xn in zip(t_samples, x_samples):
        arg = np.pi * fs * (t_eval - tn)
        term = np.where(np.abs(arg) < 1e-12, 1.0, np.sin(arg) / np.where(arg == 0, 1, arg))
        out += xn * term
    return out


def reconstruct(method, t_samples, x_samples, t_eval, fs):
    if method == 'linear':
        f = interp1d(t_samples, x_samples, kind='linear', bounds_error=False, fill_value='extrapolate')
        return f(t_eval)
    elif method == 'cubic':
        cs = CubicSpline(t_samples, x_samples)
        return cs(t_eval)
    elif method == 'sinc':
        return sinc_reconstruct(t_eval, t_samples, x_samples, fs)


def main():
    os.makedirs('figures', exist_ok=True)

    fs_list = [300, 600, 1000, 3000]
    bits_list = [8, 12, 14]
    methods = ['linear', 'cubic', 'sinc']
    fs_ref = 20000
    t_ref = np.arange(int(fs_ref * T_TOTAL)) / fs_ref
    x_ref = analog_signal(t_ref)

    results = {}
    for fs in fs_list:
        t_s, x_s = sample_signal(fs)
        for bits in bits_list:
            xq = quantize(x_s, bits)
            for m in methods:
                xr = reconstruct(m, t_s, xq, t_ref, fs)
                results[(fs, bits, m)] = np.sqrt(np.mean((x_ref - xr) ** 2))

    for bits in bits_list:
        for m in methods:
            row = [results[(fs, bits, m)] for fs in fs_list]
            print(m, 'bits', bits, [round(r, 4) for r in row])

    fs, bits = 300, 14
    t_s, x_s = sample_signal(fs)
    xq = quantize(x_s, bits)
    t_zoom = np.linspace(0, 0.05, 2000)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_ref[t_ref <= 0.05], x_ref[t_ref <= 0.05], 'k-', lw=1, label='original signal', alpha=0.5)
    ax.plot(t_s[t_s <= 0.05], xq[t_s <= 0.05], 'ko', ms=5, label='samples')
    colors = {'linear': 'tab:blue', 'cubic': 'tab:orange', 'sinc': 'tab:green'}
    for m in methods:
        xr = reconstruct(m, t_s, xq, t_zoom, fs)
        ax.plot(t_zoom, xr, color=colors[m], lw=1.3, label=f'{m} reconstruction')
    ax.set_xlabel('t [s]')
    ax.set_ylabel('amplitude [V]')
    ax.legend(fontsize=9)
    ax.set_title(f'Reconstruction methods, fs={fs} Hz, bits={bits}')
    fig.tight_layout()
    fig.savefig('figures/task2_reconstruction_methods.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, m in zip(axes, methods):
        grid = np.array([[results[(fs, bits, m)] for fs in fs_list] for bits in bits_list])
        im = ax.imshow(grid, aspect='auto', cmap='viridis')
        ax.set_xticks(range(len(fs_list)))
        ax.set_xticklabels(fs_list)
        ax.set_yticks(range(len(bits_list)))
        ax.set_yticklabels(bits_list)
        ax.set_xlabel('fs [Hz]')
        ax.set_ylabel('bits')
        ax.set_title(m)
        fig.colorbar(im, ax=ax, label='RMS error [V]')
    fig.tight_layout()
    fig.savefig('figures/task4_error_heatmaps.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, bits in zip(axes, bits_list):
        for m in methods:
            row = [results[(fs, bits, m)] for fs in fs_list]
            ax.plot(fs_list, row, marker='o', label=m, color=colors[m])
        ax.set_yscale('log')
        ax.set_xlabel('fs [Hz]')
        ax.set_title(f'bits={bits}')
        ax.grid(True, which='both', alpha=0.3)
    axes[0].set_ylabel('RMS error [V] (log)')
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig('figures/task4_error_vs_fs.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
