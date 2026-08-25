import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, freqs as freqs_resp, lsim


def main():
    os.makedirs('figures', exist_ok=True)

    fp, fstop, amax, amin = 2000.0, 8000.0, 1.0, 40.0
    n_exact = np.log10((10 ** (amin / 10) - 1) / (10 ** (amax / 10) - 1)) / (2 * np.log10(fstop / fp))
    N = int(np.ceil(n_exact))
    fc = fp / (10 ** (amax / 10) - 1) ** (1 / (2 * N))
    print('N', N, 'n_exact', n_exact, 'fc', fc)

    wc = 2 * np.pi * fc
    b, a = butter(N, wc, btype='low', analog=True)

    f_fine = np.logspace(np.log10(50), np.log10(60000), 4000)
    w_fine = 2 * np.pi * f_fine
    _, h_fine = freqs_resp(b, a, worN=w_fine)
    mag_db = 20 * np.log10(np.abs(h_fine))
    phase_deg = np.unwrap(np.angle(h_fine)) * 180 / np.pi

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(f_fine, mag_db)
    ax.axhline(-amax, ls=':', color='gray')
    ax.axhline(-amin, ls=':', color='gray')
    ax.axvline(fp, ls='--', color='r')
    ax.axvline(fstop, ls='--', color='r')
    ax.plot(fp, -amax, 'ro')
    ax.plot(fstop, -amin, 'ro')
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('|H| [dB]')
    ax.set_ylim(-80, 5)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig('figures/task3_amplitude_response.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(f_fine, phase_deg)
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('phase [deg]')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig('figures/task4_phase_response.png', dpi=150)
    plt.close(fig)

    FSAMPLING = 25000.0
    DUR = 0.05
    t = np.arange(0, DUR, 1 / FSAMPLING)
    rng = np.random.default_rng(42)
    x = (1.0 * np.sin(2 * np.pi * 1000 * t)
         + 0.6 * np.sin(2 * np.pi * 1800 * t)
         + 0.8 * np.sin(2 * np.pi * 10000 * t)
         + 0.3 * rng.standard_normal(len(t)))
    _, y, _ = lsim((b, a), U=x, T=t)
    print('RMS before', np.sqrt(np.mean(x ** 2)), 'after', np.sqrt(np.mean(y ** 2)))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    mask = t <= 0.004
    ax.plot(t[mask] * 1e3, x[mask], color='gray', lw=0.9, label='before filtering')
    ax.plot(t[mask] * 1e3, y[mask], color='tab:blue', lw=1.3, label='after filtering')
    ax.set_xlabel('time [ms]')
    ax.set_ylabel('amplitude')
    ax.legend()
    fig.tight_layout()
    fig.savefig('figures/task7_time_domain_compare.png', dpi=150)
    plt.close(fig)

    Xf = np.abs(np.fft.rfft(x))
    Yf = np.abs(np.fft.rfft(y))
    freqs_fft = np.fft.rfftfreq(len(t), 1 / FSAMPLING)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(freqs_fft, Xf, color='gray', lw=0.8, label='before filtering')
    ax.plot(freqs_fft, Yf, color='tab:blue', lw=1.0, label='after filtering')
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('|X(f)|')
    ax.legend()
    fig.tight_layout()
    fig.savefig('figures/task7_frequency_domain_compare.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
