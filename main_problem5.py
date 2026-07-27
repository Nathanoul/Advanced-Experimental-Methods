import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, buttord, freqz, group_delay, lfilter, filtfilt, firwin, kaiserord


def main():
    os.makedirs('figures', exist_ok=True)

    P0, TAU, F0, FS = 1.0, 2.0e-3, 3.0e3, 100.0e3
    A_50, F_50 = 0.15, 50.0
    A_HF, F_HF = 0.20, 30.0e3
    SIGMA_RAND = 0.05
    DUR = 5 * TAU
    SEED = 0

    n = int(round(DUR * FS))
    t = np.arange(n) / FS
    p_clean = P0 * np.exp(-t / TAU) * np.sin(2 * np.pi * F0 * t)
    rng = np.random.default_rng(SEED)
    n_50 = A_50 * np.sin(2 * np.pi * F_50 * t)
    n_hf = A_HF * np.sin(2 * np.pi * F_HF * t)
    n_rand = rng.normal(0.0, SIGMA_RAND, size=t.shape)
    p_noisy = p_clean + n_50 + n_hf + n_rand

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t * 1e3, p_noisy, color='gray', lw=0.8, label='noisy signal')
    ax.plot(t * 1e3, p_clean, color='tab:red', lw=1.4, label='clean transient')
    ax.set_xlabel('time [ms]')
    ax.set_ylabel('pressure [MPa]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('figures/task1_noisy_signal.png', dpi=150)
    plt.close(fig)

    fp, fstop, Ap, Astop = 6000.0, 20000.0, 1.0, 40.0
    N_iir, Wn = buttord(fp, fstop, Ap, Astop, fs=FS)
    b_iir, a_iir = butter(N_iir, Wn, btype='low', fs=FS)
    print('IIR N', N_iir, 'Wn', Wn)

    w, h = freqz(b_iir, a_iir, worN=8000, fs=FS)
    mag_db = 20 * np.log10(np.abs(h) + 1e-300)
    phase_deg = np.unwrap(np.angle(h)) * 180 / np.pi

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(w, mag_db)
    axes[0].axvline(fp, ls='--', color='r')
    axes[0].axvline(fstop, ls='--', color='r')
    axes[0].axhline(-Ap, ls=':', color='gray')
    axes[0].axhline(-Astop, ls=':', color='gray')
    axes[0].set_ylabel('|H| [dB]')
    axes[0].set_ylim(-80, 5)
    axes[1].plot(w, phase_deg)
    axes[1].set_xlabel('frequency [Hz]')
    axes[1].set_ylabel('phase [deg]')
    fig.tight_layout()
    fig.savefig('figures/task2_iir_response.png', dpi=150)
    plt.close(fig)

    width = fstop - fp
    ripple = Astop
    Ntaps, beta = kaiserord(ripple, width / (FS / 2))
    if Ntaps % 2 == 0:
        Ntaps += 1
    fc_fir = (fp + fstop) / 2
    h_fir = firwin(Ntaps, fc_fir, window=('kaiser', beta), fs=FS)
    print('FIR taps', Ntaps, 'beta', beta)

    w2, h2 = freqz(h_fir, [1.0], worN=8000, fs=FS)
    mag_db2 = 20 * np.log10(np.abs(h2) + 1e-300)
    phase_deg2 = np.unwrap(np.angle(h2)) * 180 / np.pi
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(w2, mag_db2)
    axes[0].set_ylabel('|H| [dB]')
    axes[0].set_ylim(-80, 5)
    axes[1].plot(w2, phase_deg2)
    axes[1].set_xlabel('frequency [Hz]')
    axes[1].set_ylabel('phase [deg]')
    fig.tight_layout()
    fig.savefig('figures/task3_fir_response.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(w, mag_db, label='IIR Butterworth')
    ax.plot(w2, mag_db2, label='FIR (Kaiser)')
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('|H| [dB]')
    ax.set_ylim(-80, 5)
    ax.legend()
    fig.tight_layout()
    fig.savefig('figures/task4_magnitude_comparison.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(w, phase_deg, label='IIR Butterworth')
    ax.plot(w2, phase_deg2, label='FIR (Kaiser)')
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('phase [deg]')
    ax.legend()
    fig.tight_layout()
    fig.savefig('figures/task4_phase_comparison.png', dpi=150)
    plt.close(fig)

    wgd, gd_iir = group_delay((b_iir, a_iir), w=w, fs=FS)
    _, gd_fir = group_delay((h_fir, [1.0]), w=w, fs=FS)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(wgd, gd_iir / FS * 1e3, label='IIR Butterworth')
    ax.plot(wgd, gd_fir / FS * 1e3, label='FIR (Kaiser)')
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('group delay [ms]')
    ax.legend()
    fig.tight_layout()
    fig.savefig('figures/task4_group_delay_comparison.png', dpi=150)
    plt.close(fig)

    p_iir_c = lfilter(b_iir, a_iir, p_noisy)
    p_fir_c = lfilter(h_fir, [1.0], p_noisy)
    rms_noisy = np.sqrt(np.mean((p_noisy - p_clean) ** 2))
    rms_iir = np.sqrt(np.mean((p_iir_c - p_clean) ** 2))
    rms_fir = np.sqrt(np.mean((p_fir_c - p_clean) ** 2))
    print('RMS noisy', rms_noisy, 'iir causal', rms_iir, 'fir causal', rms_fir)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t * 1e3, p_clean, color='tab:red', lw=1.4, label='clean transient')
    ax.plot(t * 1e3, p_iir_c, color='tab:blue', lw=1.0, label='IIR, causal')
    ax.plot(t * 1e3, p_fir_c, color='tab:orange', lw=1.0, label='FIR, causal')
    ax.set_xlabel('time [ms]')
    ax.set_ylabel('pressure [MPa]')
    ax.legend()
    fig.tight_layout()
    fig.savefig('figures/task4_waveform_comparison.png', dpi=150)
    plt.close(fig)

    p_iir_zp = filtfilt(b_iir, a_iir, p_noisy)
    p_fir_zp = filtfilt(h_fir, [1.0], p_noisy)
    rms_iir_zp = np.sqrt(np.mean((p_iir_zp - p_clean) ** 2))
    rms_fir_zp = np.sqrt(np.mean((p_fir_zp - p_clean) ** 2))
    print('RMS iir zerophase', rms_iir_zp, 'fir zerophase', rms_fir_zp)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(t * 1e3, p_clean, 'r-', lw=1.3, label='clean')
    axes[0].plot(t * 1e3, p_iir_c, 'b-', lw=0.9, label='causal')
    axes[0].plot(t * 1e3, p_iir_zp, 'g-', lw=0.9, label='zero-phase')
    axes[0].set_title('IIR')
    axes[0].legend()
    axes[1].plot(t * 1e3, p_clean, 'r-', lw=1.3, label='clean')
    axes[1].plot(t * 1e3, p_fir_c, 'b-', lw=0.9, label='causal')
    axes[1].plot(t * 1e3, p_fir_zp, 'g-', lw=0.9, label='zero-phase')
    axes[1].set_title('FIR')
    axes[1].set_xlabel('time [ms]')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig('figures/task5_causal_vs_zerophase.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
