import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    os.makedirs('figures', exist_ok=True)

    S_TC = 41e-6
    sigma_V = 3e-6
    G = 100
    gain_unc = 0.002
    bits = 12
    V_ADC = 5.0

    u_noise = sigma_V / S_TC
    lsb = (2 * V_ADC) / (2 ** bits)
    u_q = (lsb / np.sqrt(12)) / (G * S_TC)
    dT_fs = V_ADC / (G * S_TC)
    u_gain = gain_unc * dT_fs
    u_total = np.sqrt(u_noise ** 2 + u_q ** 2 + u_gain ** 2)
    print('u_noise', u_noise, 'u_q', u_q, 'dT_fs', dT_fs, 'u_gain', u_gain, 'u_total', u_total)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    labels = ['Voltage\nnoise', 'ADC\nquantization', 'Gain\nuncertainty']
    vals = [u_noise, u_q, u_gain]
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    ax.bar(labels, vals, color=colors)
    ax.axhline(u_total, color='k', ls='--', label=f'Combined RSS = {u_total:.3f} degC')
    for i, v in enumerate(vals):
        ax.annotate(f'{v:.3f}', (i, v), textcoords='offset points', xytext=(0, 5), ha='center')
    ax.set_ylabel('uncertainty [deg C]')
    ax.legend()
    fig.tight_layout()
    fig.savefig('figures/task7_uncertainty_budget.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
