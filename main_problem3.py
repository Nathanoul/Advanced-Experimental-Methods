import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

S = 0.020
PRESSURE_RANGE = 200.0
ADCS = {
    'A': dict(bits=8, rng=(-5.0, 5.0), gain_pct=0.3, offset_V=5e-3),
    'B': dict(bits=12, rng=(-5.0, 5.0), gain_pct=0.1, offset_V=2e-3),
    'C': dict(bits=16, rng=(-10.0, 10.0), gain_pct=0.05, offset_V=1e-3),
}


def main():
    os.makedirs('figures', exist_ok=True)

    results = {}
    for name, spec in ADCS.items():
        fso = spec['rng'][1] - spec['rng'][0]
        lsb = fso / (2 ** spec['bits'])
        uq_V = lsb / np.sqrt(12)
        ugain_V = spec['gain_pct'] / 100 * fso
        usys_V = np.sqrt(ugain_V ** 2 + spec['offset_V'] ** 2)
        uc_V = np.sqrt(uq_V ** 2 + usys_V ** 2)
        uq_kPa = uq_V / S
        usys_kPa = usys_V / S
        uc_kPa = uc_V / S
        pct_span = 100 * uc_kPa / (2 * PRESSURE_RANGE)
        results[name] = dict(lsb=lsb, uq=uq_kPa, usys=usys_kPa, uc=uc_kPa, pct=pct_span)
        print(name, {k: round(v, 4) for k, v in results[name].items()})

    names = list(results.keys())
    uq = [results[n]['uq'] for n in names]
    usys = [results[n]['usys'] for n in names]
    uc = [results[n]['uc'] for n in names]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(names))
    ax.bar(x, uq, label='quantization', color='tab:blue')
    ax.bar(x, usys, bottom=uq, label='gain+offset (systematic)', color='tab:orange')
    for xi, u in zip(x, uc):
        ax.plot([xi - 0.3, xi + 0.3], [u, u], color='k', lw=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f'ADC {n}' for n in names])
    ax.set_ylabel('uncertainty [kPa]')
    ax.set_title('Uncertainty budget per ADC')
    ax.legend()
    fig.tight_layout()
    fig.savefig('figures/task4_uncertainty_budget.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    pct = [results[n]['pct'] for n in names]
    ax.bar(x, pct, color='steelblue', width=0.5)
    for xi, v in zip(x, pct):
        ax.annotate(f'{v:.3f}%', (xi, v), textcoords='offset points', xytext=(0, 6), ha='center')
    ax.set_xticks(x)
    ax.set_xticklabels([f'ADC {n}' for n in names])
    ax.set_ylabel('combined uncertainty [% of pressure span]')
    fig.tight_layout()
    fig.savefig('figures/task5_adc_comparison.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
