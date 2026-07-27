import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

L9 = [
    [1, 1, 1, 1], [1, 2, 2, 2], [1, 3, 3, 3],
    [2, 1, 2, 3], [2, 2, 3, 1], [2, 3, 1, 2],
    [3, 1, 3, 2], [3, 2, 1, 3], [3, 3, 2, 1],
]
LEVEL_TO_CODED = {1: -1, 2: 0, 3: 1}
FACTORS = ['A', 'B', 'C', 'D']


def pmax(A, B, C, D):
    return 120 + 15 * A + 55 * B - 28 * C + 6 * D + 8 * B ** 2 - 5 * C ** 2 + 18 * A * B


def main():
    os.makedirs('figures', exist_ok=True)

    N_REPS = 3
    SIGMA = 6.0
    SEED = 42
    rng = np.random.default_rng(SEED)

    runs_coded = [[LEVEL_TO_CODED[l] for l in row] for row in L9]
    data = np.zeros((9, N_REPS))
    det = np.zeros(9)
    for i, (A, B, C, D) in enumerate(runs_coded):
        y = pmax(A, B, C, D)
        det[i] = y
        data[i, :] = y + rng.normal(0.0, SIGMA, size=N_REPS)

    print('deterministic Pmax per run:', np.round(det, 1))
    for i in range(9):
        print(i + 1, np.round(data[i], 2))

    sn = -10 * np.log10(np.mean(1.0 / data ** 2, axis=1))
    print('S/N per run:', np.round(sn, 3))

    level_sn = {f: {1: [], 2: [], 3: []} for f in FACTORS}
    for i, row in enumerate(L9):
        for fi, f in enumerate(FACTORS):
            level_sn[f][row[fi]].append(sn[i])
    resp = {f: [np.mean(level_sn[f][l]) for l in [1, 2, 3]] for f in FACTORS}
    for f in FACTORS:
        print(f, np.round(resp[f], 3), 'range', round(max(resp[f]) - min(resp[f]), 3))

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
    for ax, f in zip(axes, FACTORS):
        ax.plot([1, 2, 3], resp[f], marker='o')
        ax.set_title(f'Factor {f}')
        ax.set_xticks([1, 2, 3])
    axes[0].set_ylabel('mean S/N [dB]')
    fig.tight_layout()
    fig.savefig('figures/task4_main_effects_SN.png', dpi=150)
    plt.close(fig)

    best_level = {f: int(np.argmax(resp[f])) + 1 for f in FACTORS}
    print('best level', best_level)
    grand_mean_sn = sn.mean()
    sn_pred = grand_mean_sn + sum(max(resp[f]) - grand_mean_sn for f in FACTORS)
    Pmax_pred = 10 ** (sn_pred / 20)
    print('grand mean SN', grand_mean_sn, 'sn_pred', sn_pred, 'Pmax_pred', Pmax_pred)

    best_coded = [LEVEL_TO_CODED[best_level[f]] for f in FACTORS]
    Pmax_taguchi_true = pmax(*best_coded)
    print('Taguchi combo evaluated in true model:', Pmax_taguchi_true)

    best_val = -1e9
    best_combo = None
    for A in [-1, 0, 1]:
        for B in [-1, 0, 1]:
            for C in [-1, 0, 1]:
                for D in [-1, 0, 1]:
                    v = pmax(A, B, C, D)
                    if v > best_val:
                        best_val = v
                        best_combo = (A, B, C, D)
    print('true optimum', best_val, best_combo)

    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ['Taguchi\nS/N prediction', 'Taguchi combo\nin true model', 'True\noptimum']
    vals = [Pmax_pred, Pmax_taguchi_true, best_val]
    ax.bar(labels, vals, color=['tab:orange', 'tab:blue', 'tab:green'])
    for i, v in enumerate(vals):
        ax.annotate(f'{v:.1f}', (i, v), textcoords='offset points', xytext=(0, 5), ha='center')
    ax.set_ylabel('Pmax [kPa]')
    fig.tight_layout()
    fig.savefig('figures/task7_comparison.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
