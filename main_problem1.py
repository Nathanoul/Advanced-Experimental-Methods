import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def model(x, A, B, C):
    return A * np.exp(-B * x) + C


def main():
    os.makedirs('figures', exist_ok=True)

    X = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    Y = np.array([5.12, 4.21, 3.48, 2.92, 2.45, 2.12, 1.86, 1.67, 1.54])
    sigma_x = 0.01
    sigma_y = 0.05
    p0 = [4.0, 0.5, 1.5]
    n_mc = 10000
    seed = 42

    popt, pcov = curve_fit(model, X, Y, p0=p0, sigma=np.full_like(Y, sigma_y),
                            absolute_sigma=True, maxfev=5000)
    resid = Y - model(X, *popt)
    chi2 = np.sum((resid / sigma_y) ** 2)
    chi2_red = chi2 / (len(X) - 3)
    print('nominal popt', popt)
    print('chi2', chi2, 'chi2_red', chi2_red)

    rng = np.random.default_rng(seed)
    results = []
    n_failed = 0
    for _ in range(n_mc):
        x_pert = X + rng.normal(0.0, sigma_x, size=X.shape)
        y_pert = Y + rng.normal(0.0, sigma_y, size=Y.shape)
        try:
            p, _ = curve_fit(model, x_pert, y_pert, p0=p0, maxfev=5000)
            results.append(p)
        except RuntimeError:
            n_failed += 1
    results = np.array(results)
    print('converged', len(results), 'failed', n_failed)

    names = ['A', 'B', 'C']
    for i, name in enumerate(names):
        mean = results[:, i].mean()
        std = results[:, i].std(ddof=1)
        lo, hi = np.percentile(results[:, i], [2.5, 97.5])
        print(name, round(mean, 4), round(std, 4), '[', round(lo, 4), ',', round(hi, 4), ']')

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, ax in enumerate(axes):
        ax.hist(results[:, i], bins=50, color='steelblue', edgecolor='k', alpha=0.7)
        ax.axvline(popt[i], color='red', linestyle='--', label='Nominal fit')
        ax.set_title(f'Histogram of {names[i]}')
        ax.set_xlabel(names[i])
        ax.set_ylabel('Count')
        ax.legend()
    fig.tight_layout()
    fig.savefig('figures/task1_histograms_ABC.png', dpi=150)
    plt.close(fig)

    x_grid = np.linspace(-1, 7, 300)
    curves = np.array([model(x_grid, *p) for p in results])
    lo = np.percentile(curves, 2.5, axis=0)
    hi = np.percentile(curves, 97.5, axis=0)
    y_nom = model(x_grid, *popt)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(x_grid, lo, hi, color='steelblue', alpha=0.3, label='95% MC band')
    ax.plot(x_grid, y_nom, 'r-', label='Nominal fit')
    ax.errorbar(X, Y, yerr=sigma_y, fmt='ko', capsize=3, label='Measured data')
    ax.axvspan(0, 4, color='gray', alpha=0.08)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    fig.tight_layout()
    fig.savefig('figures/task1_extrapolation.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
