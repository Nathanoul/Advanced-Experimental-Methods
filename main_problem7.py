import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import studentized_range

DATA = {
    ('Conduction', 'Thermocouple'): [2.11, 2.08, 2.14, 2.10, 2.13],
    ('Conduction', 'RTD'): [2.32, 2.35, 2.29, 2.34, 2.31],
    ('Conduction', 'Thermistor'): [2.74, 2.78, 2.71, 2.76, 2.75],
    ('Convection', 'Thermocouple'): [2.43, 2.46, 2.41, 2.45, 2.44],
    ('Convection', 'RTD'): [2.66, 2.63, 2.69, 2.65, 2.67],
    ('Convection', 'Thermistor'): [3.05, 3.09, 3.02, 3.07, 3.06],
    ('Radiation', 'Thermocouple'): [2.86, 2.91, 2.88, 2.84, 2.89],
    ('Radiation', 'RTD'): [3.14, 3.18, 3.11, 3.16, 3.15],
    ('Radiation', 'Thermistor'): [3.79, 3.85, 3.76, 3.82, 3.81],
}


def anova(groups):
    all_vals = np.concatenate(list(groups.values()))
    grand_mean = all_vals.mean()
    N = len(all_vals)
    k = len(groups)
    ssb = sum(len(v) * (np.mean(v) - grand_mean) ** 2 for v in groups.values())
    ssw = sum(sum((np.array(v) - np.mean(v)) ** 2) for v in groups.values())
    dfb, dfw = k - 1, N - k
    msb, msw = ssb / dfb, ssw / dfw
    F = msb / msw
    return dict(ssb=ssb, ssw=ssw, dfb=dfb, dfw=dfw, msb=msb, msw=msw, F=F, grand_mean=grand_mean)


def tukey(groups, msw, dfw, n):
    k = len(groups)
    q = studentized_range.ppf(0.95, k, dfw)
    hsd = q * np.sqrt(msw / n)
    return q, hsd


def main():
    os.makedirs('figures', exist_ok=True)

    heating_groups = {'Conduction': [], 'Convection': [], 'Radiation': []}
    sensor_groups = {'Thermocouple': [], 'RTD': [], 'Thermistor': []}
    for (h, s), vals in DATA.items():
        heating_groups[h].extend(vals)
        sensor_groups[s].extend(vals)

    r_heat = anova(heating_groups)
    r_sens = anova(sensor_groups)
    print('heating', {k: round(v, 4) for k, v in r_heat.items()})
    print('sensor', {k: round(v, 4) for k, v in r_sens.items()})

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.boxplot([heating_groups[g] for g in ['Conduction', 'Convection', 'Radiation']],
               tick_labels=['Conduction', 'Convection', 'Radiation'])
    ax.set_ylabel('Voltage [V]')
    fig.tight_layout()
    fig.savefig('figures/task7_boxplot_heating.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.boxplot([sensor_groups[g] for g in ['Thermocouple', 'RTD', 'Thermistor']],
               tick_labels=['Thermocouple', 'RTD', 'Thermistor'])
    ax.set_ylabel('Voltage [V]')
    fig.tight_layout()
    fig.savefig('figures/task7_boxplot_sensor.png', dpi=150)
    plt.close(fig)

    q_h, hsd_h = tukey(heating_groups, r_heat['msw'], r_heat['dfw'], 15)
    q_s, hsd_s = tukey(sensor_groups, r_sens['msw'], r_sens['dfw'], 15)
    print('heating HSD', q_h, hsd_h)
    print('sensor HSD', q_s, hsd_s)

    means_h = {g: np.mean(v) for g, v in heating_groups.items()}
    means_s = {g: np.mean(v) for g, v in sensor_groups.items()}

    fig, ax = plt.subplots(figsize=(5.5, 4))
    names = list(means_h.keys())
    y = [means_h[n] for n in names]
    ax.errorbar(range(len(names)), y, yerr=hsd_h / 2, fmt='o', capsize=5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel('mean voltage +/- HSD/2 [V]')
    fig.tight_layout()
    fig.savefig('figures/task7_tukey_heating.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    names = list(means_s.keys())
    y = [means_s[n] for n in names]
    ax.errorbar(range(len(names)), y, yerr=hsd_s / 2, fmt='o', capsize=5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel('mean voltage +/- HSD/2 [V]')
    fig.tight_layout()
    fig.savefig('figures/task7_tukey_sensor.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
