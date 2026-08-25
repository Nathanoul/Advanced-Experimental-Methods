import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

groups = {
    5000: np.array([41.2, 40.8, 41.5, 41.0]),
    10000: np.array([58.6, 59.1, 58.3, 59.0, 58.8]),
    15000: np.array([72.4, 71.9, 72.8, 72.1]),
}

means = {}
stds = {}
nus = {}
for Re, h in groups.items():
    m = np.mean(h)
    s = np.std(h, ddof=1)
    means[Re] = m
    stds[Re] = s
    nus[Re] = len(h) - 1
    print(f"Re={Re}: N={len(h)}, mean={m:.4f}, std={s:.4f}, nu={nus[Re]}")

num = sum(nus[Re] * stds[Re] ** 2 for Re in groups)
den = sum(nus[Re] for Re in groups)
s_pooled = np.sqrt(num / den)
print(f"\npooled std = {s_pooled:.4f}  (nu_total = {den})")

all_h = np.concatenate(list(groups.values()))
mean_all = np.mean(all_h)
std_all = np.std(all_h, ddof=1)
print(f"combined (ungrouped) N={len(all_h)}, mean={mean_all:.4f}, std={std_all:.4f}")

fig1, ax = plt.subplots(figsize=(6.6, 3.9))
Res = sorted(groups.keys())
for Re in Res:
    h = groups[Re]
    x = np.full(h.shape, Re) + np.random.default_rng(Re).uniform(-180, 180, size=h.shape)
    ax.plot(x, h, 'o', color='tab:blue', alpha=0.7, markersize=5,
            label='individual readings' if Re == Res[0] else None)
    ax.errorbar([Re], [means[Re]], yerr=[stds[Re]], fmt='ks', capsize=5, markersize=6,
                elinewidth=1.5, label='mean $\\pm$ std' if Re == Res[0] else None)
ax.set_xlabel('Re')
ax.set_ylabel(r'$h$ [W/m$^2$K]')
ax.set_title('$h$ at each Reynolds number')
ax.set_xticks(Res)
ax.legend(fontsize=8, loc='upper left')
fig1.tight_layout()
fig1.savefig(os.path.join(FIGDIR, 'p4_heat_transfer_data.png'), dpi=160)
print("saved figure p4_heat_transfer_data.png")

fig2, ax2 = plt.subplots(figsize=(5.2, 3.9))
labels = ['pooled std\n(within groups)', 'combined std\n(ungrouped)']
vals = [s_pooled, std_all]
ax2.bar(labels, vals, color=['tab:green', 'tab:red'], alpha=0.8)
for i, v in enumerate(vals):
    ax2.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
ax2.set_ylabel(r'standard deviation [W/m$^2$K]')
ax2.set_title('Pooled vs. combined std')
fig2.tight_layout()
fig2.savefig(os.path.join(FIGDIR, 'p4_heat_transfer_pooled.png'), dpi=160)
print("saved figure p4_heat_transfer_pooled.png")
