import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from common import FIGDIR

def box(ax, x, y, w, h, text, fc="white"):
    r = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                 linewidth=1.2, edgecolor="k", facecolor=fc)
    ax.add_patch(r)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=8.5)

def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", lw=1.2))

fig, axs = plt.subplots(2, 1, figsize=(8.5, 5.6))

ax = axs[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 2.2); ax.axis("off")
ax.set_title("Strategy A - inertial (accelerometer) force inference", fontsize=10)
box(ax, 0.2, 0.7, 1.7, 0.9, "wing\nunsteady\nmotion")
box(ax, 2.3, 0.7, 1.9, 0.9, "accelerometer\n$a(t)$")
box(ax, 4.6, 0.7, 2.2, 0.9, "$F=m\\,a(t)+F_{other}$\n(Newton's 2nd law)")
box(ax, 7.2, 0.7, 2.3, 0.9, "estimated\n$F_{drag}(t)$")
arrow(ax, 1.9, 1.15, 2.3, 1.15)
arrow(ax, 4.2, 1.15, 4.6, 1.15)
arrow(ax, 6.8, 1.15, 7.2, 1.15)

ax = axs[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 2.2); ax.axis("off")
ax.set_title("Strategy B - pressure-tap surface integration", fontsize=10)
box(ax, 0.2, 0.7, 1.7, 0.9, "wing\nsurface\npressure")
box(ax, 2.3, 0.7, 1.9, 0.9, "pressure taps\n(sparse $p_i(t)$)")
box(ax, 4.6, 0.7, 2.2, 0.9, "LSQ surface\nfit $p(s,t)$")
box(ax, 7.2, 0.7, 2.3, 0.9, "$F=\\oint p\\,\\hat n\\,dA$")
arrow(ax, 1.9, 1.15, 2.3, 1.15)
arrow(ax, 4.2, 1.15, 4.6, 1.15)
arrow(ax, 6.8, 1.15, 7.2, 1.15)

fig.tight_layout()
fig.savefig(f"{FIGDIR}/a1_strategies.png")
plt.close(fig)

print("Done: figure saved to", FIGDIR)
