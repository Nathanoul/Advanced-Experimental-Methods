import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "font.size": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
})

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_output")
os.makedirs(FIGDIR, exist_ok=True)
