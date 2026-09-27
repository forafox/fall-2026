import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURVES = ["#2a78d6", "#eb6834", "#1baf7a"]


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_loo(ks, risks, title, path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for (label, risk), color in zip(risks.items(), CURVES):
        best = ks[risk.argmin()]
        ax.plot(ks, risk, marker="o", markersize=4, color=color, label=f"{label}, лучшее k = {best}")
        ax.plot(best, risk.min(), marker="*", markersize=16, color=color)
    ax.set_xlabel("число соседей k")
    ax.set_ylabel("доля ошибок LOO")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, path)
