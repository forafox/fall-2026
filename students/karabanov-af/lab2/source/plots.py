import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CURVES = ["#2a78d6", "#eb6834", "#1baf7a"]
CLASSES = ["#2a78d6", "#eb6834", "#1baf7a"]
NOISE = "#d03b3b"


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


def plot_prototypes(P, y, prototypes, noise, class_names, title, path):
    """P is the 2D projection of the training sample, prototypes and noise are index arrays."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for c, color in zip(range(y.max() + 1), CLASSES):
        mask = y == c
        ax.scatter(P[mask, 0], P[mask, 1], color=color, alpha=0.35, s=35, label=f"{class_names[c]} ({mask.sum()})")
    ax.scatter(P[prototypes, 0], P[prototypes, 1], facecolors="none", edgecolors="black", s=190, linewidths=1.8,
               label=f"эталоны ({len(prototypes)})")
    ax.scatter(P[noise, 0], P[noise, 1], color=NOISE, marker="x", s=90, linewidths=2,
               label=f"шум, отсеян ({len(noise)})")
    ax.set_xlabel("главная компонента 1")
    ax.set_ylabel("главная компонента 2")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, path)


def plot_margins(margins, title, path):
    order = np.argsort(margins)
    values = margins[order]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(np.arange(len(values))[values <= 0], values[values <= 0], color=NOISE, width=0.9,
           label=f"шум, M <= 0 ({(values <= 0).sum()})")
    ax.bar(np.arange(len(values))[values > 0], values[values > 0], color=CURVES[2], width=0.9,
           label=f"оставлены, M > 0 ({(values > 0).sum()})")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("номер объекта после сортировки")
    ax.set_ylabel("отступ M")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, path)
