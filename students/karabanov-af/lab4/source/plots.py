import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CURVES = ["#2a78d6", "#eb6834", "#1baf7a"]


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_scree(variance, ratio, stick, title, path):
    axes_numbers = np.arange(1, len(ratio) + 1)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))

    left.bar(axes_numbers, variance, color=CURVES[0], label="дисперсия компоненты")
    left.plot(axes_numbers, stick * variance.sum(), marker="o", color=CURVES[1],
              label="сломанная трость (порог случайности)")
    left.axhline(1, color="black", linestyle="--", linewidth=1, label="критерий Кайзера: дисперсия = 1")
    left.set_xlabel("номер главной компоненты")
    left.set_ylabel("дисперсия")
    left.set_title("Спектр: дисперсия по компонентам")
    left.legend()
    left.grid(alpha=0.3)

    cumulative = np.cumsum(ratio)
    right.plot(axes_numbers, cumulative, marker="o", color=CURVES[0], label="накопленная доля дисперсии")
    for level, color in zip([0.85, 0.95, 0.99], ["#9a9a9a", "#eb6834", "#1baf7a"]):
        right.axhline(level, color=color, linestyle="--", linewidth=1,
                      label=f"{level:.0%}: нужно {int(np.searchsorted(cumulative, level) + 1)} компонент")
    right.set_xlabel("число оставленных компонент")
    right.set_ylabel("доля объяснённой дисперсии")
    right.set_title("Накопленная доля дисперсии")
    right.legend(loc="lower right")
    right.grid(alpha=0.3)

    fig.suptitle(title)
    save(fig, path)
