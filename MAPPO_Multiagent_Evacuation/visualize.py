from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def plot_paths(grid_map: np.ndarray, paths: Sequence[Sequence[np.ndarray]], exits, save_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid_map, cmap="Greys", origin="upper")

    for exit_point in exits:
        ax.scatter(exit_point[1], exit_point[0], marker="s", s=100, c="limegreen", edgecolor="black")

    cmap = plt.get_cmap("tab20")
    for i, path in enumerate(paths):
        arr = np.asarray(path)
        if len(arr) == 0:
            continue
        ax.plot(arr[:, 1], arr[:, 0], color=cmap(i % 20), linewidth=1.8, label=f"agent_{i}")
        ax.scatter(arr[0, 1], arr[0, 0], color=cmap(i % 20), marker="o", s=35)
        ax.scatter(arr[-1, 1], arr[-1, 0], color=cmap(i % 20), marker="x", s=45)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("MAPPO multi-agent evacuation paths")
    if len(paths) <= 12:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_training_curves(csv_path: str | Path, save_path: str | Path) -> None:
    import csv
    import matplotlib.pyplot as plt

    csv_path = Path(csv_path)
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    episodes = [int(row["episode"]) for row in rows]
    rewards = [float(row["episode_reward"]) for row in rows]
    evacuated = [int(row["num_evacuated"]) for row in rows]
    steps = [int(row["episode_steps"]) for row in rows]

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(episodes, rewards)
    axes[0].set_ylabel("reward")
    axes[1].plot(episodes, evacuated)
    axes[1].set_ylabel("evacuated")
    axes[2].plot(episodes, steps)
    axes[2].set_ylabel("steps")
    axes[2].set_xlabel("episode")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

