from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def line_plot(
    x: NDArray[np.number] | list[float] | list[int],
    y: NDArray[np.number] | list[float],
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: Optional[str] = None,
    markers: bool = False,
) -> None:
    plt.figure(figsize=(9, 5))
    marker = "o" if markers else None
    plt.plot(x, y, marker=marker, linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        ensure_dir_for_file(save_path)
        plt.savefig(save_path, dpi=150)
    plt.show()


def shaded_ci_plot(
    x: NDArray[np.number] | list[int],
    y: NDArray[np.number],
    y_band_lo: NDArray[np.number],
    y_band_hi: NDArray[np.number],
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: Optional[str] = None,
    sample_paths: Optional[NDArray[np.number]] = None,  # shape [k, len(x)]
) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(x, y, color="C0", label="Mean P̂", linewidth=2)
    plt.fill_between(x, y_band_lo, y_band_hi, color="C0", alpha=0.2, label="95% CI")
    if sample_paths is not None:
        k = sample_paths.shape[0]
        for i in range(k):
            plt.plot(x, sample_paths[i], color="gray", alpha=0.5, linewidth=0.8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        ensure_dir_for_file(save_path)
        plt.savefig(save_path, dpi=150)
    plt.show()
