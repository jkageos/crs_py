from __future__ import annotations

import math
import os
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

from utils.random import get_seeded_rng


def poisson_pmf(
    alpha: float, i_max: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (i_values, pmf) for Poisson(α) with Δt=1."""
    if i_max is None:
        i_max = max(10, int(alpha + 6 * math.sqrt(max(alpha, 1e-9))))
    i = np.arange(0, i_max + 1)
    # Stable, typed iterative computation: p(k) = p(k-1) * α / k
    pmf = np.empty_like(i, dtype=float)
    pmf[0] = math.exp(-alpha)
    for k in range(1, i.size):
        pmf[k] = pmf[k - 1] * (alpha / k)
    return i, pmf


def plot_poisson_pmf(alphas: list[float], save_dir: str = "plots/task01") -> None:
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for a in alphas:
        i, pmf = poisson_pmf(a)
        plt.plot(i, pmf, marker="o", linestyle="-", label=f"α={a}")
    plt.xlabel("i (jobs)")
    plt.ylabel("P(X=i)")
    plt.title("Poisson PMF for different arrival rates α")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "task01_poisson_pmf.png"), dpi=150)
    plt.show()


def simulate_queue(
    alpha: float, process_time: int, steps: int = 2000, seed: int | None = None
) -> float:
    """Single-server queue with deterministic service time per job.
    Returns average waiting list length (queue length excluding job in service).
    """
    local_rng = get_seeded_rng(seed) if seed is not None else np.random.default_rng()

    queue_len_sum = 0
    queue_length = 0  # Track length instead of list
    server_remaining = 0

    # Pre-generate all arrivals at once
    arrivals = (
        local_rng.poisson(alpha, size=steps)
        if alpha > 0
        else np.zeros(steps, dtype=int)
    )

    for t in range(steps):
        # Phase 1: arrivals
        queue_length += arrivals[t]

        # Phase 2: service
        if server_remaining > 0:
            server_remaining -= 1
        if server_remaining == 0 and queue_length > 0:
            # Start next job
            queue_length -= 1
            server_remaining = process_time

        # Record queue length (waiting list only)
        queue_len_sum += queue_length

    return queue_len_sum / steps


def _simulate_single_run(args):
    """Helper for parallel execution."""
    alpha, process_time, steps, seed = args
    return simulate_queue(alpha, process_time, steps=steps, seed=seed)


def run_queue_experiments(
    alphas: np.ndarray,
    process_time: int,
    steps: int,
    runs: int,
    base_seed: int = 1234,
    parallel: bool = True,
) -> np.ndarray:
    """Average queue length for each α over multiple independent runs."""
    results = np.zeros_like(alphas, dtype=float)

    if parallel:
        # Parallel processing
        with Pool() as pool:
            for idx, a in enumerate(alphas):
                tasks = [
                    (a, process_time, steps, base_seed + r * 997 + idx)
                    for r in range(runs)
                ]
                run_results = pool.map(_simulate_single_run, tasks)
                results[idx] = np.mean(run_results)
    else:
        # Sequential (for small runs or debugging)
        for idx, a in enumerate(alphas):
            acc = 0.0
            for r in range(runs):
                acc += simulate_queue(
                    a, process_time, steps=steps, seed=base_seed + r * 997 + idx
                )
            results[idx] = acc / runs

    return results


def plot_queue_scaling(save_dir: str = "plots"):
    os.makedirs(save_dir, exist_ok=True)
    # Part (d)
    alphas_d = np.round(np.arange(0.005, 0.25 + 1e-12, 0.005), 3)
    avg_len_d = run_queue_experiments(alphas_d, process_time=4, steps=2000, runs=200)
    # Part (e)
    alphas_e = np.round(np.arange(0.005, 0.5 + 1e-12, 0.005), 3)
    avg_len_e = run_queue_experiments(alphas_e, process_time=2, steps=2000, runs=200)

    plt.figure(figsize=(9, 5))
    plt.plot(alphas_d, avg_len_d, label="service time=4")
    plt.plot(alphas_e, avg_len_e, label="service time=2")
    plt.xlabel("Arrival rate α")
    plt.ylabel("Average waiting list length")
    plt.title("Queue length vs arrival rate for different service times")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "task01_queue_scaling.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    # Example usage executing all parts quickly
    plot_poisson_pmf([0.01, 0.1, 0.5, 1.0])
    avg_len_c = simulate_queue(alpha=0.1, process_time=4, steps=2000, seed=42)
    print(f"(c) Average waiting list length: {avg_len_c:.3f}")
    plot_queue_scaling()
