from __future__ import annotations

import os
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from utils.random import get_seeded_rng, seed_rng

M_STICKS = 20
WAIT_AT_STICK = 7
C_QUAD = 0.12
ZETA_VALUES: NDArray[np.int32] = np.array([0, 1, 2], dtype=np.int32)


def _commute_times_vectorized(
    N: int, count: int, strategy: str, local_rng
) -> NDArray[np.int32]:
    """Generate multiple commute times at once."""
    zetas: NDArray[np.int32] = local_rng.choice(ZETA_VALUES, size=count).astype(
        np.int32
    )
    if strategy == "linear":
        return (np.int32(N) + zetas).astype(np.int32)
    elif strategy == "quadratic":
        # Integer time steps: floor(c*N^2) + ζ
        base = np.int32(C_QUAD * (N * N))
        return (base + zetas).astype(np.int32)
    else:
        raise ValueError("strategy must be 'linear' or 'quadratic'")


def _random_other_destinations(
    current: NDArray[np.integer], local_rng
) -> NDArray[np.integer]:
    """Sample destinations different from current stick for each robot."""
    if current.size == 0:
        return current
    dest = local_rng.integers(0, M_STICKS, size=current.size, dtype=current.dtype)
    same = dest == current
    # Resample until all differ (N <= 20, so this is cheap)
    attempts = 0
    while np.any(same) and attempts < 100:
        dest[same] = local_rng.integers(
            0, M_STICKS, size=int(np.sum(same)), dtype=current.dtype
        )
        same = dest == current
        attempts += 1
    return dest


def simulate_stick_pulling(
    N: int,
    steps: int = 1000,
    strategy: str = "linear",
    seed: int | None = None,
) -> int:
    """Simulate one run; return number of pulled sticks."""
    local_rng = get_seeded_rng(seed) if seed is not None else np.random.default_rng()

    # Robot states
    mode = np.zeros(N, dtype=np.int8)  # 0=waiting, 1=travel
    location = local_rng.integers(0, M_STICKS, size=N, dtype=np.int16)
    wait_time = np.zeros(N, dtype=np.int16)
    travel_remaining = np.zeros(
        N, dtype=np.int32
    )  # match _commute_times_vectorized dtype

    pulled = 0

    for t in range(steps):
        # 1. Advance traveling robots
        traveling = mode == 1
        travel_remaining[traveling] -= 1
        arrived = traveling & (travel_remaining <= 0)
        if np.any(arrived):
            mode[arrived] = 0
            wait_time[arrived] = 0

        # 2. Check for stick pulls: count ALL waiting robots at each stick
        # (both those who were already waiting AND those who just arrived)
        waiting_mask = mode == 0
        waiting_locations = location[waiting_mask]

        if waiting_locations.size > 0:
            # Count robots at each stick
            counts = np.bincount(waiting_locations, minlength=M_STICKS)
            sticks_with_multiple = np.where(counts >= 2)[0]

            # Pull exactly one stick per site with 2+ robots
            pulled += len(sticks_with_multiple)

            # All robots at those sticks depart immediately
            for stick in sticks_with_multiple:
                at_stick = waiting_mask & (location == stick)
                departing_idx = np.where(at_stick)[0]

                # All depart immediately to different sticks
                dest = _random_other_destinations(location[departing_idx], local_rng)
                location[departing_idx] = dest
                mode[departing_idx] = 1
                travel_times = _commute_times_vectorized(
                    N, departing_idx.size, strategy, local_rng
                )
                travel_remaining[departing_idx] = travel_times
                wait_time[departing_idx] = 0

        # 3. Increment waiting times for robots STILL waiting (not those who just pulled)
        still_waiting = mode == 0
        wait_time[still_waiting] += 1

        # 4. Robots leaving after WAIT_AT_STICK (single robots who couldn't pull)
        to_leave = still_waiting & (wait_time >= WAIT_AT_STICK)
        if np.any(to_leave):
            to_leave_idx = np.where(to_leave)[0]
            dest = _random_other_destinations(location[to_leave_idx], local_rng)
            location[to_leave_idx] = dest
            mode[to_leave_idx] = 1
            travel_times = _commute_times_vectorized(
                N, to_leave_idx.size, strategy, local_rng
            )
            travel_remaining[to_leave_idx] = travel_times
            wait_time[to_leave_idx] = 0

    return pulled


def _simulate_single_stick_run(args):
    """Helper for parallel execution."""
    N, steps, strategy, seed = args
    return simulate_stick_pulling(N, steps=steps, strategy=strategy, seed=seed)


def run_stick_experiments(
    N_values: list[int],
    steps: int,
    runs: int,
    strategy: str,
    seed: int | None = None,
    parallel: bool = True,
) -> dict[int, float]:
    """Run experiments with optional parallelization."""
    results: dict[int, float] = {}

    if parallel:
        # Parallel processing
        with Pool() as pool:
            for N in N_values:
                base_seed = seed if seed is not None else 0
                tasks = [
                    (N, steps, strategy, base_seed + r * 1009 + N * 97)
                    for r in range(runs)
                ]
                run_results = pool.map(_simulate_single_stick_run, tasks)
                results[N] = float(np.mean(run_results))
    else:
        # Sequential
        if seed is not None:
            seed_rng(seed)
        for N in N_values:
            total = 0
            for r in range(runs):
                run_seed = (seed + r * 1009 + N * 97) if seed is not None else None
                total += simulate_stick_pulling(
                    N, steps=steps, strategy=strategy, seed=run_seed
                )
            results[N] = float(total / runs)

    return results


def plot_stick_capacity(
    N_min: int = 2,
    N_max: int = 20,
    steps: int = 1000,
    runs: int = 5000,
    seed: int = 2024,
    save_dir: str = "plots/task01",
):
    os.makedirs(save_dir, exist_ok=True)
    N_values = list(range(N_min, N_max + 1))

    print(f"Running linear strategy experiments ({runs} runs per N)...")
    linear = run_stick_experiments(N_values, steps, runs, "linear", seed)
    print(f"Running quadratic strategy experiments ({runs} runs per N)...")
    quadratic = run_stick_experiments(N_values, steps, runs, "quadratic", seed + 17)

    baseline_lin = linear[2]
    baseline_quad = quadratic[2]
    rel_lin = [linear[N] / baseline_lin for N in N_values]
    rel_quad = [quadratic[N] / baseline_quad for N in N_values]

    plt.figure(figsize=(8, 5))
    plt.plot(N_values, rel_lin, marker="o", label="Linear commute")
    plt.plot(N_values, rel_quad, marker="s", label="Quadratic commute")
    plt.xlabel("System size N (robots)")
    plt.ylabel("Relative capacity (vs N=2)")
    plt.title("Stick pulling relative capacity scaling")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "task01_stick_capacity.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    # NOTE: runs=5000 can be slow; reduce for quick test
    plot_stick_capacity(runs=200)
