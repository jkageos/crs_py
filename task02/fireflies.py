from __future__ import annotations

import os
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from utils.random import get_seeded_rng


class FireflySwarm:
    """Synchronizing firefly swarm model."""

    def __init__(
        self,
        N: int,
        L: int,
        r: float,
        seed: int | None = None,
    ):
        self.N = N
        self.L = L
        self.r = r
        self.r_squared = r * r
        local_rng = (
            get_seeded_rng(seed) if seed is not None else np.random.default_rng()
        )
        self.positions = local_rng.uniform(0, 1, size=(N, 2))
        self.clocks = local_rng.integers(0, L, size=N, dtype=np.int32)
        # Track previous phase to apply correction one step AFTER flashing starts
        self.prev_phase = self.clocks % self.L

    def get_neighbors(self, i: int) -> NDArray[np.bool_]:
        """Return boolean mask of neighbors within vicinity of firefly i."""
        diff = self.positions - self.positions[i]
        dist_squared = np.sum(diff * diff, axis=1)
        neighbors = (dist_squared < self.r_squared) & (dist_squared > 0)  # Exclude self
        return neighbors

    def is_flashing(self, clock: int) -> bool:
        """Check if a firefly with given clock value is flashing."""
        phase = clock % self.L
        return phase < self.L // 2

    def step(self) -> int:
        """Advance simulation by one time step. Return number of flashing fireflies."""
        # Current phase BEFORE any updates
        phase = self.clocks % self.L

        # Identify fireflies just starting to flash (phase == 0)
        just_started = phase == 0

        # Check neighbors for those who just started
        corrections = np.zeros(self.N, dtype=np.int32)
        for i in np.where(just_started)[0]:
            neighbors = self.get_neighbors(i)
            if np.any(neighbors):
                # Check if majority of neighbors are currently flashing
                neighbor_phases = self.clocks[neighbors] % self.L
                neighbor_flashing = neighbor_phases < self.L // 2

                if np.sum(neighbor_flashing) > len(neighbor_flashing) / 2:
                    # Correct by adding 1: skip ahead in current cycle
                    corrections[i] = 1

        # Advance all clocks (base +1, plus corrections)
        self.clocks = (self.clocks + 1 + corrections) % self.L

        # Return flashing count AFTER update
        new_phase = self.clocks % self.L
        flashing = new_phase < self.L // 2

        return int(np.sum(flashing))

    def calculate_avg_neighbors(self) -> float:
        """Calculate average number of neighbors per firefly."""
        total = 0
        for i in range(self.N):
            total += int(np.sum(self.get_neighbors(i)))
        return total / self.N


def simulate_fireflies(
    N: int,
    L: int,
    r: float,
    steps: int,
    seed: int | None = None,
) -> tuple[NDArray[np.int32], float]:
    """Simulate firefly swarm and return flashing counts over time + avg neighbors."""
    swarm = FireflySwarm(N, L, r, seed=seed)
    avg_neighbors = swarm.calculate_avg_neighbors()

    flashing_counts = np.zeros(steps, dtype=np.int32)
    for t in range(steps):
        flashing_counts[t] = swarm.step()

    return flashing_counts, avg_neighbors


def _simulate_single_run(args):
    """Helper for parallel execution."""
    N, L, r, steps, seed = args
    counts, _ = simulate_fireflies(N, L, r, steps, seed)
    # Return min and max from last cycle
    last_cycle = counts[-L:]
    return int(np.min(last_cycle)), int(np.max(last_cycle))


def plot_flashing_over_time(
    N: int = 150,
    L: int = 50,
    r_values: list[float] | None = None,
    steps: int = 5000,
    seed: int = 2024,
    save_dir: str = "plots/task02",
):
    """Plot number of flashing fireflies over time for different vicinity radii."""
    if r_values is None:
        r_values = [0.05, 0.1, 0.5, 1.4]

    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, r in enumerate(r_values):
        print(f"Simulating r={r}...")
        counts, avg_neighbors = simulate_fireflies(N, L, r, steps, seed=seed + idx * 17)

        axes[idx].plot(range(steps), counts, linewidth=0.5)
        axes[idx].set_xlabel("Time step")
        axes[idx].set_ylabel("Number of flashing fireflies")
        axes[idx].set_title(f"r={r} (avg neighbors: {avg_neighbors:.2f})", fontsize=10)
        axes[idx].set_ylim(0, N)
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "task02_flashing_over_time.png"), dpi=150)
    plt.show()

    print("\nAverage neighbors per firefly:")
    for r in r_values:
        _, avg_neighbors = simulate_fireflies(N, L, r, 100, seed=seed)
        print(f"  r={r}: {avg_neighbors:.2f}")


def plot_amplitude_vs_vicinity(
    N: int = 150,
    L: int = 50,
    r_min: float = 0.025,
    r_max: float = 1.4,
    r_step: float = 0.025,
    steps: int = 5000,
    runs: int = 50,
    base_seed: int = 2024,
    save_dir: str = "plots/task02",
    parallel: bool = True,
):
    """Plot amplitude of flash cycle vs vicinity radius."""
    os.makedirs(save_dir, exist_ok=True)

    r_values = np.arange(r_min, r_max + r_step / 2, r_step)
    amplitudes = np.zeros_like(r_values, dtype=float)

    if parallel:
        print(f"Running amplitude experiments ({runs} runs per r) in parallel...")
        with Pool() as pool:
            for idx, r in enumerate(r_values):
                r_float = float(r)  # Convert numpy float to Python float
                print(f"  Processing r={r_float:.3f} ({idx + 1}/{len(r_values)})...")
                tasks = [
                    (N, L, r_float, steps, base_seed + run_idx * 1009 + idx * 97)
                    for run_idx in range(runs)
                ]
                results = pool.map(_simulate_single_run, tasks)

                # Calculate amplitude = (max - min) / 2
                amplitudes_for_r = [
                    (max_val - min_val) / 2 for min_val, max_val in results
                ]
                amplitudes[idx] = np.mean(amplitudes_for_r)
    else:
        print(f"Running amplitude experiments ({runs} runs per r) sequentially...")
        for idx, r in enumerate(r_values):
            r_float = float(r)  # Convert numpy float to Python float
            print(f"  Processing r={r_float:.3f} ({idx + 1}/{len(r_values)})...")
            run_amplitudes = []
            for run_idx in range(runs):
                counts, _ = simulate_fireflies(
                    N, L, r_float, steps, seed=base_seed + run_idx * 1009 + idx * 97
                )
                last_cycle = counts[-L:]
                amplitude = (np.max(last_cycle) - np.min(last_cycle)) / 2
                run_amplitudes.append(amplitude)
            amplitudes[idx] = np.mean(run_amplitudes)

    plt.figure(figsize=(9, 5))
    plt.plot(r_values, amplitudes, marker="o", markersize=3, linewidth=1)
    plt.xlabel("Vicinity radius r")
    plt.ylabel("Average amplitude (fireflies)")
    plt.title(f"Flash synchronization amplitude vs vicinity (N={N}, L={L})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "task02_amplitude_vs_vicinity.png"), dpi=150)
    plt.show()

    # Find optimal vicinity (maximum amplitude indicates best synchronization)
    optimal_idx = int(np.argmax(amplitudes))
    optimal_r = float(r_values[optimal_idx])
    optimal_amp = amplitudes[optimal_idx]
    print(f"\nOptimal vicinity: r={optimal_r:.3f} (amplitude={optimal_amp:.2f})")


if __name__ == "__main__":
    # Part (a): Plot flashing over time for different r values
    plot_flashing_over_time()

    # Part (b): Plot amplitude vs vicinity radius
    # NOTE: runs=50 with 5000 steps each can take a while
    # Reduce runs for quick testing (e.g., runs=10)
    plot_amplitude_vs_vicinity(runs=50)
