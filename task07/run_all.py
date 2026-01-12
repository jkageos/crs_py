import sys
from pathlib import Path

import numpy as np

# Add workspace root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from task07.locust import (
    LocustConfig,
    task7_2_global_switch_stats,
    task7_2_plot_mean_times,
    task7_2_plot_switch_counts,
    task7_a_plot,
    task7_b_plot,
    task7_b_transitions,
    task7_c_plot,
    task7_c_sample_trajectory,
)

if __name__ == "__main__":
    cfg = LocustConfig()

    # Part (a): Single trajectory
    print("(a) Simulating single trajectory...")
    task7_a_plot(cfg, steps=500, save_path="plots/task07/task07_a_left_goers.png")

    # Part (b): Transition histogram
    print("\n(b) Computing transitions (1000 runs)...")
    A, M = task7_b_transitions(cfg, steps=500, runs=1000, seed=cfg.base_seed)
    print(f"  Total transitions observed: {np.sum(A)}")
    print(f"  State occupancy: {M}")
    task7_b_plot(A, cfg, save_path="plots/task07/task07_b_transitions.png")

    # Part (c): Normalized transition probabilities and sampled trajectory
    print("\n(c) Normalizing transition probabilities...")
    P = np.zeros_like(A, dtype=float)
    for i in range(A.shape[0]):
        if M[i] > 0:
            P[i, :] = A[i, :] / M[i]

    print("  Transition probability matrix computed")
    print(f"  Non-zero rows: {np.sum(np.any(P > 0, axis=1))}")

    trajectory = task7_c_sample_trajectory(P, steps=500, seed=cfg.base_seed + 1)
    task7_c_plot(trajectory, save_path="plots/task07/task07_c_sampled_trajectory.png")

    # Part (7.2): Density-dependent global switching
    print("\n(7.2) Measuring density-dependent global switching...")
    N_values = list(range(20, 151, 10))
    mean_times, switch_counts = task7_2_global_switch_stats(
        N_values,
        base_cfg=cfg,
        steps=8000,
        runs=6,
        seed=cfg.base_seed + 10_000,
    )
    task7_2_plot_mean_times(
        N_values, mean_times, save_path="plots/task07/task07_2_mean_switch_time.png"
    )
    task7_2_plot_switch_counts(
        N_values, switch_counts, save_path="plots/task07/task07_2_switch_counts.png"
    )

    print("\n✓ Task 7.1 and 7.2 complete. Outputs saved to plots/task07/")
