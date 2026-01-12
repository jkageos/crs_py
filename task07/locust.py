from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from utils.plot import line_plot
from utils.random import get_seeded_rng


@dataclass
class LocustConfig:
    """Configuration for locust swarm simulation."""

    N: int = 20  # Number of locusts
    C: float = 1.0  # Ring circumference
    speed: float = 0.001  # Movement speed per step
    perception_range: float = 0.045  # Perception radius
    spontaneous_switch_prob: float = 0.015  # P(direction switch per step)
    base_seed: int = 2025


class LocustSwarm:
    """Simulate a swarm of locusts on a ring with direction-switching dynamics."""

    def __init__(self, cfg: LocustConfig, seed: int | None = None):
        self.cfg = cfg
        self.rng = get_seeded_rng(seed) if seed is not None else np.random.default_rng()

        # Initialize positions uniformly on ring [0, C)
        self.positions = self.rng.uniform(0, cfg.C, size=cfg.N)

        # Initialize directions: -1 (left) or +1 (right) with equal probability
        self.directions = 2 * self.rng.integers(0, 2, size=cfg.N) - 1  # {-1, +1}

    def step(self) -> int:
        """Advance simulation by one time step. Return number of left-going locusts."""
        # Pairwise circular distances (vectorized)
        diff = np.abs(self.positions[:, None] - self.positions[None, :])
        dist = np.minimum(diff, self.cfg.C - diff)
        neighbor_mask = (dist > 0) & (dist <= self.cfg.perception_range)

        left_mask = self.directions == -1
        left_counts = neighbor_mask @ left_mask.astype(int)
        total_neighbors = neighbor_mask.sum(axis=1)
        right_counts = total_neighbors - left_counts

        # Apply majority rule using previous directions
        new_dirs = self.directions.copy()
        switch_to_right = (self.directions == -1) & (right_counts > left_counts)
        switch_to_left = (self.directions == 1) & (left_counts > right_counts)
        new_dirs[switch_to_right] = 1
        new_dirs[switch_to_left] = -1

        # Spontaneous direction switches
        spontaneous_switch = (
            self.rng.random(self.cfg.N) < self.cfg.spontaneous_switch_prob
        )
        new_dirs[spontaneous_switch] *= -1

        # Update state
        self.directions = new_dirs
        self.positions = (
            self.positions + self.directions * self.cfg.speed
        ) % self.cfg.C

        return int(np.sum(self.directions == -1))


def simulate_locust_swarm(
    cfg: LocustConfig,
    steps: int,
    seed: int | None = None,
) -> tuple[NDArray[np.int_], LocustSwarm]:
    """Simulate one run and return array of left-goer counts + final swarm state."""
    swarm = LocustSwarm(cfg, seed=seed)
    left_counts = np.zeros(steps, dtype=np.int_)

    for t in range(steps):
        left_counts[t] = swarm.step()

    return left_counts, swarm


def task7_a_plot(
    cfg: LocustConfig = LocustConfig(),
    steps: int = 500,
    save_path: str = "plots/task07/task07_a_left_goers.png",
) -> None:
    """Part (a): Plot number of left-going locusts over time for one run."""
    left_counts, _ = simulate_locust_swarm(cfg, steps, seed=cfg.base_seed)

    line_plot(
        x=np.arange(steps),
        y=left_counts,
        xlabel="Time step",
        ylabel="Number of left-going locusts",
        title="Locust Swarm: Left-goers over time (N=20)",
        save_path=save_path,
        markers=False,
    )


def task7_b_transitions(
    cfg: LocustConfig = LocustConfig(),
    steps: int = 500,
    runs: int = 1000,
    seed: int | None = None,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Part (b): Count transitions L_t → L_{t+1}. Return transition matrix and model state counts."""
    # Transition count matrix: A[i][j] = count of transitions i → j
    A = np.zeros((cfg.N + 1, cfg.N + 1), dtype=np.int_)
    # Model state counts: M[i] = number of times we observed state i
    M = np.zeros(cfg.N + 1, dtype=np.int_)

    for run in range(runs):
        left_counts, _ = simulate_locust_swarm(
            cfg, steps, seed=seed + run if seed is not None else None
        )

        # Record transitions
        for t in range(steps - 1):
            L_t = left_counts[t]
            L_t1 = left_counts[t + 1]
            A[L_t, L_t1] += 1
            M[L_t] += 1

    return A, M


def task7_b_plot(
    A: NDArray[np.int_],
    cfg: LocustConfig = LocustConfig(),
    save_path: str = "plots/task07/task07_b_transitions.png",
) -> None:
    """Plot histogram of transitions as a heatmap."""
    import matplotlib.pyplot as plt

    from utils.plot import ensure_dir_for_file

    ensure_dir_for_file(save_path)

    plt.figure(figsize=(10, 8))
    plt.imshow(A, cmap="YlOrRd", aspect="auto", origin="lower")
    plt.colorbar(label="Transition count")
    plt.xlabel("L_{t+1} (next state)")
    plt.ylabel("L_t (current state)")
    plt.title("Transition histogram: L_t → L_{t+1}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def task7_c_sample_trajectory(
    P: NDArray[np.floating],
    steps: int = 500,
    seed: int | None = None,
) -> NDArray[np.int_]:
    """Part (c): Sample a trajectory using transition probabilities P."""
    rng = get_seeded_rng(seed) if seed is not None else np.random.default_rng()

    N_states = P.shape[0]
    trajectory = np.zeros(steps, dtype=np.int_)

    # Start from a random state (weighted by stationary distribution or uniform)
    trajectory[0] = rng.integers(0, N_states)

    for t in range(steps - 1):
        current_state = trajectory[t]
        # Sample next state from transition probabilities
        if np.sum(P[current_state, :]) > 0:
            next_state = rng.choice(
                N_states, p=P[current_state, :] / np.sum(P[current_state, :])
            )
        else:
            # If no outgoing transitions, stay in current state
            next_state = current_state
        trajectory[t + 1] = next_state

    return trajectory


def task7_c_plot(
    trajectory: NDArray[np.int_],
    save_path: str = "plots/task07/task07_c_sampled_trajectory.png",
) -> None:
    """Plot sampled trajectory from reduced model."""
    line_plot(
        x=np.arange(len(trajectory)),
        y=trajectory,
        xlabel="Time step",
        ylabel="Number of left-going locusts",
        title="Sampled trajectory from reduced model (transition probabilities)",
        save_path=save_path,
        markers=False,
    )


# === Task 7.2: Density-dependent global switching ===


def _zone_label(L: int, N: int) -> str:
    if L > 0.7 * N:
        return "A"
    if L < 0.3 * N:
        return "C"
    return "B"


def _measure_switch_durations_for_N(
    cfg: LocustConfig, steps: int, runs: int, seed: int | None
) -> tuple[list[int], int]:
    durations: list[int] = []

    for run in range(runs):
        left_counts, _ = simulate_locust_swarm(
            cfg, steps, seed=(seed + run) if seed is not None else None
        )

        last_nonB: str | None = None
        in_B = False
        origin_zone: str | None = None
        counter = 0

        for L in left_counts:
            zone = _zone_label(int(L), cfg.N)

            if zone != "B":
                last_nonB = zone
                if in_B:
                    if origin_zone is not None and zone != origin_zone:
                        durations.append(counter)
                    counter = 0
                    in_B = False
                    origin_zone = None
                else:
                    counter = 0
            else:
                if not in_B:
                    origin_zone = last_nonB
                    counter = 0
                    in_B = True
                counter += 1

    return durations, len(durations)


def task7_2_global_switch_stats(
    N_values: list[int],
    base_cfg: LocustConfig = LocustConfig(),
    steps: int = 8000,
    runs: int = 6,
    seed: int | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.int_]]:
    """Compute mean switch durations and counts for multiple swarm sizes."""
    means = np.full(len(N_values), np.nan, dtype=float)
    counts = np.zeros(len(N_values), dtype=int)

    for idx, N in enumerate(N_values):
        cfg = LocustConfig(
            N=N,
            C=base_cfg.C,
            speed=base_cfg.speed,
            perception_range=base_cfg.perception_range,
            spontaneous_switch_prob=base_cfg.spontaneous_switch_prob,
            base_seed=base_cfg.base_seed,
        )
        local_seed = (seed or base_cfg.base_seed) + 10_000 + N * 13
        durations, switch_count = _measure_switch_durations_for_N(
            cfg, steps=steps, runs=runs, seed=local_seed
        )
        counts[idx] = switch_count
        if durations:
            means[idx] = float(np.mean(durations))

    return means, counts


def task7_2_plot_mean_times(
    N_values: list[int],
    mean_times: NDArray[np.floating],
    save_path: str = "plots/task07/task07_2_mean_switch_time.png",
) -> None:
    line_plot(
        x=N_values,
        y=mean_times,
        xlabel="Swarm size N",
        ylabel="Mean time between global switches (steps)",
        title="Density-dependent global switching: mean duration A↔C",
        save_path=save_path,
        markers=True,
    )


def task7_2_plot_switch_counts(
    N_values: list[int],
    switch_counts: NDArray[np.int_],
    save_path: str = "plots/task07/task07_2_switch_counts.png",
) -> None:
    line_plot(
        x=N_values,
        y=switch_counts,
        xlabel="Swarm size N",
        ylabel="Number of observed switches",
        title="Density-dependent global switching: switch counts",
        save_path=save_path,
        markers=True,
    )
