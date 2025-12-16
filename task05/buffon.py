from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from utils.plot import line_plot, shaded_ci_plot
from utils.random import get_seeded_rng

# Buffon's needle parameters (defaults)
DEFAULT_B = 0.7
DEFAULT_S = 1.0


def true_probability(b: float = DEFAULT_B, s: float = DEFAULT_S) -> float:
    """True intersection probability: P = (2 b) / (π s)."""
    return (2.0 * b) / (math.pi * s)


def buffon_trial(b: float, s: float, rng: np.random.Generator) -> int:
    """Single Bernoulli trial: 1 if needle crosses a line, else 0.
    Reduced model: center uniformly in [0, s/2], angle θ uniformly in [0, π/2].
    Intersects if center distance d <= (b/2) * sin(θ).
    """
    d = rng.uniform(0.0, s / 2.0)
    theta = rng.uniform(0.0, math.pi / 2.0)
    crosses = d <= (b / 2.0) * math.sin(theta)
    return 1 if crosses else 0


def simulate_experiment(n: int, b: float, s: float, seed: int | None = None) -> float:
    """Run one experiment with n trials; return intersection probability estimate.
    Vectorized geometric simulation.
    """
    rng = get_seeded_rng(seed) if seed is not None else np.random.default_rng()
    d = rng.uniform(0.0, s / 2.0, size=n)
    theta = rng.uniform(0.0, math.pi / 2.0, size=n)
    crosses = d <= (b / 2.0) * np.sin(theta)
    return float(np.mean(crosses))


def estimate_pi_from_probability(P_hat: float, b: float, s: float) -> float:
    """Estimator: π ≈ 2b / (s P_hat)."""
    if P_hat <= 0:
        return float("inf")
    return (2.0 * b) / (s * P_hat)


def stddev_over_repeats(
    n: int,
    repeats: int,
    b: float,
    s: float,
    seed: int | None = None,
    use_binomial_shortcut: bool = False,
) -> float:
    """Run 'repeats' experiments of size n; return stddev of P_hat.
    - Default: full geometric simulation (vectorized).
    - If use_binomial_shortcut=True, draw counts ~ Binomial(n, P_true).
    """
    rng = get_seeded_rng(seed) if seed is not None else np.random.default_rng()
    if use_binomial_shortcut:
        p = true_probability(b, s)
        counts = rng.binomial(n, p, size=repeats)
        probs = counts / n
        return float(np.std(probs, ddof=1))
    # Vectorized geometric simulation across repeats
    d = rng.uniform(0.0, s / 2.0, size=(repeats, n))
    theta = rng.uniform(0.0, math.pi / 2.0, size=(repeats, n))
    crosses = d <= (b / 2.0) * np.sin(theta)
    probs = np.mean(crosses, axis=1)
    return float(np.std(probs, ddof=1))


def binomial_ci_95(P_hat: float, n: int) -> tuple[float, float]:
    """Wald 95% CI: P̂ ± 1.96 sqrt((P̂(1-P̂))/n)."""
    rad = 1.96 * math.sqrt((P_hat * (1.0 - P_hat)) / n)
    return max(0.0, P_hat - rad), min(1.0, P_hat + rad)


def outside_ci_ratio(
    n: int,
    repeats: int,
    b: float,
    s: float,
    seed: int | None = None,
    use_binomial_shortcut: bool = False,
) -> float:
    """Fraction of experiments where true P is outside the 95% CI built around P̂.
    - Default: full geometric simulation (vectorized).
    - If use_binomial_shortcut=True, draw counts ~ Binomial(n, P_true).
    """
    P_true = true_probability(b, s)
    rng = get_seeded_rng(seed) if seed is not None else np.random.default_rng()

    if use_binomial_shortcut:
        counts = rng.binomial(n, P_true, size=repeats)
        P_hat = counts / n
    else:
        d = rng.uniform(0.0, s / 2.0, size=(repeats, n))
        theta = rng.uniform(0.0, math.pi / 2.0, size=(repeats, n))
        crosses = d <= (b / 2.0) * np.sin(theta)
        P_hat = np.mean(crosses, axis=1)

    rad = 1.96 * np.sqrt((P_hat * (1.0 - P_hat)) / n)
    lo = np.maximum(0.0, P_hat - rad)
    hi = np.minimum(1.0, P_hat + rad)
    outside = (P_true < lo) | (P_true > hi)
    return float(np.mean(outside))


@dataclass
class BuffonConfig:
    b: float = DEFAULT_B
    s: float = DEFAULT_S
    base_seed: int = 2025


def task5_a_demo(
    cfg: BuffonConfig = BuffonConfig(), n: int = 100_000
) -> dict[str, float]:
    """Part (a): estimate P and π (geometric simulation)."""
    P_hat = simulate_experiment(n, cfg.b, cfg.s, seed=cfg.base_seed)
    pi_hat = estimate_pi_from_probability(P_hat, cfg.b, cfg.s)
    return {"P_hat": P_hat, "pi_hat": pi_hat, "P_true": true_probability(cfg.b, cfg.s)}


def task5_b_plot(
    cfg: BuffonConfig = BuffonConfig(),
    repeats: int = 10_000,
    save_path: str = "plots/task05/task05_stddev_vs_n.png",
    use_binomial_shortcut: bool = False,
) -> None:
    """Part (b): stddev of P̂ over n ∈ {10,20,...,1000}.
    Default uses geometric simulation; enable Binomial shortcut explicitly if desired.
    """
    n_values = np.arange(10, 1000 + 10, 10)
    y = np.array(
        [
            stddev_over_repeats(
                int(n),
                repeats,
                cfg.b,
                cfg.s,
                seed=cfg.base_seed + int(n),
                use_binomial_shortcut=use_binomial_shortcut,
            )
            for n in n_values
        ]
    )
    line_plot(
        x=n_values,
        y=y,
        xlabel="Number of trials n",
        ylabel="Std dev of intersection probability P̂",
        title="Buffon's Needle: Std dev of P̂ vs n",
        save_path=save_path,
        markers=True,
    )


def task5_c_plot(
    cfg: BuffonConfig = BuffonConfig(),
    max_n: int = 100,
    experiments: int = 200,
    save_path: str = "plots/task05/task05_prob_ci_over_n.png",
) -> None:
    """Part (c): plot P̂ over n (up to 100) with 95% CI bands from many experiments.
    Vectorized geometric simulation for accurate trajectories.
    """
    rng = get_seeded_rng(cfg.base_seed)
    # Trials: shape [experiments, max_n]
    d = rng.uniform(0.0, cfg.s / 2.0, size=(experiments, max_n))
    theta = rng.uniform(0.0, math.pi / 2.0, size=(experiments, max_n))
    crosses = d <= (cfg.b / 2.0) * np.sin(theta)

    cum_hits = np.cumsum(crosses, axis=1)
    n_values = np.arange(1, max_n + 1, dtype=float)
    paths = cum_hits / n_values  # broadcast over columns

    mean_path = np.mean(paths, axis=0)
    rad = 1.96 * np.sqrt((mean_path * (1.0 - mean_path)) / n_values)
    ci_lo = np.maximum(0.0, mean_path - rad)
    ci_hi = np.minimum(1.0, mean_path + rad)

    shaded_ci_plot(
        x=np.arange(1, max_n + 1),
        y=mean_path,
        y_band_lo=ci_lo,
        y_band_hi=ci_hi,
        xlabel="Trial number n",
        ylabel="Measured probability P̂",
        title="Buffon's Needle: P̂(n) with 95% Binomial CI (Wald)",
        save_path=save_path,
        sample_paths=paths[:10],
    )


def task5_d_plot(
    cfg: BuffonConfig = BuffonConfig(),
    repeats: int = 10_000,
    save_path: str = "plots/task05/task05_outside_ci_ratio.png",
    use_binomial_shortcut: bool = False,
) -> None:
    """Part (d): ratio of experiments where true P is outside the 95% CI vs n.
    Uses the same n-grid as part (b): n ∈ {10, 20, ..., 1000}.
    """
    n_values = np.arange(10, 1000 + 10, 10)
    ratios = np.array(
        [
            outside_ci_ratio(
                int(n),
                repeats,
                cfg.b,
                cfg.s,
                seed=cfg.base_seed + int(n),
                use_binomial_shortcut=use_binomial_shortcut,
            )
            for n in n_values
        ]
    )
    line_plot(
        x=n_values,
        y=ratios,
        xlabel="Number of trials n",
        ylabel="Fraction outside 95% CI (true P outside CI)",
        title="Buffon's Needle: Coverage vs n (Wald 95% CI)",
        save_path=save_path,
        markers=True,
    )
