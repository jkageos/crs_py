from __future__ import annotations

import numpy as np

# Global RNG (can be reused across tasks)
rng = np.random.default_rng()


def seed_rng(seed: int | None) -> None:
    """Seed the global RNG for reproducibility."""
    global rng
    rng = np.random.default_rng(seed)


def sample_poisson(alpha: float, size: int = 1) -> np.ndarray:
    """Sample Poisson(α) arrivals (λ=α because Δt=1)."""
    if alpha <= 0:
        return np.zeros(size, dtype=int)
    return rng.poisson(alpha, size=size)


def get_seeded_rng(seed: int) -> np.random.Generator:
    """Create a new RNG with given seed (for parallel processing)."""
    return np.random.default_rng(seed)
