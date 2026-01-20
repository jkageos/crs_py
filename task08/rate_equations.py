import math
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils.plot import ensure_dir_for_file

# --- Models ---------------------------------------------------------------


@dataclass
class RateEqParams:
    alpha_r: float = 0.6
    alpha_p: float = 0.2
    tau_a: float = 2.0
    tau_h: float = 15.0
    ns0: float = 1.0
    m0: float = 1.0


def _delay_index(i: int, delay_steps: int) -> int:
    return max(0, i - delay_steps)


# --- Part (a): searching + avoiding --------------------------------------


def simulate_part_a(
    params: RateEqParams, t_end: float, dt: float = 1e-4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps = int(math.ceil(t_end / dt))
    t = np.linspace(0.0, t_end, steps + 1)

    ns = np.full(steps + 1, params.ns0, dtype=float)
    m = np.full(steps + 1, params.m0, dtype=float)

    delay_a = int(round(params.tau_a / dt))

    for i in range(1, steps + 1):
        ns_delay = ns[_delay_index(i, delay_a)]

        d_ns = -params.alpha_r * ns[i - 1] * (
            ns[i - 1] + 1
        ) + params.alpha_r * ns_delay * (ns_delay + 1)
        d_m = -params.alpha_p * ns[i - 1] * m[i - 1]

        ns[i] = max(0.0, ns[i - 1] + dt * d_ns)
        m[i] = max(0.0, m[i - 1] + dt * d_m)

    return t, ns, m


# --- Part (b): add homing with delay tau_h -------------------------------


def simulate_part_b(
    params: RateEqParams,
    t_end: float,
    dt: float = 1e-4,
    reset_m_at: Optional[Tuple[float, float]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    steps = int(math.ceil(t_end / dt))
    t = np.linspace(0.0, t_end, steps + 1)

    ns = np.full(steps + 1, params.ns0, dtype=float)
    nh = np.zeros(steps + 1, dtype=float)
    m = np.full(steps + 1, params.m0, dtype=float)

    delay_a = int(round(params.tau_a / dt))
    delay_h = int(round(params.tau_h / dt))

    reset_idx = None
    reset_value: float | None = None  # ensure defined
    if reset_m_at is not None:
        t_reset, m_reset = reset_m_at
        reset_idx = int(round(t_reset / dt))
        reset_value = m_reset

    for i in range(1, steps + 1):
        ns_delay_a = ns[_delay_index(i, delay_a)]
        ns_delay_h = ns[_delay_index(i, delay_h)]
        m_delay_h = m[_delay_index(i, delay_h)]

        return_term = params.alpha_p * ns_delay_h * m_delay_h  # robots finishing homing

        d_ns = (
            -params.alpha_r * ns[i - 1] * (ns[i - 1] + 1)
            + params.alpha_r * ns_delay_a * (ns_delay_a + 1)
            - params.alpha_p * ns[i - 1] * m[i - 1]
            + return_term
        )
        d_nh = params.alpha_p * ns[i - 1] * m[i - 1] - return_term
        d_m = -params.alpha_p * ns[i - 1] * m[i - 1]

        ns[i] = max(0.0, ns[i - 1] + dt * d_ns)
        nh[i] = max(0.0, nh[i - 1] + dt * d_nh)
        m[i] = max(0.0, m[i - 1] + dt * d_m)

        if reset_idx is not None and i == reset_idx:
            m[i] = reset_value

    return t, ns, nh, m


# --- Plotting helpers -----------------------------------------------------


def _plot_lines(
    t: np.ndarray, series: dict[str, np.ndarray], title: str, save_path: str
) -> None:
    ensure_dir_for_file(save_path)
    plt.figure(figsize=(9, 5))
    for label, values in series.items():
        plt.plot(t, values, label=label, linewidth=1)
    plt.xlabel("Time t")
    plt.ylabel("Population / ratio")
    plt.title(title)
    plt.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0)
    )  # keep legend off the plot area
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def run_part_a() -> None:
    params = RateEqParams()
    t, ns, m = simulate_part_a(params, t_end=50.0, dt=1e-4)
    _plot_lines(
        t,
        {"n_s": ns, "m": m},
        "Task 8a: searching & avoiding",
        "plots/task08/task08_part_a.png",
    )


def run_part_b() -> None:
    params = RateEqParams()
    t1, ns1, nh1, m1 = simulate_part_b(params, t_end=160.0, dt=1e-4)
    _plot_lines(
        t1,
        {"n_s": ns1, "n_h": nh1, "m": m1},
        "Task 8b: with homing (no reset)",
        "plots/task08/task08_part_b_no_reset.png",
    )

    t2, ns2, nh2, m2 = simulate_part_b(
        params, t_end=160.0, dt=1e-4, reset_m_at=(80.0, 0.5)
    )
    _plot_lines(
        t2,
        {"n_s": ns2, "n_h": nh2, "m": m2},
        "Task 8b: with homing (m reset at t=80)",
        "plots/task08/task08_part_b_reset.png",
    )


def run_all() -> None:
    run_part_a()
    run_part_b()


if __name__ == "__main__":
    run_all()
