import sys
from pathlib import Path

# Add workspace root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from task05.buffon import (
    BuffonConfig,
    task5_a_demo,
    task5_b_plot,
    task5_c_plot,
    task5_d_plot,
)

if __name__ == "__main__":
    cfg = BuffonConfig()
    # (a)
    res = task5_a_demo(cfg, n=200_000)
    print(
        f"(a) P_hat={res['P_hat']:.6f}, P_true={res['P_true']:.6f}, pi_hat={res['pi_hat']:.6f}"
    )
    # (b)
    task5_b_plot(cfg, repeats=10_000, save_path="plots/task05/task05_stddev_vs_n.png")
    # (c)
    task5_c_plot(
        cfg,
        max_n=100,
        experiments=200,
        save_path="plots/task05/task05_prob_ci_over_n.png",
    )
    # (d)
    task5_d_plot(
        cfg, repeats=10_000, save_path="plots/task05/task05_outside_ci_ratio.png"
    )
