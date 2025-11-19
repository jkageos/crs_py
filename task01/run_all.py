import sys
from pathlib import Path

# Add workspace root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from task01.queueing import plot_poisson_pmf, plot_queue_scaling, simulate_queue
from task01.stick_pulling import plot_stick_capacity

if __name__ == "__main__":
    plot_poisson_pmf([0.01, 0.1, 0.5, 1.0], save_dir="plots")
    avg_len = simulate_queue(0.1, 4, 2000, seed=42)
    print(f"Average waiting list length (α=0.1, service=4): {avg_len:.3f}")
    plot_queue_scaling(save_dir="plots")
    plot_stick_capacity(runs=5000, save_dir="plots")  # full experiment
