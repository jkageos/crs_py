import sys
from pathlib import Path

# Add workspace root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from task02.fireflies import plot_amplitude_vs_vicinity, plot_flashing_over_time

if __name__ == "__main__":
    # Part (a): Flashing over time for different vicinity radii
    print("=== Part (a): Flashing over time ===")
    plot_flashing_over_time(
        N=150, L=50, r_values=[0.05, 0.1, 0.5, 1.4], steps=5000, save_dir="plots/task02"
    )

    # Part (b): Amplitude vs vicinity radius
    print("\n=== Part (b): Amplitude analysis ===")
    plot_amplitude_vs_vicinity(
        N=150,
        L=50,
        r_min=0.025,
        r_max=1.4,
        r_step=0.025,
        steps=5000,
        runs=50,
        save_dir="plots/task02",
    )
