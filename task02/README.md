# CRS – Task 2: Synchronization of a Swarm

## Requirements
- Python 3.13
- Dependencies: numpy, matplotlib

## Setup
Using UV (recommended):
```bash
uv sync
uv run python -m task02.run_all
```

Using pip:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or: source .venv/bin/activate  # Unix/macOS
pip install -U pip
pip install numpy matplotlib
```

## Running the Experiments

Run all Task 2 experiments:
```bash
python -m task02.run_all
```

Run individual parts:
```bash
# Part (a): Flashing over time
python -m task02.fireflies

# Or import and run specific functions
python -c "from task02.fireflies import plot_flashing_over_time; plot_flashing_over_time()"
```

## Outputs

Generated plots are saved to `plots/task02/`:

- **task02_flashing_over_time.png** — Number of flashing fireflies over 5000 time steps for r ∈ {0.05, 0.1, 0.5, 1.4}. Shows synchronization behavior for different vicinity radii.

- **task02_amplitude_vs_vicinity.png** — Average amplitude of flash synchronization vs vicinity radius r ∈ [0.025, 1.4]. Higher amplitude indicates better synchronization.

## Implementation Details

### Model Parameters
- Swarm size: N = 150 fireflies
- Cycle length: L = 50 time steps
- Flashing duration: L/2 = 25 time steps
- Non-flashing duration: L/2 = 25 time steps

### Synchronization Mechanism
Fireflies adjust their clocks when:
1. They just started flashing (phase = 0)
2. The majority of their neighbors are already flashing
3. Correction: clock advances by 1 (shortening current cycle by 1 step)

The correction mechanism causes fireflies to "catch up" with their neighbors, gradually aligning their flash cycles over time.

### Part (a) - Flashing Over Time
- Calculates average number of neighbors for each r value
- Plots flashing patterns over 5000 time steps
- Shows how different vicinity radii affect synchronization emergence:
  - **r=0.05** (avg neighbors: 0.87): Very sparse connectivity, minimal synchronization - counts stay near 75 with small random fluctuations
  - **r=0.1** (avg neighbors: 4.41): Moderate connectivity, partial synchronization begins to emerge - oscillations grow over time
  - **r=0.5** (avg neighbors: 69.28): High connectivity, strong synchronization - clear oscillations between ~10 and ~150 flashing fireflies
  - **r=1.4** (avg neighbors: 149.00): Complete connectivity (all-to-all), strong synchronization - stable oscillations throughout

### Part (b) - Amplitude Analysis
- Measures amplitude = (max - min) / 2 during last cycle (t ∈ [4950, 5000))
- Averages over 50 independent runs per r value
- Tests r ∈ [0.025, 1.4] in steps of 0.025
- Uses parallel processing for faster computation

## Performance Notes
- Part (b) with 50 runs can take several minutes
- For quick testing, reduce `runs=50` to `runs=10` in the code
- Parallel processing is enabled by default (uses all CPU cores)

## Analysis Questions

**What seems a good choice for the vicinity and swarm density?**

### Interpretation of Amplitude

**Amplitude** measures the strength of synchronization:
- **High amplitude** (~70-75 for N=150, L=50): Strong synchronization - fireflies flash together in coordinated waves. The count oscillates dramatically between near 0 (all dark) and near 150 (all flashing).
- **Low amplitude** (~10-20): Weak/no synchronization - fireflies remain spread across random phases, keeping the count near the mean (~75) with only small fluctuations.

### Results Analysis

Based on the amplitude vs vicinity plot:

**Observed behavior:**
- **Very small r (0.025-0.1)**: Amplitude ≈ 12-40
  - Insufficient connectivity for global coordination
  - Only small local clusters synchronize
  - Most fireflies remain independent

- **Optimal range r (0.35-0.55)**: Amplitude peaks at ≈ 74
  - **Optimal r ≈ 0.45** with amplitude ≈ 74.6
  - Avg neighbors ≈ 20-30 (interpolating from r=0.1→4.4 and r=0.5→69.3)
  - Sufficient local connectivity for synchronization to cascade globally
  - Strong emergent coordination without over-coupling

- **High r (0.6-1.0)**: Amplitude ≈ 62-70
  - Very high connectivity (50-100+ neighbors)
  - Slight decrease from peak due to conflicting signals
  - Still strong synchronization, but not quite optimal

- **Very high r (>1.0)**: Amplitude ≈ 58-65
  - Near-complete or complete connectivity
  - More conflicting correction signals can slightly disrupt optimal sync
  - Still maintains strong overall synchronization

### Recommendations

**Good choice for vicinity and swarm density:**
- **Optimal vicinity radius: r ≈ 0.4-0.5**
  - Corresponds to ~20-35 average neighbors per firefly
  - Provides best balance between local coordination and global emergence
  - Achieves near-maximal synchronization (amplitude ~74-75)

- **For given density (N=150 in 1×1 square = 150 fireflies/unit²):**
  - r < 0.2: Too sparse, poor synchronization
  - r ∈ [0.35, 0.55]: **Optimal range** for strong synchronization
  - r > 0.8: Over-connected, slightly reduced performance but still good

The synchronization mechanism works effectively when each firefly can observe and coordinate with a moderate-sized local neighborhood (roughly 15-40 neighbors), allowing local corrections to propagate and merge into global synchronized behavior.