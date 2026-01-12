# CRS – Task 7: Locust Swarm Collective Decisions

## Requirements

- Python 3.13
- Dependencies: numpy, matplotlib

## Setup

Using UV (recommended):

```bash
uv sync
uv run python -m task07.run_all
```

Using pip:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or: source .venv/bin/activate  # Unix/macOS
pip install -U pip
pip install numpy matplotlib
```

## Run Task 7

Execute all parts (a–c and 7.2):

```bash
python -m task07.run_all
```

## Outputs

Generated plots saved to `plots/task07/`:

- **task07_a_left_goers.png** – Single trajectory: number of left-going locusts over 500 steps
- **task07_b_transitions.png** – 2D heatmap of state transitions $L_t \to L_{t+1}$ (1000 runs)
- **task07_c_sampled_trajectory.png** – Sampled trajectory using learned transition probabilities
- **task07_2_mean_switch_time.png** – Mean duration between global A↔C switches vs swarm size N
- **task07_2_switch_counts.png** – Number of observed switches vs swarm size N
