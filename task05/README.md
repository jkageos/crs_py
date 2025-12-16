# CRS – Task 5: Buffon's Needle

## Requirements
- Python 3.13
- Dependencies: numpy, matplotlib

## Setup

Using UV (recommended):
```bash
uv sync
uv run python -m task05.run_all
```

Using pip:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or: source .venv/bin/activate  # Unix/macOS
pip install -U pip
pip install numpy matplotlib
```

## Run Task 5
Execute all parts (a–d) and generate plots to `plots/task05/`:
```bash
python -m task05.run_all
```

## Outputs
- `task05_stddev_vs_n.png` — Std dev of $\hat{P}$ vs $n$ (part b)
- `task05_prob_ci_over_n.png` — Running $\hat{P}(n)$ with Wald 95% CI (part c)
- `task05_outside_ci_ratio.png` — Fraction where true $P$ is outside CI vs $n$ (part d)

## Notes
- Core runner: [`task05/run_all.py`](task05/run_all.py)
- Defaults use full geometric simulation; `use_binomial_shortcut` can be enabled in part (b)/(d) functions if desired for speed.