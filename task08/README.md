# CRS – Task 8: Rate Equations

## Requirements

- Python 3.13
- Dependencies: numpy, matplotlib

## Setup

Using UV (recommended):

```bash
uv sync
uv run python -m task08.run_all
```

Using pip:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or: source .venv/bin/activate  # Unix/macOS
pip install -U pip
pip install numpy matplotlib
```

## Run Task 8

Execute all parts (a–b) and generate plots to `plots/task08/`:

```bash
python -m task08.run_all
```

## Outputs

Generated plots are saved to `plots/task08/`:

- **task08_part_a.png** — Part (a): searching & avoiding dynamics for $t \in (0, 50]$
- **task08_part_b_no_reset.png** — Part (b): with homing state for $t \in (0, 160]$
- **task08_part_b_reset.png** — Part (b): with homing state and $m$ reset to 0.5 at $t=80$

## Model Details

### Parameters

- $\alpha_r = 0.6$ — avoidance rate
- $\alpha_p = 0.2$ — puck pickup rate
- $\tau_a = 2.0$ — avoidance delay
- $\tau_h = 15.0$ — homing duration
- Initial conditions: $n_s(0) = 1.0$, $m(0) = 1.0$
- Time step: $\delta_t = 0.0001$

### Part (a): Searching & Avoiding

Rate equations:
$$\frac{dn_s(t)}{dt} = -\alpha_r n_s(t)(n_s(t)+1) + \alpha_r n_s(t-\tau_a)(n_s(t-\tau_a)+1)$$
$$\frac{dm(t)}{dt} = -\alpha_p n_s(t) m(t)$$

For $t < \tau_a$, delayed terms use initial value $n_s(0)$.

### Part (b): With Homing State

Extended model with homing state $n_h$:
$$\frac{dn_s(t)}{dt} = -\alpha_r n_s(t)(n_s(t)+1) + \alpha_r n_s(t-\tau_a)(n_s(t-\tau_a)+1) - \alpha_p n_s(t) m(t) + \alpha_p n_s(t-\tau_h) m(t-\tau_h)$$
$$\frac{dn_h(t)}{dt} = \alpha_p n_s(t) m(t) - \alpha_p n_s(t-\tau_h) m(t-\tau_h)$$
$$\frac{dm(t)}{dt} = -\alpha_p n_s(t) m(t)$$

Robots finding pucks transition to homing for duration $\tau_h$ before returning to searching.
