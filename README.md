# Swarmy – Task 1.1 and 1.2

Requirements:
- Python 3.13
- Dependencies: numpy, matplotlib

Setup:
- Windows/macOS/Linux:
  - python -m venv .venv
  - .venv\Scripts\activate (Windows) or source .venv/bin/activate (Unix)
  - pip install -U pip
  - pip install numpy matplotlib

Run all experiments (generates and saves diagrams under plots/):
- python -m task01.run_all

Outputs (characteristic diagrams):
- plots/task01_poisson_pmf.png — Poisson PMF $P(X=i)=e^{-\alpha}\alpha^i/i!$ for α in {0.01, 0.1, 0.5, 1.0}
- plots/task01_queue_scaling.png — average waiting list length vs arrival rate α for service time 4 and 2
- plots/task01_stick_capacity.png — relative capacity vs system size N for linear and quadratic commute

Notes:
- Task 1.1(c): the printed "Average waiting list length" is the mean over 2000 steps with α=0.1 and service time 4.
- Task 1.2 runs up to 5000 repetitions; this can take time. Reduce runs for quick previews (e.g., runs=200).