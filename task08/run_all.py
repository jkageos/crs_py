import sys
from pathlib import Path

# Add workspace root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from task08.rate_equations import run_all

if __name__ == "__main__":
    run_all()
