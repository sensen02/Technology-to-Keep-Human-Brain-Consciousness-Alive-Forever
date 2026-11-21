"""Run the independent BANC standing, sensory warm-up and release experiment."""
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parent / '20261120T081056Z-stance-warmup'
sys.path.insert(0, str(EXPERIMENT))
from warm_release import main

if __name__ == '__main__':
    main()
