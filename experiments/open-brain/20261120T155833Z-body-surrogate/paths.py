from pathlib import Path
import json
import numpy as np

BASE = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain')
HERE = BASE/'20261120T155833Z-body-surrogate'
ARCHIVE = BASE/'20261118T051014Z-neck-fault-assay/FORMAL'
ARRAY = Path('/run/media/sensen/Data1/omnisim-results/implant-site/banc-neck/single-array-sorting/passive-traces.npz')
SEEDS = (10901, 11311, 11701)
LEGS = ('lf','lm','lh','rf','rm','rh')
def read(path):
    return json.loads(path.read_text())
