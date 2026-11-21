"""List registered excitatory recurrent Eon inputs to bilateral DNp11."""
import csv
import gzip
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'20261118T231627Z-open-fruit-loop'))
from fastbrain import eon, load_connectome, ARCHIVE

ROOTS = (720575940620054694, 720575940629777018)
DATA = ARCHIVE/'data/flywire'

if __name__ == '__main__':
    index, _ = eon.get_hash_tables(eon.path_comp)
    weights = load_connectome(index)
    rev = {i: r for r, i in index.items()}
    with gzip.open(DATA/'fafb_consolidated_cell_types.csv.gz', 'rt') as stream:
        types = {int(row['root_id']):row for row in csv.DictReader(stream)}
    m = weights[[index[r] for r in ROOTS]].toarray()
    for j, line in enumerate(m):
        print(ROOTS[j], 'total imputs', np.count_nonzero(line), 'positive', (line>0).sum())
        for i in np.argsort(-line)[:32]:
            row=types.get(rev[i], {})
            print(rev[i], line[i], m[1-j, i], row.get('cell_type'), row.get('hemibrain_type'))
