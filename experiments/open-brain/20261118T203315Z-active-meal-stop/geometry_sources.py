"""Fetch the published BANC objects used by the set 4 morphology audit."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys


def main(archive, binding, output):
    sys.path.insert(0, str(archive / 'experiments/open-brain/20261117T051249Z-bidirectional'))
    import fetch_geometry

    rows = json.loads(binding.read_text())['set_4']
    ids = sorted({candidate['banc_888_id'] for row in rows for candidate in row['banc_matches']})
    fetch_geometry.HERE = output
    (output / 'geometry').mkdir(exist_ok=True)
    receipt = output / 'GEOMETRY_RECEIPTS.json'
    if receipt.exists():
        raise FileExistsError(receipt)
    with ThreadPoolExecutor(max_workers=len(ids)) as pool:
        records = dict(pool.map(fetch_geometry.fetch, ids))
    with receipt.open('x') as stream:
        json.dump(records, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('binding', type=Path)
    parser.add_argument('output', type=Path)
    cfg = parser.parse_args()
    main(cfg.archive, cfg.binding, cfg.output)
