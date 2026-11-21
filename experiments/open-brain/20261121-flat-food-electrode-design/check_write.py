"""Re-run the published full5 current-only assay, redirecting its output to this design directory."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = HERE.parent / '20261121T005613Z-patch226-write'
sys.path.insert(0, str(source))
import live

def save_receipt(name, result):
    (HERE / name).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')

if __name__ == '__main__':
    live.write = save_receipt
    live.main('write-smoke.json')
