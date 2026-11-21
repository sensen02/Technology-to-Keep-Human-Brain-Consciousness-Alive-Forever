"""Generate post-neck recordings anchored to the two-frame eye event."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261118T083045Z-complete-landing'))
import neck


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('discovery', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    record = json.loads((args.discovery / 'RESULT.json').read_text())
    assert record['landing_s'] is None
    assert record['visual_trigger_s'] is not None
    first = round(record['visual_trigger_s'] * 1000 / neck.DT_MS)
    neck.LANDING = first, first + 1200
    neck.TICKS = 14220
    neck.run(args.output)
