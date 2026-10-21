"""Check raw support NPZ arrays; necessary-only preflight, never authorize training.

NPZ keys match qualify_behavior_support fields. Values must summarize the whole
shared edge interval (height minimum / tilt maximum / any effective nonfoot,
and mean foot force), not single sparse instantaneous body snapshots.
Returns exit 0 for eligible arrays, 2 for ineligible arrays; malformed data fail.
No simulator, optimizer, runtime command generation or target labels are used.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from engine.embodied.motor_data_qualification import qualify_behavior_support


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--support-data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('output already exists; use a new path')
    try:
        with np.load(args.support_data, allow_pickle=False) as data:
            result = qualify_behavior_support(**{key: data[key] for key in data.files})
    except ValueError as exc:
        parser.error(str(exc))
    result.update(behavioral_optimization_authorized=False,
                  target_provenance='NOT_ESTABLISHED',
                  input_path=str(args.support_data.resolve()))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, allow_nan=False))
    return 0 if result['eligible'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
