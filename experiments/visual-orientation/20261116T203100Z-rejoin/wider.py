"""Repeat the nominal-start rejoin with additional oblique travel before the return."""
import argparse
from pathlib import Path

import rejoin


class WiderDetour(rejoin.RecoveredDetour):
    def __init__(self):
        super().__init__()
        self.walk_frames = 1570


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    rejoin.RecoveredDetour = WiderDetour
    result = rejoin.run(args.out, "full")
    raise SystemExit(0 if result["passed"] else 1)
