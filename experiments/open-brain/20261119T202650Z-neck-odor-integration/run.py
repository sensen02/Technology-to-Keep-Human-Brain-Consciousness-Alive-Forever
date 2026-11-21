"""Run a frozen mirrored-wall odor trial with a registered static neck profile."""
import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ODOR = HERE.parent / '20261119T190005Z-odor-sac'
sys.path.insert(0, str(ODOR))
import live


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--mode', choices=['intact', 'shaft', 'target_only'], required=True)
    parser.add_argument('--side', type=int, choices=[1, -1], required=True)
    parser.add_argument('--duration', type=float, default=6.4)
    args = parser.parse_args()
    os.environ['P1_OMNISIM_PROFILE'] = str(args.profile.resolve(strict=True))
    os.environ['P1_OMNISIM_MODE'] = args.mode
    live.docking.HERE = HERE
    live.run(SimpleNamespace(out=args.out.resolve(), policy=ODOR/'TRAIN_DETOUR/POLICY.npz',
                             seed=89217, stage='detour', sense='odor', assisted=True,
                             side=args.side, duration=args.duration))


if __name__ == '__main__':
    main()
