"""Reverse only the orchard adapter's post-neck yaw sign for a paired recheck."""
import argparse
from pathlib import Path

import run_case


class ReversedYawWalk(run_case.OrchardWalk):
    def action(self, phy, speed, yaw_rate, live_observation):
        return super().action(phy, speed, -yaw_rate, live_observation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',choices=('FRESH','EMPTY_GROUND'))
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    run_case.OrchardWalk = ReversedYawWalk
    run_case.run(args.case,args.output.resolve(),4.)
