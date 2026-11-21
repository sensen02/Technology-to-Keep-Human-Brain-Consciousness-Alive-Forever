"""Vary only the nominal reference clearance in the tangent-frame body adapter."""
import argparse
from pathlib import Path

import slope_solved as bench
from support_coordinates import SupportCoordinatesWalk


def walker_with_clearance(extra_cm):
    class AdjustedWalk(SupportCoordinatesWalk):
        def __init__(self,env,duration):
            super().__init__(env,duration)
            self.initial_reference_z += extra_cm
    return AdjustedWalk


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    parser.add_argument('--angles',type=int,nargs='+',default=[20])
    parser.add_argument('--yaws',type=int,nargs='+',default=[0])
    parser.add_argument('--clearance-cm',type=float,default=.015)
    args=parser.parse_args()
    bench.ReversedYawWalk=walker_with_clearance(args.clearance_cm)
    bench.main(args)
