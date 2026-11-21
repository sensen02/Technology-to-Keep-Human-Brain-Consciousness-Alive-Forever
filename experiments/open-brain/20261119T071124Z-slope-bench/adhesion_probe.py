"""Change only the available stock adhesion command magnitude for an incline probe."""
import argparse
from pathlib import Path

import numpy as np

import slope_solved as bench
from support_coordinates import SupportCoordinatesWalk


def constrained_walker(parent, limit):
    class ConstrainedWalk(parent):
        def __init__(self,env,duration):
            super().__init__(env,duration)
            self.adhesion_columns=np.array([i for i,name in enumerate(self.full.name.split('\t'))
                                            if name.startswith('adhere_claw_')])
            assert len(self.adhesion_columns)==6

        def action(self,phy,speed,yaw_rate,observation):
            action=super().action(phy,speed,yaw_rate,observation)
            action[self.adhesion_columns]*=limit
            return action
    return ConstrainedWalk


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    parser.add_argument('--angles',type=int,nargs='+',default=[20,-20])
    parser.add_argument('--yaws',type=int,nargs='+',default=[0])
    parser.add_argument('--flat-reference',action='store_true')
    parser.add_argument('--limit',type=float,default=.35)
    args=parser.parse_args()
    if not 0.<=args.limit<=1.:
        parser.error('adhesion scale must be in [0, 1]')
    parent=bench.ReversedYawWalk if args.flat_reference else SupportCoordinatesWalk
    bench.ReversedYawWalk=constrained_walker(parent,args.limit)
    bench.main(args)
