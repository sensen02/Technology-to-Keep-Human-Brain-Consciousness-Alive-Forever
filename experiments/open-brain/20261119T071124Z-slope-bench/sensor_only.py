"""Run tangent-frame walking with support estimation from local contact pressure only."""
import argparse
import ctypes as ct
from pathlib import Path

import slope_solved as bench
from support_coordinates import SupportCoordinatesWalk


class MeasuredContacts(bench.Contacts):
    def __init__(self,physics,normal):
        super().__init__(physics,normal)
        self.native=ct.CDLL(str(bench.HERE/'measured_contacts.so'))
        measure=self.native.step_observe
        measure.argtypes=self.measure.argtypes
        measure.restype=self.measure.restype
        self.measure=measure


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    parser.add_argument('--angles',type=int,nargs='+',default=(-20,20))
    parser.add_argument('--yaws',type=int,nargs='+',default=[0])
    args=parser.parse_args()
    bench.Contacts=MeasuredContacts
    bench.ReversedYawWalk=SupportCoordinatesWalk
    bench.main(args)
