"""Collect pivot demonstrations about a corrected private turn center."""
import argparse
from pathlib import Path

import numpy as np
import torch

import turning as gait


class Centered(gait.ArcReference):
    def center(self,t):
        return self.origin+(self.rotation(t)-self.rot0)@np.array([.76,0.,0.])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=gait.MODES[:2])
    p.add_argument("output",type=Path)
    a=p.parse_args()
    torch.set_num_threads(1)
    original=gait.make_teacher

    def corrected(c,mode,speed,omega):
        base=original(c,mode,speed,omega)
        old=base.reference
        base.reference=Centered(c,base,old.condition,old.order,0.,.15)
        return base

    gait.make_teacher=corrected
    try:
        gait.run(a.output.resolve(),"nominal",a.mode,"teacher",0.,.15,0,0)
    finally:
        gait.make_teacher=original


if __name__=="__main__":
    main()
