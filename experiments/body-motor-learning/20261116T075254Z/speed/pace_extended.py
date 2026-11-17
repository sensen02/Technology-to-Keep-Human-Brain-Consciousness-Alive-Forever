"""Pacing teacher whose geometrical path wraps without stepping across a pose jump."""
import argparse
from pathlib import Path

import numpy as np
import torch

import pace_bench
from speed_teacher import CYCLE_START, PhaseFollower


class ContinuousLoop(PhaseFollower):
    def __init__(self,core=None):
        super().__init__(core)
        source = self.path[CYCLE_START:]
        phase = np.linspace(0.,1.,len(source))
        smooth = phase*phase*(3.-2.*phase)
        gap = source[-1] - source[0]
        self.periodic = source + (1.-smooth)[:,None]*gap
        self.periodic[[0,-1]] = source[-1]


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("plan",type=Path)
    a=p.parse_args()
    torch.set_num_threads(1)
    pace_bench.PhaseFollower=ContinuousLoop
    pace_bench.roll(a.output.resolve(),a.plan.resolve())
