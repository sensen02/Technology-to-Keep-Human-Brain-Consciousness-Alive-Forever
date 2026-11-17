"""Hold identical anchored arc prefix, then ablate the continuous handoff."""
import argparse
from pathlib import Path
import numpy as np
import rollout


class DirectSwitch(rollout.Controller):
    def __init__(self,c,direction,plan):
        super().__init__(c,direction,plan)
        self.direct=rollout.turn.straight_policy()
        self.move_until=(plan['arc_ms']+plan['coast_ms'])*.001

    def action(self,c,t):
        if t<self.reference.arc-1e-10:
            return super().action(c,t)
        moving=int(t<self.move_until-1e-10)
        u=rollout.turn.features(c,c.measure())
        return self.direct(u,moving), np.array([rollout.BASE_PACE*moving,0.])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('direction',choices=['left','right'])
    p.add_argument('output',type=Path)
    p.add_argument('config',type=Path)
    opt=p.parse_args()
    rollout.Controller=DirectSwitch
    rollout.run(opt.output,opt.direction,opt.config,label='direct_switch_same_anchored_arc_prefix',
                subject_path=Path(__file__).resolve())
