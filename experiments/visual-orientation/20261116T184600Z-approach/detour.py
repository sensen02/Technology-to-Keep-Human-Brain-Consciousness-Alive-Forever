"""Bounded single-block visual detour experiment with actual native contact gates."""
import argparse
from pathlib import Path

import numpy as np

import approach
import obstacle
from approach_runtime import turn


class Detour(approach.Policy):
    """On detecting the green object, execute a finite temporary side course."""
    def __init__(self):
        super().__init__()
        self.diversion=None
        self.redirected=False
        self.hazard_hits=0

    def act(self,image):
        rgb=image.astype(float)
        green=(rgb[...,1]>75.)&(rgb[...,1]>2.*rgb[...,0])&(rgb[...,1]>2.*rgb[...,2])
        nominal,estimate=super().act(image) if self.diversion is None else ((0.,0),(0.,0.,0.))
        if not self.redirected and self.diversion is None and green.sum()>=6 and estimate[2]>0:
            self.diversion=0
            self.hazard_hits=int(green.sum())
        if self.diversion is None:
            return nominal,estimate
        t=self.diversion
        self.diversion+=1
        # Time is measured from the visually detected event, not the scenario start.
        if t<200:
            return (0.,1),estimate
        if t<258:
            return (0.,0),estimate
        if t<678:
            return (.08318,0),estimate
        if t<738:
            return (0.,0),estimate
        self.diversion=None
        self.redirected=True
        self.phase="align"
        self.direction=0
        self.near=False
        self.history.clear()
        return (0.,0),estimate


def score(trace,decisions,goal,actual,angle,mode,rec):
    result=obstacle.score(trace,decisions,goal,actual,angle,mode,rec)
    xy=trace["com"][:,:2]
    progress=float(result["initial_range"]-result["final_range"])
    moved_after=decisions[(decisions[:,0]>18000)&(decisions[:,3]>.06),0]
    checks=dict(native_collision_free=result["impact_count_native"]==0,
                initial_side_turn=float(np.rad2deg(np.unwrap(trace["yaw"])).max() -
                                        np.rad2deg(trace["yaw"][0])) >= 12.,
                lateral_excursion=float((xy[:,1]-xy[0,1]).max())>=.15,
                resumes_after_detour=len(moved_after)>=200,
                approaches_goal=progress>=.48, stops_at_goal=1.75<result["final_range"]<2.34,
                stable=result["standing"]["passed"],floor_nonfoot=result["nonfoot_native"]==0.)
    result.update(detour_checks={k:bool(v) for k,v in checks.items()},passed=all(checks.values()),
                  max_lateral_native=float((xy[:,1]-xy[0,1]).max()),progress=progress)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    approach.prepare=obstacle.prepare
    approach.Record=obstacle.CollisionRecord
    approach.Policy=Detour
    approach.score=score
    result=approach.trial(args.out.resolve(),0.,2.68,"intact",.12,31200)
    raise SystemExit(0 if result["passed"] else 1)
