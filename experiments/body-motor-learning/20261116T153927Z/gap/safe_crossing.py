"""Probe post-contact transfer without dragging the RF toe into the recess."""
import argparse
from pathlib import Path

import numpy as np

import crossing as q


class ClearHandoff(q.TerrainReflex):
    def __init__(self, sensor, plan, shape, mode, reach, gain, retain, holding):
        super().__init__(sensor,plan,shape,mode,reach,gain,retain)
        self.holding=holding
        self.exit_phase=0
        self.exit_contact=None

    def __call__(self, t, x, intent, local, base, measure):
        original=super().__call__(t,x,intent,local,base,measure)
        if self.reload_seen is None or t<self.reload_seen+150:
            return original
        c=self.s.body
        site=self.position(q.RF)
        if self.holding:
            if not self.exit_phase:
                self.edges.append((t,"holding_far_ground_anchor"))
                self.exit_phase=1
            # Maintain a measured far-side site throughout the stance handover.
            # The outer bench releases this option at a published command event.
            aim=self.anchor.copy()
            aim[0]+=.004
        else:
            if not self.exit_phase:
                self.edges.append((t,"unloading_far_contact_to_raise"))
                self.exit_phase=1
                self.exit_anchor=site.copy()
                self.near_anchor=np.array([self.shape["opening"][0]-.111,self.anchor[1],self.p["stance_z"]])
            if self.exit_phase==1 and measure["padz"][q.RF]>.030:
                self.edges.append((t,"lifted_clearance_for_near_side_settle"))
                self.exit_phase=2
            if self.exit_phase==2:
                balls=c.d.geom_xpos[self.pads[q.RF]]
                if (balls[:,0]+self.radii).max() < self.shape["opening"][0]-.020:
                    self.edges.append((t,"whole_toe_cleared_near_lip"))
                    self.exit_phase=3
            aim=self.exit_anchor.copy() if self.exit_phase==1 else self.near_anchor.copy()
            if self.exit_phase<3:
                aim[2]=.061
            if self.exit_phase==3 and measure["loads"][q.RF]>.03*q.W and self.exit_contact is None:
                self.edges.append((t,"near_side_settle"))
                self.exit_contact=t
            if self.exit_contact is not None and t-self.exit_contact>=140:
                return self.base(x,0)
        original[q.physical.RF_JOINTS]=self.joint_rate(q.RF,site,aim,cap=.65)
        return np.clip(original,-1.,1.)


def run(output, mode, duration, native, long_ms=None):
    wrapper=Path(__file__).resolve()
    original=q.stan.identity.capture
    if long_ms is not None:
        q.RESUMED_WALK=long_ms
        q.END=q.RELEASE+long_ms+q.TERMINAL
        duration=q.END

    def bind(core,scope,*,extra_paths):
        return original(core,{**scope,"safe_handoff":mode,"resumed_walk_ms":q.RESUMED_WALK},
                        extra_paths=[*extra_paths,wrapper])

    q.stan.identity.capture=bind
    def create(sensor, plan, shape, actor_mode, reach, gain, retain):
        return ClearHandoff(sensor,plan,shape,actor_mode,reach,gain,retain,holding=mode=="hold")
    try:
        q.run(output,"far",.036,20.,400,duration,native,branch_factory=create)
    finally:
        q.stan.identity.capture=original


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("hold","clear"))
    p.add_argument("output",type=Path)
    p.add_argument("--duration",type=int,default=q.END)
    p.add_argument("--long-ms",type=int)
    p.add_argument("--native",action="store_true")
    a=p.parse_args()
    run(a.output.resolve(),a.mode,a.duration,a.native,a.long_ms)
