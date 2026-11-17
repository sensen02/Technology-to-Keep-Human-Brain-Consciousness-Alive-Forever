"""Privileged world-foothold teacher for the unchanged body on a uniform slope."""
import argparse
from pathlib import Path

import numpy as np
import torch

import slope
import turning as turn

WARM_MS = 1600
PREVIOUS = turn.fixed.old


class Reference(PREVIOUS.PivotReference):
    def __init__(self, body, teacher, order, terrain, duration, speed):
        super().__init__(body,teacher,'left_pivot',order)
        self.teacher=teacher
        self.terrain=terrain
        self.duration=duration
        self.speed=speed
        self.rise=float(np.tan(terrain.angle))
        self.heading=self.rot0[:2,0]/np.linalg.norm(self.rot0[:2,0])
        self.warm=WARM_MS*.001
        self.braking=.8
        self.held=False

    def replace(self):
        self.anchors[self.order[self.slot%6]]=self.swing_end.copy()

    def center(self, t):
        t=min(t,self.duration)
        straight=self.speed*(PREVIOUS.ramp_integral(t)-
                 PREVIOUS.ramp_integral(max(0.,t-self.duration+self.braking),self.braking))
        run=straight*self.heading
        return self.origin+np.r_[run,run[0]*self.rise]

    def sample(self,t):
        if self.last_t is not None and not np.isclose(t-self.last_t,.005,atol=1e-10,rtol=0.):
            raise ValueError('Teacher expects contiguous commands at 5 ms')
        self.last_t=t
        local=round(max(0.,t-self.warm),8)
        center=self.center(local)
        clearance=np.zeros(6)
        stance=np.ones(6,dtype=bool)
        world=self.anchors.copy()
        phase='prepare_slope_support' if t < self.warm else 'walk'
        if t >= self.warm and local<self.duration:
            slot,leg,u=PREVIOUS.schedule(local,self.order,self.step_s)
            if slot!=self.slot:
                if self.slot>=0:
                    self.replace()
                self.slot=slot
                self.swing_start=self.anchors[leg].copy()
                self.swing_end=self.center((slot+1)*self.step_s+2.)+self.rot0@self.rest[leg]
            world=self.anchors.copy()
            stride=PREVIOUS.blend(u)
            world[leg]=(1.-stride)*self.swing_start+stride*self.swing_end
            clearance[leg]=self.lift*16.*u*u*(1.-u)*(1.-u)
            stance[leg]=False
        elif t >= self.warm:
            if not self.held:
                self.replace()
                self.held=True
            world=self.anchors.copy()
            phase='anchored_stop'
        height=(world[:,0]-slope.PIVOT)*self.rise
        planar_gain=PREVIOUS.blend(t/.45)
        self.teacher.rest_pad_bottom[:] = self.bottom0 + planar_gain*(height-(center[2]-self.origin[2]))
        relative=(world-center)@self.rot0
        self.last_snapshot=dict(t=t,phase=phase,slope_floor_at_anchor=height.tolist(),desired_center=center.tolist())
        return dict(target_xy=relative[:,:2],target_z=clearance,stance=stance,source_time_s=None,phase=phase)


def run(output,angle,walk_ms,speed):
    original_terrain=slope.Terrain
    original_factory=slope.stock_policy
    original_warm=slope.WARM
    original_capture=slope.STAND.identity.capture
    original_put=slope.STAND.put
    known=[]
    source=Path(__file__)

    class Terrain(original_terrain):
        def __init__(self,body,angle_deg):
            super().__init__(body,angle_deg)
            known.append(self)

    def controller():
        terrain,=known
        body=terrain.body
        teacher=turn.make_teacher(body,'arc_left',speed,.115)
        old=teacher.reference
        teacher.reference=Reference(body,teacher,old.order,terrain,walk_ms*.001,speed)
        origin=body.d.time

        def act(_x,_intent):
            ms=int(round((body.d.time-origin)*1000))
            action,_=teacher.action(body,ms*.001)
            return action

        return act

    def capture(body,config,*,extra_paths):
        config={**config,'warm_ms':WARM_MS,'speed_while_walking':speed,'policy':'slope world-anchor engineering teacher'}
        return original_capture(body,config,extra_paths=[*extra_paths,source,turn.TEACHER,Path(turn.fixed.old.__file__)])

    def put(path,data):
        if path == output/'RESULT.json':
            data.update(declarations='Engineering teacher plans foothold height from the actual plane and anchors the feet through the finite actuators',
                        policy_kind='phase_and_geometry_teacher',warm_ms=WARM_MS,teacher_requested_speed=speed,
                        fixed_initial_state=True,signals_to_body='42 finite actuator rates only')
        return original_put(path,data)

    slope.Terrain=Terrain
    slope.stock_policy=controller
    slope.WARM=WARM_MS
    slope.STAND.identity.capture=capture
    slope.STAND.put=put
    try:
        slope.run(output,angle,walk_ms)
    finally:
        slope.Terrain=original_terrain
        slope.stock_policy=original_factory
        slope.WARM=original_warm
        slope.STAND.identity.capture=original_capture
        slope.STAND.put=original_put


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--angle',type=float,required=True)
    p.add_argument('--speed',type=float,default=.12)
    p.add_argument('--walk-ms',type=int,default=9600)
    a=p.parse_args()
    if not 0. <= a.angle <= 3. or not .07 <= a.speed <= .15 or a.walk_ms not in (9600,14400):
        p.error('Unexpected angle, speed or whole-cycle duration')
    torch.set_num_threads(1)
    run(a.output.resolve(),a.angle,a.walk_ms,a.speed)
