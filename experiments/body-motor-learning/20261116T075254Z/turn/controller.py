"""Continuous-curvature world-anchor teacher in the original finite body."""
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
TURN_SOURCE=HERE.parents[1]/'20261116T004235Z'
sys.path.insert(0,str(TURN_SOURCE))
import turning as turn
from tarsus_clearance import add_clearance


class ArcBlend(turn.ArcReference):
    """Preserves swing anchors across turn, coast, and stationary handoff.

    Foothold lookahead and the interpolation clock are privileged teacher
    planning variables. Actual pose is read for heading alignment; the actuator
    is the only interface that can move the physical body.
    """
    def __init__(self,c,teacher,order,direction,plan):
        super().__init__(c,teacher,direction+'_pivot',order,plan['arc_forward'],plan['arc_yaw'])
        self.arc=plan['arc_ms']*.001
        self.taper=plan['taper_ms']*.001
        self.coast=plan['coast_ms']*.001
        self.duration=self.arc+self.coast
        self.brake=plan['brake_ms']*.001
        self.new_speed=plan['coast_forward']
        self.trim=plan['measure_heading']
        self.align_time=self.arc+self.taper
        self.align_ramp=plan.get('heading_blend_ms',450)*.001
        self.heading_error=None
        self.hold=False
        self.plan=plan
        self.reset_edges()

    def reset_edges(self):
        # At/after taper there is one optional measured-yaw adjustment; build
        # and update the trajectory only in private planning coordinates.
        self.dt=.005
        last=int(np.ceil((self.duration+3.)/self.dt))
        self.relative_time=np.arange(0,last+1)*self.dt
        t=self.arc+self.relative_time
        ff=np.asarray([self.speed_at(k) for k in t])
        ang=np.asarray([self.angle_at(k) for k in t])
        velocity=ff[:,None]*(np.cos(ang[:,None])*self.heading+np.sin(ang[:,None])*self.left)
        integral=np.vstack([np.zeros(2),np.cumsum(.5*self.dt*(velocity[1:]+velocity[:-1]),axis=0)])
        self.arc_end_center=super().center(self.arc)
        self.post_points=integral

    def angle_at(self,t):
        integral=turn.fixed.old.ramp_integral
        if t<=self.arc:
            return self.omega*integral(t)
        elapsed=t-self.arc
        theta=self.omega*(integral(self.arc)+elapsed-integral(elapsed,self.taper))
        if self.heading_error is not None:
            theta+=self.heading_error*turn.fixed.old.blend((t-self.align_time)/self.align_ramp)
        return theta

    def speed_at(self,t):
        if t<=self.arc:
            return self.speed*turn.fixed.old.blend(t)
        if t>=self.duration:
            return 0.
        s=turn.fixed.old.blend((t-self.arc)/self.taper)
        speed=(1.-s)*self.speed+s*self.new_speed
        if self.brake>0.:
            speed*=1.-turn.fixed.old.blend((t-self.duration+self.brake)/self.brake)
        return speed

    def center(self,t):
        if t<=self.arc:
            return super().center(t)
        shift=np.array([np.interp(t-self.arc,self.relative_time,self.post_points[:,j]) for j in (0,1)])
        return self.arc_end_center+np.r_[shift,0.]

    def rotation(self,t):
        theta=self.angle_at(t)
        cs,sn=np.cos(theta),np.sin(theta)
        return np.array([[cs,-sn,0.],[sn,cs,0.],[0.,0.,1.]])@self.rot0

    def handle_plant_read(self,c,t):
        if not self.trim or self.heading_error is not None or t<self.align_time:
            return
        actual=np.arctan2(c.d.xmat[c.th,3],c.d.xmat[c.th,0])
        R=self.rotation(t)
        target=np.arctan2(R[1,0],R[0,0])
        self.heading_error=float(np.arctan2(np.sin(actual-target),np.cos(actual-target)))
        self.reset_edges()

    def command(self,t):
        if t<self.arc:
            turn_rate=self.omega*turn.fixed.old.blend(t)
        elif t<self.align_time:
            turn_rate=self.omega*(1.-turn.fixed.old.blend((t-self.arc)/self.taper))
        else:
            turn_rate=0.
        if self.heading_error is not None and self.align_time<t<self.align_time+self.align_ramp:
            u=(t-self.align_time)/self.align_ramp
            turn_rate+=self.heading_error*30*u*u*(1.-u)*(1.-u)/self.align_ramp
        return np.array([self.speed_at(t),turn_rate])

    def sample(self,t):
        if t<self.duration-1e-10:
            result=super().sample(t)
            result['phase']='arc' if t<self.arc else 'turn_exit_blend' if t<self.align_time else 'exit_straight'
            return result
        if self.last_t is None or abs(t-self.last_t-.005)>1e-8:
            raise ValueError('Expected contiguous 5 ms decisions')
        self.last_t=t
        if not self.hold:
            self.anchors[self.order[self.slot%6]]=self.swing_end.copy()
            self.hold=True
        desired=self.center(t)
        target=(self.anchors-desired)@self.rotation(t)
        self.last_snapshot=dict(t=t,phase='post_arc_hold',yaw=self.angle_at(t),anchors=self.anchors.copy().tolist())
        return dict(target_xy=target[:,:2],target_z=np.zeros(6),stance=np.ones(6,dtype=bool),
                    source_time_s=None,phase='post_arc_hold')


class Controller:
    def __init__(self,c,direction,plan):
        if direction not in ('left','right'):
            raise ValueError('Expected left or right')
        source='arc_'+direction
        self.teacher=turn.make_teacher(c,source,plan['arc_forward'],plan['arc_yaw'])
        old=self.teacher.reference
        self.reference=ArcBlend(c,self.teacher,old.order,direction,plan)
        self.teacher.reference=self.reference
        if plan['right_hind_margin']>0:
            add_clearance(self.teacher,c,plan['right_hind_margin'])

    def action(self,c,t):
        self.reference.handle_plant_read(c,t)
        a,diag=self.teacher.action(c,t)
        return a,self.reference.command(t)
