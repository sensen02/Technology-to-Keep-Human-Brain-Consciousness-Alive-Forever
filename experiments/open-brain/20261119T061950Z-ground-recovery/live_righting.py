"""Body-local righting mode using the same continuous orchard physics state."""
import numpy as np

from fixture import Plant
from body_righting import RightingController, PULSE


class LiveRighting:
    def __init__(self,env):
        self.p=Plant(env)
        self.control=RightingController(self.p,6)
        self.command=None
        self.started=None
        self.since=None
        self.events=[]
        self.change=None
        self.n=0
        self.finished=0
        original=env.task.before_step

        def before_step(physics,action,rng):
            original(physics,action,rng)
            if self.command is not None:
                physics.data.ctrl[:]=self.command

        env.task.before_step=before_step

    def begin(self,now):
        (action,count),index=self.control.predict(self.p.state())
        self.started=now
        self.change=now+PULSE*count
        self.command=action
        self.n+=1
        self.events.append(dict(time_s=now,event='pulse',index=index,count=count,
                                up=float(self.p.data.xmat[self.p.th,8])))

    def step(self,now,tick,has_body_contact,feet):
        p=self.p
        if self.command is None:
            fallen=has_body_contact and len(feet)<=2 and p.data.xmat[p.th,8]<.55
            resting=np.linalg.norm(p.data.qvel[3:6])<12.
            if fallen and resting:
                self.since=now if self.since is None else self.since
                if now-self.since>=.026 and self.n<5:
                    self.control.trace=[]
                    self.control.poses=[]
                    self.control.t0=now
                    self.begin(now)
            else:
                self.since=None
        if self.command is not None:
            if now>=self.change:
                self.command=p.ctrl_neutral
            if tick%10==0:
                self.control.observe(1 if now<self.change else 2)
                if self.control.steady(self.control.trace):
                    self.finished+=1
                    self.events.append(dict(time_s=now,event='stood',index=self.n))
                    self.command=None
                    self.since=None
                    return False,True
                if now-self.started>.48 and self.n<5:
                    self.begin(now)
            return True,False
        return False,False

    def close(self):
        self.control.pool.shutdown()
