"""Reuse the validated abdomen-clearance route for the FlyBody wing hinges."""
import numpy as np

from fixture import land


class WingStow:
    tolerance=np.deg2rad(2.)

    def __init__(self,p,now):
        self.p=p
        self.q=p.model.jnt_qposadr[p.wing_joint]
        self.initial=p.data.qpos[self.q].copy()
        self.start=now
        self.lift=np.tile((1.10,-.10,-1.20),2)
        self.dorsal=np.tile((1.50,.10,-1.20),2)

    def command(self,now):
        age=now-self.start
        if age<.020:
            target=self.initial+land.smooth(age/.020)*(self.lift-self.initial)
        elif age<.045:
            target=self.lift+land.smooth((age-.020)/.025)*(self.dorsal-self.lift)
        else:
            target=self.dorsal+land.smooth((age-.115)/.025)*(self.p.base[self.q]-self.dorsal)
        action=self.p.ctrl_neutral.copy()
        idx=self.p.wing_act
        action[idx]=np.clip(target-self.p.data.qpos[self.q],self.p.min[idx],self.p.max[idx])
        return action
