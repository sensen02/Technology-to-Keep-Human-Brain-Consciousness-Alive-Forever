"""Current-state body correction below the recorded neck. No scene or goal input."""
import json
import numpy as np
from scipy.special import expit
from context import HERE,rollout,tr
from stance import features, LEGS
from body_search import NATIVE_WALK

def controller(params):
    support_gain, swing_gain, adhesion, speed_gain=params
    class Recovery(NATIVE_WALK):
        def __init__(self,env,duration):
            super().__init__(env,duration)
            with np.load(HERE/'STANCE.npz') as z:
                self.mean,self.scale,self.coef,self.stance=[z[k] for k in ('mean','scale','coef','stance')]
            # Use exactly the existing claw/tarsus classification, without running its logging hooks.
            self.scan=BaseContacts(env.physics)
            self.joints=[]
            for j,name in enumerate(self.commands):
                for i,(leg,side) in enumerate(LEGS):
                    if not name.endswith('_'+leg+'_'+side):continue
                    if 'adhere_claw' in name:
                        self.joints.append((j,i,-1))
                    else:
                        k=env.physics.model.name2id('walker/'+name,'joint')
                        self.joints.append((j,i,int(env.physics.model.jnt_qposadr[k])))
            self.j=np.array([s[0] for s in self.joints]); self.leg=np.array([s[1] for s in self.joints])
            self.addr=np.array([s[2] for s in self.joints]); self.isjoint=self.addr>=0
            self.cols=np.array(self.columns)[self.j]
            self.good=self.stance[self.leg,self.addr]
        def action(self,phy,speed,turn,obs):
            feet,_,_=self.scan.scan(phy)
            loaded=np.array([key in feet for key in LEGS])
            f=features(loaded,phy.data.qpos,phy.data.qvel[2])
            risk=expit(self.coef[0]+((f-self.mean)/self.scale)@self.coef[1:])
            a=super().action(phy,speed*speed_gain,turn,obs)
            bearing=loaded[self.leg]
            rate=risk*np.where(bearing,support_gain,swing_gain)
            target=np.where(self.isjoint,self.good,np.where(bearing,adhesion,0.))
            a[self.cols]+=rate*(target-a[self.cols])
            return np.clip(a,self.full.minimum,self.full.maximum)
    return Recovery

BaseContacts=tr.docking.Contacts
if __name__=='__main__':
    best=json.loads((HERE/'SELECTED.json').read_text())
    tr.docking.ReversedYawWalk=controller(best['params'])
    rollout.main()
