"""Adjust rear tarsal clearance inside the teacher's private inverse kinematics."""
import argparse
from pathlib import Path
import types

import mujoco as mj
import numpy as np
import torch

import turning as gait


def add_clearance(teacher,c,margin):
    baseline=teacher._ik
    li=gait.LEGS.index("RH")
    inds=teacher.leg_act[li]
    qa=teacher.qa[inds]
    va=teacher.va[inds]
    m=c.plant
    geom=m.geom("RHTarsus4_geom").id
    body=int(m.geom_bodyid[geom])
    mesh=int(m.geom_dataid[geom])
    i,n=int(m.mesh_vertadr[mesh]),int(m.mesh_vertnum[mesh])
    vertices=m.mesh_vert[i:i+n]
    jac=np.zeros((3,m.nv))
    ridge=.0005*np.eye(3)

    def task(self,c,desired):
        former=self.q_des[self.qa].copy()
        achieved,_,_=baseline(c,desired)
        for _ in range(5):
            shadow=self.shadow
            rot=shadow.geom_xmat[geom].reshape(3,3)
            points=vertices@rot.T+shadow.geom_xpos[geom]
            lowest=points[np.argmin(points[:,2])]
            error=margin-lowest[2]
            if error<=0.:
                break
            mj.mj_jac(m,shadow,jac,None,lowest,body)
            z=jac[2,va]
            _, tip=self._leg_kinematics(c,li)
            normal=z-tip.T@np.linalg.solve(tip@tip.T+ridge,tip@z)
            increment=np.clip(.75*error*normal/(normal@normal+.001),-.075,.075)
            self.q_des[qa]=np.clip(self.q_des[qa]+increment,self.bounds[inds,0],self.bounds[inds,1])
            self._fk()
        achieved[li]=self._leg_kinematics(c,li,False)
        raw=(self.q_des[self.qa]-former)/self.config["command_dt_s"]
        return achieved,np.clip(raw,-2.,2.),raw

    teacher._ik=types.MethodType(task,teacher)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    p.add_argument("margin",type=float)
    a=p.parse_args()
    torch.set_num_threads(1)
    old=gait.make_teacher

    def make(c,mode,speed,omega):
        teacher=old(c,mode,speed,omega)
        add_clearance(teacher,c,a.margin)
        return teacher

    gait.make_teacher=make
    try:
        gait.run(a.out.resolve(),"nominal","arc_left","teacher",.13,.115,4800,0)
    finally:
        gait.make_teacher=old


if __name__=="__main__":
    main()
