"""Search joint angles, thorax height and tonic activations against native inverse dynamics."""
from pathlib import Path
import argparse
import sys

import mujoco as mj
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / '20261120T074805Z-linear-interfaces'
sys.path.insert(0, str(PREVIOUS))
from assay import BASE, BodyObservations, MODEL, load_pose
LEG_PREFIXES=('LF','LM','LH','RF','RM','RH')
SEGMENTS=('Coxa','Femur','Trochanter','Tibia','Tarsus')


def body(contact_ms=20.):
    model = mj.MjModel.from_xml_path(str(MODEL))
    # Optional explicitly identified alternate floor, only the plane has priority.
    if contact_ms != 20.:
        floor = model.geom('floor').id
        model.geom_priority[floor] = 1
        model.geom_solref[floor, 0] = contact_ms * .001
    return model


class Equilibrium:
    def __init__(self, model):
        self.m, self.d = model, mj.MjData(model)
        self.obs = BodyObservations(model)
        self.qidx = np.r_[2, model.jnt_qposadr[1:]]
        self.lo = np.r_[1.25, model.jnt_range[1:,0]+1e-5, -.14,-.30, np.full(model.nu,.0001)]
        self.hi = np.r_[2.25, model.jnt_range[1:,1]-1e-5, .14,.30, np.ones(model.nu)]
        self.qref = np.load(BASE/'body/archived_pose.npz')['qpos']
        self.n = len(self.qidx)+2
        self.reference = np.r_[self.qref[self.qidx],0.,0.]
        self.ids = np.array([model.geom(leg+'Tarsus5_geom').id for leg in ('LF','LM','LH','RF','RM','RH')])
        self.verts = []
        for gid in self.ids:
            mesh = model.geom_dataid[gid]
            j = model.mesh_vertadr[mesh]
            self.verts.append(model.mesh_vert[j:j+model.mesh_vertnum[mesh]].astype(float))
        self.d.qpos[:] = self.qref
        self.foot_geoms = np.array(list(self.obs.foot_by_geom))
        self.signs = np.array([1 if model.geom(g).name.startswith('L') else -1 for g in self.foot_geoms])

    def footprint(self):
        d=self.d
        xy=d.geom_xpos[self.ids,:2]
        margin=d.geom_xpos[self.foot_geoms,1]*self.signs
        fore_hind=xy[[0,3],0]-xy[[2,5],0]
        middle_hind=xy[[1,4],0]-xy[[2,5],0]
        distances=[np.linalg.norm(xy[i]-xy[j]) for i,j in ((0,1),(1,2),(0,2),(3,4),(4,5),(3,5))]
        return np.maximum(np.r_[.24-margin,.40-fore_hind,.22-middle_hind,.25-np.array(distances)],0.)

    def set_pose(self, values):
        self.d.qpos[self.qidx] = values[:-2]
        mj.mju_euler2Quat(self.d.qpos[3:7], np.r_[values[-2:],0.], 'xyz')

    def residual(self, x, keep):
        m, d = self.m, self.d
        self.set_pose(x[:self.n])
        d.act[:] = x[self.n:]
        d.ctrl[:] = d.act
        d.qacc[:] = 0.
        mj.mj_inverse(m, d)
        mj.mj_fwdActuation(m, d)
        force = (d.qfrc_inverse - d.qfrc_actuator) / self.obs.weight
        dist = np.array([d.geom_xpos[g,2] + (v @ d.geom_xmat[g,6:9]).min()
                         for g,v in zip(self.ids,self.verts,strict=True)])
        return np.r_[force, 3*np.maximum(dist-.0005,0), 3*np.minimum(dist+.011,0), 2*self.footprint(),
                     keep*(x[:self.n]-self.reference), keep*.1*(d.act-.3)]

    def fit(self, out, seed, iterations):
        init = np.r_[self.reference, np.full(self.m.nu,.25)] if seed is None else np.load(seed)['x']
        x = np.clip(init,self.lo+1e-9,self.hi-1e-9)
        for keep in (.012, .0003, .000001):
            def fun(v):
                return self.residual(v, keep)
            def jac(v):
                orig = fun(v)
                j = np.zeros((len(orig),len(v)))
                for i in range(self.n):
                    w=v.copy()
                    w[i]+=1e-7
                    j[:,i]=(fun(w)-orig)/1e-7
                fun(v)
                d,m=self.d,self.m
                d.act[:]=0
                mj.mj_fwdActuation(m,d)
                f=d.actuator_force.copy()
                d.act[:]=1
                mj.mj_fwdActuation(m,d)
                from run_body import moment_matrix
                j[:m.nv,self.n:] = -moment_matrix(m,d).T * (d.actuator_force-f)/self.obs.weight
                j[-m.nu:,self.n:] = np.eye(m.nu) * keep*.1
                return j
            fit = least_squares(fun,x,jac=jac,bounds=(self.lo,self.hi),max_nfev=iterations,
                                ftol=1e-13,gtol=1e-12,xtol=1e-13)
            x = fit.x
            r = fun(x)
            q=self.d.qpos.copy()
            d=load_pose(self.m,q)
            d.act[:]=x[self.n:]
            d.ctrl[:]=d.act
            mj.mj_forward(self.m,d)
            z=self.obs.measure(d)
            print(keep,fit.nfev,'balance',round(max(abs(r[:self.m.nv])),7),
                  'loads',z['ground_load_w'].round(4),'depth',round(min(c.dist for c in d.contact),5),flush=True)
            np.savez_compressed(out,x=x,qpos=q,act=d.act,balance=r[:self.m.nv],contact_ms=self.contact_ms)
        return x


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--contact-ms',type=float,default=20.)
    parser.add_argument('--seed',type=Path)
    parser.add_argument('--iterations',type=int,default=100)
    args=parser.parse_args()
    eq=Equilibrium(body(args.contact_ms))
    eq.contact_ms=args.contact_ms
    eq.fit(args.output,args.seed,args.iterations)
