"""Linearize the actual free body at the equilibrium and stabilize it via native muscle controls."""
import numpy as np
from scipy.linalg import solve_discrete_are, solve
from solve import HERE, body, load_pose, mj


class MuscleTeacher:
    def __init__(self, model, pose, activation):
        self.model,self.pose,self.activation=model,pose,activation
        d=load_pose(model, pose)
        d.act[:]=activation
        d.ctrl[:]=activation
        mj.mj_forward(model,d)
        nv,na=model.nv,model.na
        n=2*nv+na
        A=np.empty((n,n));B=np.empty((n,model.nu))
        mj.mjd_transitionFD(model,d,1e-7,True,A,B,None,None)
        # Five native steps per command. Ignore absolute horizontal position and heading.
        block_B=sum(np.linalg.matrix_power(A,k) @ B for k in range(5))
        block_A=np.linalg.matrix_power(A,5)
        self.states=np.delete(np.arange(n),[0,1,5])
        a=block_A[np.ix_(self.states,self.states)]
        self.arms=np.flatnonzero((activation>.001) & (activation<.999))
        b=block_B[np.ix_(self.states,self.arms)]
        q=np.diag(np.r_[np.full(nv,120.),np.full(nv,.007),np.full(na,.015)][self.states])
        r=np.eye(len(self.arms))*.04
        p=solve_discrete_are(a,b,q,r)
        self.gain=solve(r+b.T@p@b,b.T@p@a,assume_a='pos')
        self.radius=float(np.max(abs(np.linalg.eigvals(a-b@self.gain))))

    def command(self,data):
        pos=np.empty(self.model.nv)
        mj.mj_differentiatePos(self.model,pos,1.,self.pose,data.qpos)
        state=np.r_[pos,data.qvel,data.act-self.activation]
        u=self.activation.copy()
        u[self.arms]-=self.gain@state[self.states]
        return np.clip(u,self.model.actuator_ctrlrange[:,0],self.model.actuator_ctrlrange[:,1])


if __name__=='__main__':
    from solve import BodyObservations
    import json
    stance=json.loads((HERE/'stance.json').read_text())
    a={'qpos':np.array(stance['qpos']), 'act':np.array(stance['tonic_activation'])}
    m=body(1);t=MuscleTeacher(m,a['qpos'],a['act']);o=BodyObservations(m)
    print(t.radius,len(t.arms),flush=True)
    for sign in (0,-1,1):
        d=load_pose(m,a['qpos']);d.act[:]=a['act'];d.ctrl[:]=d.act
        d.qvel[0]=sign*.03
        mj.mj_forward(m,d)
        for i in range(10000):
            if i%5==0:
                d.ctrl[:]=t.command(d)
            mj.mj_step(m,d)
        mj.mj_forward(m,d)
        print(sign,o.measure(d)['ground_load_w'],d.qpos[:3],max(abs(d.qvel)),flush=True)
