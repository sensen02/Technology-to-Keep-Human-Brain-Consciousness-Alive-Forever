"""Where does the world-anchored gait break down: reach saturation, grounding, or height?"""
import numpy as np, json
from pathlib import Path
import gait_world as W
from gait_world import (Core, rc, T, LEGS, NL, NJ, MAX_RATE, W as WEIGHT,
                        rel_and_world, phase_of, TRIPOD_A)
HERE=Path(__file__).resolve().parent

def run(period=120,stride=0.02,lift=0.08,settle=300,ticks=1600,tau=0.05,kz=4.0):
    core=Core(); core.reset()
    rest=[np.asarray(core.measure()['foot'][l],float) for l in range(NL)]
    for k in range(settle):
        a=np.zeros(NJ)
        for leg in range(NL):
            rel,world=rel_and_world(core,leg); ids=T.leg_joint_action_index(core,leg); ds=core.a_dof[ids]
            cur=np.asarray(core.measure()['foot'][leg],float); des=(rest[leg]-cur)/tau
            A=rel[:,ds]; rv=core.d.qvel.copy(); rv[core.a_dof]=0
            dq=A.T@np.linalg.solve(A@A.T+1e-2*np.eye(3),des-rel@rv)
            a[ids]=dq/MAX_RATE/(1+np.abs(dq)/MAX_RATE)
        core.tick(np.clip(a,-1,1))
    anchors=[np.mean([core.d.geom_xpos[g] for g in core.pads[L]],axis=0) for L in LEGS]
    com0=np.array(core.d.subtree_com[core.th],float)
    log=[]
    for k in range(ticks):
        a=np.zeros(NJ); sat=0; reach=[]
        for leg in range(NL):
            rel,world=rel_and_world(core,leg); ids=T.leg_joint_action_index(core,leg); ds=core.a_dof[ids]
            ph=phase_of(leg,k,period)
            if ph<0.5: tgt=anchors[leg]; extra=0.
            else:
                u=(ph-0.5)/0.5; tgt=anchors[leg]+np.array([stride*u,0.,0.]); extra=lift*np.sin(np.pi*u)
                if u>=0.999: anchors[leg]=tgt.copy()
            pos=np.mean([core.d.geom_xpos[g] for g in core.pads[LEGS[leg]]],axis=0)
            err=float(np.linalg.norm(tgt-pos)); reach.append(err)
            des=(tgt-pos)/tau; des[2]+=kz*extra
            A=world[:,ds]; rv=core.d.qvel.copy(); rv[core.a_dof]=0
            dq=A.T@np.linalg.solve(A@A.T+1e-2*np.eye(3),des-world@rv)
            raw=dq/MAX_RATE
            if np.abs(raw).max()>1: sat+=1
            a[ids]=raw/(1+np.abs(raw))
        m=core.tick(np.clip(a,-1,1))
        if k%100==0:
            log.append(dict(k=k,com_z=float(core.d.subtree_com[core.th][2]),
                            com_x=float(core.d.subtree_com[core.th][0]),
                            com_y=float(core.d.subtree_com[core.th][1]),
                            sup=float(m['loads'].sum()/WEIGHT),nf=float(m['nonfoot']),
                            sat_legs=int(sat),reach_max=float(max(reach)),reach_mean=float(np.mean(reach)),
                            padz=np.round(m['padz'],3).tolist()))
            print(log[-1],flush=True)
    (HERE/'DIAG_WORLD.json').write_text(json.dumps(dict(label='reach and saturation trace',log=log),indent=2)+'\n')
    return log
if __name__=='__main__': run()
