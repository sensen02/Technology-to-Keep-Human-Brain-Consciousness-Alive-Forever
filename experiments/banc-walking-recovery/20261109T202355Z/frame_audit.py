"""S2/S5 coordinate audit; training-only teacher, no motion acceptance claims."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import mujoco as mj
HERE=Path(__file__).resolve().parent
OLD=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-repair-pipeline/20261109T055352Z')
sys.path.insert(0,str(OLD))
import repair_core as rc
import train_single_leg as T
import teacher_eval as E

def skew(v):
    x,y,z=v
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])

def jacobians(c,leg):
    m,d=c.plant,c.d
    jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv))
    mj.mj_jac(m,d,jp,jr,d.xpos[c.th],c.th)
    pads=c.pads[rc.LEG_ORDER[leg]]
    p=np.mean([d.geom_xpos[g] for g in pads],axis=0)
    J=np.zeros((3,m.nv))
    for g in pads:
        j=np.zeros_like(J)
        mj.mj_jac(m,d,j,None,d.geom_xpos[g],int(m.geom_bodyid[g]))
        J+=j/len(pads)
    R=d.xmat[c.th].reshape(3,3)
    relative=R.T@(J-jp+skew(p-d.xpos[c.th])@jr)
    return relative,J

def direction(c,leg,axis):
    rel,world=jacobians(c,leg)
    j=rel[0] if axis=='forward' else -world[2]
    a=np.zeros(rc.NJ)
    for i in T.leg_joint_action_index(c,leg):
        a[i]=np.sign(j[int(c.actuator.dof_addresses[i])])
    return a

def corrected(c,leg,k):
    rise=-direction(c,leg,'down')
    if k<60:return rise
    if k<150:return np.clip(direction(c,leg,'forward')+.5*rise,-1,1)
    return direction(c,leg,'down')

def audit():
    c=rc.Core(); c.reset(); rows=[]
    for name,angle in [('nominal',0.),('yaw90',np.pi/2),('yaw180',np.pi)]:
        c.reset()
        if angle:
            q=np.zeros(4);mj.mju_axisAngle2Quat(q,np.array([0.,0.,1.]),angle)
            c.d.qpos[3:7]=q
        mj.mj_forward(c.plant,c.d)
        q0=c.d.qpos.copy()
        for leg in range(rc.NL):
            analytic,world=jacobians(c,leg)
            numeric=np.zeros_like(analytic);nz=np.zeros(c.plant.nv)
            eps=1e-6
            for j in range(c.plant.nv):
                vel=np.zeros(c.plant.nv);vel[j]=1
                vals=[]
                for sign in (1,-1):
                    c.d.qpos[:]=q0
                    mj.mj_integratePos(c.plant,c.d.qpos,vel,sign*eps)
                    mj.mj_forward(c.plant,c.d)
                    meas=c.measure();vals.append((meas['foot'][leg].copy(),meas['padz'][leg]))
                numeric[:,j]=(vals[0][0]-vals[1][0])/(2*eps)
                nz[j]=(vals[0][1]-vals[1][1])/(2*eps)
            c.d.qpos[:]=q0;mj.mj_forward(c.plant,c.d)
            # Check mean center-z, since min across equal-height spheres is non-differentiable.
            oldx=T.foot_horizontal_jacobian(c,leg)
            rows.append(dict(pose=name,leg=rc.LEG_ORDER[leg],relative_error=float(np.max(np.abs(analytic-numeric))),old_worldx_vs_measured_error=float(np.max(np.abs(oldx-numeric[0])))))
    return rows

def main():
    rows=audit(); print('Jacobian max error',max(r['relative_error'] for r in rows),flush=True)
    assert max(r['relative_error'] for r in rows)<1e-5
    outcomes={}
    for name,policy in [('old',T.teacher_action),('coordinate_corrected',corrected)]:
        outcomes[name]={}
        for leg in range(rc.NL):
            r=E.score(leg,policy)
            outcomes[name][rc.LEG_ORDER[leg]]=r
            print(name,rc.LEG_ORDER[leg],r['passed_implemented'],round(r['advance_implemented'],5),round(r['support_min'],4),round(r['com_min'],4),flush=True)
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OLD/'repair_core.py',OLD/'train_single_leg.py',OLD/'teacher_eval.py',Path(rc.PAD_XML)]}
    result=dict(label='coordinate audit and declared teacher comparison; NOT walking',jacobian_checks=rows,teacher_comparison=outcomes,source_hashes=sources,acceptance_thresholds=T.THRESH)
    (HERE/'FRAME_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
