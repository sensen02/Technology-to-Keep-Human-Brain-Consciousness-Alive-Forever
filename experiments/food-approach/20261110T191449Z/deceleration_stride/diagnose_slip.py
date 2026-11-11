"""Saved slip numerator/denominator and forward-only contact-velocity decomposition."""
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import json
import numpy as np
from api import HERE
import physics
import mujoco as mj
LEGS=physics.LEGS

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def summarize(a):
    post=a['control_source']==5;l=a['loads'][post];s=a['speed'][post];com=a['com'][post];t=a['time'][post]
    den=l.sum(0);num=(l*s).sum(0);duration=t[-1]-t[0];xy=np.linalg.norm(com[-1,:2]-com[0,:2])/duration
    per={}
    for i,leg in enumerate(LEGS):
        masks=dict(loaded=l[:,i]>=.1*physics.W,unloaded=l[:,i]<=.05*physics.W,intermediate=(l[:,i]>.05*physics.W)&(l[:,i]<.1*physics.W))
        per[leg]=dict(vt=float(num[i]/den[i]),slip=float(num[i]/den[i]/xy),load_denominator=float(den[i]),
            loadweighted_speed_numerator=float(num[i]),by_load_band={k:dict(samples=int(m.sum()),numerator=float((l[m,i]*s[m,i]).sum()),
            numerator_fraction=float((l[m,i]*s[m,i]).sum()/max(num[i],1e-30)),load_denominator=float(l[m,i].sum())) for k,m in masks.items()})
    return dict(xy_progress_denominator=float(xy),signed_worldX=float((com[-1,0]-com[0,0])/duration),per_leg=per)

def reconstruct(a,c):
    ids=np.flatnonzero(a['control_source']==5)[::5]
    jac=np.zeros((3,c.plant.nv));den=np.zeros(6);total=np.zeros(6);absx=np.zeros(6);absy=np.zeros(6);sx=np.zeros(6);sy=np.zeros(6);x2=np.zeros(6);y2=np.zeros(6)
    load_err=speed_err=0.;load_errsum=speed_errsum=0.;actual_total=np.zeros(6);actual_den=np.zeros(6)
    for j in ids:
        # Private reconstruction only; no mj_step/tick, integration, or actualrun edits.
        c.d.qpos[:]=a['qpos'][j];c.d.qvel[:]=a['qvel'][j];c.d.ctrl[:]=a['actualctrl'][j]
        c.d.qacc_warmstart[:]=0;c.d.xfrc_applied[:]=0;c.d.qfrc_applied[:]=0
        mj.mj_forward(c.plant,c.d)
        loads=np.zeros(6);wt=np.zeros(6)
        for ci in range(c.d.ncon):
            contact=c.d.contact[ci];ga,gb=int(contact.geom1),int(contact.geom2)
            if not(c.planes[ga] or c.planes[gb]):continue
            geom=gb if c.planes[ga] else ga;leg=int(c.geom_leg[geom])
            if leg<0:continue
            force=np.zeros(6);mj.mj_contactForce(c.plant,c.d,ci,force)
            normal=max(0.,float((contact.frame.reshape(3,3).T@force[:3])[2])*(1 if c.planes[ga] else -1))
            mj.mj_jac(c.plant,c.d,jac,None,np.asarray(contact.pos),int(c.plant.geom_bodyid[geom]))
            velocity=jac@c.d.qvel;vx,vy=velocity[:2];sp=float(np.hypot(vx,vy))
            loads[leg]+=normal;wt[leg]+=normal*sp
            den[leg]+=normal;total[leg]+=normal*sp;absx[leg]+=normal*abs(vx);absy[leg]+=normal*abs(vy)
            sx[leg]+=normal*vx;sy[leg]+=normal*vy;x2[leg]+=normal*vx*vx;y2[leg]+=normal*vy*vy
        speed=np.where(loads>0,wt/np.maximum(loads,1e-12),0)
        le=float(np.max(np.abs(loads-a['loads'][j])));se=float(np.max(np.abs(speed-a['speed'][j])))
        load_err=max(load_err,le);speed_err=max(speed_err,se);load_errsum+=le;speed_errsum+=se
        actual_den+=a['loads'][j];actual_total+=a['loads'][j]*a['speed'][j]
    per={leg:dict(contact_loadweighted_tangent_speed=float(total[i]/den[i]),absolute_vx=float(absx[i]/den[i]),absolute_vy=float(absy[i]/den[i]),
        signed_vx=float(sx[i]/den[i]),signed_vy=float(sy[i]/den[i]),rms_vx=float(np.sqrt(x2[i]/den[i])),rms_vy=float(np.sqrt(y2[i]/den[i])),
        saved5ms_vt=float(actual_total[i]/actual_den[i]),reconstructed5ms_vt=float(total[i]/den[i])) for i,leg in enumerate(LEGS)}
    return dict(samples=len(ids),native_integrations=0,method='Private savedqpos/qvel/actualctrl +mj_forward, cleared unrecorded warmstart; original measure contactpoint Jacobian and normalforce weights; every5ms only',
        max_load_difference_native=load_err,mean_sample_max_load_difference_native=load_errsum/len(ids),max_speed_difference=speed_err,mean_sample_max_speed_difference=speed_errsum/len(ids),per_leg=per,
        limitation='Unrecorded solverwarmstart may change reconstructed contactforce weights; discrepancies reported, no reconstructed values substituted into original acceptance. Decomposition worldX/Y, fullteacher turning+008 is existing comparison, not a straightfull matchingcontrol.')

def main():
    paths=dict(stride_018=HERE/'stride_018'/'TRACE_1MS.npz',full_teacher_left=HERE.parent/'teacher'/'left_full'/'TRACE_1MS.npz')
    c=physics.Core(adhesion=False);out={}
    for name,path in paths.items():
        a=load(path);out[name]=dict(saved_exact_1ms=summarize(a),reconstructed5ms=reconstruct(a,c))
    full=out['full_teacher_left']['saved_exact_1ms'];half=out['stride_018']['saved_exact_1ms']
    comparison={leg:dict(vt_ratio_half_over_full=half['per_leg'][leg]['vt']/full['per_leg'][leg]['vt'],
        xy_denominator_ratio_half_over_full=half['xy_progress_denominator']/full['xy_progress_denominator'],
        absolute_vx_ratio_half_over_full=out['stride_018']['reconstructed5ms']['per_leg'][leg]['absolute_vx']/out['full_teacher_left']['reconstructed5ms']['per_leg'][leg]['absolute_vx'],
        absolute_vy_ratio_half_over_full=out['stride_018']['reconstructed5ms']['per_leg'][leg]['absolute_vy']/out['full_teacher_left']['reconstructed5ms']['per_leg'][leg]['absolute_vy']) for leg in LEGS}
    result=dict(cases=out,comparison=comparison,
        interpretation='Original metric numerator/load denominator contributions computed exactly from saved1ms evidence. Contactx/y reconstruction is descriptive at5ms, no extra physics trajectory. Existing fullteacher turns yaw+.008 while stride018 is straight: comparison cannot isolate unchanged lateral reference as sole causal factor.')
    (HERE/'SLIP_DIAGNOSIS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
