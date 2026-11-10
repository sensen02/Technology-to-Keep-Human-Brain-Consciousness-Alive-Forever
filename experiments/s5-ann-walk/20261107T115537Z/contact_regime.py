"""Saved-contact friction/velocity regime diagnosis; scratch FK only, no mj_step.

Ellipse, Linfinity and L1 normalized traction ratios are all reported. None is
silently called the exact friction bound for a different native cone formulation.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE=ROOT/'experiments/s5-ann-incremental/20261107T010000Z'
sys.path.insert(0,str(BASE))
import mujoco as mj
from plant_probe import Plant
from incremental_actuator import DEFAULT_MODEL_PATH
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T115537Z')
RUN=OUT.parent/'20261107T112003Z/selected/teacher_selected_4s'
WAVE=OUT.parent/'20261107T081009Z/METADATA.json'
LEGS=('LF','LM','LH','RF','RM','RH')


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def summarize(a):
    if not len(a):return dict(contacts=0)
    # Columns t,leg,stance,Fn,Fz,vt,|vn|,ellipse,Linf,L1,jointcap,localphase,Ft1,Ft2,mu1,mu2
    w=a[:,3];total=w.sum()
    avg=lambda v:float(np.sum(w*v)/total)
    result=dict(contacts=len(a),summed_normal_weight=float(total),force_weighted_tangential_speed=avg(a[:,5]),
        force_weighted_abs_normal_speed=avg(a[:,6]),tangential_speed_max=float(a[:,5].max()),normal_speed_abs_max=float(a[:,6].max()),
        ellipse_utilization_mean=avg(a[:,7]),ellipse_utilization_max=float(a[:,7].max()),
        Linfinity_normalized_traction_mean=avg(a[:,8]),L1_normalized_traction_mean=avg(a[:,9]),
        L1_normalized_traction_max=float(a[:,9].max()),ellipse_gt_0p9_weight_fraction=avg(a[:,7]>.9),
        L1_gt_0p9_weight_fraction=avg(a[:,9]>.9),Linf_gt_0p9_weight_fraction=avg(a[:,8]>.9),
        slip_gt_0p1_weight_fraction=avg(a[:,5]>.1),ellipse_gt_0p9_AND_slip_gt_0p1_weight_fraction=avg((a[:,7]>.9)&(a[:,5]>.1)),
        L1_gt_0p9_AND_slip_gt_0p1_weight_fraction=avg((a[:,9]>.9)&(a[:,5]>.1)),
        ellipse_le_0p9_AND_slip_gt_0p1_weight_fraction=avg((a[:,7]<=.9)&(a[:,5]>.1)),
        L1_le_0p9_AND_slip_gt_0p1_weight_fraction=avg((a[:,9]<=.9)&(a[:,5]>.1)),
        joint_cap_weight_fraction=avg(a[:,10]>0),joint_cap_AND_slip_gt_0p1_weight_fraction=avg((a[:,10]>0)&(a[:,5]>.1)),
        normal_force_range=[float(w.min()),float(w.max())],actual_mu1_range=[float(a[:,14].min()),float(a[:,14].max())],
        actual_mu2_range=[float(a[:,15].min()),float(a[:,15].max())])
    for label,mask in [('cap',a[:,10]>0),('not_cap',a[:,10]==0)]:
        result['force_weighted_slip_when_'+label]=float(np.sum(w[mask]*a[mask,5])/w[mask].sum()) if mask.any() else None
    return result


def main():
    def forbidden(*a,**kw):raise AssertionError('No physics in saved-contact regime analysis')
    mj.mj_step=forbidden
    output=OUT/'CONTACT_REGIME.json';image=OUT/'CONTACT_REGIME.png'
    if output.exists() or image.exists():raise FileExistsError('Preserve existing contact diagnosis')
    files=[Path(__file__),RUN/'BODY.npz',RUN/'NATIVE.npz',RUN/'COMMANDS.npz',RUN/'RESULT.json',WAVE,
        DEFAULT_MODEL_PATH,BASE/'plant_probe.py',BASE/'incremental_actuator.py']
    hashes={str(f):sha(f) for f in files};B=load(RUN/'BODY.npz');N=load(RUN/'NATIVE.npz')['rows'];C=load(RUN/'COMMANDS.npz')
    wave=json.loads(WAVE.read_text());offset=np.array(wave['phase_offsets']);duty=wave['parameters']['duty']
    p=Plant();m=p.m;d=mj.MjData(m);rows=[];force_rows=[];velocity_error=0.;dims=set();friction=[];solref=[];solimp=[]
    names=[m.joint(int(j)).name for j in p.jids];indices=[np.array([i for i,n in enumerate(names) if n.startswith('joint_'+leg)]) for leg in LEGS]
    max_solver_iterations=0;max_solver_gradient=0.;max_solver_improvement=0.;max_solver_fwdinv=0.
    assert len(N)==len(B['time'])*400
    for i,t in enumerate(B['time']):
        d.qpos[:]=B['qpos'][i];d.qvel[:]=B['qvel'][i];d.ctrl[:]=B['ctrl'][i];d.time=float(t);d.qacc_warmstart[:]=0.
        mj.mj_forward(m,d);force_rows.append(p.contact(d))
        max_solver_iterations=max(max_solver_iterations,int(np.max(d.solver_niter)))
        if hasattr(d,'solver_fwdinv'):max_solver_fwdinv=max(max_solver_fwdinv,float(abs(d.solver_fwdinv).max()))
        for field in ('gradient','improvement'):
            try:value=float(np.max(abs(getattr(d.solver,field))))
            except (AttributeError,TypeError,ValueError):value=0.
            if field=='gradient':max_solver_gradient=max(max_solver_gradient,value)
            else:max_solver_improvement=max(max_solver_improvement,value)
        if t-.005<1.-1e-9:continue
        local=(float(t)-.25+offset)%1;caps=[bool(np.isclose(abs(B['ctrl'][i,ix]),C['limits'][ix],rtol=1e-7,atol=1e-9).any()) for ix in indices]
        for ci in range(d.ncon):
            co=d.contact[ci];ga,gb=int(co.geom1),int(co.geom2)
            if not(p.planes[ga] or p.planes[gb]):continue
            geom=gb if p.planes[ga] else ga;leg=int(p.feet[geom])
            if leg<0:continue
            f=np.zeros(6);mj.mj_contactForce(m,d,ci,f)
            if f[0]<=0:continue
            if co.dim<3:raise RuntimeError('Positive foot contact lacks sliding friction dimensions')
            mu=np.asarray(co.friction[:2]);assert np.all(mu>0)
            normalized=f[1:3]/(mu*f[0]);ellipse=float(np.linalg.norm(normalized));linf=float(abs(normalized).max());l1=float(abs(normalized).sum())
            bid=int(m.geom_bodyid[geom]);J=np.zeros((3,m.nv));mj.mj_jac(m,d,J,None,co.pos,bid);v=J@d.qvel
            twist=np.zeros(6);mj.mj_objectVelocity(m,d,mj.mjtObj.mjOBJ_BODY,bid,twist,0)
            rigid=twist[3:]+np.cross(twist[:3],co.pos-d.xipos[bid]);velocity_error=max(velocity_error,float(abs(v-rigid).max()))
            vc=co.frame.reshape(3,3)@v;fz=float((co.frame.reshape(3,3).T@f[:3])[2])*(1 if p.planes[ga] else -1)
            rows.append(np.r_[t,leg,local[leg]<duty,f[0],fz,np.linalg.norm(vc[1:3]),abs(vc[0]),ellipse,linf,l1,caps[leg],local[leg],f[1:3],mu])
            dims.add(int(co.dim));friction.append(co.friction.copy());solref.append(co.solref.copy());solimp.append(co.solimp.copy())
    a=np.asarray(rows);expected=N[399::400];force_rows=np.asarray(force_rows)
    force_error=float(abs(force_rows[:,1:9]-expected[:,1:9]).max());trusted=force_error<=1e-6 and velocity_error<=1e-8
    phasebins=np.zeros((6,36,3));phasebins[:]=np.nan;perleg={}
    for li,leg in enumerate(LEGS):
        contacts=a[a[:,1]==li]
        perleg[leg]={label:summarize(contacts[contacts[:,2]==stance]) for label,stance in [('stance',1),('swing',0)]}
        perleg[leg]['all']=summarize(contacts)
        for k in range(36):
            selected=contacts[(contacts[:,11]>=k/36)&(contacts[:,11]<(k+1)/36)]
            if len(selected):
                s=summarize(selected);phasebins[li,k]=[s['force_weighted_tangential_speed'],s['ellipse_utilization_mean'],s['L1_normalized_traction_mean']]
    friction=np.array(friction);solref=np.array(solref);solimp=np.array(solimp)
    report=dict(stage='S5 saved-contact friction/compliance regime diagnostic only',folder=str(RUN),physics_steps=0,
        trusted_reconstruction=trusted,normal_force_native_max_error=force_error,point_J_vs_xipos_twist_max_error=velocity_error,
        model=dict(cone=int(m.opt.cone),cone_name=str(mj.mjtCone(int(m.opt.cone)).name),contact_dimensions=sorted(dims),
            solver=int(m.opt.solver),solver_name=str(mj.mjtSolver(int(m.opt.solver)).name),iterations=int(m.opt.iterations),
            tolerance=float(m.opt.tolerance),noslip_iterations=int(m.opt.noslip_iterations),
            actual_contact_friction_min=friction.min(0),actual_contact_friction_max=friction.max(0),
            actual_contact_solref_min=solref.min(0),actual_contact_solref_max=solref.max(0),
            actual_contact_solimp_min=solimp.min(0),actual_contact_solimp_max=solimp.max(0)),
        scratch_solver=dict(max_iterations=max_solver_iterations,max_recorded_gradient=max_solver_gradient,
            max_recorded_improvement=max_solver_improvement,max_solver_fwdinv=max_solver_fwdinv,warnings=int(d.warning.number.sum()),
            caveat='Scratch forward reconstruction withzeroaccwarmstart; solverarraypadding can include unusedentries. Originalnativeintegration residualhistory notsaved.'),
        definitions='Fn=coframeforce[0]>0; Ft=coframeforce[1:3]; ellipse=norm(Ft_i/(mu_i*Fn)); Linf=maxabsnormalizedcomponents; L1=sumabsnormalizedcomponents. vt=norm((co.frame@Jmaterial*qvel)[1:3]), |vn|=abs(normalcomponent). Fnweights overcontacts andequal5mstimes.',
        phase='ReleaseBODYframe begins>=1s, local=(BODY.time-.25+081009offset)%1, stance<5/6; nofittedshift.',
        interpretation='Fornativepyramidalcondim3, L1normalizedratio corresponds tofourfriction-edge facet condition; ellipticalratio aloneisnot nativepyramidalboundary. Near.9threshold isdescriptive, notexactstick/slip proof. Significantinside-coneslipmayreflectsoftcontactregularization/dynamics; doesnotisolatecausality.',
        per_leg=perleg,aggregate={label:summarize(a[a[:,2]==stance]) for label,stance in [('stance',1),('swing',0)]},
        limitations='Forceweights trustedonlyifnativeFzmatch andindependentpointvelocitycheck pass. BODYsamples5ms missnativeevents. Actualcontactcoframeorientationvnormal signnotinferredfromforce; absolute normalvelocityreported. Capassociationnoncausal. No friction/compliance/model/solver tuning, no physicalacceptance or neuralclaim.',
        source_hashes=hashes,source_hashes_unchanged={f:sha(f)==h for f,h in hashes.items()})
    OUT.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(12,10),constrained_layout=True)
    for i,(title,vmax) in enumerate([('Actual material tangential speed',None),('Elliptical normalized traction ratio',1.),('Pyramidal L1 normalized traction ratio',1.)]):
        im=axes[i].imshow(phasebins[:,:,i],aspect='auto',origin='lower',extent=[0,1,-.5,5.5],vmin=0,vmax=vmax)
        axes[i].axvline(duty,color='w',ls='--');axes[i].set(yticks=np.arange(6),yticklabels=LEGS,xlabel='Nominal local cycle',title=title);fig.colorbar(im,ax=axes[i])
    fig.suptitle('Selected teacher saved contacts; cone '+report['model']['cone_name']+'; no physical changes;0mj_step')
    fig.savefig(image,dpi=150);plt.close(fig)
    print(json.dumps(dict(output=str(output),trusted=trusted,force_error=force_error,velocity_check=velocity_error,model=report['model'],hind={leg:perleg[leg]['stance'] for leg in ('LH','RH')}),default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x),indent=2),flush=True)
    return 0 if trusted else 2


if __name__=='__main__':raise SystemExit(main())
