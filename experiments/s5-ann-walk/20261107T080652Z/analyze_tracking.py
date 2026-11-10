"""Analyze saved teacher tracking with FK only; never integrate physics.

Default: compare teacher_gain2_4s and teacher_gain8_4s, write TRACKING_DIAGNOSIS.json
and TRACKING.png in the authorized080652 output directory after BOTH complete.
--folder PATH --no-write permits read-only analysis of an earlier saved baseline.
BODY frames end at command ticks4,9,... . Integrated targets are reconstructed
AFTER those ticks, including rate/angle clipping. Counterfactual target feet use
that actual free-root pose with42hinges replaced FOR FK ONLY. All clearances and
foot drift here refer to Tarsus5 BODY ORIGINS, not sole/material contact points.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE=ROOT/'experiments/s5-ann-incremental/20261107T010000Z'
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T080652Z')
sys.path[:0]=[str(BASE),str(ROOT/'tools')]
import dynamic_support_metrics as dsm
LEGS=('LF','LM','LH','RF','RM','RH')
CDT,PDT=.001,.005


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    with np.load(p,allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def runs(mask):
    edges=np.diff(np.r_[False,np.asarray(mask,bool),False].astype(np.int8))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))


def sustained(load,on,off):
    state=bool(load[0]>=on);output=np.full(len(load),state,bool);count=0
    for i,value in enumerate(load):
        changed=value<=off if state else value>=on
        count=count+1 if changed else 0
        if count>=2:
            state=not state;output[i-count+1:i+1]=state;count=0
        output[i]=state
    return output


def rms(x):
    x=np.asarray(x)
    return float(np.sqrt(np.mean(x*x))) if x.size else None


def summary(x):
    x=np.asarray(x)
    return dict(rms=rms(x),max_abs=float(np.abs(x).max()) if x.size else None)


def complete_intervals(mask,t):
    return [dict(start_s=float(t[a]-PDT),end_s=float(t[b-1]),seconds=float((b-a)*PDT))
            for a,b in runs(mask) if a>0 and b<len(mask) and b-a>=2]


def signal_lag(actual,target,max_frames=20):
    """Exploratory target-to-actual lag, using only same-release overlapping samples."""
    tests=[]
    for lag in range(-max_frames,max_frames+1):
        if lag>0:a,b=actual[lag:],target[:-lag]
        elif lag<0:a,b=actual[:lag],target[-lag:]
        else:a,b=actual,target
        if len(a)>=10:
            tests.append((float(np.mean((a-b)**2)),lag))
    if not tests:return None
    error,lag=min(tests)
    return dict(milliseconds=lag*PDT*1000,rms_after_shift=float(np.sqrt(error)),
        convention='positive means actual lags target; exploratory overlap search +/-100ms, not a causal delay estimate')


def analyze(folder,m,mj,handover,cap):
    bd=load(folder/'BODY.npz');cc=load(folder/'COMMANDS.npz')
    native=load(folder/'NATIVE.npz')['rows']
    result=json.loads((folder/'RESULT.json').read_text())
    commands=cc['rows'];t=bd['time']
    assert native.shape==(80*len(commands),20)
    ticks=np.arange(4,len(commands),5)
    assert len(ticks)==len(t)
    assert np.allclose(t,commands[ticks,0]+CDT,rtol=0,atol=1e-9)
    assert np.allclose(t,native[399::400,0],rtol=0,atol=1e-9)
    assert np.allclose(np.diff(t),PDT,rtol=0,atol=1e-10)
    assert all(np.isfinite(a).all() for a in [commands,native,*bd.values()])
    jids=m.actuator_trnid[:,0].copy();qa=m.jnt_qposadr[jids]
    names=[m.joint(int(j)).name for j in jids]
    leg_indices=[np.array([i for i,name in enumerate(names) if name.startswith('joint_'+leg)],int) for leg in LEGS]
    assert all(len(i)==7 for i in leg_indices) and len(set(np.concatenate(leg_indices)))==42
    limits=cc['limits'];kp,kd=cc['kp'],cc['kd']
    native_limits=np.minimum(-m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
    assert np.all(limits<=native_limits+1e-12)
    low,high=m.jnt_range[jids,0],m.jnt_range[jids,1]
    # Commands log targets BEFORE1ms update. BODY ends AFTER update and80steps.
    integrated=np.clip(commands[:,1:43]+CDT*np.clip(commands[:,43:85]+commands[:,85:127],-cap,cap),low,high)
    consistency=float(np.max(np.abs(integrated[:-1]-commands[1:,1:43])))
    assert consistency<1e-10
    target=integrated[ticks]
    joint_error=bd['qpos'][:,qa]-target
    ctrl=bd['ctrl'];saturated=np.abs(ctrl)>=limits*(1-1e-6)
    assert np.all(np.abs(ctrl)<=limits+1e-9)
    force=native[:,1:7].reshape(len(t),400,6).mean(axis=1)
    loaded=np.column_stack([sustained(force[:,j],.01*dsm.WEIGHT,.002*dsm.WEIGHT) for j in range(6)])
    keep=t-PDT>=handover-1e-9
    assert keep.sum()>20
    actual_data=mj.MjData(m);target_data=mj.MjData(m)
    bids=[m.body(leg+'Tarsus5').id for leg in LEGS]
    thorax=m.body('Thorax').id
    anchor=mj.MjData(m);q0=m.key_qpos[0].copy();q0[qa]=np.clip(q0[qa],low,high)
    anchor.qpos[:]=q0;mj.mj_forward(m,anchor)
    anchor_rot=anchor.xmat[thorax].reshape(3,3)
    anchors=(anchor.xpos[bids]-anchor.xpos[thorax])@anchor_rot
    actual_world=np.empty((len(t),6,3));target_world=np.empty_like(actual_world)
    actual_local=np.empty_like(actual_world);target_local=np.empty_like(actual_world)
    saved_feet_error=0.
    plane=m.geom_type==mj.mjtGeom.mjGEOM_PLANE
    geom_leg=np.array([next((j for j,leg in enumerate(LEGS) if (m.geom(g).name or '').startswith(leg) and 'Tars' in (m.geom(g).name or '')),-1) for g in range(m.ngeom)])
    target_penetration=np.zeros((len(t),6));actual_penetration=np.zeros_like(target_penetration)
    for i,pose in enumerate(bd['qpos']):
        actual_data.qpos[:]=pose
        mj.mj_forward(m,actual_data)
        target_data.qpos[:]=pose
        target_data.qpos[qa]=target[i]  # COUNTERFACTUAL FK ONLY, never a physics step.
        mj.mj_forward(m,target_data)
        assert np.array_equal(actual_data.qpos[:7],target_data.qpos[:7])
        assert actual_data.time==target_data.time==anchor.time==0.
        origin=actual_data.xpos[thorax];rot=actual_data.xmat[thorax].reshape(3,3)
        assert np.allclose(target_data.xpos[thorax],origin,atol=1e-12,rtol=0)
        actual_world[i]=actual_data.xpos[bids]
        target_world[i]=target_data.xpos[bids]
        actual_local[i]=(actual_world[i]-origin)@rot
        target_local[i]=(target_world[i]-origin)@rot
        saved_feet_error=max(saved_feet_error,float(np.abs(actual_world[i]-bd['feet'][i]).max()))
        for scratch,penetration in ((actual_data,actual_penetration),(target_data,target_penetration)):
            for contact in scratch.contact[:scratch.ncon]:
                ga,gb=int(contact.geom1),int(contact.geom2)
                if not (plane[ga] or plane[gb]):
                    continue
                leg=geom_leg[gb if plane[ga] else ga]
                if leg>=0:
                    penetration[i,leg]=max(penetration[i,leg],max(0.,-float(contact.dist)))
    assert saved_feet_error<1e-12
    foot_error=actual_world-target_world
    target_height=target_local[:,:,2]-anchors[None,:,2]
    requested_swing=target_height>=.05
    assist=native[(native[:,0]>=handover-.2)&(native[:,0]<handover)]
    release=native[native[:,0]>=handover]
    gates=dsm.evaluate_native(dict(foot_support_mean=float(assist[:,1:7].sum(1).mean()/dsm.WEIGHT),
                                   com_z_mean=float(assist[:,10].mean())),release)
    gates['checks']['D1_no_nonfoot_ground_force']=bool(release[:,7].max()==0)
    gates['passed']=all(gates['checks'].values())
    per_joint=[]
    for i,name in enumerate(names):
        per_joint.append(dict(actuator_index=i,joint=name,qpos_address=int(qa[i]),
            error_rad=summary(joint_error[keep,i]),sampled_saturation_fraction=float(saturated[keep,i].mean()),
            torque_limit_native=float(limits[i]),kp=float(kp[i]),kd=float(kd[i])))
    per_leg=[]
    for j,leg in enumerate(LEGS):
        ii=leg_indices[j]
        local_t=t[keep];load_state=loaded[keep,j];wanted=requested_swing[keep,j]
        mask=keep & requested_swing[:,j]
        err=joint_error[:,ii]
        gap=np.linalg.norm(foot_error[:,j],axis=1)
        saturation_any=saturated[:,ii].any(axis=1)
        loaded_baseline=float(np.median(actual_world[keep & loaded[:,j],j,2])) if np.any(keep & loaded[:,j]) else None
        actual_clear=actual_world[:,j,2]-loaded_baseline if loaded_baseline is not None else np.full(len(t),np.nan)
        target_clear=target_world[:,j,2]-loaded_baseline if loaded_baseline is not None else np.full(len(t),np.nan)
        requested=complete_intervals(wanted,local_t)
        unload=complete_intervals(~load_state,local_t)
        leg_report=dict(leg=leg,actuator_indices=ii.tolist(),joint_names=[names[i] for i in ii],
            joint_error_rad=summary(err[keep]),sampled_saturation_joint_fraction=float(saturated[keep][:,ii].mean()),
            sampled_saturation_any_joint_fraction=float(saturation_any[keep].mean()),
            foot_position_error_3d_rms=float(np.sqrt(np.mean(gap[keep]**2))),
            foot_position_error_3d_max=float(gap[keep].max()),
            foot_position_error_component_rms=np.sqrt(np.mean(foot_error[keep,j]**2,axis=0)).tolist(),
            mean_foot_load_weight=float(force[keep,j].mean()/dsm.WEIGHT),
            actual_sustained_loaded_fraction=float(load_state.mean()),
            actual_complete_unloads=len(unload),actual_unload_intervals=unload,
            requested_complete_lifts=len(requested),requested_lift_intervals=requested,
            target_local_origin_lift_peak=float(target_height[keep,j].max()),
            actual_local_origin_lift_peak=float((actual_local[keep,j,2]-anchors[j,2]).max()),
            actual_loaded_origin_z_baseline=loaded_baseline,
            target_world_origin_clearance_peak=float(np.nanmax(target_clear[keep])) if loaded_baseline is not None else None,
            actual_world_origin_clearance_peak=float(np.nanmax(actual_clear[keep])) if loaded_baseline is not None else None,
            target_FK_tarsal_ground_penetration_max=float(target_penetration[keep,j].max()),
            actual_FK_tarsal_ground_penetration_max=float(actual_penetration[keep,j].max()),
            target_FK_ground_penetration_frame_fraction=float((target_penetration[keep,j]>1e-9).mean()),
            joint_signal_lag=signal_lag(bd['qpos'][keep][:,qa[ii]],target[keep][:,ii]))
        if mask.any():
            leg_report['during_requested_lift']=dict(frames=int(mask.sum()),
                actual_still_loaded_fraction=float(loaded[mask,j].mean()),
                actual_load_mean_weight=float(force[mask,j].mean()/dsm.WEIGHT),
                joint_error_rad=summary(err[mask]),
                foot_position_error_3d_rms=float(np.sqrt(np.mean(gap[mask]**2))),
                foot_vertical_error_mean=float(foot_error[mask,j,2].mean()),
                sampled_saturation_any_joint_fraction=float(saturation_any[mask].mean()),
                target_FK_tarsal_ground_penetration_max=float(target_penetration[mask,j].max()),
                target_FK_ground_penetration_frame_fraction=float((target_penetration[mask,j]>1e-9).mean()),
                actual_FK_tarsal_ground_penetration_max=float(actual_penetration[mask,j].max()),
                actual_origin_clearance_peak=float(np.nanmax(actual_clear[mask])) if loaded_baseline is not None else None)
        else:
            leg_report['during_requested_lift']=None
        if leg_report['requested_complete_lifts']==0:
            leg_report['interpretation']='Integrated targets do not supply a complete >=0.05native origin-lift interval in release; inspect targetgeometry/cadence first.'
        elif leg_report['during_requested_lift'] and leg_report['during_requested_lift']['actual_still_loaded_fraction']>.5:
            leg_report['interpretation']='Integrated target FK asks for lift, while actual foot remains loaded during most requested-lift frames. Mapped joint/foot errors demonstrate a tracking discrepancy; force/collision/load causation is not established by this observation alone.'
        else:
            leg_report['interpretation']='Actual sustained unloading occurs during target lift; compare mapped residuals and torque samples quantitatively.'
        if leg_report['during_requested_lift'] and leg_report['during_requested_lift']['target_FK_tarsal_ground_penetration_max']>.01:
            leg_report['interpretation']+=' At the same recorded root pose, integrated target tarsal collision geometry penetrates the ground during requested lift; intended body-relative lift does not ensure world-ground clearance. This is geometric counterfactual evidence, not a force or causality measurement.'
        per_leg.append(leg_report)
    files=[folder/name for name in ('BODY.npz','COMMANDS.npz','NATIVE.npz','RESULT.json')]
    report=dict(folder=str(folder),controller=result['controller'],seconds=result['seconds'],
        score_window_s=[handover,float(t[-1])],native_support=gates,per_leg=per_leg,per_joint=per_joint,
        checks=dict(target_integration_max_error=consistency,actual_FK_saved_feet_max_error=saved_feet_error,
                    actual_root_preserved_for_target_FK=True,physics_steps=0),
        counts=dict(FK_forward=2*len(t)+1,physics_step=0,body_frames=len(t),release_body_frames=int(keep.sum()),native_samples=len(native)),
        model=dict(nq=m.nq,nv=m.nv,nu=m.nu,nmesh=m.nmesh,neq=m.neq,gravity=m.opt.gravity.tolist()),
        source_hashes={str(p):sha(p) for p in files})
    series=dict(time=t,joint_error=joint_error,foot_error=foot_error,
        actual_z=actual_world[:,:,2],target_z=target_world[:,:,2],target_height=target_height,
        force=force,loaded=loaded,saturated=saturated,leg_indices=leg_indices)
    return report,series


def plot(reports,series,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(6,3,figsize=(17,18),sharex=True,constrained_layout=True)
    for r,(rep,ss) in enumerate(zip(reports,series)):
        color=plt.cm.tab10(r)
        label=Path(rep['folder']).name
        for j,leg in enumerate(LEGS):
            ii=ss['leg_indices'][j]
            error=np.sqrt(np.mean(ss['joint_error'][:,ii]**2,axis=1))
            axes[j,0].plot(ss['time'],error,color=color,label=label,linewidth=.8)
            axes[j,1].plot(ss['time'],ss['actual_z'][:,j],color=color,linewidth=.9,label=label+' actual')
            axes[j,1].plot(ss['time'],ss['target_z'][:,j],color=color,ls='--',linewidth=.7,label=label+' target FK')
            axes[j,2].plot(ss['time'],ss['force'][:,j]/dsm.WEIGHT,color=color,linewidth=.8,label=label+' load/W')
            axes[j,2].plot(ss['time'],ss['saturated'][:,ii].any(axis=1)*.45,color=color,ls=':',alpha=.5,linewidth=.7,label=label+' any joint at cap x0.45')
    for j,leg in enumerate(LEGS):
        axes[j,0].set(ylabel=leg+' joint RMS error (rad)')
        axes[j,1].set(ylabel=leg+' Tarsus5 origin world z')
        axes[j,2].set(ylabel=leg+' force / weight')
        for ax in axes[j]:ax.axvline(1.,color='k',ls='--',linewidth=.6)
    axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=7);axes[0,2].legend(fontsize=7)
    for ax in axes[-1]:ax.set_xlabel('Simulation seconds')
    fig.suptitle('Teacher tracking diagnostic: post-tick integrated targets, same-root counterfactual FK; no new physics\nFoot origin heights are proxies, not material-contact clearance/slip. Torque caps sampled every5ms.')
    fig.savefig(path,dpi=150);plt.close(fig)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--folder',type=Path,action='append')
    ap.add_argument('--no-write',action='store_true')
    ap.add_argument('--handover',type=float,default=1.)
    ap.add_argument('--rate-cap',type=float,default=30.)
    args=ap.parse_args(argv)
    folders=args.folder or [OUT/'teacher_gain2_4s',OUT/'teacher_gain8_4s']
    if args.folder and not args.no_write:
        raise SystemExit('Explicit --folder analysis requires --no-write; default matched comparison writes only080652.')
    required=('BODY.npz','COMMANDS.npz','NATIVE.npz','RESULT.json')
    missing=[str(f/name) for f in folders for name in required if not (f/name).exists()]
    if missing:
        print(json.dumps(dict(status='INPUTS_NOT_READY',missing=missing)),flush=True)
        return 3
    import mujoco as mj
    from incremental_actuator import load_full_model,DEFAULT_MODEL_PATH
    m=load_full_model()
    start=time.monotonic();reports=[];series=[]
    for folder in folders:
        report,ss=analyze(folder,m,mj,args.handover,args.rate_cap)
        reports.append(report);series.append(ss)
        print(json.dumps(dict(analyzed=str(folder),D_pass=report['native_support']['passed'],
            legs=[dict(leg=x['leg'],joint_rms=x['joint_error_rad']['rms'],foot_rms=x['foot_position_error_3d_rms'],
                unloads=x['actual_complete_unloads'],requested=x['requested_complete_lifts'],
                requested_but_loaded=x['during_requested_lift']['actual_still_loaded_fraction'] if x['during_requested_lift'] else None,
                saturation_any=x['sampled_saturation_any_joint_fraction']) for x in report['per_leg']])),flush=True)
    comparison=[]
    if len(reports)==2:
        for a,b in zip(reports[0]['per_leg'],reports[1]['per_leg']):
            comparison.append(dict(leg=a['leg'],joint_rms_first=a['joint_error_rad']['rms'],
                joint_rms_second=b['joint_error_rad']['rms'],foot_rms_first=a['foot_position_error_3d_rms'],
                foot_rms_second=b['foot_position_error_3d_rms'],unloads_first=a['actual_complete_unloads'],
                unloads_second=b['actual_complete_unloads'],saturation_first=a['sampled_saturation_any_joint_fraction'],
                saturation_second=b['sampled_saturation_any_joint_fraction']))
    outputs=[OUT/'TRACKING_DIAGNOSIS.json',OUT/'TRACKING.png']
    previous={str(p):sha(p) for p in outputs if p.exists()}  # Read before any overwrite.
    doc=dict(stage='S5 finite-actuator tracking diagnostic, algorithm teachers only',runs=reports,comparison=comparison,
        alignment='BODY i at tick k=5*i+4 after80substeps; integrated_target=clip(cmd[k,target]+.001*clip(rate+perturb,+/-cap),joint_bounds). Actual and target FK share exact recorded free-root pose.',
        thresholds=dict(load_on_weight_fraction=.01,load_off_weight_fraction=.002,confirmation_ms=10,
            target_origin_lift_native=.05,saturation_fraction_of_limit=.999999),
        counts=dict(physics_step=0,FK_forward=sum(r['counts']['FK_forward'] for r in reports)),
        method_hashes={str(Path(__file__)):sha(__file__),str(DEFAULT_MODEL_PATH):sha(DEFAULT_MODEL_PATH),
            str(ROOT/'tools/dynamic_support_metrics.py'):sha(ROOT/'tools/dynamic_support_metrics.py')},
        previous_outputs=previous,wall_seconds=time.monotonic()-start,
        limitations=['All runs analyzed here are declared algorithm teachers, not ANN locomotion.',
            'Sampled torque saturation is evaluated at5ms BODY frames; it is not a native-resolution fraction.',
            'Tarsus5 body-origin z, excursion and FK error are proxies, not distal sole clearance or material-point slip.',
            'Target FK tarsal-plane penetration uses collision contact.dist at the same actual root pose, without integration; it measures geometric overlap, not actual force or material-point slip.',
            'Actual qpos is assigned only to scratch FK data, and jointtargets are overwritten only in a second scratch FK data object. No mj_step is called.',
            'Changing gains can change body/contact trajectories; mapped errors and torque associations alone do not isolate a unique cause.',
            'Unchanged D gates are recomputed from every native release sample with the strict positive-nonfoot override.'])
    if not args.no_write:
        OUT.mkdir(parents=True,exist_ok=True)
        outputs[0].write_text(json.dumps(doc,indent=2,allow_nan=False)+'\n')
        plot(reports,series,outputs[1])
    print(json.dumps(dict(status='ANALYZED',written=not args.no_write,FK_forward=doc['counts']['FK_forward'],physics_steps=0,wall_s=doc['wall_seconds'])),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
