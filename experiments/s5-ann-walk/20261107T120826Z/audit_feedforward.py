"""Independent saved static-support feedforward audit: scratch FK only, no mj_step.
No controller/planner imports. Reconstruct direct source PCHIP, contact-J weighting,
normalized equilibrium, preload mapping and actual integrated endpoints separately.
"""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
from scipy.interpolate import PchipInterpolator
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE=ROOT/'experiments/s5-ann-incremental/20261107T010000Z'
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T120826Z')
GAIT=OUT.parent/'20261107T101114Z/GAIT.npz'
W=24.446354755878005

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def error(a,b):return float(abs(np.asarray(a)-np.asarray(b)).max())
def stats(x):
    x=np.asarray(x,float);return dict(max_abs=float(abs(x).max()),RMS=float(np.sqrt(np.mean(x*x))))
def encoder(v):
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,np.ndarray):return v.tolist()
    raise TypeError(type(v).__name__)
def s5(v):
    x=np.clip(v,0,1);return x**3*(10-15*x+6*x*x)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('folder',type=Path,nargs='?',default=OUT/'support_feedforward_screen');ap.add_argument('--no-write',action='store_true');args=ap.parse_args(argv)
    folder=args.folder;paths=[folder/name for name in ('FEEDFORWARD.npz','SENSORY.npz','POLICY.npz','COMMANDS.npz','BODY.npz','NATIVE.npz','PREREGISTRATION.json','RESULT.json')]+[GAIT,Path(__file__)]
    missing=[str(p) for p in paths if not p.exists()]
    if missing:print(json.dumps(dict(status='INPUTS_NOT_READY',missing=missing,physics_steps=0)));return 3
    before={str(p):sha(p) for p in paths};ff,sens,policy,command,body,native=[load(p) for p in paths[:6]];protocol=json.loads(paths[6].read_text());result=json.loads(paths[7].read_text());g=load(GAIT)
    params=protocol['parameters'];time=ff['time'];n=len(time);assert n>0
    q=g['qpos_hinges'];phase=g['phase'];f=PchipInterpolator(np.r_[phase[-3:-1]-2*np.pi,phase,phase[1:3]+2*np.pi],np.concatenate((q[-3:-1],q,q[1:3]),axis=0),axis=0,extrapolate=False)
    metadata=json.loads(str(g['metadata_json'].item()));offset=np.asarray(protocol['wave_metadata']['phase_offsets'])
    sys.path.insert(0,str(BASE));import mujoco as mj
    from plant_probe import Plant
    from incremental_actuator import DEFAULT_MODEL_PATH
    p=Plant();m=p.m;data=mj.MjData(m);assert (m.nq,m.nv,m.nu,m.nmesh,m.neq)==(49,48,42,71,0)
    qa,va=p.qa,p.va;lo=m.jnt_range[p.jids,0];hi=m.jnt_range[p.jids,1];limits=np.minimum(-m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);fl=m.actuator_forcelimited.astype(bool)
    limits[fl]=np.minimum(limits[fl],np.minimum(-m.actuator_forcerange[fl,0],m.actuator_forcerange[fl,1]))
    assert np.array_equal(command['limits'],limits)
    kp,kd=command['kp'],command['kd'];assert np.allclose(kp,limits/.2*128,rtol=1e-12,atol=1e-12) and np.allclose(kd,limits*.005*np.sqrt(128),rtol=1e-12,atol=1e-12)
    assert np.array_equal(g['qpos_addresses'],qa)
    names=[m.joint(int(j)).name for j in p.jids];rh=names.index('joint_RHTrochanter_yaw');records=[];fallback=np.asarray(ff['fallback'],bool)
    maximum={key:0. for key in ('normal_load','contact_point','Jz','COM','active','equilibrium_log','raw_FF','clipped_FF','nominal','correction_bound','requested_correction','target_correction','commanded_target','alpha','clip_mask','before_target','actual_q','actual_v','actual_root','actual_ctrl','policy_rate','after_target','native_force')}
    eq=[];forcegap=[];corrections=[];actualtau=[];planned=[];actual=[]
    for k,t in enumerate(time):
        si=int(abs(sens['time']-t).argmin());bi=int(abs(body['time']-t).argmin());end=int(abs(body['time']-(t+.005)).argmin());tick=int(round(t/.001))
        assert abs(sens['time'][si]-t)<1e-9 and abs(body['time'][bi]-t)<1e-8 and abs(body['time'][end]-t-.005)<1e-8
        data.qpos[:]=sens['qpos'][si];data.qvel[:]=sens['qvel'][si];data.ctrl[:]=body['ctrl'][bi];data.time=float(t);data.qacc_warmstart[:]=0
        mj.mj_forward(m,data)
        loads=np.zeros(6);points=np.zeros((6,3));Jz=np.zeros((6,48))
        for ci in range(data.ncon):
            co=data.contact[ci];ga,gb=int(co.geom1),int(co.geom2)
            if not(p.planes[ga] or p.planes[gb]):continue
            geom=gb if p.planes[ga] else ga;leg=int(p.feet[geom])
            if leg<0:continue
            wrench=np.zeros(6);mj.mj_contactForce(m,data,ci,wrench)
            normal=float((co.frame.reshape(3,3).T@wrench[:3])[2])*(1 if p.planes[ga] else -1)
            if normal<=0:continue
            jac=np.zeros((3,48));mj.mj_jac(m,data,jac,None,co.pos,int(m.geom_bodyid[geom]))
            loads[leg]+=normal;points[leg]+=normal*co.pos;Jz[leg]+=normal*jac[2]
        positive=loads>0;points[positive]/=loads[positive,None];Jz[positive]/=loads[positive,None]
        stance=((t-.25)+offset)%1<5/6;active=stance&(loads>params['active_normal_force_threshold']);Jz*=active[:,None]
        force=ff['fz'][k];com=data.subtree_com[p.thorax];A=np.vstack((np.ones(6),points[:,0]-com[0],points[:,1]-com[1]))
        residual=A@(force/W)-np.array([1.,0,0]);eq.append(residual)
        raw=(data.qfrc_bias-data.qfrc_passive-Jz.T@force)[va];clipped=np.clip(raw,-limits,limits)
        nominal=f((2*np.pi*(t-.25))%(2*np.pi));nominal=nominal.copy();nominal[rh]+=.02*s5(t/.5);nominal=np.clip(nominal,lo,hi)
        cap=np.minimum(limits/kp,.003);alpha=float(np.clip((t-1)/.1,0,1));correction=np.zeros(42) if fallback[k] else alpha*np.clip(clipped/kp,-cap,cap)
        unbounded=nominal+correction;desired=np.clip(unbounded,lo,hi)
        target=policy['targets'][si];rate=np.clip((desired-target)/.005,-30,30);after=target.copy()
        for j in range(5):
            row=command['rows'][tick+j];assert error(row[1:43],after)<1e-10
            after=np.clip(after+.001*np.clip(row[43:85]+row[85:127],-30,30),lo,hi)
        comparisons=dict(normal_load=error(loads,ff['actual_normal_load'][k]),contact_point=error(points,ff['contact_point'][k]),Jz=error(Jz,ff['Jz'][k]),COM=error(com,ff['actual_com'][k]),
            active=error(active.astype(int),ff['active'][k].astype(int)),equilibrium_log=error(residual*W,ff['equilibrium_residual'][k]),raw_FF=error(raw,ff['tau_feedforward_raw'][k]),
            clipped_FF=error(clipped,ff['tau_feedforward_clipped'][k]),nominal=error(nominal,ff['nominal_qref'][k]),correction_bound=error(cap,ff['correction_bound'][k]),
            requested_correction=error(correction,ff['requested_correction'][k]),target_correction=error(desired-nominal,ff['target_correction'][k]),commanded_target=error(desired,ff['commanded_target'][k]),
            alpha=abs(alpha-ff['alpha'][k]),clip_mask=error((desired!=unbounded).astype(int),ff['target_native_clip_mask'][k].astype(int)),
            before_target=max(error(target,ff['PD_target_before'][k]),error(target,command['rows'][tick,1:43])),
            actual_q=error(data.qpos[qa],ff['actual_q'][k]),actual_v=error(data.qvel[va],ff['actual_qvel'][k]),actual_root=error(data.qpos[:7],ff['actual_root'][k]),
            actual_ctrl=error(data.ctrl,ff['actual_ctrl'][k]),policy_rate=error(rate,policy['actions'][si]),after_target=error(after,ff['PD_target_after_5ms'][k]),
            native_force=error(loads,native['rows'][int(round(t/.0000125))-1,1:7]))
        for key,value in comparisons.items():maximum[key]=max(maximum[key],value)
        forcegap.append(force-loads);corrections.append(correction);actualtau.append(body['ctrl'][end]);planned.append(force);actual.append(loads)
        records.append(dict(time=float(t),fallback=bool(fallback[k]),status=str(ff['status'][k]),equilibrium_max_normalized=float(abs(residual).max()),
            correction_max_rad=float(abs(correction).max()),planned_Fz_W=float(force.sum()/W),actual_Fz_W=float(loads.sum()/W),
            planned_vs_actual_Fz_rms=float(np.sqrt(np.mean((force-loads)**2))),sampled_endpoint_torque_cap_fraction=float(np.mean(abs(body['ctrl'][end])>=limits*(1-1e-6)))))
    eq=np.asarray(eq);corrections=np.asarray(corrections);fz=ff['fz'];active=ff['active'].astype(bool);valid=~fallback
    bounds=max(float(-fz.min()),float((fz-2*W).max()),float(abs(fz[~active]).max()) if (~active).any() else 0.,0.)
    checks=dict(full_model=True,original_native_caps_exact=True,gain128=True,mapping=True,sidechannel_reconstruction=max(maximum.values())<=1e-6,
        planned_force_bounds=bounds<=1e-9,nonfallback_equilibrium=bool(not valid.any() or abs(eq[valid]).max()<=1e-6),
        fallback_zero_correction=bool(np.all(corrections[fallback]==0)),tiny_correction_bound=bool(abs(corrections).max()<=.0015625+1e-12),
        actual_saved_torque_caps=bool(np.all(abs(body['ctrl'])<=limits+1e-9)),trace_hash=sha(folder/'FEEDFORWARD.npz')==result['feedforward_trace_sha256'])
    frozen={}
    for key in ('source_hashes','additional_sources'):
        for path,digest in protocol.get(key,{}).items():frozen[path]=Path(path).exists() and sha(path)==digest
    checks['all_registered_sources_unchanged']=all(frozen.values());assert all(sha(path)==digest for path,digest in before.items())
    report=dict(status='FEEDFORWARD_EQUATIONS_VERIFIED_NOT_PHYSICAL_ACCEPTANCE' if all(checks.values()) else 'FEEDFORWARD_AUDIT_FAILED',checks=checks,
        solver=dict(ticks=n,fallback_count=int(fallback.sum()),fallback_fraction=float(fallback.mean()),statuses={str(s):int(np.sum(ff['status']==s)) for s in np.unique(ff['status'])}),
        maximum_reconstruction_errors=maximum,nonfallback_equilibrium_max_normalized=float(abs(eq[valid]).max()) if valid.any() else None,
        equilibrium_tolerance=dict(normalized_by_W=1e-6,native=1e-6*W,note='The tolerance is1e-6*W for native residual, NOT1e-6absoluteforce; momentrows retain nativeleverarmunits.'),
        correction_max_rad=float(abs(corrections).max()),force_bound_violation=bounds,planned_minus_actual_normal_force=stats(forcegap),
        per_leg_planned_minus_actual_normal_force_RMS=np.sqrt(np.mean(np.asarray(forcegap)**2,axis=0)),
        sampled_actual_torque_cap_fraction=float(np.mean(abs(body['ctrl'])>=limits*(1-1e-6))),policy_records=records,
        sources_unchanged=frozen,input_hashes={**before,str(DEFAULT_MODEL_PATH):sha(DEFAULT_MODEL_PATH)},counts=dict(FK_forward=n,physics_steps=0,neural_steps=0),
        limitations=['No allocation optimizer or controller was imported or rerun. Loggedplannedforces are checked for feasibility, not unique/minimumobjective optimality.',
            'Staticfz planning ignoresacceleration,tangentialforces,contactmoments andcontactcompliance. Planned fz versus actual Fn RMS is reported separately; they are not force assignments.',
            'Root,qpos/qvel/ctrl are assigned only to scratchdata for mj_forward; zero newphysicalintegration. Actualnoauxforce/rootwritecontract is reviewedsourceevidence; strictparentreplay isseparate.',
            'Nominalreference remains fullshape plusmicropreloads; no neural or physical walking acceptance follows from these equations.'])
    if not args.no_write:
        output=OUT/'AUDIT_FEEDFORWARD.json'
        if output.exists():output.read_bytes()
        output.write_text(json.dumps(report,indent=2,allow_nan=False,default=encoder)+'\n')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True,constrained_layout=True)
        axes[0].semilogy(time,np.maximum(abs(eq).max(1),1e-16),label='allocation equilibrium residual /W');axes[0].axhline(1e-6,color='r',ls=':');axes[0].legend()
        axes[1].plot(time,np.asarray(planned).sum(1)/W,label='planned normalforce/W');axes[1].plot(time,np.asarray(actual).sum(1)/W,label='actualcontact Fn/W');axes[1].legend()
        axes[2].plot(time,abs(corrections).max(1),label='max requested target correction rad');axes[2].fill_between(time,0,.0015625,where=fallback,color='r',alpha=.2,label='fallback');axes[2].legend();axes[2].set_xlabel('Policy time seconds')
        fig.suptitle('Support feedforward equation audit; plannedforces not physicalassignments; no newphysics')
        fig.savefig(OUT/'AUDIT_FEEDFORWARD.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(status=report['status'],checks=checks,solver=report['solver'],max_errors=maximum,correctionmax=report['correction_max_rad'],planned_actual_force_RMS=report['planned_minus_actual_normal_force'],physics_steps=0),default=encoder),flush=True)
    return 0 if all(checks.values()) else 2
if __name__=='__main__':raise SystemExit(main())
