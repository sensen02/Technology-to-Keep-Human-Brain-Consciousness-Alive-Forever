"""Independent saved whole-body-QP audit. Scratch mj_forward only; no mj_step.

Hard model consistency, actual force/model mismatch and finite-actuator realization
are separate results. This never establishes the neural locomotion goal.
"""
import argparse,json,sys,hashlib,time
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE=ROOT/'experiments/s5-ann-incremental/20261107T010000Z'
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T092817Z')
CDT,PDT,W=.001,.005,24.446354755878005
LEGS=('LF','LM','LH','RF','RM','RH')


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def err(a,b):return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
def stats(a):
    a=np.asarray(a,float)
    return dict(max_abs=float(abs(a).max()),rms=float(np.sqrt(np.mean(a*a))),p95_abs=float(np.percentile(abs(a),95)))
def field(data,*names):
    for name in names:
        if name in data:return data[name]
    raise KeyError('Missing any of '+str(names))


def contacts(mj,p,d):
    J=np.zeros((6,3,48));jd=np.zeros((6,3));load=np.zeros(6);mu=np.full(6,np.inf)
    forces=np.zeros((6,3));generalized=np.zeros(48);all_ground=np.zeros(48)
    for i in range(d.ncon):
        c=d.contact[i];ga,gb=int(c.geom1),int(c.geom2)
        if not(p.planes[ga] or p.planes[gb]):continue
        geom=gb if p.planes[ga] else ga;leg=int(p.feet[geom]);bid=int(p.m.geom_bodyid[geom])
        wrench=np.zeros(6);mj.mj_contactForce(mj_model:=p.m,d,i,wrench)
        sign=1 if p.planes[ga] else -1
        F=sign*(c.frame.reshape(3,3).T@wrench[:3]);T=sign*(c.frame.reshape(3,3).T@wrench[3:])
        jp=np.zeros((3,48));jr=np.zeros_like(jp);dp=np.zeros_like(jp);dr=np.zeros_like(jp)
        mj.mj_jac(mj_model,d,jp,jr,c.pos,bid);mj.mj_jacDot(mj_model,d,dp,dr,c.pos,bid)
        all_ground+=jp.T@F+jr.T@T
        if leg<0:continue
        generalized+=jp.T@F+jr.T@T;forces[leg]+=F
        if F[2]<=0:continue
        J[leg]+=F[2]*jp;jd[leg]+=F[2]*(dp@d.qvel);load[leg]+=F[2]
        mu[leg]=min(mu[leg],float(c.friction[0]),float(c.friction[1]))
    good=load>0;J[good]/=load[good,None,None];jd[good]/=load[good,None]
    mu[~np.isfinite(mu)]=0
    return J,jd,load,mu,forces,generalized,all_ground


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('folder',type=Path,nargs='?',default=OUT/'wbc_screen')
    ap.add_argument('--no-write',action='store_true');args=ap.parse_args(argv);folder=args.folder
    paths=[folder/n for n in ('QP.npz','SENSORY.npz','POLICY.npz','COMMANDS.npz','BODY.npz','NATIVE.npz','PREREGISTRATION.json')]
    missing=[str(p) for p in paths if not p.exists()]
    if missing:print(json.dumps(dict(status='INPUTS_NOT_READY',missing=missing)));return 3
    before={str(p):sha(p) for p in paths};qp,sens,policy,cmd,body,native=[load(p) for p in paths[:6]]
    prereg=json.loads(paths[6].read_text());param=prereg['parameters'];dyn_tol=float(param['dynamics_tolerance_W']);con_tol=float(param['constraint_tolerance'])
    qtime=field(qp,'time','policy_times');stime=sens['time'];indices=np.array([int(np.argmin(abs(stime-t))) for t in qtime])
    assert err(stime[indices],qtime)<1e-9
    sys.path[:0]=[str(BASE),str(ROOT/'tools')]
    import mujoco as mj
    from plant_probe import Plant
    from incremental_actuator import DEFAULT_MODEL_PATH
    p=Plant();m=p.m;d=mj.MjData(m);assert (m.nq,m.nv,m.nu,m.nmesh,m.neq)==(49,48,42,71,0)
    native_limit=np.minimum(-m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);limited=m.actuator_forcelimited.astype(bool)
    native_limit[limited]=np.minimum(native_limit[limited],np.minimum(-m.actuator_forcerange[limited,0],m.actuator_forcerange[limited,1]))
    assert np.array_equal(cmd['limits'],native_limit),'Recorded torque caps differ from original native limits'
    kp,kd=cmd['kp'],cmd['kd'];assert np.allclose(kp,native_limit/.2*32.,rtol=1e-12,atol=1e-12)
    assert np.allclose(kd,native_limit*.005*np.sqrt(32.),rtol=1e-12,atol=1e-12)
    S=np.zeros((48,42));S[p.va,np.arange(42)]=1.;low=m.jnt_range[p.jids,0];high=m.jnt_range[p.jids,1]
    times=body['time'];rows=cmd['rows'];M=np.empty((48,48));offset=np.asarray(prereg['wave_metadata']['phase_offsets'])
    frequency=float(prereg['knobs']['frequency_hz']);active_log=field(qp,'activelegs','active_contact')
    Js=field(qp,'averaged_contact_J','J');jds=field(qp,'averaged_Jdotv','Jdotv');taus=field(qp,'tau_optimal','tau')
    fs=field(qp,'contact_force','forces');rhs=qp['dynamics_rhs'];predlog=field(qp,'tau_predicted_reachable','reachabletau')
    n=len(qtime);assert all(len(x)==n for x in (active_log,Js,jds,taus,fs,rhs,predlog))
    fallback=np.asarray(qp['fallback'],bool);solved=~fallback;metrics={k:[] for k in ('J','Jdotv','rhs','dynamics_W','logged_dynamics','constraint','active','mu','load','reachable_log','endpoint_target','actual_endpoint_log','unmodeled_force_W','ground_reaction_mismatch_W','tau_pred_actual','tau_opt_actual','real_dynamics_W','force_reconstruction')}
    all_dyn=[];all_pred=[];all_actual=[];active_count=[];diagnostic_rows=[];begin=time.monotonic()
    for k,(t,i) in enumerate(zip(qtime,indices)):
        tick=int(round(t/CDT));assert abs(tick*CDT-t)<1e-8
        assert tick+4<len(rows) and err(rows[tick,1:43],policy['targets'][i])<1e-10
        prev=int(np.argmin(abs(times-t)))
        assert abs(times[prev]-t)<1e-8,'SENSORY must align with previous BODY endpoint'
        d.qpos[:]=sens['qpos'][i];d.qvel[:]=sens['qvel'][i];d.ctrl[:]=body['ctrl'][prev];d.time=float(t);d.qacc_warmstart[:]=0
        mj.mj_forward(m,d);mj.mj_fullM(m,M,d.qM);bias=d.qfrc_bias-d.qfrc_passive
        J,jdv,loads,mu,actual_forces,actual_tarsal,actual_ground=contacts(mj,p,d)
        nominal=((t-.25)*frequency+offset)%1<5/6;active=nominal&(loads>float(param['contact_threshold_W'])*W)
        masked=J*active[:,None,None];masked_jd=jdv*active[:,None]
        tau=taus[k];force=fs[k];acc=qp['qacc'][k];modeled=np.einsum('lij,li->j',masked,force)
        residual=M@acc+bias-S@tau-modeled
        square=np.max(abs(force[:,:2])-mu[:,None]/np.sqrt(2)*force[:,2,None])
        circle=np.max(np.linalg.norm(force[:,:2],axis=1)-mu*force[:,2])
        normal=max(float(-force[:,2].min()),float((force[:,2]-float(param['max_normal_force_W'])*W*active).max()))
        violation=max(float((abs(tau)-native_limit).max()),float(square),float(circle),normal,float(abs(force[~active]).max()) if (~active).any() else 0.,0.)
        start_target=policy['targets'][i];rate=policy['actions'][i]
        reachable=start_target.copy()
        for j in range(5):
            command=rows[tick+j];assert err(command[43:85],rate)<1e-10
            reachable=np.clip(reachable+CDT*np.clip(command[43:85]+command[85:127],-30,30),low,high)
        predicted=np.clip(kp*(reachable-d.qpos[p.qa])-kd*d.qvel[p.va],-native_limit,native_limit)
        endpoint=int(np.argmin(abs(times-(t+PDT))));assert abs(times[endpoint]-(t+PDT))<1e-8
        applied=body['ctrl'][endpoint]
        expected_before=native['rows'][int(round(t/.0000125))-1,1:7]
        constraint_eq=M@d.qacc+bias-S@d.ctrl-d.qfrc_constraint
        values=dict(J=err(masked,Js[k]),Jdotv=err(masked_jd,jds[k]),rhs=err(-bias,rhs[k]),dynamics_W=float(abs(residual).max()/W),
            logged_dynamics=err(residual,qp['dynamics_residual'][k]) if 'dynamics_residual' in qp else 0.,constraint=violation,
            active=float(np.any(active!=active_log[k])),mu=err(mu,qp['friction_mu'][k]),load=err(loads,qp['actual_contact_load'][k]),
            reachable_log=err(predicted,predlog[k]),endpoint_target=err(reachable,rows[tick+5,1:43]) if tick+5<len(rows) else 0.,
            actual_endpoint_log=err(applied,qp['tau_actual_after_policy'][k]) if 'tau_actual_after_policy' in qp else 0.,
            unmodeled_force_W=float(actual_forces[~active,2].sum()/W),ground_reaction_mismatch_W=float(abs(actual_ground-modeled).max()/W),
            tau_pred_actual=float(np.sqrt(np.mean((predicted-applied)**2))),tau_opt_actual=float(np.sqrt(np.mean((tau-applied)**2))),
            real_dynamics_W=float(abs(constraint_eq).max()/W),force_reconstruction=err(loads,expected_before))
        for key,value in values.items():metrics[key].append(value)
        endq=body['qpos'][endpoint,p.qa];endv=body['qvel'][endpoint,p.va]
        raw_frozen=kp*(reachable-d.qpos[p.qa])-kd*d.qvel[p.va]
        state_change=-kp*(endq-d.qpos[p.qa])-kd*(endv-d.qvel[p.va])
        end_formula=np.clip(raw_frozen+state_change,-native_limit,native_limit)
        diagnostic_rows.append(dict(time=float(t),fallback=bool(fallback[k]),active_legs=int(active.sum()),
            QP_root_z_acceleration=float(acc[2]),scratch_actual_root_z_acceleration=float(d.qacc[2]),
            realized_mean_root_z_acceleration=float((body['qvel'][endpoint,2]-d.qvel[2])/PDT),
            optimal_torque_rms=float(np.sqrt(np.mean(tau*tau))),frozen_predicted_vs_optimal_rms=float(np.sqrt(np.mean((predicted-tau)**2))),
            optimal_vs_actual_endpoint_rms=values['tau_opt_actual'],frozen_prediction_vs_actual_endpoint_rms=values['tau_pred_actual'],
            position_change_raw_torque_rms=float(np.sqrt(np.mean((kp*(endq-d.qpos[p.qa]))**2))),
            velocity_change_raw_torque_rms=float(np.sqrt(np.mean((kd*(endv-d.qvel[p.va]))**2))),
            endpoint_formula_vs_saved_ctrl_rms=float(np.sqrt(np.mean((end_formula-applied)**2))),
            actual_ground_Fz_W=float(actual_forces[:,2].sum()/W),QP_ground_Fz_W=float(force[:,2].sum()/W),
            excluded_leg_Fz_W=values['unmodeled_force_W'],ground_generalized_mismatch_W=values['ground_reaction_mismatch_W'],
            source_state_constant_assumption='prediction freezes q/qvel; state-change torque terms are diagnostic only, no endpoint compensation implemented'))
        all_dyn.append(residual);all_pred.append(predicted);all_actual.append(applied);active_count.append(int(active.sum()))
    a={k:np.asarray(v) for k,v in metrics.items()};valid=solved if solved.any() else np.ones(n,bool)
    reconstruction=bool(max(a['J'].max(),a['Jdotv'].max(),a['rhs'].max(),a['active'].max(),a['mu'].max(),a['load'].max(),a['force_reconstruction'].max())<=1e-6)
    hard=bool(solved.any() and np.max(a['dynamics_W'][solved])<=dyn_tol and np.max(a['constraint'][solved])<=con_tol)
    checks=dict(native_caps_exact=True,gains_declared=True,QP_logged_model_reconstructed=reconstruction,
        logged_dynamics_residual_matches=bool(a['logged_dynamics'].max()<=1e-6),solved_hard_constraints_consistent=hard,
        predicted_endpoint_torque_reconstructed=bool(a['reachable_log'].max()<=1e-6),endpoint_targets_consistent=bool(a['endpoint_target'].max()<=1e-10),
        logged_actual_endpoint_matches_BODY=bool(a['actual_endpoint_log'].max()<=1e-10))
    unmodeled=bool(a['unmodeled_force_W'].max()>.01 or a['ground_reaction_mismatch_W'].max()>.1)
    nr=native['rows'];release=nr[:,0]>=1.;rr=nr[release]
    def first_native(mask):
        ii=np.flatnonzero(mask)
        return float(rr[ii[0],0]) if len(ii) else None
    def first_policy(mask):
        ii=np.flatnonzero(mask)
        return float(qtime[ii[0]]) if len(ii) else None
    events=dict(first_positive_nonfoot_force=first_native(rr[:,7]>1e-9),
        first_native_foot_load_below_08W=first_native(rr[:,1:7].sum(1)<.8*W),
        first_com_below_08h0=first_native(rr[:,10]<.8*1.2072528224791905),
        first_excluded_contact_load_above_001W=first_policy(a['unmodeled_force_W']>.01),
        first_QP_actual_generalized_reaction_gap_above_01W=first_policy(a['ground_reaction_mismatch_W']>.1),
        first_frozen_endpoint_torque_prediction_rms_gap_above1=first_policy(a['tau_pred_actual']>1.),
        first_fallback=first_policy(fallback),note='Native load thresholds here locate events; they do not replace unchanged physical D gates.')
    selected=sorted(set([int(np.argmin(abs(qtime-t))) for t in [1.,1.025,1.05,1.075,1.1,1.125,1.15,1.2,1.3,1.5,2.,3.]]))
    paths_extra=[Path(__file__),DEFAULT_MODEL_PATH];source_valid={}
    for mapping in ('source_hashes','additional_sources'):
        for path,digest in prereg.get(mapping,{}).items():source_valid[path]=Path(path).exists() and sha(path)==digest
    checks['registered_sources_unchanged']=all(source_valid.values())
    checks['saved_BODY_torque_within_native_caps']=bool(np.all(abs(body['ctrl'])<=native_limit+1e-9))
    checks['QP_saved_root_matches_SENSORY']=bool(err(qp['actual_root'],sens['qpos'][indices,:7])<=1e-12) if 'actual_root' in qp else False
    checks['QP_saved_velocity_matches_SENSORY']=bool(err(qp['actual_qvel'],sens['qvel'][indices])<=1e-12) if 'actual_qvel' in qp else False
    report=dict(stage='S5 independent hard-QP model and actuator realization audit',folder=str(folder),
        status='MODEL_CONSISTENCY_ONLY' if all(checks.values()) else 'MODEL_AUDIT_FAILED',checks=checks,
        actual_QP_goal='NOT_ESTABLISHED',goal_not_established_reasons=[reason for reason,yes in (
            ('fallback occurred',bool(fallback.any())),('unmodeled or mismatched actual ground reaction',unmodeled),
            ('hard-model or reconstruction check failed',not all(checks.values())),
            ('finite-PD endpoint torque differs from frozen-state reachable prediction',bool(a['tau_pred_actual'].max()>1e-3))) if yes],
        solver=dict(ticks=n,solved_count=int(solved.sum()),fallback_count=int(fallback.sum()),fallback_fraction=float(fallback.mean()),
            statuses={str(s):int(np.sum(qp['status']==s)) for s in np.unique(qp['status'])}),
        tolerances=dict(dynamics_W=dyn_tol,constraint=con_tol,contact_reconstruction=1e-6,
            unmodeled_normal_force_W=.01,actual_ground_generalized_mismatch_W=.1,
            note='Last two are disclosed diagnostic flags, not locomotion acceptance thresholds.'),
        metrics={k:stats(v) for k,v in a.items()},solved_dynamics_W=stats(a['dynamics_W'][valid]),solved_constraint=stats(a['constraint'][valid]),
        active_legs_minmax=[min(active_count),max(active_count)],early_failure_timing=events,
        selected_policy_diagnostics=[diagnostic_rows[i] for i in selected],
        policy_diagnostics=diagnostic_rows,
        model=dict(nq=m.nq,nv=m.nv,nu=m.nu,nmesh=m.nmesh,neq=m.neq,gravity=m.opt.gravity.tolist(),dt=m.opt.timestep),
        counts=dict(FK_forward=n,physics_step=0,neural_step=0),
        input_hashes={**before,**{str(p):sha(p) for p in paths_extra}},registered_sources_unchanged=source_valid,
        limitations=['No QP solver/runtime was imported or rerun. Saved optimal variables are independently checked at saved policy states.',
            'Scratch reconstruction lacks saved solver warmstart; contact forces/J weighting are verified against logs and native samples, not assumed.',
            'A force-weighted J represents the logged proportional contact-force distribution; real multiple-contact forces, swing reactions, contact torques and limits may differ.',
            'Generalized residual normalized by weight mixes root-force and rotational/joint-torque units; full per-coordinate residuals remain in QP logs and diagnostic metrics.',
            'Predicted endpoint torque freezes q/qvel for5ms. Actual BODY endpoint torque is from finite-PD execution on changing physics; neither proves physically optimal control.',
            'This audit cannot verify neural locomotion. All fallback windows and mismatch remain in the full sample; parent frozen locomotion assessment is separate.'])
    assert all(sha(path)==digest for path,digest in before.items())
    if not args.no_write:
        path=OUT/'AUDIT_QP.json'
        if path.exists():path.read_bytes()
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True,constrained_layout=True)
        axes[0].semilogy(qtime,np.maximum(a['dynamics_W'],1e-16),label='QP model dynamics residual /W');axes[0].axhline(dyn_tol,color='r',ls=':');axes[0].legend()
        axes[1].plot(qtime,a['unmodeled_force_W'],label='actual excluded-leg ground Fz /W');axes[1].plot(qtime,a['ground_reaction_mismatch_W'],label='actual ground minus modeled generalized reaction /W');axes[1].legend()
        axes[2].plot(qtime,a['tau_pred_actual'],label='reachable prediction vs actual endpoint RMS torque');axes[2].plot(qtime,a['tau_opt_actual'],label='QP optimal vs actual endpoint RMS torque');axes[2].fill_between(qtime,0,1,where=fallback,color='r',alpha=.15,label='fallback');axes[2].legend();axes[2].set_xlabel('Policy time seconds')
        fig.suptitle('Hard QP model consistency differs from actual contact/actuator realization; no physics run by auditor')
        fig.savefig(OUT/'AUDIT_QP.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(status=report['status'],checks=checks,solver=report['solver'],metrics=report['metrics'],physics_steps=0,written=not args.no_write)),flush=True)
    return 0 if all(checks.values()) and not fallback.any() else 2


if __name__=='__main__':raise SystemExit(main())
