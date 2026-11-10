"""Independent saved contact-damping audit; scratch FK only, no mj_step/solver.
Parent owns final execution. Optional --filter-tau .02 reconstructs causal state
from raw contact velocities and clock, not logged filter outputs. No passivity claim.
"""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
from scipy.interpolate import PchipInterpolator
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE=ROOT/'experiments/s5-ann-incremental/20261107T010000Z'
GAIT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T101114Z/GAIT.npz')

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def error(a,b):return float(abs(np.asarray(a,float)-np.asarray(b,float)).max())
def encode(v):
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,np.ndarray):return v.tolist()
    raise TypeError(type(v).__name__)

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('folder',type=Path);ap.add_argument('--filter-tau',type=float);ap.add_argument('--no-write',action='store_true');args=ap.parse_args(argv)
    if args.filter_tau is not None and args.filter_tau!=.02:raise ValueError('Only declared single-variable tau.02 is supported')
    folder=args.folder;tracefile=folder/('FILTERED_DAMPING.npz' if args.filter_tau is not None else 'CONTACT_DAMPING.npz')
    files=[tracefile]+[folder/(name+'.npz') for name in ('SENSORY','POLICY','COMMANDS','BODY','NATIVE')]+[folder/'PREREGISTRATION.json',folder/'RESULT.json',GAIT,Path(__file__)]
    missing=[str(p) for p in files if not p.exists()]
    if missing:print(json.dumps(dict(status='INPUTS_NOT_READY',missing=missing,physics_steps=0)));return 3
    before={str(p):sha(p) for p in files};trace,sens,policy,commands,body,native=[load(p) for p in files[:6]];protocol=json.loads(files[6].read_text());result=json.loads(files[7].read_text());g=load(GAIT)
    if args.filter_tau is None and 'filtered_contact_velocity' in trace:raise ValueError('Filtered trace requires explicit --filter-tau .02')
    if args.filter_tau is not None:assert protocol['velocity_filter']['time_constant_s']==.02
    sys.path.insert(0,str(BASE));import mujoco as mj
    from plant_probe import Plant
    from incremental_actuator import DEFAULT_MODEL_PATH
    p=Plant();m=p.m;d=mj.MjData(m);assert (m.nq,m.nv,m.nu,m.nmesh,m.neq)==(49,48,42,71,0)
    limits=np.minimum(-m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);forced=m.actuator_forcelimited.astype(bool)
    limits[forced]=np.minimum(limits[forced],np.minimum(-m.actuator_forcerange[forced,0],m.actuator_forcerange[forced,1]))
    kp,kd=commands['kp'],commands['kd'];assert np.array_equal(limits,commands['limits'])
    assert np.allclose(kp,limits/.2*128,rtol=1e-12,atol=1e-12) and np.allclose(kd,limits*.005*np.sqrt(128),rtol=1e-12,atol=1e-12)
    qa,va=p.qa,p.va;lo,hi=m.jnt_range[p.jids].T;assert np.array_equal(qa,g['qpos_addresses'])
    rh=[m.joint(int(j)).name for j in p.jids].index('joint_RHTrochanter_yaw');meta=json.loads(str(g['metadata_json'].item()));offset=np.asarray(protocol['wave_metadata']['phase_offsets'])
    phase,q=g['phase'],g['qpos_hinges'];f=PchipInterpolator(np.r_[phase[-3:-1]-2*np.pi,phase,phase[1:3]+2*np.pi],np.concatenate((q[-3:-1],q,q[1:3]),axis=0),axis=0,extrapolate=False)
    assert protocol['parameters']['damping_native_force_per_velocity']==5.
    maxima={};records=[];state=np.zeros((6,3));was_active=np.zeros(6,bool);last=None;rows=commands['rows'];time=trace['time'];n=len(time)
    def compare(name,a,b):maxima[name]=max(maxima.get(name,0.),error(a,b))
    for k,t in enumerate(time):
        i=int(abs(sens['time']-t).argmin());bi=int(abs(body['time']-t).argmin());end=int(abs(body['time']-(t+.005)).argmin());tick=int(round(t/.001))
        assert abs(sens['time'][i]-t)<1e-9 and abs(body['time'][bi]-t)<1e-8 and abs(body['time'][end]-t-.005)<1e-8
        d.qpos[:]=sens['qpos'][i];d.qvel[:]=sens['qvel'][i];d.ctrl[:]=body['ctrl'][bi];d.time=float(t);d.qacc_warmstart[:]=0;mj.mj_forward(m,d)
        stance=((t-.25)+offset)%1<5/6;loads=np.zeros(6);velocity=np.zeros((6,3));J=np.zeros((6,3,48));cross=0.
        for ci in range(d.ncon):
            co=d.contact[ci];ga,gb=int(co.geom1),int(co.geom2)
            if not(p.planes[ga] or p.planes[gb]):continue
            geom=gb if p.planes[ga] else ga;leg=int(p.feet[geom])
            if leg<0 or not stance[leg]:continue
            force=np.zeros(6);mj.mj_contactForce(m,d,ci,force)
            normal=float((co.frame.reshape(3,3).T@force[:3])[2])*(1 if p.planes[ga] else -1)
            if normal<=0:continue
            bid=int(m.geom_bodyid[geom]);jac=np.zeros((3,48));mj.mj_jac(m,d,jac,None,co.pos,bid);pointv=jac@d.qvel
            twist=np.zeros(6);mj.mj_objectVelocity(m,d,mj.mjtObj.mjOBJ_BODY,bid,twist,0)
            rigid=twist[3:]+np.cross(twist[:3],co.pos-d.xipos[bid]);cross=max(cross,error(pointv,rigid))
            loads[leg]+=normal;velocity[leg]+=normal*pointv;J[leg]+=normal*jac
        active=loads>0;velocity[active]/=loads[active,None];J[active]/=loads[active,None,None]
        used=velocity
        if args.filter_tau is not None:
            dt=.005 if last is None else float(t-last);assert dt>=0
            beta=float(-np.expm1(-dt/.02));new=active&~was_active;state[~active|new]=0.;state[active]+=beta*(velocity[active]-state[active]);last=float(t);was_active=active.copy();used=state.copy()
            for name,value in [('filtered_contact_velocity',used),('filter_state',used),('filter_dt',dt),('filter_beta',beta),('newly_active_contact',new),('filter_active',active)]:compare(name,value,trace[name][k])
        raw=-5*np.einsum('lkj,lk->j',J[:,:2,va],used[:,:2]);clipped=np.clip(raw,-limits,limits);alpha=float(np.clip((t-1)/.1,0,1));correction=alpha*clipped/kp
        nominal=f((2*np.pi*(t-.25))%(2*np.pi)).copy();u=np.clip(t/.5,0,1);nominal[rh]+=.02*u**3*(10-15*u+6*u*u);nominal=np.clip(nominal,lo,hi)
        unbounded=nominal+correction;desired=np.clip(unbounded,lo,hi);before_target=policy['targets'][i];rate=np.clip((desired-before_target)/.005,-30,30);after=before_target.copy()
        for j in range(5):
            command=rows[tick+j];compare('within_hold_target_before',after,command[1:43]);compare('held_requested_rate',rate,command[43:85])
            after=np.clip(after+.001*np.clip(command[43:85]+command[85:127],-30,30),lo,hi)
        computed=dict(stance=stance,active=active,actual_normal_load=loads,contact_velocity=velocity,contact_J=J,nominal_qref=nominal,
            tau_damping_raw=raw,tau_damping_clipped=clipped,alpha=alpha,requested_correction=correction,target_correction=desired-nominal,
            commanded_target=desired,target_native_clip_mask=desired!=unbounded,PD_target_before=before_target,
            actual_q=d.qpos[qa],actual_qvel=d.qvel[va],actual_root=d.qpos[:7],actual_ctrl=d.ctrl,PD_target_after_5ms=after)
        for name,value in computed.items():compare(name,value,trace[name][k])
        compare('policy_actions',rate,policy['actions'][i]);compare('velocity_crosscheck_logged',cross,trace['velocity_crosscheck'][k])
        compare('native_nominalstance_Fn',loads,np.where(stance,native['rows'][int(round(t/.0000125))-1,1:7],0.))
        if args.filter_tau is not None:
            compare('joint_damping_power_raw',raw@d.qvel[va],trace['joint_damping_power_raw'][k]);compare('full_virtual_damping_power',-5*np.sum(used[:,:2]*velocity[:,:2]),trace['full_virtual_damping_power'][k])
        records.append(dict(time=float(t),active_legs=int(active.sum()),measured_xy_speed_max=float(np.linalg.norm(velocity[:,:2],axis=1).max()),
            used_xy_speed_max=float(np.linalg.norm(used[:,:2],axis=1).max()),correction_max_rad=float(abs(correction).max()),
            raw_damping_torque_max=float(abs(raw).max()),joint_raw_power=float(raw@d.qvel[va]),full_virtual_power=float(-5*np.sum(used[:,:2]*velocity[:,:2])),
            raw_torque_clip_fraction=float(np.mean(abs(raw)>limits)),actual_endpoint_torque_cap_fraction=float(np.mean(abs(body['ctrl'][end])>=limits*(1-1e-6)))))
    frozen={}
    for key in ('source_hashes','additional_sources'):
        for path,h in protocol.get(key,{}).items():frozen[path]=Path(path).exists() and sha(path)==h
    for key,hkey in [('baseline','baseline_sha256'),('wrapper','wrapper_sha256'),('effective_source','effective_sha256')]:
        adaptation=protocol.get('adaptation',{})
        if key in adaptation:frozen[adaptation[key]]=sha(adaptation[key])==adaptation[hkey]
    checks=dict(full_model=True,native_caps_exact=True,gain128=True,mapping=True,all_logged_equations=bool(max(maxima.values())<=1e-6),
        target_bias_bound=all(r['correction_max_rad']<=.0015625+1e-12 for r in records),saved_actual_torque_caps=bool(np.all(abs(body['ctrl'])<=limits+1e-9)),
        recorded_trace_hash=sha(tracefile)==result['contact_damping_trace_sha256'],registered_sources_unchanged=all(frozen.values()),no_static_force_plan=result['static_feedforward_used'] is False)
    assert all(sha(path)==h for path,h in before.items())
    report=dict(status='DAMPING_EQUATIONS_VERIFIED_NOT_PHYSICAL_ACCEPTANCE' if all(checks.values()) else 'DAMPING_EQUATION_AUDIT_FAILED',checks=checks,folder=str(folder),filter_tau=args.filter_tau,
        maximum_reconstruction_errors=maxima,ticks=n,correction_max_rad=max(r['correction_max_rad'] for r in records),
        raw_joint_positive_power_fraction=float(np.mean([r['joint_raw_power']>0 for r in records])),virtual_positive_power_fraction=float(np.mean([r['full_virtual_power']>0 for r in records])),
        maximum_velocity_crosscheck=max(float(np.max(trace['velocity_crosscheck'])),maxima['velocity_crosscheck_logged']),
        sources_unchanged=frozen,input_hashes={**before,str(DEFAULT_MODEL_PATH):sha(DEFAULT_MODEL_PATH)},counts=dict(FK_forward=n,physics_steps=0,neural_steps=0),policy_records=records,
        limitations=['No damping planner/filter/runtime was imported; contactJacobian/velocity and filter were independently recomputed fromsavedstates.',
            'Source snapshot assertion and source review support actualno-writecontract; scratch state reconstruction is not independentphysicalreplay.',
            'Full-generalized virtual damping and selectedjointmotorpower differ because rootDOFsareunactuated. Filterlag and PDtarget realization neednotbepassive.',
            'Tiny boundednominalpreloadequation consistency doesnotestablish D5/heading/materialslip or neuralwalking. Parentfrozenstrictassessment remainsseparate.'])
    if not args.no_write:
        output=folder/'AUDIT_DAMPING.json'
        if output.exists():output.read_bytes()
        output.write_text(json.dumps(report,indent=2,allow_nan=False,default=encode)+'\n')
    print(json.dumps(dict(status=report['status'],checks=checks,max_errors=maxima,ticks=n,filter_tau=args.filter_tau,physics_steps=0,written=not args.no_write),default=encode),flush=True)
    return 0 if all(checks.values()) else 2
if __name__=='__main__':raise SystemExit(main())
