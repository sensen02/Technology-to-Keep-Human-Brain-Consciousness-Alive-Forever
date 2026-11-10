"""Current full body with actual qualified-sensory->BANC point-neurons->MN readout.
First1s teacher-rate warmup; then learned readout only. No brain/feeding claim.
"""
from pathlib import Path
import sys,json,time,hashlib,argparse
import numpy as np
import mujoco as mj
HERE=Path(__file__).resolve().parent;ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype');OUT=Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T185910Z')
BASE=ROOT/'experiments/s5-ann-incremental/20261107T010000Z';sys.path[:0]=[str(HERE/'dynamics'),str(HERE/'encoder'),str(BASE),str(ROOT/'tools')]
from module import LocalCircuit
from encoder import SensoryEncoder
from incremental_actuator import IncrementalActuator,DEFAULT_MODEL_PATH
from plant_probe import Plant
import dynamic_support_metrics as dsm
SOURCE=Path('/home/sensen/Desktop/engine修复验收/flybody-retarget/20261107T182940Z/teacherderived_slow4_physics')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def dump(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--control',choices=['learned','hold'],default='learned');args=ap.parse_args()
    folder=OUT/('closed_loop' if args.control=='learned' else 'output_hold');folder.mkdir(parents=True,exist_ok=False)
    circuit=LocalCircuit();enc=SensoryEncoder();weights=load(OUT/'readout/MODEL.npz');W=weights['weights']
    assert np.array_equal(weights['mn_root_ids'],circuit.root_ids[circuit.mn_local]) and np.array_equal(enc.root_ids,circuit.root_ids)
    assert np.isin(enc.local_indices,circuit.allowed_sensory_local).all()
    p=Plant();m=p.m;d=p.fresh();source_initial=load(SOURCE/'INITIAL.npz');spec=mj.mjtState.mjSTATE_INTEGRATION
    mj.mj_setState(m,d,source_initial['state'],spec);mj.mj_forward(m,d)
    assert np.array_equal(weights['joint_names'],[m.joint(int(j)).name for j in p.jids])
    actuator=IncrementalActuator(m,d,kp=p.limits/.2*128,kd=p.limits*.005*np.sqrt(128),max_rate_rad_s=30.)
    # Only warmup prefix is retained in deployment memory; no teacher after1s.
    warm=load(SOURCE/'COMMANDS.npz')['rows'][:1000,1:43].copy()
    warm_expected=load(SOURCE/'BODY.npz')['qpos'][1000].copy()
    initial=np.zeros(mj.mj_stateSize(m,spec));mj.mj_getState(m,d,initial,spec)
    np.savez_compressed(folder/'INITIAL.npz',state=initial,target=actuator.target_rad,kp=actuator.kp,kd=actuator.kd,limits=actuator.torque_limit,qa=p.qa,va=p.va,joint_bounds=m.jnt_range[p.jids])
    mj.mj_saveModel(m,str(folder/'MODEL.mjb'),None)
    dump(folder/'PROTOCOL.json',dict(condition=args.control,controller='qualifiedafferencts->strictreal-IDconductance network->MN-onlylearnedreadout->bounded42rateintegrator',hold_s=1.,neural_handover_s=1.,learner_s=2.,total_s=3.,physics_dt=m.opt.timestep,command_dt=.001,policy_dt=.005,
      input_roles='locallegq/qvel/footload proxies only; zeroDN, no brain implemented',teacher_access='first1000commands warmup prefix only; noneafter1s',body_source=str(DEFAULT_MODEL_PATH),body_source_sha256=sha(DEFAULT_MODEL_PATH),runner_sha256=sha(__file__),readout_sha256=sha(OUT/'readout/MODEL.npz'),network_sha256=sha(HERE/'dynamics/module.py'),encoder_sha256=sha(HERE/'encoder/encoder.py'),timing='start-of-interval MNfeatures; held startphysical sensory input integrated neurally aftersamephysical5ms interval'))
    nt=3000;native=np.empty((nt*80,20));commands=[];body=[];policy=[];slip=[];start=time.monotonic();saturated=0;neural_vmin=0.;neural_vmax=-1e10;warm_error=None
    def state():return dict(time=float(d.time),qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),qacc_warmstart=d.qacc_warmstart.copy(),com=d.subtree_com[p.thorax].copy())
    body.append(state());zero=np.zeros(20000)
    for tick in range(600):
        k0=tick*5;mn=circuit.filtered_rates_hz[circuit.mn_local].copy();rate=30*np.tanh((mn/100.)@W)
        foot=np.zeros(6) if k0==0 else native[k0*80-1,1:7].copy();q=d.qpos[p.qa].copy();vel=d.qvel[p.va].copy();current=enc.encode(q,vel,foot)
        if k0==1000:
            before=p.snapshot(d);warm_error=float(np.abs(d.qpos-warm_expected).max());after=p.snapshot(d);assert np.array_equal(before,after)
            np.savez_compressed(folder/'HANDOVER.npz',physics_state=before,actuator_target=actuator.target_rad,mn_rates_hz=mn,neural_v=circuit.net.v,neural_ge=circuit.net.ge,neural_gi=circuit.net.gi,neural_adaptation=circuit.net.adaptation,neural_delay_ring=circuit.net._ring,neural_ring_i=circuit.net._ring_i)
        policy.append(dict(time_s=float(d.time),mn_rates_hz=mn,predicted_rate=rate.copy(),local_physical_input=np.r_[q,vel,foot],sensory_current_nA=current[enc.local_indices].copy(),mn_voltages_mV=circuit.net.v[circuit.mn_local].copy(),teacher_warmup=k0<1000))
        for offset in range(5):
            k=k0+offset;actual_rate=warm[k] if k<1000 else (rate if args.control=='learned' else np.zeros(42))
            actuator.update(actual_rate,.001);commands.append(np.r_[k*.001,actual_rate,actuator.target_rad])
            for sub in range(80):
                d.ctrl[:]=actuator.torque(d);saturated+=int(np.count_nonzero(np.abs(d.ctrl)>=actuator.torque_limit-1e-12))
                assert not d.qfrc_applied.any() and not d.xfrc_applied.any()
                mj.mj_step(m,d);mj.mj_forward(m,d);native[k*80+sub]=p.contact(d)
                if sub%40==39:
                    loads=np.zeros(6);speeds=np.zeros(6)
                    for ci,c in enumerate(d.contact):
                        ga,gb=int(c.geom1),int(c.geom2)
                        if not(p.planes[ga] or p.planes[gb]):continue
                        g=gb if p.planes[ga] else ga;leg=p.feet[g]
                        if leg<0:continue
                        f=np.zeros(6);mj.mj_contactForce(m,d,ci,f);fz=max(0.,float((c.frame.reshape(3,3).T@f[:3])[2])*(1 if p.planes[ga] else -1))
                        jac=np.zeros((3,m.nv));mj.mj_jac(m,d,jac,np.zeros_like(jac),c.pos,int(m.geom_bodyid[g]));loads[leg]+=fz;speeds[leg]+=fz*np.linalg.norm((jac@d.qvel)[:2])
                    slip.append(np.r_[d.time,loads,speeds])
            body.append(state());assert np.isfinite(d.qpos).all() and not d.warning.number.any()
        circuit.advance(current,zero,5.)
        neural_vmin=min(neural_vmin,float(circuit.net.v.min()));neural_vmax=max(neural_vmax,float(circuit.net.v.max()))
        assert abs(circuit.net.t_ms/1000-d.time)<1e-8
        if (tick+1)%100==0:print(json.dumps(dict(physical_s=float(d.time),neural_s=circuit.net.t_ms/1000,wall_s=time.monotonic()-start)),flush=True)
    for name,values in [('BODY',{key:np.asarray([b[key] for b in body]) for key in body[0]}),('NATIVE',dict(rows=native)),('COMMANDS',dict(rows=np.asarray(commands))),('SLIP',dict(rows=np.asarray(slip))),('POLICY',{key:np.asarray([r[key] for r in policy]) for key in policy[0]})]:np.savez_compressed(folder/(name+'.npz'),**values)
    np.savez_compressed(folder/'NEURAL_FINAL.npz',v=circuit.net.v,ge=circuit.net.ge,gi=circuit.net.gi,adaptation=circuit.net.adaptation,refractory_until_ms=circuit.net.refractory_until_ms,spike_count=circuit.net.spike_count,delay_ring=circuit.net._ring,ring_i=circuit.net._ring_i,filtered_rates_hz=circuit.filtered_rates_hz,steps=circuit.net.steps)
    assist=native[(native[:,0]>=.8)&(native[:,0]<1.)];release=native[native[:,0]>=1.]
    gate=dsm.evaluate_native(dict(foot_support_mean=float(assist[:,1:7].sum(1).mean()/dsm.WEIGHT),com_z_mean=float(assist[:,10].mean())),release);gate['checks']['D1_no_nonfoot_ground_force']=bool(release[:,7].max()==0);gate['passed']=all(gate['checks'].values())
    com=np.array([b['com'] for b in body]);speed=np.linalg.norm(np.diff(com[1000:,:2],axis=0),axis=1).sum()/2.;s=np.asarray(slip);mask=s[:,0]>=1;sv=s[mask,7:13].sum(0)/np.maximum(s[mask,1:7].sum(0),1e-15)
    result=dict(status='LOCAL_NEURAL_PILOT_NOT_STABLE_WALKING_ACCEPTANCE',native_steps=len(native),neural_steps=circuit.net.steps,sim_s=float(d.time),learner_controlled_s=2. if args.control=='learned' else 0.,condition=args.control,physical_gate=gate,
      warmup_qpos_max_error=warm_error,neural_state_preserved_at_handover=True,after_handover_source='MN_only_learned_readout' if args.control=='learned' else 'zero_rates_PD_target_hold',MN_spikes=int(circuit.net.spike_count[circuit.mn_local].sum()),active_MN_count=int(np.count_nonzero(circuit.net.spike_count[circuit.mn_local])),neural_voltage_range_mV=[neural_vmin,neural_vmax],
      root_displacement=(body[-1]['qpos'][:3]-body[1000]['qpos'][:3]).tolist(),posthold_COM_path_speed=float(speed),material_slip_weighted_speed=sv.tolist(),slip_to_pathspeed_ratio=(sv/max(speed,1e-12)).tolist(),nonfoot_force_max=float(release[:,7].max()),torque_saturated_entries=saturated,torque_total_entries=nt*80*42,warnings=d.warning.number.tolist(),wall_s=time.monotonic()-start,
      source_quality='failedreference used onlyfor warmup/learning; no teacher accessed in2sneuralphase',files={p.name:sha(p) for p in folder.iterdir() if p.is_file()})
    dump(folder/'RESULT.json',result);print(json.dumps({k:result[k] for k in ['status','physical_gate','warmup_qpos_max_error','MN_spikes','slip_to_pathspeed_ratio','wall_s']},indent=2))
if __name__=='__main__':main()
