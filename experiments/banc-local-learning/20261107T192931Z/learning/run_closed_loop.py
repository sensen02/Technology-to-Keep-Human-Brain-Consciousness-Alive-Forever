"""Round9 physical runner for parent execution only; never run by fitting task.

New adaptation-conductance circuit + frozen old physical sensory encoder +
preregistered newhistory MN-only readout. Old fullbody/initial/physics unchanged.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
import mujoco as mj

HERE=Path(__file__).resolve().parent
ROUND=HERE.parent
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
OLD=ROOT/'experiments/banc-local-learning/20261107T185910Z'
OUT=Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T192931Z')
PLANTBASE=ROOT/'experiments/s5-ann-incremental/20261107T010000Z'
sys.path[:0]=[str(ROUND/'dynamics'),str(OLD/'encoder'),str(PLANTBASE),str(ROOT/'tools')]
from module import LocalCircuit
from encoder import SensoryEncoder
from runtime import Readout
from incremental_actuator import IncrementalActuator,DEFAULT_MODEL_PATH
from plant_probe import Plant
import dynamic_support_metrics as dsm
SOURCE=Path('/home/sensen/Desktop/engine修复验收/flybody-retarget/20261107T182940Z/teacherderived_slow4_physics')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def dump(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--control',choices=['learned','hold'],default='learned')
    parser.add_argument('--model',type=Path,default=HERE/'cases/newhistory/MODEL.npz')
    args=parser.parse_args()
    verification=json.loads((HERE/'INDEPENDENT_VERIFICATION.json').read_text())
    if verification['status']!='PASS_FIXED_FOUR_CASE_ENGINEERING':raise RuntimeError('Independent readout verification not complete')
    verified_hash=verification['case_model_sha256']['newhistory']
    if sha(args.model)!=verified_hash:raise ValueError('Primary model differs from independently verified weights')
    folder=OUT/('closed_loop' if args.control=='learned' else 'output_hold');folder.mkdir(parents=True,exist_ok=False)
    circuit=LocalCircuit();enc=SensoryEncoder();readout=Readout(args.model)
    metadata=load(args.model)
    assert str(metadata['case'])=='newhistory' and bool(metadata['primary']) and np.array_equal(readout.lags_bins,[0,1,4,10])
    assert np.array_equal(readout.mn_root_ids,circuit.root_ids[circuit.mn_local]) and np.array_equal(enc.root_ids,circuit.root_ids)
    assert np.array_equal(readout.mn_local,circuit.mn_local) and np.array_equal(readout.mn_leg,circuit.mn_leg)
    assert np.isin(enc.local_indices,circuit.allowed_sensory_local).all()
    assert hasattr(circuit.net,'g_adapt'), 'New mechanism must expose adaptation conductance uS'
    p=Plant();m=p.m;d=p.fresh();source_initial=load(SOURCE/'INITIAL.npz');spec=mj.mjtState.mjSTATE_INTEGRATION
    mj.mj_setState(m,d,source_initial['state'],spec);mj.mj_forward(m,d)
    assert np.array_equal(readout.joint_names,[m.joint(int(j)).name for j in p.jids])
    actuator=IncrementalActuator(m,d,kp=p.limits/.2*128,kd=p.limits*.005*np.sqrt(128),max_rate_rad_s=30.)
    warm=load(SOURCE/'COMMANDS.npz')['rows'][:1000,1:43].copy()
    warm_expected=load(SOURCE/'BODY.npz')['qpos'][1000].copy()
    initial=np.zeros(mj.mj_stateSize(m,spec));mj.mj_getState(m,d,initial,spec)
    np.savez_compressed(folder/'INITIAL.npz',state=initial,target=actuator.target_rad,kp=actuator.kp,kd=actuator.kd,limits=actuator.torque_limit,qa=p.qa,va=p.va,joint_bounds=m.jnt_range[p.jids])
    mj.mj_saveModel(m,str(folder/'MODEL.mjb'),None)
    dump(folder/'PROTOCOL.json',dict(condition=args.control,controller='qualified556afferents->realIDnewadaptationconductance->samelegMNhistory[0,1,4,10]->bounded42rateintegrator',hold_s=1.,neural_handover_s=1.,learner_s=2.,total_s=3.,physics_dt=m.opt.timestep,command_dt=.001,policy_dt=.005,
        input_roles='same old locallegq/qvel/footload; zeroDN; no brain implemented',teacher_access='first1000commands warmup prefix only; noneafter1s',body_source=str(DEFAULT_MODEL_PATH),body_source_sha256=sha(DEFAULT_MODEL_PATH),runner_sha256=sha(__file__),readout_path=str(args.model.resolve()),readout_sha256=sha(args.model),runtime_sha256=sha(HERE/'runtime.py'),network_sha256=sha(ROUND/'dynamics/module.py'),encoder_sha256=sha(OLD/'encoder/encoder.py'),verification_sha256=sha(HERE/'INDEPENDENT_VERIFICATION.json'),
        timing='observe START MNfilteredHz into history, action uses lag0/1/4/10, hold START physical sensory and integrate neural5ms after physical interval; no handover resets',adaptation='conductance g_adapt in uS; EA=-80mV, increment=.002/15uS, tau100ms; engineering assumption not biological calibration',history='oldest50ms; sample at exactage50ms included, absent at55ms; zerocurrent alone can retain olderactivity',historical_hold='185910Z/output_hold can serve body-only matched historical zero-rate baseline after parent verifies initial/body/command identity; not new-mechanism neural evidence'))
    nt=3000;native=np.empty((nt*80,20));commands=[];body=[];policy=[];slip=[];start=time.monotonic();saturated=0;neural_vmin=0.;neural_vmax=-1e10;warm_error=None
    def state():return dict(time=float(d.time),qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),qacc_warmstart=d.qacc_warmstart.copy(),com=d.subtree_com[p.thorax].copy())
    body.append(state());zero=np.zeros(20000)
    for tick in range(600):
        k0=tick*5;mn=circuit.filtered_rates_hz[circuit.mn_local].copy()
        readout.observe(mn);rate=readout.action()
        foot=np.zeros(6) if k0==0 else native[k0*80-1,1:7].copy();q=d.qpos[p.qa].copy();vel=d.qvel[p.va].copy();current=enc.encode(q,vel,foot)
        if k0==1000:
            before=p.snapshot(d);warm_error=float(np.abs(d.qpos-warm_expected).max());after=p.snapshot(d);assert np.array_equal(before,after)
            predictor={'predictor_'+key:value for key,value in readout.state_dict().items()}
            np.savez_compressed(folder/'HANDOVER.npz',physics_state=before,actuator_target=actuator.target_rad,mn_rates_hz=mn,neural_v=circuit.net.v,neural_ge=circuit.net.ge,neural_gi=circuit.net.gi,neural_g_adapt_uS=circuit.net.g_adapt,neural_delay_ring=circuit.net._ring,neural_ring_i=circuit.net._ring_i,**predictor)
        policy.append(dict(time_s=float(d.time),mn_rates_hz=mn,predicted_rate=rate.copy(),local_physical_input=np.r_[q,vel,foot],sensory_current_nA=current[enc.local_indices].copy(),mn_voltages_mV=circuit.net.v[circuit.mn_local].copy(),teacher_warmup=k0<1000,readout_features=readout.feature_vector(),predictor_observations_seen=readout.observations_seen))
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
    np.savez_compressed(folder/'NEURAL_FINAL.npz',v=circuit.net.v,ge=circuit.net.ge,gi=circuit.net.gi,g_adapt_uS=circuit.net.g_adapt,refractory_until_ms=circuit.net.refractory_until_ms,spike_count=circuit.net.spike_count,delay_ring=circuit.net._ring,ring_i=circuit.net._ring_i,filtered_rates_hz=circuit.filtered_rates_hz,steps=circuit.net.steps)
    np.savez_compressed(folder/'READOUT_FINAL.npz',**readout.state_dict())
    assist=native[(native[:,0]>=.8)&(native[:,0]<1.)];release=native[native[:,0]>=1.]
    gate=dsm.evaluate_native(dict(foot_support_mean=float(assist[:,1:7].sum(1).mean()/dsm.WEIGHT),com_z_mean=float(assist[:,10].mean())),release);gate['checks']['D1_no_nonfoot_ground_force']=bool(release[:,7].max()==0);gate['passed']=all(gate['checks'].values())
    com=np.array([b['com'] for b in body]);speed=np.linalg.norm(np.diff(com[1000:,:2],axis=0),axis=1).sum()/2.;s=np.asarray(slip);mask=s[:,0]>=1;sv=s[mask,7:13].sum(0)/np.maximum(s[mask,1:7].sum(0),1e-15)
    result=dict(status='NEW_MECHANISM_HISTORY_PILOT_NOT_WALKING_ACCEPTANCE',native_steps=len(native),neural_steps=circuit.net.steps,sim_s=float(d.time),learner_controlled_s=2. if args.control=='learned' else 0.,condition=args.control,physical_gate=gate,
        warmup_qpos_max_error=warm_error,neural_state_preserved_at_handover=True,predictor_history_preserved_at_handover=True,after_handover_source='MN_only_causal_history_readout' if args.control=='learned' else 'zero_rates_PD_target_hold',MN_spikes=int(circuit.net.spike_count[circuit.mn_local].sum()),active_MN_count=int(np.count_nonzero(circuit.net.spike_count[circuit.mn_local])),neural_voltage_range_mV=[neural_vmin,neural_vmax],
        root_displacement=(body[-1]['qpos'][:3]-body[1000]['qpos'][:3]).tolist(),posthold_COM_path_speed=float(speed),material_slip_weighted_speed=sv.tolist(),slip_to_pathspeed_ratio=(sv/max(speed,1e-12)).tolist(),nonfoot_force_max=float(release[:,7].max()),torque_saturated_entries=saturated,torque_total_entries=nt*80*42,warnings=d.warning.number.tolist(),wall_s=time.monotonic()-start,
        source_quality='failedreference onlywarmup/learning; primarynewhistoryfixedbeforetemporalresults; no teacher accessed in2sneuralphase',files={p.name:sha(p) for p in folder.iterdir() if p.is_file()})
    dump(folder/'RESULT.json',result);print(json.dumps({k:result[k] for k in ['status','physical_gate','warmup_qpos_max_error','MN_spikes','slip_to_pathspeed_ratio','wall_s']},indent=2))

if __name__=='__main__':main()
