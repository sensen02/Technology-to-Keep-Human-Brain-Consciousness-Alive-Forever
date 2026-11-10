"""Prepared NEW 1.4s unchanged teacher recorder. NOT run by offline tests.
No heavy import until main receives a parent-written, hashed execution decision.
This records physical raw input; it does not run sensory candidates or select a timestep.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys

SRC=Path(__file__).resolve().parent
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/sensory')
NUM=SRC.parent/'numerics/run.py'
PHASES=(1.225,1.275,1.3,1.325,1.375)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate_decision(path):
    decision=json.loads(Path(path).read_text())
    if decision.get('explicit_parent_heavy_GO') is not True or decision.get('operation')!='record_sensory_raw_teacher_1p4s':
        raise ValueError('requires parent heavy-process GO for this recording')
    if decision.get('numerics_source_sha256')!=sha(NUM):
        raise ValueError('numerics runtime source not frozen at supplied hash')
    if decision.get('numerical_status') not in ('converged','nonconverged_factorial_diagnostic'):
        raise ValueError('numerical result must be declared, never inferred stable')
    if decision.get('neural_dt_s') not in (.0005,.00025,.000125,.0000625):
        raise ValueError('explicit previously tested neural timestep required')
    # Freeze all prep code/identity, not just the numerical runtime.
    required=list(SRC.glob('*.py'))+[OUT/'IDENTITY.json',OUT/'PROTOCOL.json',OUT/'OFFLINE_TESTS.json']
    expected=decision.get('sensory_input_sha256',{})
    for p in required:
        if expected.get(str(p))!=sha(p):raise ValueError('sensory source/evidence not frozen: '+str(p))
    if not json.loads((OUT/'OFFLINE_TESTS.json').read_text())['passed']:
        raise ValueError('offline gate failed')
    return decision


def execute(decision,decision_path):
    # These imports are deliberately beyond the explicit gate.
    import copy
    import pickle
    import numpy as np
    from scipy import sparse
    from adapter import Layout, SensoryAdapter, capture_raw, save_stream
    module_name='sensory_frozen_numerics'
    spec=importlib.util.spec_from_file_location(module_name,NUM);num=importlib.util.module_from_spec(spec)
    sys.modules[module_name]=num;spec.loader.exec_module(num)
    dt=decision['neural_dt_s'];folder=OUT/('TEACHER_dt_'+str(dt*1000).replace('.','p')+'ms')
    folder.mkdir(parents=True,exist_ok=False)
    # All constructor writes are redirected before construction, including legacy globals.
    num.O=folder;num.rt.O=folder;num.rt.b.O=folder
    num.dump(folder/'EXECUTION_DECISION.json',dict(decision=decision,source=str(decision_path),sha256=sha(decision_path)))
    t=num.NumericalRuntime();t.set_dt(dt)
    from engine.embodied.sensory_channels import ChannelSensoryEncoder, ChannelConfig
    from engine.neural_cond import ConductanceNetwork, ConductanceParams
    from run_neural_muscle_closure import observe_quantities
    m=t.m;mj=t.mj
    layout=Layout.from_model(m,t.q0,t.joints,t.geom,int(mj.mjtGeom.mjGEOM_PLANE),
        dict(path=str(num.rt.b.P/'MODEL_JOINT_TORQUE.xml'),sha256=sha(num.rt.b.P/'MODEL_JOINT_TORQUE.xml')))
    num.dump(folder/'LAYOUT.json',layout.to_dict())
    identity=json.loads((OUT/'IDENTITY.json').read_text());enc=ChannelSensoryEncoder(t.groups,12000,ChannelConfig())
    adapter=SensoryAdapter(enc,layout,identity,'baseline')
    np.random.seed(0);net=ConductanceNetwork(12000,ConductanceParams(),dt_ms=dt*1000)
    net.set_connectivity(*[sparse.load_npz(num.rt.b.AB/'A'/('ACTUAL_'+key+'.npz')) for key in ('We','Wi')])
    d=mj.MjData(m);mj.mj_resetDataKeyframe(m,d,0);mj.mj_forward(m,d)
    t.history.reset();rates=np.zeros(66);raws=[];currents=[];native=[];body=[t.bodyrow(d)];cache={};packed=[]
    capture_steps={round(ph/num.PDT):ph for ph in (*PHASES,1.4)}
    kp=t.gains['kp'];kd=np.asarray(t.gains['kd']);clip=t.gains['clip'];old_current_checks=0
    try:
        for step in range(round(1.4/num.PDT)):
            if step%80==0 and step>=round(.2/num.PDT):
                raw=capture_raw(m,d,layout,mj.mj_contactForce)
                current,record=adapter.encode(raw)
                old_current,old_record=enc.encode(observe_quantities(m,d,t.q0,t.joints,t.geom))
                if current.tobytes()!=old_current.tobytes():raise AssertionError('actual physical raw baseline not bitwise exact')
                old_current_checks+=1;raws.append(raw);currents.append(current.copy())
                rates,counts,_=t.integrate(net,current,rates);packed.extend(t.last_full)
            time=float(d.time);ramp=np.clip((time-.15)/.10,0,1);phi=max(0,time-.15)*2*np.pi*5
            qg=t.gait(phi%(2*np.pi));desired=(1-ramp)*t.q0[t.qa]+ramp*qg
            velocity=ramp*t.gait(phi%(2*np.pi),1)*2*np.pi*5+(qg-t.q0[t.qa])*(10. if .15<time<.25 else 0.)
            torque=kp*(desired-d.qpos[t.qa])+kd*(velocity-d.qvel[t.va])+d.qfrc_bias[t.va]-d.qfrc_passive[t.va]
            d.ctrl[:]=np.clip(np.clip(torque,-clip,clip),-t.limits,t.limits);d.qfrc_applied[:]=0;d.xfrc_applied[:]=0
            mj.mj_step(m,d);mj.mj_forward(m,d)
            if step>=round(1./num.PDT):native.append(t.contact(d))
            if (step+1)%40==0:body.append(t.bodyrow(d))
            assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and not d.warning.number.any()
            if step+1 in capture_steps:
                ph=capture_steps[step+1];window=np.asarray(native)[-round(.2/num.PDT):]
                cache[ph]=dict(integration=t.snapshot(d),net=copy.deepcopy(net),enc=copy.deepcopy(enc),rates=rates.copy(),
                    history=t.history.state_dict(),duration=ph,phase_offset=0.,assist=dict(
                    foot_support_mean=float(window[:,1:7].sum(1).mean()/(m.body_mass.sum()*abs(m.opt.gravity[2]))),
                    com_z_mean=float(window[:,10].mean())),warm_folder=str(folder))
        assert len(raws)==1200 and old_current_checks==1200
        historical=num.arrays(num.S/'TRUE_CURRENTS.npz')
        assert np.array_equal(np.asarray(currents),historical['currents']), 'new unchanged teacher current differs from actual old teacher'
        assert np.array_equal(np.asarray([r.time for r in raws]),historical['time']), 'teacher time differs'
        with (num.S/'DATA/PHYSICAL_CACHE.pkl').open('rb') as f:physical=pickle.load(f)
        physical_checks=[]
        for ph in PHASES:
            assert np.array_equal(cache[ph]['integration'],physical[ph]['integration'])
            if dt==.0005:
                assert num.exact(vars(cache[ph]['net']),vars(physical[ph]['net']))
                assert num.exact(cache[ph]['rates'],physical[ph]['rates']) and num.exact(cache[ph]['history'],physical[ph]['history'])
            physical_checks.append(dict(phase=ph,full_physical_integration_exact=True,neural_baseline_exact=dt==.0005))
        with (folder/'BASELINE_CACHE.pkl').open('xb') as f:pickle.dump(cache,f,protocol=5)
        num.dump(folder/'RECORDING_RESULT.json',dict(passed=True,actual_old_observer_bitwise_frames=old_current_checks,
            historical_actual_teacher_current_exact=True,physical_checks=physical_checks,neural_dt_s=dt,numerical_status=decision['numerical_status'],
            autonomous_behavior='NOT_TESTED',new_physical_seconds=1.4,new_neural_steps=1200*t.n,importance='核心必需'))
    except BaseException:
        import traceback
        num.dump(folder/'FAILURE.json',dict(traceback=traceback.format_exc(),completed_raw_samples=len(raws)))
        raise
    finally:
        if raws:save_stream(folder/'RAW_STREAM.npz',raws,currents,layout)
        np.savez_compressed(folder/'BODY.npz',**{key:np.array([r[key] for r in body]) for key in body[0]})
        np.savez_compressed(folder/'NATIVE.npz',rows=np.asarray(native))
        np.savez_compressed(folder/'BASELINE_SPIKES.npz',spikes12000_packed=np.asarray(packed),neural_dt_s=dt)
    print('RAW_TEACHER_READY',str(folder),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--authorization-file',type=Path,required=True)
    args=p.parse_args();decision=validate_decision(args.authorization_file);execute(decision,args.authorization_file)


if __name__=='__main__':main()
