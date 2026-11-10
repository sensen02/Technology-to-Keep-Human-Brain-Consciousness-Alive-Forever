"""Deferred integration hooks. Parent supplies the frozen numerical runtime module.
Import is pure. Functions below that construct/step a runtime require parent heavy GO.
No timeline, pose or contact features enter old_command / neural history decoder.
"""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import numpy as np
from adapter import Layout, SensoryAdapter, capture_raw, save_stream, load_stream, ARMS


@contextmanager
def output_scope(num, root):
    previous=(num.O,num.rt.O,num.rt.b.O)
    num.O=root;num.rt.O=root;num.rt.b.O=root
    try:yield
    finally:num.O,num.rt.O,num.rt.b.O=previous


def runtime_class(num, identity, output_root, numerical_status):
    """Define (do not construct) a runtime under an explicit new sensory output root.
    The parent freezes PREREGISTRATION.json before construction and grants heavy GO.
    """
    root=Path(output_root).resolve()
    allowed=Path('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/sensory').resolve()
    if root==allowed or not root.is_relative_to(allowed):raise ValueError('fresh sensory subdirectory required')
    if numerical_status not in ('converged','nonconverged_factorial_diagnostic'):
        raise ValueError('explicit numerical result required')
    class SensoryRuntime(num.NumericalRuntime):
        def __init__(self, arm='baseline'):
            if not (root/'PREREGISTRATION.json').is_file():raise ValueError('parent must freeze runtime preregistration before model construction')
            with output_scope(num,root):super().__init__()
            self.sensory_arm=arm
            self.sensory_output_root=root
            self.numerical_status=numerical_status
            self.sensory_identity=copy.deepcopy(identity)
            self.raw_layout=Layout.from_model(self.m,self.q0,self.joints,self.geom,
                int(self.mj.mjtGeom.mjGEOM_PLANE),dict(path=str(num.rt.b.P/'MODEL_JOINT_TORQUE.xml'),
                sha256=num.sha(num.rt.b.P/'MODEL_JOINT_TORQUE.xml')))
            self.actual_raw=[];self.actual_currents=[];self.actual_records=[]

        def observe_neural(self,d,net,enc,rates):
            raw=capture_raw(self.m,d,self.raw_layout,self.mj.mj_contactForce)
            adapter=SensoryAdapter(enc,self.raw_layout,self.sensory_identity,self.sensory_arm)
            current,record=adapter.encode(raw)
            if self.sensory_arm=='baseline':
                from run_neural_muscle_closure import observe_quantities
                original=enc.encode(observe_quantities(self.m,d,self.q0,self.joints,self.geom))[0]
                if current.tobytes()!=original.tobytes():raise AssertionError('live baseline raw equivalence failed')
            if self.zero_sensory:current=np.zeros_like(current)
            record['actual_sensory_cut']=bool(self.zero_sensory)
            self.actual_raw.append(raw);self.actual_currents.append(current.copy());self.actual_records.append(record)
            return self.integrate(net,current,rates)

        def episode(self,label,ph,config,cut=False,sensory_cut=False):
            self.actual_raw=[];self.actual_currents=[];self.actual_records=[]
            if Path(label).name!=label or label in ('.','..'):raise ValueError('single component run label required')
            try:
                with output_scope(num,root):
                    return super().episode(label,ph,config,cut=cut,sensory_cut=sensory_cut)
            finally:
                folder=root/'RUNS'/label
                if self.actual_raw:
                    save_stream(folder/'RAW_STREAM.npz',self.actual_raw,self.actual_currents,self.raw_layout)
                    num.dump(folder/'SENSORY_RECORDS.json',dict(arm=self.sensory_arm,
                        actual_records=self.actual_records,numerical_status=self.numerical_status,
                        decoder_input='completed neural history and motor pool rates only',importance='核心必需'))
    return SensoryRuntime


def reencode_actual_stream(stream_path, identity, encoder, arm):
    """Offline arrays only: checks recorded baseline before producing one new input stream."""
    if arm not in ARMS:raise ValueError('unknown arm')
    layout,raws,actual=load_stream(stream_path)
    baseline=SensoryAdapter(copy.deepcopy(encoder),layout,identity,'baseline')
    candidate=SensoryAdapter(copy.deepcopy(encoder),layout,identity,arm)
    encoded=[]
    for raw,reference in zip(raws,actual):
        previous,_=baseline.encode(raw)
        if previous.tobytes()!=reference.tobytes():raise AssertionError('recorded stream is not unchanged baseline actual current')
        current,_=candidate.encode(raw);encoded.append(current)
    return np.array([r.time for r in raws]),np.stack(encoded),layout


def replay_warm(num,runtime,teacher_cache,currents,dt,arm,out_dir):
    """Heavy neural replay, never called by pure tests. Own neural state for EVERY arm/dt.
    teacher_cache contributes unchanged physical integration and same-run assist references.
    RH scaling remains release-only in NumericalRuntime.episode, never in this replay.
    """
    from scipy import sparse
    from engine.neural_cond import ConductanceNetwork,ConductanceParams
    from engine.embodied.sensory_channels import ChannelSensoryEncoder,ChannelConfig
    if arm not in ARMS:raise ValueError('unknown arm')
    out=Path(out_dir).resolve()
    if not out.is_relative_to(runtime.sensory_output_root) or out==runtime.sensory_output_root:
        raise ValueError('warm output must be inside new sensory runtime root')
    out.mkdir(parents=True,exist_ok=False)
    runtime.set_dt(dt);runtime.sensory_arm=arm
    net=ConductanceNetwork(12000,ConductanceParams(),dt_ms=dt*1000)
    net.set_connectivity(*[sparse.load_npz(num.rt.b.AB/'A'/('ACTUAL_'+k+'.npz')) for k in ('We','Wi')])
    enc=ChannelSensoryEncoder(runtime.groups,12000,ChannelConfig())
    runtime.history.reset();rates=np.zeros(66);runtime.cache={};trace=[];packed=[]
    capture_steps={round((float(ph)-.2)/.001):float(ph) for ph in teacher_cache}
    if np.asarray(currents).shape!=(1200,12000) or np.asarray(currents).dtype!=np.float64:
        raise ValueError('1200 actual/reencoded float64 1ms-held samples required')
    for k,current in enumerate(currents):
        rates,counts,_=runtime.integrate(net,current,rates);packed.extend(runtime.last_full);trace.append(rates.copy())
        if k+1 in capture_steps:
            ph=capture_steps[k+1];c=copy.deepcopy(teacher_cache[ph])
            c.update(net=copy.deepcopy(net),enc=copy.deepcopy(enc),rates=rates.copy(),history=runtime.history.state_dict())
            runtime.cache[ph]=c
    assert set(runtime.cache)==set(teacher_cache)
    np.savez_compressed(out/'WARM_NEURAL.npz',pool_rates=np.asarray(trace),spikes12000_packed=np.asarray(packed),
                        end_time=.2+np.arange(1,1201)*.001,neural_dt_s=dt)
    cfg=out/'PARAMETERS.json'
    num.dump(cfg,dict(arm=arm,neural_dt_s=dt,command_dt_s=.001,physics_dt_s=.0000125,
        gains='original .3 nA/unit; club abs(v)/10 clipped[0,1]; C maxabs(block deviation)/.30 clipped[0,1]',
        gain_status='ENGINEERING',direction_restored=False,vibration_frequency_tuning='NOT_MODELLED',
        RH_scale=.375,RH_scope='release_only',decoder='unchanged neural-only',numerical_status='PARENT_DECISION_REQUIRED',importance='核心必需'))
    return cfg
