"""Short interface/causal-cut checks: real Core construction, ZERO physics steps.

Shared handover is exercised for 20ms on a mocked plant, never a real gait.
Default checkpoint is the immutable original 300-step role-aux SlowGRU.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import student_wave as sw
import sensory_ablation as ab
from identity import atomic_json, sha256

DEFAULT_CHECKPOINT=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T114148Z/BC_GRU_ROLE_AUX_300/final.pt')


class FakeCore:
    def __init__(self):
        self.d=SimpleNamespace(time=1.,qpos=np.zeros(3),qvel=np.ones(3),ctrl=np.zeros(42),
            xfrc_applied=np.zeros((1,6)),qfrc_applied=np.zeros(3))
        self.actuator=SimpleNamespace(target_rad=np.zeros(42))
        self.buf=np.zeros((2,42)); self._padz_prev=np.zeros(6); self._pad_rate=np.ones(6)
        self.requested_rate=np.zeros(42); self.adhesion_on=False
    def obs238(self,previous,target_speed):
        x=np.linspace(-.2,.2,238); x[189:195]=np.arange(1,7)*.1
        x[195:237]=previous; return x
    def critic_extras(self):
        return np.linspace(-.1,.1,16)

class FakeTrace:
    def __init__(self):
        self.actions=[]; self.sources=[]
    def tick(self,c,action,source):
        sw.no_external(c)
        self.actions.append(action.copy()); self.sources.append(source)
        c.d.time+=.001; c.d.qpos+=.0001
        c.requested_rate=action.copy()*sw.MAX_RATE
        sw.no_external(c)

class FakeTracker:
    config=sw.CycleConfig(timeout=4.)
    def measure(self,c): return dict(events=[])
    def finalize(self): return dict(events=[])

class FakeTeacher:
    def __init__(self): self.calls=[]
    def action(self,c,t):
        self.calls.append(t)
        return np.full(42,.01+len(self.calls)*.001), {}


def check(checkpoint):
    sw.seed_all(20261110)
    checkpoint=Path(checkpoint).resolve()
    old_checkpoint=sha256(checkpoint)
    core=sw.Core(adhesion=False)  # construction only; tick is NEVER called
    model,payload=sw.load_student(checkpoint,core)
    assert model.architecture=='gru238'
    old_weights=ab.weights_hash(model)
    results={}
    def passed(name, value):
        assert value,name
        results[name]=True
    passed('original_checkpoint_full_load_student_closure',True)
    passed('teacher_actual_093347_preflight',bool(sw.teacher_preflight()))
    compiled={name:np.asarray(getattr(core.plant,name)).copy() for name in dir(core.plant)
              if not name.startswith('_') and isinstance(getattr(core.plant,name,None),np.ndarray)}
    x,v=sw.observations(core,np.linspace(-.1,.1,42),.02)
    h=model.actor.initial_state(1)
    real_before=sw.continuity_snapshot(core,x[195:237],h)
    with torch.no_grad():
        expected=model.act(x,v,deterministic=True,state=h)
        for condition in ab.CONDITIONS:
            proxy=ab.ModelProxy(model,core,condition,2,float(core.d.time))
            passed(condition+'_tracking_interface',proxy.actor is model.actor and proxy.architecture==model.architecture)
            actual=proxy.act(x,v,deterministic=True,state=h)
            passed(condition+'_false_flag_bitwise_baseline',all(torch.equal(actual[k],expected[k])
                for k in expected if torch.is_tensor(expected[k])))
            passed(condition+'_false_flag_raw_input',np.array_equal(proxy.records[0]['actor_input_actual'],x))
            passed(condition+'_critic_raw',np.array_equal(proxy.records[0]['raw_V255'],v))
        identity=ab.ModelProxy(model,core,'identity',1,float(core.d.time))
        baseline_state=h
        identity_state=h
        for i in range(3):
            baseline=model.act(x,v,deterministic=True,state=baseline_state)
            wrapped=identity.act(x,v,deterministic=True,state=identity_state)
            passed('identity_action_hidden_value_exact_'+str(i),all(torch.equal(wrapped[k],baseline[k])
                for k in baseline if torch.is_tensor(baseline[k])))
            baseline_state=baseline['state']; identity_state=wrapped['state']
        passed('identity_flag_true_still_raw',identity.intervene and all(np.array_equal(r['actor_input_actual'],x) for r in identity.records))
        masked=ab.ModelProxy(model,core,'foot_velocity_zero',1,float(core.d.time))
        prefix=masked.act(x,v,deterministic=True,state=h)
        cut=masked.act(x,v,deterministic=True,state=prefix['state'])
        desired=x.copy(); desired[189:195]=0
        reference=model.act(desired,v,deterministic=True,state=prefix['state'])
        passed('post_cut_exact_original_model_act',all(torch.equal(cut[k],reference[k]) for k in cut if torch.is_tensor(cut[k])))
        passed('six_entries_zeroed_and_other_232_unchanged',np.array_equal(masked.records[1]['actor_input_actual'],desired)
            and np.all(np.isin(np.flatnonzero(desired!=x),np.arange(189,195))))
        passed('cut_hidden_carried_without_reset',np.array_equal(masked.records[1]['hidden_in'],masked.records[0]['hidden_out']))
        passed('critic_value_from_raw_V255',torch.equal(cut['value'],model.value(torch.as_tensor(v).unsqueeze(0))))
        passed('cut_state_continuity',masked.cut_continuity['all_exact'])
    real_after=sw.continuity_snapshot(core,x[195:237],h)
    passed('real_core_zero_steps_all_state_unchanged',all(np.array_equal(real_before[k],real_after[k]) for k in real_before))
    # Use the actual shared handover implementation; physics and observer alone mocked.
    for condition in ab.CONDITIONS:
        fake=FakeCore(); teacher=FakeTeacher(); trace=FakeTrace()
        proxy=ab.ModelProxy(model,fake,condition,2,1.010)
        with patch.object(sw,'tracker',lambda c:FakeTracker()):
            run=sw.continuous_handover(fake,teacher,proxy,trace,10,10,control='student')
        mapping=ab.validate_records(proxy,run['decisions'],10,10)
        passed(condition+'_shared_short_strict_decision_counts',mapping['total']==4 and len(trace.actions)==20)
        passed(condition+'_shared_prefix_false_post_true',[r['intervene'] for r in proxy.records]==[False,False,True,True])
        passed(condition+'_teacher_not_queried_after_handover',len(teacher.calls)==2)
        passed(condition+'_shared_exact_handover',run['transition']['all_exact'])
        saved={k:np.asarray([d[k] for d in run['decisions']]) for k in run['decisions'][0]}
        saved.update({k:np.asarray([r[k] for r in proxy.records]) for k in proxy.records[0]})
        passed(condition+'_saved_actual_actions_not_primed_predictions',np.array_equal(saved['action'][0],run['decisions'][0]['action'])
            and 'policy_action' in saved and np.array_equal(saved['policy_action'][2],saved['action'][2]))
        try:
            ab.validate_records(proxy,run['decisions'][:-1],10,10)
        except RuntimeError:
            passed(condition+'_bad_decision_count_rejected',True)
        else:
            raise AssertionError('bad count accepted')
        passed(condition+'_previous_actions_are_actual',all(np.array_equal(proxy.records[i]['previous_actual_action'],
            run['decisions'][i-1]['action']) for i in range(1,4)))
        passed(condition+'_source_student_is_3',trace.sources==[2]*10+[3]*10)
        if condition=='foot_velocity_zero':
            passed('mock_nonzero_probe_exactly_six_differences',np.array_equal(
                np.flatnonzero(proxy.records[2]['raw_X238']!=proxy.records[2]['actor_input_actual']),np.arange(189,195)))
    passed('immutable_original_checkpoint',sha256(checkpoint)==old_checkpoint)
    passed('immutable_model_weights',ab.weights_hash(model)==old_weights)
    passed('compiled_arrays_unchanged',all(np.array_equal(getattr(core.plant,k),v) for k,v in compiled.items()))
    passed('real_core_time_zero',float(core.d.time)==0.)
    report=dict(all_passed=all(results.values()),checks=results,check_count=len(results),
        source_hashes=ab.source_hashes(),checkpoint=dict(path=str(checkpoint),sha256=old_checkpoint,
            model_identity=payload['model_identity']),
        scope='real Core 0 physics steps; direct real checkpoint act; mocked shared handover 20ms per condition',
        real_gait_executed=False,training_executed=False,native_recording=False,
        claim='interface and intervention correctness only; no behavioral or complete-goal pass')
    atomic_json(ab.HERE/'ABLATION_CHECKS.json',report)
    print(json.dumps(dict(all_passed=report['all_passed'],checks=len(results),real_physics_steps=0,mocked_ms_per_condition=20)),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default=str(DEFAULT_CHECKPOINT))
    args=p.parse_args()
    check(args.checkpoint)
