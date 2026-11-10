"""Transparent synchronous neural observation + same-run physical/native audit."""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
import student_wave as sw
from physics import Core,NativeRecorder
from identity import HERE,capture,verify,verify_runtime,atomic_json,sha256


class NeuralObserver:
    def __init__(self,model,core):
        self.model=model;self.core=core;self.actor=model.actor;self.architecture=model.architecture
        self.rows=[]

    def act(self,obs,critic_obs=None,**kwargs):
        state=kwargs.get('state')
        before=state.detach().cpu().numpy().copy()
        output=self.model.act(obs,critic_obs,**kwargs)
        def a(value):
            return value.detach().cpu().numpy().copy() if torch.is_tensor(value) else np.asarray(value).copy()
        self.rows.append(dict(time=float(self.core.d.time),X238=a(obs),V255=a(critic_obs),
            hidden_before=before[0],hidden_after=a(output['state'])[0],
            policy_action=a(output['action'])[0],mean=a(output['mean'])[0],latent=a(output['latent'])[0],
            logstd=a(output['logstd'])[0],log_prob=float(output['log_prob'][0]),value=float(output['value'][0])))
        return output


def self_check(checkpoint):
    sw.seed_all(20261110)
    core=Core(adhesion=False);model,_=sw.load_student(checkpoint,core)
    if model.architecture!='gru238':raise ValueError('this synchronous audit requires explicit GRU state')
    frozen={k:v.clone() for k,v in model.state_dict().items()}
    qpos=core.d.qpos.copy();qvel=core.d.qvel.copy();target=core.actuator.target_rad.copy()
    x,v=sw.observations(core,np.zeros(42),9.6);state=model.actor.initial_state(1)
    baseline=model.act(x,v,state=state,deterministic=True)
    observer=NeuralObserver(model,core);result=observer.act(x,v,state=state,deterministic=True)
    exact={k:bool(torch.equal(baseline[k],result[k])) for k in baseline}
    assert all(exact.values())
    assert all(torch.equal(val,model.state_dict()[k]) for k,val in frozen.items())
    assert np.array_equal(qpos,core.d.qpos) and np.array_equal(qvel,core.d.qvel) and np.array_equal(target,core.actuator.target_rad)
    assert float(core.d.time)==0.
    assert observer.rows[0]['hidden_before'].shape==(128,) and observer.rows[0]['X238'].shape==(238,)
    return dict(all_passed=True,outputs_bitexact=exact,physical_steps=0,weights_unchanged=True,
        checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
        source_hashes={str(p):sha256(p) for p in sw.SOURCE_FILES+[Path(__file__).resolve()]})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--out')
    parser.add_argument('--native',action='store_true');parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();checkpoint=Path(args.checkpoint).resolve()
    checks=self_check(checkpoint)
    if args.check_only:
        atomic_json(HERE/'AUDIT_CHECKS.json',checks);print(json.dumps(checks),flush=True);return
    if not args.out:raise ValueError('actualaudit requires exclusive --out')
    sw.teacher_preflight();out=sw.exclusive_out(args.out);sw.seed_all(20261110)
    recorder=NativeRecorder(out/'native') if args.native else None
    core=Core(adhesion=False,recorder=recorder);model,payload=sw.load_student(checkpoint,core)
    before={k:v.clone() for k,v in model.state_dict().items()}
    observer=NeuralObserver(model,core)
    ident=capture(core,dict(kind='synchronous_neural_physical_audit',cli=vars(args),seed=20261110,
       settle_s=1.,teacher_s=4.8,autonomous_s=9.6,model_identity=payload['model_identity'],
       role_head_used=False,neural_observer='transparent actualruntime copies; returnsoriginaloutputdict'),
       extra_paths=sw.SOURCE_FILES+[Path(__file__).resolve(),checkpoint]+[Path(p) for p in payload['identity']['files']])
    atomic_json(out/'IDENTITY.json',ident);atomic_json(out/'AUDIT_CHECKS.json',checks)
    trace=sw.Trace(full=True)
    try:
        sw.settle(core,trace)
        teacher=sw.StaticTeacher(core,sw.teacher_config(3))
        run=sw.continuous_handover(core,teacher,observer,trace,4800,9600,control='student',sampled=False)
    finally:
        native=recorder.close() if recorder is not None else None
    decisions=run.pop('decisions')
    assert len(observer.rows)==len(decisions)==2880
    sources=np.asarray([d['source'] for d in decisions]);assert np.array_equal(sources,np.r_[np.full(960,2),np.full(1920,3)])
    arrays={k:np.asarray([row[k] for row in observer.rows]) for k in observer.rows[0]}
    arrays['control_source']=sources
    arrays['actually_applied_action']=np.asarray([d['action'] for d in decisions])
    arrays['actually_applied_action_is_policy']=sources==3
    assert np.array_equal(arrays['hidden_before'][1:],arrays['hidden_after'][:-1])
    assert np.array_equal(arrays['policy_action'][sources==3],arrays['actually_applied_action'][sources==3])
    assert np.array_equal(arrays['time'],np.asarray([d['time'] for d in decisions]))
    for k in ('X238','V255'):assert np.array_equal(arrays[k],np.asarray([d[k] for d in decisions]))
    assert all(torch.equal(val,model.state_dict()[key]) for key,val in before.items())
    assert all(np.isfinite(value).all() for value in arrays.values())
    neural_path=out/'neural_decisions.npz';np.savez_compressed(neural_path,**arrays)
    with np.load(neural_path,allow_pickle=False) as z:assert all(np.array_equal(z[k],value) for k,value in arrays.items())
    trace_info=trace.save(out/'trace_1ms.npz')
    decision_path=out/'decisions_5ms.npz';np.savez_compressed(decision_path,**{k:np.asarray([d[k] for d in decisions]) for k in decisions[0]})
    report=dict(control='student',sampled=False,settle=trace.summary(core,0),teacher_warmup=trace.summary(core,2),
       autonomous=trace.summary(core,3),**run,trace=trace_info,decisions=dict(path=str(decision_path),sha256=sha256(decision_path)),
       neural=dict(path=str(neural_path),sha256=sha256(neural_path),samples=2880,teacher_priming_samples=960,
           actual_neural_control_samples=1920,hidden_chain_exact=True,neural_actions_equal_applied_actions=True,
           runtime_recording=True,role_head_used=False,weights_unchanged=True),
       native=native,control_sources=sw.SOURCE,
       interpretation='Actualsynchronousneuralsamples+samephysicalrun; algorithmteacherprefixseparate, originalcompletewalkingacceptance requiresallgates/sourceclosure/nativechecks.')
    verify(ident);verify_runtime(payload['identity'],core);atomic_json(out/'RESULT.json',report)
    print(json.dumps(dict(all_gates=report['autonomous']['all_gates'],forward=report['autonomous']['forward_signed_native_per_s'],
        COM=report['autonomous']['com_height_min'],nonfoot=report['autonomous']['nonfoot_force_max_native'],
        slip=report['autonomous']['slip_ratio_max'],neural_rows=2880,native_rows=0 if native is None else native['native_samples'])),flush=True)

if __name__=='__main__':main()
