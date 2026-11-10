"""Fast mocked burst contracts; no real physics steps or main training."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import burst_recovery as b
import recovery_bc as r
import student_wave as sw
from identity import atomic_json, verify, sha256
from ppo import save_checkpoint


class MockCore:
    def __init__(self):
        self.d = SimpleNamespace(time=1.02,subtree_com=np.array([[0.,0.,1.45]]),
            qpos=np.array([0.,0.,1.4,1.,0.,0.,0.]))
        self.th=0;self.position=np.zeros(42);self.expected_previous=None
        self.initial_heading=np.array([1.,0.,0.])

    def obs238(self,previous_actual_action,target_speed):
        if self.expected_previous is not None:
            assert np.array_equal(previous_actual_action,self.expected_previous)
        x=np.zeros(238);x[:42]=self.position
        x[-43:-1]=previous_actual_action;x[-1]=target_speed
        return x

    def critic_extras(self):
        return np.zeros(16)


class MockTrace:
    def __init__(self,unsafe_tick=None):
        self.actions=[];self.sources=[];self.rec=[];self.unsafe_tick=unsafe_tick

    def tick(self,c,action,source):
        self.actions.append(action.copy());self.sources.append(source)
        c.position+=action*.001;c.d.time+=.001
        # Actual body responds to executed action; no stage reset or oracle overwrite.
        c.d.qpos[2]+=float(action[0])*.00001
        c.d.subtree_com[0,0]+=.0001
        c.expected_previous=action.copy()
        nonfoot=1. if len(self.actions)==self.unsafe_tick else 0.
        self.rec.append((c.d.time,c.d.subtree_com[0].copy(),np.ones(6),nonfoot,np.full(6,.01)))
        return dict(nonfoot=nonfoot)


class MockTeacher:
    def __init__(self):
        self.calls=[];self.private=np.zeros(42)

    def action(self,c,phase):
        self.calls.append((c.position.copy(),float(c.d.time),phase,self.private.copy()))
        self.private+=.001
        return np.clip(.1+self.private-c.position,-1,1),dict(root_balance_residual_norm_over_W=.001,
            planned_stance=[True]*6,ik_error_norm_per_leg=[.002]*6)


def must_reject(fn):
    try:
        fn()
    except (ValueError,RuntimeError,AssertionError,FileExistsError):
        return
    raise AssertionError('invalid input was accepted')


def manifest(paths,config=None):
    value=dict(files={str(Path(p).resolve()):dict(sha256=sha256(p)) for p in paths})
    if config is not None:
        value['config']=config
    value['identity_sha256']=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return value


def check_trajectory(architecture):
    sw.seed_all(20261301)
    model=sw.new_model(architecture)
    c,teacher,trace=MockCore(),MockTeacher(),MockTrace()
    previous=np.full(42,.012);c.expected_previous=previous.copy()
    arrays,report=b.trajectory(c,teacher,model,trace,previous,np.random.default_rng(50),20,20,.01,5,5)
    b.validate_arrays(arrays,20,20,5,5)
    assert np.array_equal(arrays['stage'],[0,0,0,0,1,2,1,2])
    assert np.array_equal(arrays['beta'],[1,1,1,1,0,1,0,1])
    assert report['physical_ms']==40 and report['teacher_queries']==8 and report['state_resets']==0
    assert len(trace.actions)==40 and len(teacher.calls)==8
    assert report['stage_counts']==dict(prefix=4,student_burst=2,teacher_recovery=2)
    for name,count in [('prefix',20),('student_burst',10),('teacher_recovery',10),('whole_postprefix',20)]:
        assert report['stage_metrics'][name]['physical_ms']==count
        assert report['stage_metrics'][name]['physical_metrics']['sample_count']==count
    assert np.array_equal(arrays['X238'][0,-43:-1],previous)
    for i in range(8):
        assert np.array_equal(arrays['X238'][i,:42],teacher.calls[i][0])
        assert np.array_equal(arrays['Y42'][i],.1+(i+1)*.001-teacher.calls[i][0])
        assert np.array_equal(teacher.calls[i][3],np.full(42,i*.001))
        assert teacher.calls[i][2]==i*.005
        assert abs(teacher.calls[i][1]-arrays['time'][i])<1e-12
        assert all(np.array_equal(a,arrays['executed_action'][i]) for a in trace.actions[i*5:i*5+5])
        assert all(s==(sw.SOURCE['student'] if arrays['stage'][i]==1 else sw.SOURCE['teacher']) for s in trace.sources[i*5:i*5+5])
    assert np.all(arrays['gaussian_noise'][arrays['stage']!=1]==0)
    assert np.any(arrays['gaussian_noise'][arrays['stage']==1]!=0)
    assert np.array_equal(arrays['executed_action'][arrays['stage']!=1],arrays['Y42'][arrays['stage']!=1])
    with torch.no_grad():
        replay=torch.tanh(sw.bc_mean(model,torch.as_tensor(arrays['X238'],dtype=torch.float64))).numpy()
    assert np.allclose(replay,arrays['student_action'],rtol=0,atol=1e-16)
    if architecture=='gru238':
        x=torch.tensor(arrays['X238'],dtype=torch.float64,requires_grad=True)
        sw.bc_mean(model,x)[4:].square().sum().backward()
        assert float(x.grad[:4].abs().sum())>0
    boundaries=report['boundary_comparisons']
    assert len(boundaries)==2
    assert boundaries[0]['before_burst']['time_s']<boundaries[0]['after_burst']['time_s']<boundaries[0]['after_recovery']['time_s']
    assert boundaries[0]['after_recovery']==boundaries[1]['before_burst']
    assert not boundaries[-1]['terminal_partial']
    assert report['correction_statistics']['root_height_error_absolute_reduction']['count']==2
    assert boundaries[0]['before_burst']['root_height']!=boundaries[0]['after_burst']['root_height']
    for key,index in [('stage',(5,)),('beta',(4,)),('time',(4,)),('teacher_phase_s',(4,)),
            ('executed_action',(4,0)),('gaussian_noise',(5,0)),('X238',(4,195)),('applied_ticks',(3,))]:
        wrong={k:v.copy() for k,v in arrays.items()};wrong[key][index]+=1
        must_reject(lambda:b.validate_arrays(wrong,20,20,5,5))
    wrong={k:v.copy() for k,v in arrays.items()};wrong['time'][2]=np.nan
    must_reject(lambda:b.validate_arrays(wrong,20,20,5,5))
    must_reject(lambda:b.validate_arrays(arrays,20,20,10,5))
    for unsafe_tick in (7,22,27):
        unsafe,ur=b.trajectory(MockCore(),MockTeacher(),model,MockTrace(unsafe_tick),np.zeros(42),np.random.default_rng(9),20,20,.01,5,5)
        assert ur['early_stop'] is not None and ur['physical_ms']==unsafe_tick
        assert unsafe['applied_ticks'][-1]==2
        b.validate_arrays(unsafe,20,20,5,5)
    low=MockCore();low.d.subtree_com[0,2]=1.19
    _,lr=b.trajectory(low,MockTeacher(),model,MockTrace(),np.zeros(42),np.random.default_rng(1),20,20,.01,5,5)
    assert lr['physical_ms']==1 and lr['early_stop']['com_z']==1.19
    partial,pr=b.trajectory(MockCore(),MockTeacher(),model,MockTrace(),np.zeros(42),np.random.default_rng(1),20,20,.01,5,10)
    b.validate_arrays(partial,20,20,5,10)
    assert pr['boundary_comparisons'][-1]['terminal_partial']
    for bad in (0,1,5.5,True):
        must_reject(lambda:b.durations(20,20,bad,5))
    for noise in (-1,float('nan')):
        must_reject(lambda:b.trajectory(MockCore(),MockTeacher(),model,MockTrace(),np.zeros(42),np.random.default_rng(1),20,20,noise,5,5))
    return dict(mocked_physical_ms=40,teacher_queries=8,pure_student_bursts=True,teacher_only_recovery=True,
        exact_stage_schedule=True,noise_only_burst=True,actual_state_labels_and_teacher_history=True,
        previous_actual_action_continuous=True,full_hidden_replay=True,prefix_BPTT=architecture=='gru238',
        measured_boundary_comparison=True,stage_trace_selection=True,prefix_and_burst_recovery_safety=True,
        partial_last_segment=True,action_time_phase_stage_tamper_rejected=True)


def make_data(root,model):
    base_dir=root/'base';base_dir.mkdir();rec_dir=root/'burst';rec_dir.mkdir()
    checkpoint=root/'parent.pt';trainer=sw.make_trainer(model)
    optimizer=torch.optim.Adam(model.actor.parameters(),lr=.001)
    loss=sw.bc_mean(model,torch.zeros((3,238),dtype=torch.float64)).square().sum()+sum(p.square().sum() for p in model.actor.parameters())*.0001
    loss.backward();optimizer.step()
    trainer.actor_optimizer.zero_grad(set_to_none=True)
    sum(p.square().sum() for p in model.actor_parameters()).backward();trainer.actor_optimizer.step();trainer.updates=17
    parent_identity=manifest([b.HERE/'burst_recovery.py'])
    save_checkpoint(checkpoint,trainer,parent_identity,extra=dict(kind='synthetic_parent',bc_optimizer=optimizer.state_dict()))
    metas=[]
    for directory,recovery in [(base_dir,False),(rec_dir,True)]:
        config=dict(mode='collect',checkpoint=str(checkpoint),out=str(directory),episodes=3,
            burst_ms=5,recover_ms=5,noise_std=.01,seed=20261301 if recovery else 20,trace=False)
        entries=[]
        for i in range(3):
            seed=config['seed']+i
            arrays,report=b.trajectory(MockCore(),MockTeacher(),model,MockTrace(),np.full(42,i*.001),np.random.default_rng(seed),20,20,.01,5,5)
            if not recovery:
                rng=np.random.default_rng(seed);arrays['X238'][:,:20]+=rng.normal(0,.01,(8,20))
                arrays['Y42']+=rng.normal(0,.001,(8,42))
            ep=directory/f'episode_{i:03d}.npz';np.savez_compressed(ep,**arrays)
            ip=directory/f'episode_{i:03d}_IDENTITY.json'
            atomic_json(ip,manifest([checkpoint]+[Path(p) for p in b.source_hashes()],dict(cli=config,episode=i,seed=seed,prefix_ms=20,mixed_ms=20)))
            rp=directory/f'episode_{i:03d}_RESULT.json';atomic_json(rp,report)
            entries.append(dict(episode=i,seed=seed,split='validation' if i==2 else 'train',path=str(ep),
                sha256=sha256(ep),samples=8,identity_path=str(ip),identity_sha256=sha256(ip),
                report_path=str(rp),report_sha256=sha256(rp),trace=None,mixed_samples=4,early_stop=None))
        meta=dict(format_version=1,kind='bounded_burst_recovery' if recovery else 'synthetic_base',collection_complete=True,
            config=config,source_hashes=b.source_hashes(),teacher_reference_evidence=dict(files={str(b.HERE/'burst_recovery.py'):sha256(b.HERE/'burst_recovery.py')}),
            checkpoint_path=str(checkpoint),checkpoint_sha256=sha256(checkpoint),episodes=entries,all_episode_conditions_passed=True)
        atomic_json(directory/'DATASET.json',meta);metas.append(meta)
    return base_dir,rec_dir,checkpoint,metas


def check_training(root,architecture):
    root.mkdir();sw.seed_all(20261311)
    model=sw.new_model(architecture)
    base_dir,rec_dir,checkpoint,(base,recovery)=make_data(root,model)
    groups=r.partition(base,recovery)
    base_tensors=r.tensors(groups['base_train']);rec=r.tensors(groups['recovery_train'])
    short=[(rec[0][0][:1],rec[0][1][:1])]
    assert torch.allclose(r.aggregation_loss(model,base_tensors,short),sw.bounded_bc_loss(model,base_tensors+short),rtol=0,atol=1e-17)
    assert [e['episode'] for e in groups['base_train']]==[0,1]
    assert [e['episode'] for e in groups['recovery_train']]==[0,1]
    assert [e['episode'] for e in groups['base_validation']]==[2]
    assert [e['episode'] for e in groups['recovery_validation']]==[2]
    overlap=json.loads(json.dumps(recovery));overlap['episodes'][0]['seed']=20
    must_reject(lambda:r.partition(base,overlap))
    incomplete=json.loads(json.dumps(recovery));incomplete['episodes'][0]['mixed_samples']=0
    must_reject(lambda:r.partition(base,incomplete))
    with patch.object(b,'PREFIX_MS',20),patch.object(b,'MIXED_MS',20),patch.object(sw,'teacher_preflight',lambda:recovery['teacher_reference_evidence']):
        b.validate_burst(rec_dir)
        target=rec_dir/'DATASET.json';original=target.read_text()
        for key,value in [('burst_ms',10),('recover_ms',10),('noise_std',.02),('seed',42),('checkpoint','/tmp/other.pt')]:
            wrong=json.loads(original);wrong['config'][key]=value;atomic_json(target,wrong)
            try:
                must_reject(lambda:b.validate_burst(rec_dir))
            finally:
                target.write_text(original)
        for kind in ('checkpoint_substitution','removed_teacher_evidence'):
            wrong=json.loads(original)
            if kind=='checkpoint_substitution':
                wrong['checkpoint_path']=str(b.HERE/'burst_recovery.py')
                wrong['checkpoint_sha256']=sha256(b.HERE/'burst_recovery.py')
            else:
                wrong['teacher_reference_evidence']['files']={}
            atomic_json(target,wrong)
            try:
                must_reject(lambda:b.validate_burst(rec_dir))
            finally:
                target.write_text(original)
        checkpoint_bytes=checkpoint.read_bytes()
        try:
            for kind in ('weight','rng','bytes'):
                payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
                if kind=='weight':
                    next(iter(payload['state']['model'].values())).add_(.01);torch.save(payload,checkpoint)
                elif kind=='rng':
                    payload['rng']['torch']=torch.Generator().manual_seed(7).get_state();torch.save(payload,checkpoint)
                else:
                    checkpoint.write_bytes(checkpoint_bytes+b'tamper')
                must_reject(lambda:b.validate_burst(rec_dir));checkpoint.write_bytes(checkpoint_bytes)
        finally:
            checkpoint.write_bytes(checkpoint_bytes)
        args=SimpleNamespace(checkpoint=str(checkpoint),base_dataset=str(base_dir),recovery=str(rec_dir),
            out=str(root/'trained'),steps=2,lr=.0003,seed=20261311,mode='train')
        def fake_capture(core,config,extra_paths):
            # BURST_CHECKS is generated only after these checks finish. Its identity
            # path is present in production; excluding it here avoids circularity.
            return manifest([p for p in extra_paths if Path(p).name!='BURST_CHECKS.json'],config)
        with ExitStack() as stack:
            stack.enter_context(patch.object(sw,'Core',lambda **kw:MockCore()))
            stack.enter_context(patch.object(sw,'verify_runtime',lambda ident,core:verify(ident)))
            stack.enter_context(patch.object(sw,'validate_dataset',lambda path:base))
            stack.enter_context(patch.object(b,'capture',fake_capture))
            result=b.train(args)
            loaded,payload=b.load_burst_student(result['checkpoint'],MockCore())
            assert payload['extra']['kind']=='burst_bc' and payload['extra']['warm_start']=='model_only'
            assert payload['extra']['weights']['base']==payload['extra']['weights']['recovery']==1
            assert payload['state']['updates']==0
            assert payload['state']['actor']['state']=={} and payload['state']['critic']['state']=={}
            assert payload['extra']['bc_optimizer']['state']
            assert all(v['step']==2 for v in payload['extra']['bc_optimizer']['state'].values())
            assert torch.equal(payload['rng']['torch'],torch.Generator().manual_seed(args.seed).get_state())
            paths=payload['identity']['files']
            for directory in (base_dir,rec_dir):
                assert str(directory/'DATASET.json') in paths
            for meta in (base,recovery):
                for e in meta['episodes']:
                    assert all(e[k] in paths for k in ('path','identity_path','report_path'))
            parent=torch.load(checkpoint,map_location='cpu',weights_only=False)
            for k,v in parent['state']['model'].items():
                if not k.startswith('actor.'):
                    assert torch.equal(v,loaded.state_dict()[k])
            curve=payload['extra']['curve']
            assert all(np.isfinite(v) for row in curve for k,v in row.items())
            assert curve[0]['base_train_MSE']!=curve[-1]['base_train_MSE']
            must_reject(lambda:sw.exclusive_out(root/'trained'))
            original=target.read_text();target.write_text(original+' ')
            try:
                must_reject(lambda:sw.load_student(result['checkpoint'],MockCore()))
            finally:
                target.write_text(original)
    return dict(synthetic_steps=2,per_sample_equal_weight=True,whole_episode_holdouts=True,
        failed_prefix_excluded_from_training_only=True,fresh_BC_Adam=True,fresh_PPO_optimizers_counters=True,
        fresh_declared_rng=True,model_only_warm_start=True,checkpoint_reload_exact=True,
        source_command_metadata_checkpoint_weight_rng_tamper_rejected=True,
        original_student_loader_checks_full_identity=True,base_and_burst_full_identity_closure=True)


def main():
    torch.set_num_threads(1)
    checks={architecture+'_trajectory':check_trajectory(architecture) for architecture in ('ff238','gru238')}
    with tempfile.TemporaryDirectory(prefix='burst_contracts_') as tmp:
        for architecture in ('ff238','gru238'):
            checks[architecture+'_training']=check_training(Path(tmp)/architecture,architecture)
    args=b.parser().parse_args(['collect','--checkpoint','/tmp/model.pt','--out','/tmp/new'])
    assert (args.episodes,args.burst_ms,args.recover_ms,args.noise_std,args.seed)==(3,50,200,.01,20261301)
    must_reject(lambda:b.absolute('relative'))
    result=dict(all_passed=True,checks=checks,source_hashes=b.source_hashes(),
        scope='mocked 20ms prefix + 20ms postprefix; synthetic two-update FF/GRU only; no real physics step or collection',
        limitation='engineering contracts only; actual teacher correction and neural autonomous walking NOT_ESTABLISHED')
    atomic_json(b.HERE/'BURST_CHECKS.json',result)
    b.check_preflight()
    print(json.dumps(dict(all_passed=True,checks=list(checks),source_files=len(result['source_hashes']))),flush=True)
    return result


if __name__=='__main__':
    main()
