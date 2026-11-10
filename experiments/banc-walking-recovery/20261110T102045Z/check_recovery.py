"""Fast mocked recovery contracts; no real gait, collection, or main training."""
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

import recovery_bc as r
import student_wave as sw
from identity import atomic_json, verify, sha256
from ppo import save_checkpoint


class MockCore:
    def __init__(self):
        self.d = SimpleNamespace(time=1.02,subtree_com=np.array([[0.,0.,1.45]]),qpos=np.array([0.,0.,1.4]))
        self.th=0
        self.position=np.zeros(42)
        self.history=[]
        self.expected_previous=None

    def obs238(self,previous_actual_action,target_speed):
        if self.expected_previous is not None:
            assert np.array_equal(previous_actual_action,self.expected_previous)
        x=np.zeros(238)
        x[:42]=self.position
        x[42:84]=self.history[-1] if self.history else 0
        x[-43:-1]=previous_actual_action
        x[-1]=target_speed
        return x

    def critic_extras(self):
        return np.zeros(16)


class MockTrace:
    def __init__(self,unsafe_tick=None):
        self.actions=[]
        self.sources=[]
        self.unsafe_tick=unsafe_tick

    def tick(self,c,action,source):
        self.actions.append(action.copy())
        self.sources.append(source)
        c.position += action*.001
        c.d.time += .001
        c.history.append(action.copy())
        c.expected_previous=action.copy()
        return dict(nonfoot=1. if len(self.actions)==self.unsafe_tick else 0.)


class MockTeacher:
    def __init__(self):
        self.calls=[]
        self.private=np.zeros(42)

    def action(self,c,phase):
        self.calls.append((c.position.copy(),float(c.d.time),phase,self.private.copy()))
        self.private += .001
        return np.clip(.1+self.private-c.position,-1,1),dict(root_balance_residual_norm_over_W=.001)


def must_reject(fn):
    try:
        fn()
    except (ValueError,RuntimeError,AssertionError,FileExistsError):
        return
    raise AssertionError('invalid input was accepted')


def manifest(paths):
    value=dict(files={str(Path(p).resolve()):dict(sha256=sha256(p)) for p in paths})
    value['identity_sha256']=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return value


def check_trajectory(architecture,beta):
    sw.seed_all(20261201)
    model=sw.new_model(architecture)
    c,teacher,trace=MockCore(),MockTeacher(),MockTrace()
    initial_pulse=np.full(42,.012)
    c.expected_previous=initial_pulse.copy()
    arrays,report=r.trajectory(c,teacher,model,trace,initial_pulse,np.random.default_rng(500),10,10,.01,beta=beta)
    r.validate_arrays(SimpleNamespaceAdapter(arrays))
    r.validate_arrays(SimpleNamespaceAdapter(arrays),expected_beta=beta)
    must_reject(lambda:r.validate_arrays(SimpleNamespaceAdapter(arrays),expected_beta=.1))
    assert report['stage_metrics']['mixed']['com_min']==1.45
    assert report['stage_metrics']['mixed']['actual_root_height_final']==1.4
    assert report['stage_metrics']['mixed']['nonfoot_max']==0.
    assert report['stage_metrics']['prefix']['physical_ms']==10
    assert report['teacher_queries']==4 and report['physical_ms']==20 and report['state_resets']==0
    assert len(trace.actions)==20 and len(teacher.calls)==4
    assert np.array_equal(arrays['stage'],[0,0,1,1])
    assert np.array_equal(arrays['beta'],[1.,1.,beta,beta])
    expected=np.clip(beta*arrays['Y42'][2:]+(1-beta)*arrays['student_action'][2:]+arrays['gaussian_noise'][2:],-1,1)
    assert np.array_equal(arrays['executed_action'][2:],expected)
    wrong={k:v.copy() for k,v in arrays.items()}
    wrong['executed_action'][2,0]+=.001
    must_reject(lambda:r.validate_arrays(SimpleNamespaceAdapter(wrong)))
    wrong={k:v.copy() for k,v in arrays.items()}
    wrong['beta'][3]=.2
    must_reject(lambda:r.validate_arrays(SimpleNamespaceAdapter(wrong)))
    for bad in (-.01,1.01,float('nan')):
        must_reject(lambda:r.trajectory(MockCore(),MockTeacher(),model,MockTrace(),np.zeros(42),np.random.default_rng(1),10,10,beta=bad))
    assert np.array_equal(arrays['X238'][0,-43:-1],initial_pulse)
    for i in range(4):
        assert np.array_equal(arrays['X238'][i,:42],teacher.calls[i][0])
        assert np.array_equal(arrays['Y42'][i],.1+(i+1)*.001-teacher.calls[i][0])
        assert np.array_equal(teacher.calls[i][3],np.full(42,i*.001))
        assert teacher.calls[i][2]==i*.005
        assert abs(teacher.calls[i][1]-arrays['time'][i])<1e-12
        assert all(np.array_equal(a,arrays['executed_action'][i]) for a in trace.actions[i*5:i*5+5])
    with torch.no_grad():
        replay=torch.tanh(sw.bc_mean(model,torch.as_tensor(arrays['X238'],dtype=torch.float64))).numpy()
    assert np.allclose(replay,arrays['student_action'],rtol=0,atol=1e-16)
    if architecture=='gru238':
        # Pure mixed loss must flow backward THROUGH the stored prefix inputs.
        x=torch.tensor(arrays['X238'],dtype=torch.float64,requires_grad=True)
        means=sw.bc_mean(model,x)
        means[2:].square().sum().backward()
        assert float(x.grad[:2].abs().sum())>0
    broken={k:v.copy() for k,v in arrays.items()}
    broken['X238'][2,-43:-1]=0
    must_reject(lambda:r.validate_arrays(SimpleNamespaceAdapter(broken)))
    unsafe,ur=r.trajectory(MockCore(),MockTeacher(),model,MockTrace(7),np.zeros(42),np.random.default_rng(9),10,10,.01)
    assert ur['early_stop'] is not None and ur['physical_ms']==7 and list(unsafe['applied_ticks'])==[5,2]
    r.validate_arrays(SimpleNamespaceAdapter(unsafe))
    low=MockCore();low.d.subtree_com[0,2]=1.19
    _,lr=r.trajectory(low,MockTeacher(),model,MockTrace(),np.zeros(42),np.random.default_rng(1),10,10,.01)
    assert lr['physical_ms']==1 and lr['early_stop']['com_z']==1.19
    return dict(decisions=4,mocked_physical_ms=20,teacher_queries=4,previous_actual_continuous=True,
        teacher_current_state=True,holds_5ms=True,hidden_replay_exact=True,safety_stops_checked=True,
        prefix_gradient_reaches_prefix=architecture=='gru238')


class SimpleNamespaceAdapter(dict):
    @property
    def files(self):
        return list(self.keys())


def make_data(root,model):
    base_dir=root/'base';base_dir.mkdir()
    rec_dir=root/'recovery';rec_dir.mkdir()
    checkpoint=root/'parent.pt'
    trainer=sw.make_trainer(model)
    # Deliberately populate parent BC/PPO optimizer momentum and counters.
    optimizer=torch.optim.Adam(model.actor.parameters(),lr=.001)
    x=torch.zeros((3,238),dtype=torch.float64)
    loss=sw.bc_mean(model,x).square().sum()+sum(p.square().sum() for p in model.actor.parameters())*.0001
    loss.backward();optimizer.step()
    trainer.actor_optimizer.zero_grad(set_to_none=True)
    loss=sum(p.square().sum() for p in model.actor_parameters())
    loss.backward();trainer.actor_optimizer.step()
    trainer.updates=17
    parent_identity=manifest([r.HERE/'recovery_bc.py'])
    save_checkpoint(checkpoint,trainer,parent_identity,extra=dict(kind='synthetic_parent',bc_optimizer=optimizer.state_dict()))
    metas=[]
    for directory,recovery in [(base_dir,False),(rec_dir,True)]:
        entries=[]
        for i in range(3):
            seed=(20261201 if recovery else 20)+i
            arrays,report=r.trajectory(MockCore(),MockTeacher(),model,MockTrace(),np.full(42,i*.001),np.random.default_rng(seed),10,10,.01)
            if not recovery:
                rng=np.random.default_rng(seed)
                arrays['X238'][:,:20]+=rng.normal(0,.01,(4,20))
                arrays['Y42']+=rng.normal(0,.001,(4,42))
            ep=directory/f'episode_{i:03d}.npz'
            np.savez_compressed(ep,**arrays)
            ip=directory/f'episode_{i:03d}_IDENTITY.json';atomic_json(ip,parent_identity)
            rp=directory/f'episode_{i:03d}_RESULT.json';atomic_json(rp,report)
            entries.append(dict(episode=i,seed=seed,split='validation' if i==2 else 'train',path=str(ep),
                sha256=sha256(ep),samples=4,identity_path=str(ip),identity_sha256=sha256(ip),
                report_path=str(rp),report_sha256=sha256(rp),trace=None,mixed_samples=2,early_stop=None))
        meta=dict(format_version=1,kind='bounded_recovery_aggregation' if recovery else 'synthetic_base',
            collection_complete=True,config=dict(beta=.5),source_hashes=r.source_hashes(),teacher_reference_evidence=dict(files={}),
            checkpoint_path=str(checkpoint),checkpoint_sha256=sha256(checkpoint),episodes=entries,
            all_episode_conditions_passed=True)
        atomic_json(directory/'DATASET.json',meta);metas.append(meta)
    return base_dir,rec_dir,checkpoint,metas


def check_training(root,architecture):
    root.mkdir()
    sw.seed_all(20261211)
    model=sw.new_model(architecture)
    base_dir,rec_dir,checkpoint,(base,recovery)=make_data(root,model)
    groups=r.partition(base,recovery)
    b=r.tensors(groups['base_train']);rec=r.tensors(groups['recovery_train'])
    short=[(rec[0][0][:1],rec[0][1][:1])]
    actual=r.aggregation_loss(model,b,short)
    expected=sw.bounded_bc_loss(model,b+short)
    assert torch.allclose(actual,expected,rtol=0,atol=1e-17)
    assert [e['episode'] for e in groups['base_train']]==[0,1]
    assert [e['episode'] for e in groups['recovery_train']]==[0,1]
    assert [e['episode'] for e in groups['base_validation']]==[2]
    assert [e['episode'] for e in groups['recovery_validation']]==[2]
    overlap=json.loads(json.dumps(recovery));overlap['episodes'][0]['seed']=20
    must_reject(lambda:r.partition(base,overlap))
    one=json.loads(json.dumps(recovery));one['episodes']=one['episodes'][:1]
    must_reject(lambda:r.partition(base,one))
    # Real validator against four-decision mocked episodes and real SHA manifests.
    with patch.object(r,'PREFIX_MS',10),patch.object(r,'MIXED_MS',10):
        r.validate_recovery(rec_dir)
        target=rec_dir/'DATASET.json';original=target.read_text()
        wrong=json.loads(original);wrong['config']['beta']=.9
        atomic_json(target,wrong)
        try:
            must_reject(lambda:r.validate_recovery(rec_dir))
        finally:
            target.write_text(original)
        checkpoint_bytes=checkpoint.read_bytes()
        try:
            for kind in ('weight','rng','bytes'):
                payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
                if kind=='weight':
                    next(iter(payload['state']['model'].values())).add_(.01)
                    torch.save(payload,checkpoint)
                elif kind=='rng':
                    payload['rng']['torch']=torch.Generator().manual_seed(7).get_state()
                    torch.save(payload,checkpoint)
                else:
                    checkpoint.write_bytes(checkpoint_bytes+b'tamper')
                must_reject(lambda:r.validate_recovery(rec_dir))
                checkpoint.write_bytes(checkpoint_bytes)
        finally:
            checkpoint.write_bytes(checkpoint_bytes)
        args=SimpleNamespace(checkpoint=str(checkpoint),base_dataset=str(base_dir),recovery=str(rec_dir),
            out=str(root/'trained'),steps=2,lr=.0003,seed=20261211,mode='train')
        def fake_capture(core,config,extra_paths):
            result=manifest(extra_paths);result['config']=config
            result.pop('identity_sha256')
            result['identity_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            return result
        with ExitStack() as stack:
            stack.enter_context(patch.object(sw,'Core',lambda **kw:MockCore()))
            stack.enter_context(patch.object(sw,'verify_runtime',lambda ident,core:verify(ident)))
            stack.enter_context(patch.object(sw,'validate_dataset',lambda path:base))
            stack.enter_context(patch.object(r,'capture',fake_capture))
            result=r.train(args)
            loaded,payload=r.load_recovery_student(result['checkpoint'],MockCore())
            assert payload['extra']['kind']=='recovery_bc'
            assert payload['extra']['weights']['base']==payload['extra']['weights']['recovery']==1
            assert payload['state']['updates']==0
            assert payload['state']['actor']['state']=={} and payload['state']['critic']['state']=={}
            assert payload['extra']['bc_optimizer']['state']
            assert all(v['step']==2 for v in payload['extra']['bc_optimizer']['state'].values())
            assert payload['extra']['warm_start']=='model_only'
            assert torch.equal(payload['rng']['torch'],torch.Generator().manual_seed(args.seed).get_state())
            identity_paths=payload['identity']['files']
            for directory in (base_dir,rec_dir):
                assert str(directory/'DATASET.json') in identity_paths
            for meta in (base,recovery):
                for e in meta['episodes']:
                    assert e['path'] in identity_paths and e['identity_path'] in identity_paths and e['report_path'] in identity_paths
            parent=torch.load(checkpoint,map_location='cpu',weights_only=False)
            for k,v in parent['state']['model'].items():
                if not k.startswith('actor.'):
                    assert torch.equal(v,loaded.state_dict()[k])
            curve=payload['extra']['curve']
            assert all(np.isfinite(v) for row in curve for k,v in row.items())
            assert curve[0]['base_train_MSE']!=curve[-1]['base_train_MSE']
            must_reject(lambda:sw.exclusive_out(root/'trained'))
            # Evaluator's unmodified student loader verifies recovery metadata via
            # the checkpoint identity, independently of our wrapper.
            target=rec_dir/'DATASET.json';original=target.read_text()
            target.write_text(original+' ')
            try:
                must_reject(lambda:sw.load_student(result['checkpoint'],MockCore()))
            finally:
                target.write_text(original)
            assert not list((root/'trained').glob('*.tmp'))
    return dict(synthetic_steps=2,finite_loss_and_gradients=True,whole_episode_holdouts=True,
        fresh_BC_Adam=True,fresh_PPO_states=True,declared_fresh_rng=True,model_only_warm_start=True,
        checkpoint_atomic_reload=True,weight_rng_checkpoint_tamper_rejected=True,
        metadata_expected_beta_rejection=True,recovery_metadata_tamper_rejected_by_original_loader=True,
        full_base_and_recovery_identity_closure=True,output_overwrite_rejected=True)


def main():
    torch.set_num_threads(1)
    checks={}
    for architecture in ('ff238','gru238'):
        for beta in (.5,.9,0.,1.):
            checks[architecture+'_trajectory_beta_'+str(beta)]=check_trajectory(architecture,beta)
    with tempfile.TemporaryDirectory(prefix='recovery_contracts_') as tmp:
        for architecture in ('ff238','gru238'):
            checks[architecture+'_training']=check_training(Path(tmp)/architecture,architecture)
    p=r.parser()
    args=p.parse_args(['collect','--checkpoint','/tmp/model.pt','--out','/tmp/new'])
    assert args.episodes==3 and args.cycles==2 and args.beta==.5 and args.noise_std==.01 and args.seed==20261221
    assert p.parse_args(['collect','--checkpoint','/tmp/model.pt','--out','/tmp/new','--episodes','2']).episodes==2
    must_reject(lambda:r.absolute('relative'))
    result=dict(all_passed=True,checks=checks,source_hashes={**r.source_hashes(),str(r.HERE/'check_recovery.py'):sha256(r.HERE/'check_recovery.py')},
        scope='mocked 10ms teacher + 10ms mixed only; synthetic two-update FF/GRU; no real physics step, gait collection, or main training',
        limitation='engineering contracts only; neural autonomy and teacher corrective recovery effectiveness NOT_ESTABLISHED')
    atomic_json(r.HERE/'RECOVERY_CHECKS.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    main()
