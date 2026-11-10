"""Bounded wiring checks only: tiny synthetic BC and 20ms actual handover.

Never runs collection, long BC, or walking evaluation. Temporary synthetic artifacts
are removed on exit. STUDENT_CHECKS.json records absolute executable source hashes.
"""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import student_wave as s
from identity import HERE, atomic_json, capture, sha256, verify


def main():
    s.seed_all(731)
    rows=[]
    def check(name,condition,**details):
        passed=bool(condition)
        rows.append(dict(name=name,passed=passed,**details))
        if not passed:
            raise AssertionError(name)

    def rejects(name,fn):
        try:
            fn()
        except (ValueError,RuntimeError,FileExistsError,KeyError):
            check(name,True)
        else:
            check(name,False)

    source=Path(s.__file__).resolve().read_text()
    tree=ast.parse(source)
    obs_function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='observations')
    obs_source=ast.get_source_segment(source,obs_function)
    check('actor observation uses actual obs238 and constant .075',
          'c.obs238(previous_actual_action, target_speed=TARGET_SPEED)' in obs_source and s.TARGET_SPEED==.075)
    check('no phase/reference/teacher input in observation builder',
          not any(word in obs_source for word in ('phase','reference','teacher','leg_order')))
    check('critic concatenates actual extras and remaining time only',
          "np.r_[x, c.critic_extras(), float(remaining_s)]" in obs_source)
    check('collector excludes landing tail targets',
          'if k < round(CYCLE_S*args.cycles*1000):' in source)
    check('reuse excludes provenance phase and landing tail from actor',
          "keep=phase < CYCLE_S*cycles-1e-10" in source and
          'np.savez_compressed(path,X238=x,Y42=y,V255=v)' in source)
    check('bounded tanh mean loss, no expert atanh or .95 clipping',
          'torch.tanh(bc_mean(model,x))-y' in source and 'atanh' not in source and '.95' not in source)
    check('declared unchanged physics step and zero force requirement',
          'mj_step' not in source and not any(token in source for token in ('qpos[:]', 'qvel[:]', 'qfrc_applied[:]','xfrc_applied[:]')))
    check('actor file contains no teacher implementation',
          'StaticTeacher' not in (HERE/'models.py').read_text())
    defaults=s.parser().parse_args(['collect','--out','/tmp/unused_student_checker'])
    check('CLI collection defaults four episodes two cycles',defaults.episodes==4 and defaults.cycles==2)
    reuse=s.parser().parse_args(['collect','--out','/tmp/unused_student_checker','--reuse-episode','a','--reuse-episode','b'])
    check('CLI accepts ordered repeated independent reuse',reuse.reuse_episode==['a','b'])
    defaults=s.parser().parse_args(['bc','--out','/tmp/unused_student_checker','--dataset','/tmp/unused'])
    check('CLI BC defaults 300 steps lr .001',defaults.steps==300 and defaults.lr==.001)
    defaults=s.parser().parse_args(['evaluate','--out','/tmp/unused_student_checker'])
    check('CLI evaluation defaults two autonomous cycles and no native trace',defaults.cycles==2 and not defaults.native)
    evidence=s.teacher_preflight()
    check('existing strict teacher reference gates and source closure',evidence['all_gates'])
    c=s.Core(adhesion=False)
    x,v=s.observations(c,np.zeros(42),.02)
    check('actual obs dimensions and float64',x.shape==(238,) and v.shape==(255,) and x.dtype==np.float64)
    check('critic exact actor prefix',np.array_equal(v[:238],x))
    rejects('reject wrong previous actual action dimension',lambda:s.observations(c,np.zeros(41),1.))
    rejects('reject out of bound action',lambda:s.bounded_action(np.full(42,1.01)))
    rejects('reject nonfinite action',lambda:s.bounded_action(np.full(42,np.nan)))

    synthetic_summary={}
    with tempfile.TemporaryDirectory(prefix='student-wave-check-') as temporary:
        root=Path(temporary)
        test_identity=capture(c,dict(purpose='synthetic wiring only; not gait evidence'),
                              extra_paths=s.SOURCE_FILES+[Path(__file__).resolve()])
        dataset=root/'synthetic';dataset.mkdir()
        entries=[]
        rng=np.random.default_rng(92)
        for episode in range(3):
            n=6
            target=rng.uniform(-1,1,(n,42));target[0,0]=1.;target[1,0]=-1.
            xx=rng.normal(0,.1,(n,238));xx[:,-1]=s.TARGET_SPEED
            xx[0,-43:-1]=0.;xx[1:,-43:-1]=target[:-1]
            vv=np.column_stack([xx,rng.normal(0,.1,(n,16)),np.arange(n,0,-1)*.005])
            path=dataset/f'episode_{episode:03d}.npz'
            np.savez_compressed(path,X238=xx,Y42=target,V255=vv)
            ipath=dataset/f'episode_{episode:03d}_IDENTITY.json'
            atomic_json(ipath,test_identity)
            rpath=dataset/f'episode_{episode:03d}_RESULT.json'
            atomic_json(rpath,dict(teacher_episode_condition_passed=True,kind='SYNTHETIC ONLY'))
            entries.append(dict(episode=episode,seed=episode+100,
                split='validation' if episode==2 else 'train',path=str(path),sha256=sha256(path),samples=n,
                identity_path=str(ipath),identity_sha256=sha256(ipath),report_path=str(rpath),report_sha256=sha256(rpath),
                trace=None,teacher_episode_condition_passed=True))
        meta=dict(format_version=1,collection_complete=True,source_hashes=s.source_hashes(),
            teacher_reference_evidence=evidence,episodes=entries,all_episode_conditions_passed=True,
            supervision=dict(samples_per_episode=6,landing_tail_excluded=True),kind='SYNTHETIC ONLY')
        mpath=dataset/'DATASET.json';atomic_json(mpath,meta)
        check('synthetic independent train/holdout metadata accepted',s.validate_dataset(dataset)['collection_complete'])
        bad=deepcopy(meta);bad['episodes'][-1]['split']='train';atomic_json(mpath,bad)
        rejects('reject wrong holdout split',lambda:s.validate_dataset(dataset))
        bad=deepcopy(meta);bad['episodes'][-1]['sha256']=bad['episodes'][0]['sha256'];atomic_json(mpath,bad)
        rejects('reject same exact trajectory train/validation',lambda:s.validate_dataset(dataset))
        bad=deepcopy(meta);bad['episodes'][-1]['seed']=bad['episodes'][0]['seed'];atomic_json(mpath,bad)
        rejects('reject repeated episode seed identity',lambda:s.validate_dataset(dataset))
        bad=deepcopy(meta);bad['source_hashes'][str(Path(s.__file__).resolve())]='0'*64;atomic_json(mpath,bad)
        rejects('reject dataset executable source drift',lambda:s.validate_dataset(dataset))
        bad=deepcopy(meta);bad['episodes'][0]['report_sha256']='0'*64;atomic_json(mpath,bad)
        rejects('reject dataset report digest drift',lambda:s.validate_dataset(dataset))
        bad=deepcopy(meta);bad['supervision']['landing_tail_excluded']=False;atomic_json(mpath,bad)
        rejects('reject metadata permitting supervised landing tail',lambda:s.validate_dataset(dataset))
        bad=deepcopy(meta);bad['all_episode_conditions_passed']=False;atomic_json(mpath,bad)
        rejects('reject inconsistent aggregate teacher condition',lambda:s.validate_dataset(dataset))
        atomic_json(mpath,meta)
        existing=root/'exclusive';s.exclusive_out(existing)
        rejects('CLI refuses existing output directory',lambda:s.exclusive_out(existing))
        for architecture in ('ff238','gru238'):
            model=s.new_model(architecture)
            xxx=torch.as_tensor(xx,dtype=torch.float64);yyy=torch.as_tensor(target,dtype=torch.float64)
            critic_before={k:t.clone() for k,t in model.critic.state_dict().items()}
            std_before=model.raw_logstd.detach().clone()
            loss=s.bounded_bc_loss(model,[(xxx,yyy)])
            loss.backward()
            check(architecture+' finite differentiable synthetic bounded loss',
                torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.actor.parameters()))
            check(architecture+' synthetic actor gradients nonzero',
                sum(float(p.grad.square().sum()) for p in model.actor.parameters())>0)
            check(architecture+' std initialized .05',torch.allclose(model.logstd().exp(),torch.full((42,),.05,dtype=torch.float64)))
            check(architecture+' BC critic and std excluded from loss',
                all(p.grad is None for p in model.critic.parameters()) and model.raw_logstd.grad is None)
            if architecture=='gru238':
                with torch.no_grad():
                    full=s.bc_mean(model,xxx)
                    state=model.actor.initial_state(1);pieces=[]
                    for tick in xxx:
                        mean,state=model.actor.step(tick[None],state);pieces.append(mean[0])
                    check('GRU full episode replay equals sequential recurrence',torch.equal(full,torch.stack(pieces)))
            out=root/architecture
            args=SimpleNamespace(out=str(out),dataset=str(dataset),architecture=architecture,steps=2,lr=.001,
                                 seed=123,allow_failed_teacher_episodes=False)
            with redirect_stdout(io.StringIO()):
                result=s.bc(args)
            payload=torch.load(out/'final.pt',map_location='cpu',weights_only=False)
            check(architecture+' BC complete path and exact compatible checkpoint reload',result['checkpoint_reload_exact'])
            check(architecture+' fresh PPO actor/critic optimizers',
                not payload['state']['actor']['state'] and not payload['state']['critic']['state'])
            check(architecture+' retained BC Adam state',bool(payload['extra']['bc_optimizer']['state']))
            check(architecture+' retained complete RNG/config/dataset closure',
                all(k in payload['rng'] for k in ('numpy','python','torch','cuda')) and
                payload['extra']['dataset_sha256']==sha256(mpath) and
                len(payload['extra']['collection_identities'])==3)
            reloaded,_=s.load_student(out/'final.pt',c)
            check(architecture+' deployment load verifies physical/data identity',reloaded.obs_dim==238 and reloaded.critic_obs_dim==255)
            action=reloaded.act(x,v,deterministic=True,state=reloaded.actor.initial_state(1) if architecture=='gru238' else None)['action']
            check(architecture+' deployment bounded42 action',action.shape==(1,42) and torch.isfinite(action).all() and action.abs().max()<=1.)
            check(architecture+' two tiny BC steps decrease synthetic training loss',
                payload['extra']['curve'][-1]['train_MSE']<payload['extra']['curve'][0]['train_MSE'])
            synthetic_summary[architecture]=dict(steps=2,initial=payload['extra']['curve'][0],final=payload['extra']['curve'][-1])

        # Exercise reused episode conversion with synthetic 2000 decisions only;
        # runtime identity is real, sample/report are explicitly synthetic fixtures.
        verified=root/'verified_fixture';verified.mkdir()
        nn=2000;phase=np.arange(nn)*.005
        sx=rng.normal(0,.1,(nn,238));sy=rng.uniform(-1,1,(nn,42));sx[:,-1]=s.TARGET_SPEED
        sx[0,-43:-1]=0.;sx[1:,-43:-1]=sy[:-1]
        sv=np.column_stack([sx,np.zeros((nn,16)),10.-phase])
        spath=verified/'BC_SAMPLES.npz'
        np.savez_compressed(spath,obs=sx,critic_obs=sv,action=sy,time=1.+phase,source_phase=phase)
        vi=capture(c,dict(config=s.teacher_config(2),
            pulse_seed=None,settle_s=1.,landing_tail_s=.4),extra_paths=s.SOURCE_FILES)
        atomic_json(verified/'IDENTITY.json',vi)
        atomic_json(verified/'RESULT.json',dict(all_gates=True,complete_per_leg={leg:1 for leg in s.LEGS},
            events=[],sample_file_sha256=sha256(spath),kind='SYNTHETIC ONLY'))
        converted=root/'converted';converted.mkdir()
        entry=s.import_verified_episode(verified,converted,2,123)
        with np.load(entry['path']) as z:
            check('reuse excludes 80 diagnostic tail decisions from BC',z['X238'].shape==(1920,238) and z['Y42'].shape==(1920,42))
            check('reuse phase provenance excluded from saved actor arrays',set(z.files)=={'X238','Y42','V255'} and np.array_equal(z['X238'],sx[:1920]))
        check('reuse binds all three original artifact hashes',len(entry['reused_original_files'])==3)
        # Distinct seeded pulse episode import and final validation ordering.
        vi['config']['pulse_seed']=20261120
        vi.pop('identity_sha256')
        import hashlib
        vi['identity_sha256']=hashlib.sha256(json.dumps(vi,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        atomic_json(verified/'IDENTITY.json',vi)
        entry2=s.import_verified_episode(verified,converted,2,124,episode=1,validation=True)
        check('reuse preserves independent physical pulse seed and holdout role',
              entry2['seed']==20261120 and entry2['split']=='validation' and entry2['perturbation']['duration_ms']==20)
        # End-to-end all-reuse collect path with synthetic files, zero physical ticks.
        pulsed=root/'verified_pulse_fixture';pulsed.mkdir()
        px=sx.copy();px[:,0]+=.001
        np.savez_compressed(pulsed/'BC_SAMPLES.npz',obs=px,critic_obs=np.column_stack([px,sv[:,238:]]),
                            action=sy,time=1.02+phase,source_phase=phase)
        atomic_json(pulsed/'IDENTITY.json',vi)
        atomic_json(pulsed/'RESULT.json',dict(all_gates=True,complete_per_leg={leg:1 for leg in s.LEGS},
            events=[],sample_file_sha256=sha256(pulsed/'BC_SAMPLES.npz'),kind='SYNTHETIC ONLY'))
        vi['config']['pulse_seed']=None
        vi.pop('identity_sha256')
        vi['identity_sha256']=hashlib.sha256(json.dumps(vi,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        atomic_json(verified/'IDENTITY.json',vi)
        import_args=SimpleNamespace(out=str(root/'all_reused'),episodes=2,cycles=2,seed=123,trace=False,
            reuse_episode=[str(verified),str(pulsed)])
        with patch.object(s,'settle',side_effect=AssertionError('reuse must not execute physical settle')),\
             patch.object(s,'StaticTeacher',side_effect=AssertionError('reuse must not execute teacher')),\
             redirect_stdout(io.StringIO()):
            imported=s.collect(import_args)
        check('all-reuse collect completes without physical replay',imported['collection_complete'] and len(imported['episodes'])==2)
        check('all-reuse last independent episode held out',
              [e['split'] for e in imported['episodes']]==['train','validation'] and imported['episodes'][1]['seed']==20261120)
        check('all-reuse dataset closure validates',s.validate_dataset(import_args.out)['all_episode_conditions_passed'])

    # Actual continuous run: exactly 20 physical milliseconds, no settle/gait trial.
    s.seed_all(731)
    c=s.Core(adhesion=False)
    teacher=s.StaticTeacher(c,s.teacher_config(1))
    model=s.new_model('gru238')
    trace=s.Trace(full=True)
    boundary=float(c.d.time)+.010
    real_teacher_action=teacher.action
    teacher_calls=[];actor_calls=[]
    def guarded_teacher(core,elapsed):
        if float(core.d.time)>=boundary-1e-10:
            raise AssertionError('teacher.action called after student handover')
        teacher_calls.append(float(core.d.time))
        return real_teacher_action(core,elapsed)
    real_act=model.act
    def observed_act(obs,critic_obs=None,**kwargs):
        expected=s.observations(c,obs[-43:-1],critic_obs[-1])[0]
        if not np.array_equal(obs,expected):
            raise AssertionError('actor observation is not current actual state')
        actor_calls.append(dict(time=float(c.d.time),previous=np.array(obs[-43:-1]),dim=len(obs)))
        return real_act(obs,critic_obs,**kwargs)
    teacher.action=guarded_teacher;model.act=observed_act
    def forbidden_reset(*args,**kwargs):
        raise AssertionError('physical reset forbidden during handover')
    with patch.object(c,'reset',forbidden_reset),patch.object(c,'reset_from',forbidden_reset):
        run=s.continuous_handover(c,teacher,model,trace,10,10)
    check('actual 20ms total physical ticks exactly',len(trace.rec)==20 and abs(c.d.time-.020)<1e-10)
    check('actual handover exact all state/history/hidden hashes',run['transition']['all_exact'] and
          run['transition']['before_sha256']==run['transition']['after_sha256'])
    check('actual GRU primed before every teacher command and continued',len(actor_calls)==4 and len(teacher_calls)==2)
    check('actual no teacher query after switch instrumented',all(t<boundary for t in teacher_calls))
    check('actual 5ms action hold and previous actual action handover',
          all(np.array_equal(trace.rows[j]['action'],trace.rows[5*(j//5)]['action']) for j in range(20)) and
          np.array_equal(actor_calls[2]['previous'],trace.rows[9]['action']))
    check('actual no external force / no adhesion throughout',all(r['external_body_force_max']==0 and
          r['external_generalized_force_max']==0 for r in trace.rows) and not c.adhesion_on)
    check('actual control sources exactly teacher then student',trace.sources==[s.SOURCE['teacher']]*10+[s.SOURCE['student']]*10)
    check('actual native timestep unchanged 12.5us',abs(c.plant.opt.timestep-12.5e-6)<1e-15)
    check('all compiled physical model arrays unchanged',all(
        hashlib.sha256(np.asarray(getattr(c.plant,key)).tobytes()).hexdigest()==old['sha256']
        for key,old in test_identity['compiled_arrays'].items()))
    check('actual actuator requested action equals bounded command',all(np.array_equal(r['action'],r['actual_requested_action']) for r in trace.rows))
    check('full trace required physical fields present',all(k in trace.rows[0] for k in
          ('qpos','qvel','com','loads','nonfoot','padz','speed','relativefoot','target','actualctrl','action','control_source')))

    # Mock tick only for branch instrumentation: no additional physical gait.
    class MockTrace:
        def __init__(self): self.actions=[]
        def tick(self,core,action,source): self.actions.append((np.array(action),source))
    class MockTracker:
        config=s.CycleConfig(timeout=4.)
        def measure(self,core): return dict(events=[])
        def finalize(self): return dict(events=[])
    for control in ('zero','teacher'):
        actor_count=[];teacher_count=[]
        m=s.new_model('gru238');t=s.StaticTeacher(c,s.teacher_config(3));mt=MockTrace()
        def mock_act(*args,**kwargs):
            actor_count.append(1);return dict(state=kwargs['state'])
        def mock_teacher(*args,**kwargs):
            teacher_count.append(1);return np.zeros(42),{}
        m.act=mock_act;t.action=mock_teacher
        with patch.object(s,'tracker',lambda core:MockTracker()):
            s.continuous_handover(c,t,m,mt,10,10,control=control)
        check(control+' control actor called only for warmup priming',len(actor_count)==2)
        check(control+' branch teacher query count exact',len(teacher_count)==(2 if control=='zero' else 4))
        check(control+' branch control-source flags',all(src==s.SOURCE['zero' if control=='zero' else 'teacher_replay'] for a,src in mt.actions[10:]))
    outputs=dict(status='PASS',purpose='S2/S5 tiny BC and 20ms wiring only; no walking evidence',
        command=[sys.executable,'-B',str(Path(__file__).resolve())],
        all_passed=all(row['passed'] for row in rows),passed=sum(row['passed'] for row in rows),total=len(rows),checks=rows,
        source_hashes={str(p.resolve()):sha256(p) for p in s.SOURCE_FILES+[Path(__file__).resolve()]},
        synthetic_bc=synthetic_summary,
        actual_physical=dict(duration_s=.020,decision_ms=5,teacher_ms=10,student_ms=10,
            transition=run['transition'],teacher_calls=len(teacher_calls),actor_calls=len(actor_calls),
            no_external_force=True,no_adhesion=True,native_dt_s=float(c.plant.opt.timestep)),
        not_executed=['full collection','long BC','full walking evaluation','native long recording'],
        interpretation='engineering checks only; parent schedules real independent teacher episodes and closed-loop evaluation')
    atomic_json(HERE/'STUDENT_CHECKS.json',outputs)
    print(json.dumps(dict(all_passed=outputs['all_passed'],passed=outputs['passed'],total=outputs['total'],
                         report=str(HERE/'STUDENT_CHECKS.json'))),flush=True)


if __name__=='__main__':
    main()
