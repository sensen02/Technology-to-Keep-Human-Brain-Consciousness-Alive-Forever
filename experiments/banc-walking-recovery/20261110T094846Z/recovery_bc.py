"""Bounded DART/DAgger-style physical trajectory aggregation, NOT neural autonomy.

Only the mixed-stage actor is observation-only. A virtual, state-querying algorithm
teacher supplies labels AND 90% of mixed commands; it is not a proven recovery
oracle. Prefix+mixed labels train full contiguous sequences, with zero GRU hidden.
No execution on import; CLI creates exclusive output directories.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import student_wave as sw
from identity import HERE, atomic_json, capture, verify, sha256
from ppo import save_checkpoint, load_checkpoint

OWN_SOURCES = [HERE/'recovery_bc.py']
BETA = .9
DECISION_MS = 5
PREFIX_MS = 4800
MIXED_MS = 4800
INTERPRETATION = ('virtual algorithm teacher prefix and teacher/student mixed physical commands; '
    'NOT neural autonomy; corrective labels are current-state teacher targets, '
    'not evidence of a perfect recovery controller; native units, absolute SI NOT_ESTABLISHED')


def absolute(path):
    p = Path(path).expanduser()
    if not p.is_absolute():
        raise ValueError('absolute path required: '+str(p))
    return p.resolve()


def source_hashes():
    return {str(p.resolve()): sha256(p) for p in sw.SOURCE_FILES+OWN_SOURCES}


def checked_files(files):
    for p, digest in files.items():
        if not Path(p).is_absolute() or sha256(p) != digest:
            raise ValueError('input identity drift: '+p)


def check_preflight():
    checks = json.loads((HERE/'RECOVERY_CHECKS.json').read_text())
    if not checks['all_passed']:
        raise ValueError('recovery checks failed')
    # This snapshot need not contain STUDENT_CHECKS.json: our checked source
    # closure includes the unchanged student/model/physics dependencies directly.
    checked_files(checks['source_hashes'])


def actor_action(model, x, state):
    # No critic extras, reference clock, teacher phase, or privileged input here.
    with torch.no_grad():
        xt = torch.as_tensor(x.copy(), dtype=torch.float64)
        if model.architecture == 'gru238':
            mean, state = model.actor.step(xt[None, :], state)
            action = torch.tanh(mean)[0]
        else:
            action = torch.tanh(model.actor(xt))
    return sw.bounded_action(action.detach().cpu().numpy()), state


def trajectory(c, teacher, model, trace, previous, rng, prefix_ms=PREFIX_MS,
               mixed_ms=MIXED_MS, noise_std=.01):
    """Shared instrumentable path: continuous plant, teacher history, and GRU state.

    Safety checks occur after each physical 1ms tick. An unsafe final decision may
    have fewer than five ticks, explicitly recorded; never execute a recovery reset.
    """
    if min(prefix_ms,mixed_ms) <= 0 or prefix_ms % 5 or mixed_ms % 5:
        raise ValueError('positive five-ms durations required')
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError('finite nonnegative noise required')
    previous = sw.bounded_action(previous)
    state = model.actor.initial_state(1) if model.architecture == 'gru238' else None
    rows, early_stop = [], None
    for k in range(0,prefix_ms+mixed_ms,DECISION_MS):
        mixed = k >= prefix_ms
        x,_ = sw.observations(c,previous,(prefix_ms+mixed_ms-k)*.001)
        student,state = actor_action(model,x,state)
        # Query at this exact actual physical state on every decision. The SAME
        # teacher instance retains private IK/qdes history across the handover.
        expert,diag = teacher.action(c,k*.001)
        expert = sw.bounded_action(expert)
        noise = rng.normal(0,noise_std,42) if mixed else np.zeros(42)
        selected = BETA*expert+(1-BETA)*student if mixed else expert.copy()
        executed = sw.bounded_action(np.clip(selected+noise,-1,1))
        row = dict(X238=x.copy(),Y42=expert,student_action=student,
            executed_action=executed,beta=BETA if mixed else 1.,
            time=float(c.d.time),teacher_phase_s=k*.001,stage=int(mixed),
            gaussian_noise=noise,applied_ticks=0,
            model_root_residual=float(diag.get('root_balance_residual_norm_over_W',0.)))
        rows.append(row)
        for _ in range(DECISION_MS):
            m = trace.tick(c,executed,sw.SOURCE['teacher_replay'] if mixed else sw.SOURCE['teacher'])
            row['applied_ticks'] += 1
            com_z = float(c.d.subtree_com[c.th][2])
            if com_z < 1.2 or float(m['nonfoot']) > 0:
                early_stop = dict(time_s=float(c.d.time),com_z=com_z,
                    nonfoot=float(m['nonfoot']),decision=len(rows)-1,
                    reason='collection safety only: COM<1.2 or nonfoot>0; acceptance unchanged')
                break
        previous = executed.copy()
        if early_stop is not None:
            break
    arrays = {key: np.asarray([r[key] for r in rows]) for key in rows[0]}
    return arrays, dict(early_stop=early_stop,decisions=len(rows),
        prefix_decisions=int(np.sum(arrays['stage']==0)),mixed_decisions=int(np.sum(arrays['stage']==1)),
        physical_ms=int(arrays['applied_ticks'].sum()),teacher_queries=len(rows),
        state_resets=0,hidden_reset='zero only at episode start',
        model_unsupported=bool(np.any(arrays['model_root_residual']>.02)),
        model_root_residual_max=float(np.max(arrays['model_root_residual'])), 
        teacher_history='one continuous instance; current-state query each decision',
        teacher_limit='static balance/pose correction and private target history; not a validated recovery oracle',
        model_support='static model residual is diagnostic only; no optimized-force writes to physics',
        interpretation=INTERPRETATION)


def collect(args):
    checkpoint = absolute(args.checkpoint)
    if args.episodes < 2 or args.cycles != 2 or args.beta != BETA:
        raise ValueError('require >=2 episodes, exactly 2 cycles, fixed beta=.9')
    if not math.isfinite(args.noise_std) or args.noise_std < 0:
        raise ValueError('finite nonnegative noise required')
    absolute(args.out)
    evidence = sw.teacher_preflight()
    out = sw.exclusive_out(args.out)
    entries = []
    for episode in range(args.episodes):
        seed = args.seed+episode
        sw.seed_all(seed)
        c = sw.Core(adhesion=False)
        model,payload = sw.load_student(checkpoint,c)
        ident = capture(c,dict(mode='recovery_collect',cli=vars(args),episode=episode,
            seed=seed,teacher=sw.teacher_config(2),decision_ms=5,settle_ms=1000,
            prefix_ms=PREFIX_MS,mixed_ms=MIXED_MS,interpretation=INTERPRETATION),
            extra_paths=sw.SOURCE_FILES+OWN_SOURCES+[checkpoint]+[Path(p) for p in evidence['files']])
        ip = out/f'episode_{episode:03d}_IDENTITY.json'
        atomic_json(ip,ident)
        trace = sw.Trace(full=args.trace)
        sw.settle(c,trace)
        # Every new episode gets an independent 20ms motor pulse, including ep0.
        previous,pulse = sw.perturb(c,trace,1,seed)
        teacher = sw.StaticTeacher(c,sw.teacher_config(2))
        arrays,report = trajectory(c,teacher,model,trace,previous,np.random.default_rng(seed+1000000),
            noise_std=args.noise_std)
        ep = out/f'episode_{episode:03d}.npz'
        np.savez_compressed(ep,**arrays)
        report.update(perturbation=pulse,teacher=teacher.metadata,
            checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
            model_identity=payload['model_identity'])
        rp = out/f'episode_{episode:03d}_RESULT.json'
        atomic_json(rp,report)
        trace_info = trace.save(out/f'episode_{episode:03d}_trace.npz') if args.trace else None
        verify(ident)
        entries.append(dict(episode=episode,seed=seed,split='validation' if episode==args.episodes-1 else 'train',
            path=str(ep),sha256=sha256(ep),samples=len(arrays['stage']),
            identity_path=str(ip),identity_sha256=sha256(ip),report_path=str(rp),report_sha256=sha256(rp),
            mixed_samples=report['mixed_decisions'],early_stop=report['early_stop'],trace=trace_info))
        meta = dict(format_version=1,kind='bounded_recovery_aggregation',config=vars(args),
            source_hashes=source_hashes(),checkpoint_path=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
            teacher_reference_evidence=evidence,episodes=entries,collection_complete=len(entries)==args.episodes,
            split_contract='last independent episode held out; no intra-episode random split',
            label_coverage='full prefix + mixed; no loss mask; full zero-hidden GRU sequence without detach',
            actor_inputs='raw X238 actual observation and previous executed action; no clock/reference/phase',
            noise='independent Gaussian on all 42 normalized selected rates in mixed stage, clip [-1,1], hold 5ms',
            safety='COM<1.2 or nonfoot>0 stops collection only; acceptance unchanged',
            interpretation=INTERPRETATION)
        atomic_json(out/'DATASET.json',meta)
        print(json.dumps(dict(episode=episode,split=entries[-1]['split'],**report)),flush=True)
    validate_recovery(out)
    return meta


def validate_arrays(z):
    n = len(z['stage'])
    vector = ('X238','Y42','student_action','executed_action','gaussian_noise')
    for k in vector:
        if z[k].shape != (n,238 if k=='X238' else 42):
            raise ValueError('recovery array dimensions: '+k)
    for k in ('stage','beta','time','teacher_phase_s','applied_ticks','model_root_residual'):
        if z[k].shape != (n,):
            raise ValueError('recovery scalar dimensions: '+k)
    if n < 1 or any(not np.isfinite(z[k]).all() for k in z.files):
        raise ValueError('empty/nonfinite recovery sequence')
    for k in ('Y42','student_action','executed_action'):
        if np.any(np.abs(z[k])>1+1e-12):
            raise ValueError('unbounded recovery action')
    stage = z['stage']
    if not np.isin(stage,[0,1]).all() or stage[0]!=0 or np.any(np.diff(stage)<0):
        raise ValueError('prefix must precede mixed stage')
    if not np.array_equal(z['beta'],np.where(stage==0,1.,BETA)):
        raise ValueError('fixed beta mismatch')
    if not np.array_equal(z['executed_action'][stage==0],z['Y42'][stage==0]):
        raise ValueError('prefix must execute actual expert')
    if not np.all(z['gaussian_noise'][stage==0]==0):
        raise ValueError('prefix must be noise free')
    expected = np.clip(z['beta'][:,None]*z['Y42']+(1-z['beta'][:,None])*z['student_action']+z['gaussian_noise'],-1,1)
    if not np.allclose(z['executed_action'],expected,rtol=0,atol=2e-16):
        raise ValueError('executed mixture mismatch')
    if n>1 and not np.array_equal(z['X238'][1:,-43:-1],z['executed_action'][:-1]):
        raise ValueError('previous actual action mismatch')
    if not np.all(z['applied_ticks'][:-1]==5) or not 1<=z['applied_ticks'][-1]<=5:
        raise ValueError('five-ms hold mismatch')
    if not np.allclose(z['teacher_phase_s'],np.arange(n)*.005,rtol=0,atol=1e-12):
        raise ValueError('teacher phase provenance mismatch')
    if n>1 and not np.allclose(np.diff(z['time']),.005,rtol=0,atol=1e-10):
        raise ValueError('actual decision time continuity mismatch')
    if not np.allclose(z['X238'][:,-1],sw.TARGET_SPEED,rtol=0,atol=1e-15):
        raise ValueError('target speed mismatch')


def validate_recovery(directory):
    path = absolute(directory)/'DATASET.json'
    meta = json.loads(path.read_text())
    if meta['format_version']!=1 or meta['kind']!='bounded_recovery_aggregation' or not meta['collection_complete']:
        raise ValueError('incomplete recovery dataset')
    checked_files(meta['source_hashes'])
    checked_files(meta['teacher_reference_evidence']['files'])
    checked_files({meta['checkpoint_path']:meta['checkpoint_sha256']})
    entries = meta['episodes']
    if len(entries)<2 or [e['split'] for e in entries]!=['train']*(len(entries)-1)+['validation']:
        raise ValueError('recovery requires independent final episode holdout')
    for key in ('episode','seed','sha256','path'):
        if len({e[key] for e in entries})!=len(entries):
            raise ValueError('duplicate recovery episode: '+key)
    for e in entries:
        checked_files({e[k]:e[d] for k,d in (('path','sha256'),('identity_path','identity_sha256'),('report_path','report_sha256'))})
        verify(json.loads(Path(e['identity_path']).read_text()))
        report = json.loads(Path(e['report_path']).read_text())
        if e.get('trace'):
            checked_files({e['trace']['path']:e['trace']['sha256']})
        with np.load(e['path'],allow_pickle=False) as z:
            validate_arrays(z)
            n = len(z['stage'])
            if n!=e['samples'] or int(np.sum(z['stage']==1))!=e['mixed_samples']:
                raise ValueError('sample count mismatch')
            if not np.array_equal(z['stage'],(np.arange(n)>=PREFIX_MS//5).astype(int)):
                raise ValueError('one-cycle prefix mismatch')
            if n>(PREFIX_MS+MIXED_MS)//5:
                raise ValueError('bounded duration exceeded')
            if report['early_stop'] is None and n!=(PREFIX_MS+MIXED_MS)//5:
                raise ValueError('unreported incomplete trajectory')
            if report['early_stop']!=e['early_stop'] or report['physical_ms']!=int(z['applied_ticks'].sum()):
                raise ValueError('physical report mismatch')
    return meta


def dataset_closure(directory,meta):
    paths = [absolute(directory)/'DATASET.json']
    paths += [Path(p) for p in meta['source_hashes']]
    paths += [Path(p) for p in meta['teacher_reference_evidence']['files']]
    if 'checkpoint_path' in meta:
        paths.append(Path(meta['checkpoint_path']))
    for e in meta['episodes']:
        paths += [Path(e[k]) for k in ('path','identity_path','report_path')]
        paths += [Path(p) for p in e.get('reused_original_files',{})]
        if e.get('trace'):
            paths.append(Path(e['trace']['path']))
        manifest = json.loads(Path(e['identity_path']).read_text())
        paths += [Path(p) for p in manifest['files']]
    return sorted(set(paths))


def partition(base,recovery):
    # No random sample split. Fail on overlap rather than silently leaking holdouts.
    if {e['seed'] for e in base['episodes']} & {e['seed'] for e in recovery['episodes']}:
        raise ValueError('base/recovery seeds must be independent')
    if {e['sha256'] for e in base['episodes']} & {e['sha256'] for e in recovery['episodes']}:
        raise ValueError('base/recovery trajectory overlap')
    groups = {}
    for name,meta in [('base',base),('recovery',recovery)]:
        for split in ('train','validation'):
            entries = [e for e in meta['episodes'] if e['split']==split]
            if not entries:
                raise ValueError('missing episode partition')
            groups[name+'_'+split] = entries
    if any(e['mixed_samples']<1 for e in recovery['episodes']):
        raise ValueError('each recovery episode must reach mixed stage before training')
    return groups


def tensors(entries):
    episodes = []
    for e in entries:
        with np.load(e['path'],allow_pickle=False) as z:
            episodes.append(tuple(torch.as_tensor(z[k].copy(),dtype=torch.float64) for k in ('X238','Y42')))
    return episodes


def aggregation_loss(model,base,recovery):
    # Explicit per-sample weights base=1, recovery=1, including shorter unsafe
    # episodes. Do not accidentally upweight short recovery sources/episodes.
    nb = sum(x.shape[0] for x,y in base)
    nr = sum(x.shape[0] for x,y in recovery)
    return (nb*sw.bounded_bc_loss(model,base)+nr*sw.bounded_bc_loss(model,recovery))/(nb+nr)


def load_recovery_student(path,core):
    model,payload = sw.load_student(absolute(path),core)
    extra = payload.get('extra') or {}
    if extra.get('kind')=='recovery_bc':
        checked_files(extra['recovery_closure'])
        validate_recovery(Path(extra['recovery_dataset_path']).parent)
    return model,payload


def train(args):
    if args.steps<1 or not math.isfinite(args.lr) or args.lr<=0:
        raise ValueError('positive training steps/lr required')
    checkpoint = absolute(args.checkpoint)
    base_dir,recovery_dir = absolute(args.base_dataset),absolute(args.recovery)
    absolute(args.out)
    base = sw.validate_dataset(base_dir)
    recovery = validate_recovery(recovery_dir)
    if not base['all_episode_conditions_passed']:
        raise ValueError('base teacher episode failed')
    groups = partition(base,recovery)
    out = sw.exclusive_out(args.out)
    sw.seed_all(args.seed)
    c = sw.Core(adhesion=False)
    model,parent = load_recovery_student(checkpoint,c)
    # load_student validates full parent but never restores its RNG. Copy ONLY the
    # model into a fresh trainer; inherited PPO/BC Adam/counters are discarded.
    mi = parent['model_identity']
    fresh = sw.ActorCritic(mi['architecture'],critic_obs_dim=255,dtype=torch.float64,
        logstd_bounds=mi['logstd_bounds'])
    fresh.load_state_dict(model.state_dict(),strict=True)
    model = fresh
    # Constructor consumed RNG; explicitly restart the declared training RNG.
    sw.seed_all(args.seed)
    optimizer = torch.optim.Adam(model.actor.parameters(),lr=args.lr)
    episodes = {k:tensors(v) for k,v in groups.items()}
    closure = dataset_closure(base_dir,base)+dataset_closure(recovery_dir,recovery)
    closure += [checkpoint]+[Path(p) for p in parent['identity']['files']]
    closure = sorted(set(closure))
    ident = capture(c,dict(mode='recovery_bc',cli=vars(args),parent_sha256=sha256(checkpoint),
        warm_start='MODEL ONLY; fresh BC Adam, PPO states/counters and declared RNG seed',
        loss='sample-weighted bounded-action MSE; base sample weight=1, recovery sample weight=1',
        label_coverage='all prefix + mixed labels; GRU full sequence from zero without detach',
        partitions={k:[dict(episode=e['episode'],seed=e['seed'],path=e['path']) for e in v] for k,v in groups.items()}),
        extra_paths=sw.SOURCE_FILES+OWN_SOURCES+closure)
    atomic_json(out/'IDENTITY.json',ident)
    untouched = {k:v.clone() for k,v in model.state_dict().items() if not k.startswith('actor.')}
    curve=[]
    for step in range(args.steps+1):
        if step:
            optimizer.zero_grad(set_to_none=True)
            loss = aggregation_loss(model,episodes['base_train'],episodes['recovery_train'])
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite recovery loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.actor.parameters()):
                raise FloatingPointError('nonfinite recovery gradient')
            optimizer.step()
        with torch.no_grad():
            row = dict(step=step,**{k+'_MSE':float(sw.bounded_bc_loss(model,v)) for k,v in episodes.items()})
        curve.append(row)
        atomic_json(out/'CURVE.json',dict(curve=curve,interpretation='supervised MSE only; not closed-loop gait evidence'))
        if step==0 or step==args.steps or step%25==0:
            print(json.dumps(row),flush=True)
    if any(not torch.equal(v,model.state_dict()[k]) for k,v in untouched.items()):
        raise RuntimeError('BC changed critic or exploration parameters')
    trainer = sw.make_trainer(model)
    verify(ident)
    dp = base_dir/'DATASET.json'
    rp = recovery_dir/'DATASET.json'
    recovery_closure = {str(p):sha256(p) for p in dataset_closure(recovery_dir,recovery)}
    manifests = [json.loads(Path(e['identity_path']).read_text()) for m in (base,recovery) for e in m['episodes']]
    save_checkpoint(out/'final.pt',trainer,ident,extra=dict(kind='recovery_bc',
        bc_optimizer=optimizer.state_dict(),bc_config=vars(args),curve=curve,
        dataset_path=str(dp),dataset_sha256=sha256(dp),dataset_metadata=base,
        recovery_dataset_path=str(rp),recovery_dataset_sha256=sha256(rp),recovery_closure=recovery_closure,
        recovery_metadata=recovery,collection_identities=manifests,
        parent_checkpoint=str(checkpoint),parent_checkpoint_sha256=sha256(checkpoint),
        warm_start='model_only',rng_seed=args.seed,ppo_optimizers='fresh',actor_only_bc=True,
        weights=dict(base=1,recovery=1,normalization='weighted sum squared errors / (42 * weighted decision count)'),
        interpretation=INTERPRETATION))
    check = sw.ActorCritic(mi['architecture'],critic_obs_dim=255,dtype=torch.float64,logstd_bounds=mi['logstd_bounds'])
    load_checkpoint(out/'final.pt',sw.make_trainer(check),ident,restore_rng=False)
    if any(not torch.equal(v,check.state_dict()[k]) for k,v in model.state_dict().items()):
        raise RuntimeError('checkpoint reload mismatch')
    payload = torch.load(out/'final.pt',map_location='cpu',weights_only=False)
    if payload['state']['actor']['state'] or payload['state']['critic']['state']:
        raise RuntimeError('PPO optimizer was not fresh')
    result = dict(checkpoint=str(out/'final.pt'),checkpoint_sha256=sha256(out/'final.pt'),
        final=curve[-1],checkpoint_reload_exact=True,ppo_optimizer_entries=0,
        parent_checkpoint_sha256=sha256(checkpoint),identity_sha256=ident['identity_sha256'],
        base_metadata_sha256=sha256(dp),recovery_metadata_sha256=sha256(rp),
        closed_loop='NOT_EVALUATED',interpretation=INTERPRETATION)
    atomic_json(out/'RESULT.json',result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode',required=True)
    co = sub.add_parser('collect')
    co.add_argument('--checkpoint',required=True);co.add_argument('--out',required=True)
    co.add_argument('--episodes',type=int,default=3);co.add_argument('--cycles',type=int,default=2)
    co.add_argument('--beta',type=float,default=.9);co.add_argument('--noise-std',type=float,default=.01)
    co.add_argument('--seed',type=int,default=20261201);co.add_argument('--trace',action='store_true')
    tr = sub.add_parser('train')
    for flag in ('checkpoint','base-dataset','recovery','out'):
        tr.add_argument('--'+flag,required=True)
    tr.add_argument('--steps',type=int,default=1000);tr.add_argument('--lr',type=float,default=.0003)
    tr.add_argument('--seed',type=int,default=20261211)
    return p


def main():
    args = parser().parse_args()
    check_preflight()
    return dict(collect=collect,train=train)[args.mode](args)


if __name__=='__main__':
    main()
