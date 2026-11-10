"""Pure-student short bursts with teacher-only scheduled recovery, bounded data only.

The algorithm teacher is queried on EVERY actual decision, including student
bursts, so its private history advances continuously. It is an adaptation/data
schedule, not an autonomous actor or a proven recovery oracle. No execution on
import. Raw observation-only FF238/GRU238; unchanged geometry and physics.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import recovery_bc as r
import student_wave as sw
from identity import HERE, atomic_json, capture, verify, sha256
from ppo import save_checkpoint, load_checkpoint

PREFIX_MS = 4800
MIXED_MS = 4800
DECISION_MS = 5
OWN_SOURCES = [HERE/'burst_recovery.py', HERE/'check_burst.py']
INTERPRETATION = ('bounded pure-student bursts and teacher-only adaptation/recovery commands; '
    'NOT autonomous walking evidence; actual correction effectiveness must be measured; '
    'static root residual is diagnostic, not recovery oracle proof; native units, SI NOT_ESTABLISHED')
absolute = r.absolute
checked_files = r.checked_files


def source_hashes():
    return {str(p.resolve()): sha256(p) for p in sw.SOURCE_FILES+[HERE/'recovery_bc.py']+OWN_SOURCES}


def check_preflight():
    checks = json.loads((HERE/'BURST_CHECKS.json').read_text())
    if not checks['all_passed'] or checks['source_hashes'] != source_hashes():
        raise ValueError('burst checks failed or source closure mismatch')
    checked_files(checks['source_hashes'])
    return sw.teacher_preflight()


def durations(prefix_ms, mixed_ms, burst_ms, recover_ms):
    for value in (prefix_ms, mixed_ms, burst_ms, recover_ms):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0 or value % 5:
            raise ValueError('positive integer five-ms durations required')


def schedule(k, prefix_ms, burst_ms, recover_ms):
    return 0 if k < prefix_ms else (1 if (k-prefix_ms) % (burst_ms+recover_ms) < burst_ms else 2)


def body_measurement(c, reference):
    """Measured body departure from initial teacher pose; excludes forward travel.

    Height error and quaternion angle are separate native/radian diagnostics,
    not an invented scalar gait gate or the teacher's static force residual.
    """
    root = float(c.d.qpos[2])
    result = dict(time_s=float(c.d.time), root_height=root,
        root_height_error=root-reference['root_height'],
        com_z=float(c.d.subtree_com[c.th][2]))
    if len(c.d.qpos) >= 7:
        q = np.asarray(c.d.qpos[3:7], float)
        q0 = reference['quaternion']
        cosine = abs(float(np.dot(q,q0)/(np.linalg.norm(q)*np.linalg.norm(q0))))
        result['root_orientation_error_rad'] = float(2*np.arccos(np.clip(cosine,0,1)))
    return result


def bounded_stats(values):
    a = np.asarray(values, float)
    return dict(count=int(a.size), mean=float(a.mean()), min=float(a.min()),
        max=float(a.max()), p50=float(np.quantile(a,.5)), p95=float(np.quantile(a,.95))) if a.size else dict(count=0)


def trajectory(c, teacher, model, trace, previous, rng, prefix_ms=PREFIX_MS,
               mixed_ms=MIXED_MS, noise_std=.01, burst_ms=50, recover_ms=200):
    durations(prefix_ms,mixed_ms,burst_ms,recover_ms)
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError('finite nonnegative burst noise required')
    previous = sw.bounded_action(previous)
    state = model.actor.initial_state(1) if model.architecture == 'gru238' else None
    reference = dict(root_height=float(c.d.qpos[2]), quaternion=np.asarray(c.d.qpos[3:7],float).copy())
    trace_start = len(trace.rec) if hasattr(trace,'rec') else 0
    stage_metrics, tick_stages, rows, boundaries = {}, [], [], []
    stance_errors = {stage:[] for stage in (0,1,2)}
    early_stop, current_boundary, last_stage = None, None, None
    names = {0:'prefix',1:'student_burst',2:'teacher_recovery'}
    for k in range(0,prefix_ms+mixed_ms,DECISION_MS):
        stage = schedule(k,prefix_ms,burst_ms,recover_ms)
        # Boundary measurements are actual plant states, without an extra teacher query.
        if stage != last_stage:
            point = body_measurement(c,reference)
            if stage == 1:
                if current_boundary is not None:
                    current_boundary['after_recovery'] = point
                current_boundary = dict(cycle=len(boundaries),before_burst=point)
                boundaries.append(current_boundary)
            elif stage == 2 and current_boundary is not None:
                current_boundary['after_burst'] = point
            last_stage = stage
        x,_ = sw.observations(c,previous,(prefix_ms+mixed_ms-k)*.001)
        student,state = r.actor_action(model,x,state)
        expert,diag = teacher.action(c,k*.001)
        expert = sw.bounded_action(expert)
        if 'ik_error_norm_per_leg' in diag and 'planned_stance' in diag:
            errors = np.asarray(diag['ik_error_norm_per_leg'],float)
            stance = np.asarray(diag['planned_stance'],bool)
            if errors.shape != (6,) or stance.shape != (6,) or not np.isfinite(errors).all():
                raise ValueError('invalid actual stance IK diagnostics')
            stance_errors[stage].extend(errors[stance].tolist())
        beta = 0. if stage == 1 else 1.
        noise = rng.normal(0,noise_std,42) if stage == 1 else np.zeros(42)
        executed = sw.bounded_action(np.clip((student if stage == 1 else expert)+noise,-1,1))
        row = dict(X238=x.copy(),Y42=expert,student_action=student,
            executed_action=executed,gaussian_noise=noise,time=float(c.d.time),
            teacher_phase_s=k*.001,beta=beta,applied_ticks=0,stage=stage,
            model_root_residual=float(diag.get('root_balance_residual_norm_over_W',0.)))
        rows.append(row)
        for _ in range(DECISION_MS):
            m = trace.tick(c,executed,sw.SOURCE['student'] if stage == 1 else sw.SOURCE['teacher'])
            row['applied_ticks'] += 1
            tick_stages.append(stage)
            com = float(c.d.subtree_com[c.th][2]); root = float(c.d.qpos[2])
            metrics = stage_metrics.setdefault(names[stage],dict(physical_ms=0,com_min=com,
                nonfoot_max=0.,actual_root_height_min=root,actual_root_height_max=root))
            metrics['physical_ms'] += 1
            metrics['com_min'] = min(metrics['com_min'],com)
            metrics['nonfoot_max'] = max(metrics['nonfoot_max'],float(m['nonfoot']))
            metrics['actual_root_height_min'] = min(metrics['actual_root_height_min'],root)
            metrics['actual_root_height_max'] = max(metrics['actual_root_height_max'],root)
            metrics['actual_root_height_final'] = root
            if com < 1.2 or float(m['nonfoot']) > 0:
                early_stop = dict(time_s=float(c.d.time),com_z=com,nonfoot=float(m['nonfoot']),
                    decision=len(rows)-1,reason='collection safety only: COM<1.2 or nonfoot>0; acceptance unchanged')
                break
        previous = executed.copy()
        if early_stop is not None:
            break
    if current_boundary is not None:
        key = 'after_burst' if last_stage == 1 else 'after_recovery'
        current_boundary[key] = body_measurement(c,reference)
        current_boundary['terminal_partial'] = bool(early_stop or (mixed_ms % (burst_ms+recover_ms)))
    arrays = {key:np.asarray([row[key] for row in rows]) for key in rows[0]}
    for stage,name in names.items():
        mask = arrays['stage']==stage
        if np.any(mask):
            stage_metrics[name]['model_root_residual'] = bounded_stats(arrays['model_root_residual'][mask])
            stage_metrics[name]['actual_stance_foot_IK_error_native'] = bounded_stats(stance_errors[stage])
    if hasattr(trace,'rec'):
        records = trace.rec[trace_start:]
        if len(records) != len(tick_stages):
            raise ValueError('trace/tick count mismatch')
        for stage,name in names.items():
            rec = [v for v,s in zip(records,tick_stages) if s==stage]
            if len(rec)>=2:
                stage_metrics[name]['physical_metrics'] = sw.summarise(rec,len(rec)*80,initial_heading=c.initial_heading)
                stage_metrics[name]['physical_metrics_scope'] = ('selected actual 1ms stage rows; timestamps retain gaps; '
                    'speed spans first-to-last selected row; pooled stage diagnostics, not contiguous gait acceptance')
        post = [v for v,s in zip(records,tick_stages) if s!=0]
        if len(post)>=2:
            stage_metrics['whole_postprefix'] = dict(physical_ms=len(post),
                physical_metrics=sw.summarise(post,len(post)*80,initial_heading=c.initial_heading))
    corrections = {}
    for key in ('root_height_error','root_orientation_error_rad'):
        deltas = [abs(b['after_burst'][key])-abs(b['after_recovery'][key]) for b in boundaries
            if 'after_burst' in b and 'after_recovery' in b and not b.get('terminal_partial',False) and key in b['after_burst']]
        corrections[key+'_absolute_reduction'] = bounded_stats(deltas)
    return arrays,dict(early_stop=early_stop,decisions=len(rows),stage_metrics=stage_metrics,
        prefix_decisions=int(np.sum(arrays['stage']==0)),mixed_decisions=int(np.sum(arrays['stage']!=0)),
        stage_counts={name:int(np.sum(arrays['stage']==stage)) for stage,name in names.items()},
        physical_ms=int(arrays['applied_ticks'].sum()),teacher_queries=len(rows),state_resets=0,
        hidden_reset='zero only at episode start; actual observations throughout all stages',
        teacher_history='one continuous instance; queried at actual state every decision including burst labels',
        teacher_limit='adaptation schedule; correction effectiveness NOT_ESTABLISHED until physically measured',
        boundary_comparisons=boundaries,correction_statistics=corrections,
        body_error_reference='teacher initialization root height and quaternion; excludes root xy progression',
        boundary_limit='terminal_partial marks incomplete final cycle/safety truncation; correction statistics exclude it; no assumed recovery',
        model_root_residual_max=float(np.max(arrays['model_root_residual'])),
        interpretation=INTERPRETATION)


def validate_arrays(z, prefix_ms=PREFIX_MS, mixed_ms=MIXED_MS, burst_ms=50, recover_ms=200):
    durations(prefix_ms,mixed_ms,burst_ms,recover_ms)
    n = len(z['stage'])
    for key in ('X238','Y42','student_action','executed_action','gaussian_noise'):
        if z[key].shape != (n,238 if key=='X238' else 42):
            raise ValueError('burst vector dimensions: '+key)
    for key in ('stage','beta','time','teacher_phase_s','applied_ticks','model_root_residual'):
        if z[key].shape != (n,):
            raise ValueError('burst scalar dimensions: '+key)
    if n<1 or n>(prefix_ms+mixed_ms)//5 or any(not np.isfinite(z[k]).all() for k in z):
        raise ValueError('empty, overlong or nonfinite burst sequence')
    expected_stage = np.asarray([schedule(k,prefix_ms,burst_ms,recover_ms) for k in range(0,n*5,5)])
    if not np.array_equal(z['stage'],expected_stage):
        raise ValueError('actual scheduled prefix/burst/recovery duration mismatch')
    beta = np.where(expected_stage==1,0.,1.)
    if not np.array_equal(z['beta'],beta):
        raise ValueError('per-stage beta mismatch')
    for key in ('Y42','student_action','executed_action'):
        if np.any(np.abs(z[key])>1+1e-12):
            raise ValueError('unbounded burst action')
    teacher_mask = expected_stage!=1
    if not np.all(z['gaussian_noise'][teacher_mask]==0):
        raise ValueError('noise allowed only during student bursts')
    if not np.array_equal(z['executed_action'][teacher_mask],z['Y42'][teacher_mask]):
        raise ValueError('teacher-only stage must execute exact expert')
    expected = np.clip(beta[:,None]*z['Y42']+(1-beta[:,None])*z['student_action']+z['gaussian_noise'],-1,1)
    if not np.allclose(z['executed_action'],expected,rtol=0,atol=2e-16):
        raise ValueError('per-stage executed action mismatch')
    if n>1 and not np.array_equal(z['X238'][1:,-43:-1],z['executed_action'][:-1]):
        raise ValueError('previous executed action discontinuity')
    if not np.all(z['applied_ticks'][:-1]==5) or not 1<=z['applied_ticks'][-1]<=5 or not np.all(z['applied_ticks']==np.floor(z['applied_ticks'])):
        raise ValueError('five-ms hold mismatch')
    if not np.allclose(z['teacher_phase_s'],np.arange(n)*.005,rtol=0,atol=1e-12):
        raise ValueError('teacher phase mismatch')
    if n>1 and not np.allclose(np.diff(z['time']),.005,rtol=0,atol=1e-10):
        raise ValueError('actual time discontinuity')
    if not np.allclose(z['X238'][:,-1],sw.TARGET_SPEED,rtol=0,atol=1e-15):
        raise ValueError('target speed mismatch')


def validate_burst(directory):
    path = absolute(directory)/'DATASET.json'
    meta = json.loads(path.read_text())
    if meta['format_version']!=1 or meta['kind']!='bounded_burst_recovery' or not meta['collection_complete']:
        raise ValueError('incomplete burst dataset')
    if meta['source_hashes'] != source_hashes():
        raise ValueError('burst source closure mismatch')
    checked_files(meta['source_hashes'])
    checked_files(meta['teacher_reference_evidence']['files'])
    if meta['teacher_reference_evidence'] != sw.teacher_preflight():
        raise ValueError('teacher reference evidence mismatch')
    checked_files({meta['checkpoint_path']:meta['checkpoint_sha256']})
    config = meta['config']
    if str(absolute(config['checkpoint'])) != meta['checkpoint_path']:
        raise ValueError('checkpoint metadata differs from collection command')
    durations(PREFIX_MS,MIXED_MS,config['burst_ms'],config['recover_ms'])
    if not math.isfinite(config['noise_std']) or config['noise_std']<0:
        raise ValueError('invalid configured burst noise')
    entries = meta['episodes']
    if len(entries)<2 or len(entries)!=config['episodes'] or [e['split'] for e in entries]!=['train']*(len(entries)-1)+['validation']:
        raise ValueError('independent final episode holdout required')
    for key in ('episode','seed','sha256','path'):
        if len({e[key] for e in entries})!=len(entries):
            raise ValueError('duplicate burst episode: '+key)
    for index,e in enumerate(entries):
        if e['episode']!=index or e['seed']!=config['seed']+index:
            raise ValueError('episode/seed provenance mismatch')
        checked_files({e[k]:e[d] for k,d in (('path','sha256'),('identity_path','identity_sha256'),('report_path','report_sha256'))})
        identity = json.loads(Path(e['identity_path']).read_text())
        verify(identity)
        required = {**meta['teacher_reference_evidence']['files'],
            meta['checkpoint_path']:meta['checkpoint_sha256'],**meta['source_hashes']}
        if any(identity['files'].get(p,{}).get('sha256')!=digest for p,digest in required.items()):
            raise ValueError('captured identity omits or changes collection inputs')
        if identity['config']['cli'] != config or identity['config']['episode']!=index or identity['config']['seed']!=e['seed']:
            raise ValueError('command metadata differs from captured collection identity')
        if identity['config']['prefix_ms']!=PREFIX_MS or identity['config']['mixed_ms']!=MIXED_MS:
            raise ValueError('captured duration mismatch')
        report = json.loads(Path(e['report_path']).read_text())
        if e.get('trace'):
            checked_files({e['trace']['path']:e['trace']['sha256']})
        with np.load(e['path'],allow_pickle=False) as z:
            validate_arrays(z,PREFIX_MS,MIXED_MS,config['burst_ms'],config['recover_ms'])
            n = len(z['stage']); ticks = int(z['applied_ticks'].sum())
            if n!=e['samples'] or int(np.sum(z['stage']!=0))!=e['mixed_samples']:
                raise ValueError('sample count mismatch')
            if report['early_stop'] is None and ticks!=PREFIX_MS+MIXED_MS:
                raise ValueError('unreported incomplete trajectory')
            if report['early_stop']!=e['early_stop'] or report['physical_ms']!=ticks or report['teacher_queries']!=n:
                raise ValueError('physical report mismatch')
    return meta


def collect(args):
    checkpoint = absolute(args.checkpoint)
    absolute(args.out)
    durations(PREFIX_MS,MIXED_MS,args.burst_ms,args.recover_ms)
    if args.episodes<2 or not math.isfinite(args.noise_std) or args.noise_std<0:
        raise ValueError('>=2 episodes and finite nonnegative noise required')
    evidence = sw.teacher_preflight()
    out = sw.exclusive_out(args.out)
    entries = []
    for episode in range(args.episodes):
        seed = args.seed+episode
        sw.seed_all(seed)
        c = sw.Core(adhesion=False)
        model,payload = load_burst_student(checkpoint,c)
        ident = capture(c,dict(mode='burst_collect',cli=vars(args),episode=episode,seed=seed,
            teacher=sw.teacher_config(2),decision_ms=5,settle_ms=1000,prefix_ms=PREFIX_MS,mixed_ms=MIXED_MS,
            interpretation=INTERPRETATION),extra_paths=sw.SOURCE_FILES+[HERE/'recovery_bc.py']+OWN_SOURCES+
            [checkpoint,HERE/'BURST_CHECKS.json']+[Path(p) for p in evidence['files']])
        ip = out/f'episode_{episode:03d}_IDENTITY.json';atomic_json(ip,ident)
        trace = sw.Trace(full=args.trace)
        sw.settle(c,trace)
        previous,pulse = sw.perturb(c,trace,1,seed)
        teacher = sw.StaticTeacher(c,sw.teacher_config(2))
        arrays,report = trajectory(c,teacher,model,trace,previous,np.random.default_rng(seed+1000000),
            noise_std=args.noise_std,burst_ms=args.burst_ms,recover_ms=args.recover_ms)
        ep = out/f'episode_{episode:03d}.npz';np.savez_compressed(ep,**arrays)
        report.update(perturbation=pulse,teacher=teacher.metadata,checkpoint=str(checkpoint),
            checkpoint_sha256=sha256(checkpoint),model_identity=payload['model_identity'])
        rp = out/f'episode_{episode:03d}_RESULT.json';atomic_json(rp,report)
        trace_info = trace.save(out/f'episode_{episode:03d}_trace.npz') if args.trace else None
        verify(ident)
        entries.append(dict(episode=episode,seed=seed,split='validation' if episode==args.episodes-1 else 'train',
            path=str(ep),sha256=sha256(ep),samples=len(arrays['stage']),identity_path=str(ip),identity_sha256=sha256(ip),
            report_path=str(rp),report_sha256=sha256(rp),mixed_samples=report['mixed_decisions'],
            early_stop=report['early_stop'],trace=trace_info))
        meta = dict(format_version=1,kind='bounded_burst_recovery',config=vars(args),source_hashes=source_hashes(),
            checkpoint_path=str(checkpoint),checkpoint_sha256=sha256(checkpoint),teacher_reference_evidence=evidence,
            episodes=entries,collection_complete=len(entries)==args.episodes,
            split_contract='last independent episode held out; no random intra-episode split',
            label_coverage='every prefix/burst/recovery decision; sample weight 1; full zero-hidden GRU sequence',
            actor_inputs='raw actual X238 including previous executed action; no clock/teacher phase/reference',
            noise='independent Gaussian on all 42 student burst rates only; clip [-1,1], hold 5ms',
            schedule='4800ms exact teacher prefix; 4800ms alternating beta0 burst/beta1 recovery; final partial segment retained',
            safety='COM<1.2 or nonfoot>0 stops only collection; failed prefix retained; acceptance unchanged',
            interpretation=INTERPRETATION)
        atomic_json(out/'DATASET.json',meta)
        print(json.dumps(dict(episode=episode,stage_counts=report['stage_counts'],early_stop=report['early_stop'],
            stage_COM_min={k:v['com_min'] for k,v in report['stage_metrics'].items() if 'com_min' in v})),flush=True)
    validate_burst(out)
    return meta


def load_burst_student(path,core):
    model,payload = sw.load_student(absolute(path),core)
    extra = payload.get('extra') or {}
    if extra.get('kind')=='burst_bc':
        checked_files(extra['burst_closure'])
        validate_burst(Path(extra['burst_dataset_path']).parent)
    return model,payload


def train(args):
    if args.steps<1 or not math.isfinite(args.lr) or args.lr<=0:
        raise ValueError('positive training steps/lr required')
    checkpoint = absolute(args.checkpoint)
    base_dir,recovery_dir = absolute(args.base_dataset),absolute(args.recovery)
    absolute(args.out)
    base = sw.validate_dataset(base_dir)
    recovery = validate_burst(recovery_dir)
    if not base['all_episode_conditions_passed']:
        raise ValueError('base teacher episode failed')
    groups = r.partition(base,recovery)
    out = sw.exclusive_out(args.out)
    sw.seed_all(args.seed)
    c = sw.Core(adhesion=False)
    model,parent = load_burst_student(checkpoint,c)
    mi = parent['model_identity']
    fresh = sw.ActorCritic(mi['architecture'],critic_obs_dim=255,dtype=torch.float64,logstd_bounds=mi['logstd_bounds'])
    fresh.load_state_dict(model.state_dict(),strict=True)
    model = fresh
    sw.seed_all(args.seed)
    optimizer = torch.optim.Adam(model.actor.parameters(),lr=args.lr)
    episodes = {k:r.tensors(v) for k,v in groups.items()}
    closure = r.dataset_closure(base_dir,base)+r.dataset_closure(recovery_dir,recovery)
    closure += [checkpoint]+[Path(p) for p in parent['identity']['files']]
    closure = sorted(set(closure))
    ident = capture(c,dict(mode='burst_bc',cli=vars(args),parent_sha256=sha256(checkpoint),
        warm_start='MODEL ONLY; fresh BC Adam, PPO states/counters, and declared RNG seed',
        loss='all supervised labels, base and burst sample weights 1; full GRU BPTT without detach',
        partitions={k:[dict(episode=e['episode'],seed=e['seed'],path=e['path']) for e in v] for k,v in groups.items()}),
        extra_paths=sw.SOURCE_FILES+[HERE/'recovery_bc.py']+OWN_SOURCES+[HERE/'BURST_CHECKS.json']+closure)
    atomic_json(out/'IDENTITY.json',ident)
    untouched = {k:v.clone() for k,v in model.state_dict().items() if not k.startswith('actor.')}
    curve = []
    for step in range(args.steps+1):
        if step:
            optimizer.zero_grad(set_to_none=True)
            loss = r.aggregation_loss(model,episodes['base_train'],episodes['recovery_train'])
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite burst loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.actor.parameters()):
                raise FloatingPointError('nonfinite burst gradient')
            optimizer.step()
        with torch.no_grad():
            row = dict(step=step,**{k+'_MSE':float(sw.bounded_bc_loss(model,v)) for k,v in episodes.items()})
        curve.append(row)
        atomic_json(out/'CURVE.json',dict(curve=curve,interpretation='supervised MSE only; not gait evidence'))
        if step==0 or step==args.steps or step%25==0:
            print(json.dumps(row),flush=True)
    if any(not torch.equal(v,model.state_dict()[k]) for k,v in untouched.items()):
        raise RuntimeError('BC changed critic or exploration parameters')
    trainer = sw.make_trainer(model)
    verify(ident)
    dp,rp = base_dir/'DATASET.json',recovery_dir/'DATASET.json'
    burst_closure = {str(p):sha256(p) for p in r.dataset_closure(recovery_dir,recovery)}
    manifests = [json.loads(Path(e['identity_path']).read_text()) for m in (base,recovery) for e in m['episodes']]
    save_checkpoint(out/'final.pt',trainer,ident,extra=dict(kind='burst_bc',bc_optimizer=optimizer.state_dict(),
        bc_config=vars(args),curve=curve,dataset_path=str(dp),dataset_sha256=sha256(dp),dataset_metadata=base,
        burst_dataset_path=str(rp),burst_dataset_sha256=sha256(rp),burst_closure=burst_closure,
        burst_metadata=recovery,collection_identities=manifests,parent_checkpoint=str(checkpoint),
        parent_checkpoint_sha256=sha256(checkpoint),warm_start='model_only',rng_seed=args.seed,
        ppo_optimizers='fresh',actor_only_bc=True,weights=dict(base=1,recovery=1,
        normalization='sum squared errors / (42 * decision count)'),interpretation=INTERPRETATION))
    check = sw.ActorCritic(mi['architecture'],critic_obs_dim=255,dtype=torch.float64,logstd_bounds=mi['logstd_bounds'])
    load_checkpoint(out/'final.pt',sw.make_trainer(check),ident,restore_rng=False)
    if any(not torch.equal(v,check.state_dict()[k]) for k,v in model.state_dict().items()):
        raise RuntimeError('checkpoint reload mismatch')
    payload = torch.load(out/'final.pt',map_location='cpu',weights_only=False)
    if payload['state']['actor']['state'] or payload['state']['critic']['state'] or payload['state']['updates']!=0:
        raise RuntimeError('PPO optimizer/counters not fresh')
    result = dict(checkpoint=str(out/'final.pt'),checkpoint_sha256=sha256(out/'final.pt'),final=curve[-1],
        checkpoint_reload_exact=True,ppo_optimizer_entries=0,parent_checkpoint_sha256=sha256(checkpoint),
        identity_sha256=ident['identity_sha256'],base_metadata_sha256=sha256(dp),burst_metadata_sha256=sha256(rp),
        closed_loop='NOT_EVALUATED',interpretation=INTERPRETATION)
    atomic_json(out/'RESULT.json',result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode',required=True)
    co = sub.add_parser('collect')
    co.add_argument('--checkpoint',required=True);co.add_argument('--out',required=True)
    co.add_argument('--episodes',type=int,default=3)
    co.add_argument('--burst-ms',type=int,default=50);co.add_argument('--recover-ms',type=int,default=200)
    co.add_argument('--noise-std',type=float,default=.01);co.add_argument('--seed',type=int,default=20261301)
    co.add_argument('--trace',action='store_true')
    tr = sub.add_parser('train')
    for flag in ('checkpoint','base-dataset','recovery','out'):
        tr.add_argument('--'+flag,required=True)
    tr.add_argument('--steps',type=int,default=1000);tr.add_argument('--lr',type=float,default=.0003)
    tr.add_argument('--seed',type=int,default=20261311)
    return p


def main():
    args = parser().parse_args()
    check_preflight()
    return dict(collect=collect,train=train)[args.mode](args)


if __name__=='__main__':
    main()
