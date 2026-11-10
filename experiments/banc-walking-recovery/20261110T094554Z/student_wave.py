"""S2/S5 observation-only wave imitation and continuous physical handover.

No execution on import. CLI outputs are exclusive new directories. Collection is
algorithm teacher evidence; BC MSE is not closed-loop walking evidence. Units are
native; absolute SI NOT_ESTABLISHED. Physical model and native timestep unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch

from physics import Core, NativeRecorder, LEGS, MAX_RATE, summarise
from static_teacher import StaticTeacher
from cycle_reward import CycleTracker, CycleConfig
from models import ActorCritic, NORMALIZATION_FLOOR, ACTOR_INPUT_TRANSFORM
from ppo import PPOTrainer, PPOConfig, save_checkpoint, load_checkpoint
from recurrent import RecurrentPPOTrainer, RecurrentPPOConfig
from identity import HERE, atomic_json, capture, verify, verify_runtime, sha256

TARGET_SPEED = .075
DECISION_MS = 5
CYCLE_S = 4.8
SOURCE = dict(settle=0, perturb=1, teacher=2, student=3, zero=4, teacher_replay=5)
SOURCE_FILES = [HERE / x for x in ('student_wave.py', 'models.py', 'ppo.py',
    'recurrent.py', 'physics.py', 'static_teacher.py', 'wave_reference.py',
    'cycle_reward.py', 'identity.py')]
TEACHER_BASE = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T093347Z')
TEACHER_RESULT = TEACHER_BASE / 'early_unload/RESULT.json'
TEACHER_IDENTITY = TEACHER_BASE / 'early_unload/IDENTITY.json'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def exclusive_out(path):
    path = Path(path).expanduser().absolute()
    path.mkdir(parents=True, exist_ok=False)
    return path


def teacher_config(cycles):
    return dict(mode='measuredwave', gravity_ff=True, stance_force_pref='uniform',
                step_s=.8, cycles=int(cycles), stride=.36, lift=.12, contact_transition_height=.01)


def source_hashes():
    return {str(p.resolve()): sha256(p) for p in SOURCE_FILES}


def student_preflight():
    checks = json.loads((HERE/'STUDENT_CHECKS.json').read_text())
    if not checks['all_passed']:
        raise ValueError('student checks failed')
    for path, digest in checks['source_hashes'].items():
        if not Path(path).is_absolute() or sha256(path) != digest:
            raise ValueError('student check source drift: '+path)
    return True


def teacher_preflight():
    checks_path = TEACHER_BASE / 'SMOOTH_CHECKS.json'
    checks = json.loads(checks_path.read_text())
    if not checks['all_passed']:
        raise ValueError('static teacher checks failed')
    for path, digest in checks['source_hashes'].items():
        if sha256(path) != digest:
            raise ValueError('static check source drift: ' + path)
    result = json.loads(TEACHER_RESULT.read_text())
    manifest = json.loads(TEACHER_IDENTITY.read_text())
    verify(manifest)
    expected = teacher_config(2)
    # The original result stores only overrides, with defaults supplied by teacher.
    from static_teacher import DEFAULT_CONFIG
    actual = {**DEFAULT_CONFIG, **result['config']}
    if any(actual[k] != v for k, v in expected.items()):
        raise ValueError('teacher reference condition mismatch')
    if not result['all_gates'] or result['early_stop'] is not None:
        raise ValueError('reference teacher did not pass physical gates')
    if not all(result['complete_per_leg'][leg] >= 1 for leg in LEGS):
        raise ValueError('reference teacher lacks strict completions')
    trace = TEACHER_RESULT.with_name('TRACE_1MS.npz')
    if sha256(trace) != result['trace_sha256']:
        raise ValueError('teacher trace digest mismatch')
    return dict(condition=expected, all_gates=True,
                complete_per_leg=result['complete_per_leg'],
                files={str(p.resolve()): sha256(p) for p in
                       (checks_path, TEACHER_RESULT, TEACHER_IDENTITY, trace)})


def observations(c, previous_actual_action, remaining_s):
    x = np.asarray(c.obs238(previous_actual_action, target_speed=TARGET_SPEED), dtype=np.float64)
    v = np.r_[x, c.critic_extras(), float(remaining_s)]
    if x.shape != (238,) or v.shape != (255,) or not np.isfinite(v).all():
        raise ValueError('observation contract requires finite 238 actor / 255 critic')
    return x, v


def no_external(c):
    if c.adhesion_on or np.any(c.d.xfrc_applied) or np.any(c.d.qfrc_applied):
        raise ValueError('requires adhesion off and zero external force')


def bounded_action(action):
    a = np.asarray(action, dtype=np.float64)
    if a.shape != (42,) or not np.isfinite(a).all() or np.any(np.abs(a) > 1 + 1e-12):
        raise ValueError('action requires finite bounded 42-rate')
    return a.copy()


class Trace:
    """One trajectory at a time; optional full 1ms arrays, always gate samples."""
    def __init__(self, full=True):
        self.full = full
        self.rows = []
        self.rec = []
        self.sources = []

    def tick(self, c, action, source):
        no_external(c)
        action = bounded_action(action)
        m = c.tick(action)
        no_external(c)
        t = float(c.d.time)
        com = c.d.subtree_com[c.th].copy()
        self.rec.append((t, com, m['loads'].copy(), float(m['nonfoot']), m['speed'].copy()))
        self.sources.append(int(source))
        if self.full:
            self.rows.append(dict(time=t, qpos=c.d.qpos.copy(), qvel=c.d.qvel.copy(),
                com=com, loads=m['loads'].copy(), nonfoot=float(m['nonfoot']),
                padz=m['padz'].copy(), speed=m['speed'].copy(), relativefoot=m['foot'].copy(),
                action=action, actual_requested_action=c.requested_rate.copy()/MAX_RATE,
                target=c.actuator.target_rad.copy(), actualctrl=c.d.ctrl.copy(),
                actuator_force=c.d.actuator_force.copy(), control_source=int(source),
                external_body_force_max=float(np.max(np.abs(c.d.xfrc_applied))),
                external_generalized_force_max=float(np.max(np.abs(c.d.qfrc_applied)))))
        return m

    def save(self, path):
        if not self.full or not self.rows:
            raise ValueError('no full trace')
        np.savez_compressed(path, **{k: np.asarray([r[k] for r in self.rows]) for k in self.rows[0]})
        return dict(path=str(Path(path).resolve()), sha256=sha256(path), samples=len(self.rows))

    def summary(self, c, source=None):
        rec = self.rec if source is None else [r for r, s in zip(self.rec, self.sources) if s == source]
        if len(rec) < 2:
            return dict(all_gates=False, sample_count=len(rec), reason='less than two samples')
        report = summarise(rec, len(rec)*80, initial_heading=c.initial_heading)
        report['window'] = dict(first_sample_s=rec[0][0], last_sample_s=rec[-1][0], sample_count=len(rec))
        return report


def settle(c, trace, ms=1000):
    for _ in range(ms):
        trace.tick(c, np.zeros(42), SOURCE['settle'])


def perturb(c, trace, episode, seed):
    """Declared physical command pulse, no state writes; rest calibrated afterwards."""
    pulse = np.zeros(42) if episode == 0 else np.clip(np.random.default_rng(seed).normal(0, .02, 42), -1, 1)
    previous = np.zeros(42)
    if episode != 0:
        for _ in range(20):
            trace.tick(c, pulse, SOURCE['perturb'])
        previous = pulse.copy()
    return previous, dict(kind='none' if episode == 0 else 'seeded constant motor-command pulse',
        seed=int(seed), duration_ms=0 if episode == 0 else 20, normal_std=.02,
        action=pulse.tolist(), state_writes=False, external_forces=False,
        rest_calibration='StaticTeacher constructed after physical pulse')


def tracker(c):
    return CycleTracker(CycleConfig(timeout=4.)).reset(start_time=float(c.d.time))


def cycle_report(observer, events):
    events.extend(observer.finalize()['events'])
    counts = {leg: sum(e['type'] == 'complete' and e['leg'] == leg for e in events) for leg in LEGS}
    return dict(complete_per_leg=counts, all_legs_completed=all(n >= 1 for n in counts.values()),
                events=events, config=asdict(observer.config),
                interpretation='strict sensed cycle diagnostic; no actions generated')


def import_verified_episode(directory, out, cycles, seed, episode=0, validation=False):
    """Reuse one actual independent replay, never duplicate the baseline trajectory."""
    directory=Path(directory).resolve()
    samples_path=directory/'BC_SAMPLES.npz'
    result_path=directory/'RESULT.json'
    identity_path=directory/'IDENTITY.json'
    report=json.loads(result_path.read_text())
    manifest=json.loads(identity_path.read_text())
    c=Core(adhesion=False)
    verify_runtime(manifest,c)
    del c
    from static_teacher import DEFAULT_CONFIG
    cfg={**DEFAULT_CONFIG,**manifest['config']['config']}
    if any(cfg[k] != value for k,value in teacher_config(cycles).items()):
        raise ValueError('reused teacher condition mismatch')
    pulse_seed=manifest['config'].get('pulse_seed')
    if manifest['config'].get('settle_s') != 1. or manifest['config'].get('landing_tail_s') != .4:
        raise ValueError('reuse requires 1s physical settle and .4s diagnostic tail')
    if episode > 0 and pulse_seed is None:
        raise ValueError('only episode0 may reuse unperturbed baseline')
    seed=int(seed if pulse_seed is None else pulse_seed)
    cycles_passed=all(report['complete_per_leg'][leg]>=1 for leg in LEGS)
    condition_passed=bool(report['all_gates'] and cycles_passed)
    if sha256(samples_path) != report['sample_file_sha256']:
        raise ValueError('reused BC sample digest mismatch')
    with np.load(samples_path,allow_pickle=False) as z:
        phase=z['source_phase']
        expected=round((CYCLE_S*cycles+.4)/.005)
        if phase.shape != (expected,) or not np.allclose(phase,np.arange(expected)*.005,rtol=0,atol=1e-10):
            raise ValueError('reused source phase coverage mismatch')
        keep=phase < CYCLE_S*cycles-1e-10
        x,y,v=(z[key][keep].copy() for key in ('obs','action','critic_obs'))
        actual_time=z['time'][keep].copy()
    if not np.allclose(x[:,-1],TARGET_SPEED,rtol=0,atol=1e-15):
        raise ValueError('reused target speed mismatch')
    path=out/f'episode_{episode:03d}.npz'
    # Provenance phase is deliberately excluded from every actor observation.
    np.savez_compressed(path,X238=x,Y42=y,V255=v)
    own_identity=out/f'episode_{episode:03d}_IDENTITY.json'
    atomic_json(own_identity,manifest)
    originals={str(p):sha256(p) for p in (samples_path,result_path,identity_path)}
    report['cycles']=dict(complete_per_leg=report['complete_per_leg'],all_legs_completed=cycles_passed,
        events=report['events'],config=asdict(CycleConfig(timeout=4.)))
    report['teacher_episode_condition_passed']=condition_passed
    report['reused_episode']=dict(original_files=originals,landing_tail_excluded=True,
        source_phase='provenance only; excluded from actor',first_actual_command_time_s=float(actual_time[0]))
    own_report=out/f'episode_{episode:03d}_RESULT.json'
    atomic_json(own_report,report)
    pulse_action=np.zeros(42) if pulse_seed is None else np.clip(np.random.default_rng(pulse_seed).normal(0,.02,42),-1,1)
    pulse=dict(kind='none' if pulse_seed is None else 'seeded constant motor-command pulse',
               seed=seed,duration_ms=0 if pulse_seed is None else 20,normal_std=.02,action=pulse_action.tolist(),
               state_writes=False,external_forces=False,rest_calibration='original teacher after physical pulse/settle')
    return dict(episode=episode,seed=seed,split='validation' if validation else 'train',path=str(path),sha256=sha256(path),samples=len(x),
        identity_path=str(own_identity),identity_sha256=sha256(own_identity),
        report_path=str(own_report),report_sha256=sha256(own_report),trace=None,
        perturbation=pulse,teacher_episode_condition_passed=condition_passed,reused_original_files=originals)


def dataset_metadata(out,args,evidence,entries):
    meta=dict(format_version=1,kind='independent physical teacher episodes',config=vars(args).copy(),
        source_hashes=source_hashes(),teacher_reference_evidence=evidence,episodes=entries,
        collection_complete=len(entries)==args.episodes,
        X238='actual before command obs238(previous_actual_action,.075)',
        Y42='bounded teacher command held five actual 1ms ticks',
        V255='X238 + actual core.critic_extras() + remaining_s',
        actor_inputs_exclude=['clock','phase','leg_order','reference','teacher_future'],
        split_contract='last independent episode held out; no intra-trajectory split',
        supervision=dict(gait_duration_s=CYCLE_S*args.cycles,physical_landing_tail_s=.4,
            landing_tail_excluded=True,samples_per_episode=round(CYCLE_S*args.cycles/.005)),
        all_episode_conditions_passed=all(e['teacher_episode_condition_passed'] for e in entries),
        dataset_validity='PARENT_DECIDES; new failures never inherited from reference teacher pass')
    atomic_json(out/'DATASET.json',meta)
    return meta


def collect(args):
    if args.episodes < 2 or args.cycles < 1:
        raise ValueError('require >=2 independent episodes and >=1 cycles')
    evidence = teacher_preflight()
    out = exclusive_out(args.out)
    config = vars(args).copy()
    entries = []
    reuse=args.reuse_episode or []
    if len(reuse)>args.episodes or len(set(map(str,reuse)))!=len(reuse):
        raise ValueError('too many or duplicate reused episodes')
    for episode,directory in enumerate(reuse):
        entries.append(import_verified_episode(directory,out,args.cycles,args.seed+episode,
            episode=episode,validation=episode==args.episodes-1))
        meta=dataset_metadata(out,args,evidence,entries)
        print(json.dumps(dict(episode=episode,reused=True,samples=entries[-1]['samples'],
            condition=entries[-1]['teacher_episode_condition_passed'])),flush=True)
    for episode in range(len(entries),args.episodes):
        episode_seed = args.seed + episode
        seed_all(episode_seed)
        c = Core(adhesion=False)
        ident = capture(c, dict(mode='collect', cli=config, episode=episode, seed=episode_seed,
            teacher=teacher_config(args.cycles), command_dt_s=.005, settle_s=1., landing_tail_s=.4),
            extra_paths=SOURCE_FILES + [HERE/'STATIC_CHECKS.json', TEACHER_RESULT, TEACHER_IDENTITY])
        ident_path = out / f'episode_{episode:03d}_IDENTITY.json'
        atomic_json(ident_path, ident)
        tr = Trace(full=args.trace)
        settle(c, tr)
        previous, pulse = perturb(c, tr, episode, episode_seed)
        teacher = StaticTeacher(c, teacher_config(args.cycles))
        observer, events = tracker(c), []
        total_ms = round((CYCLE_S*args.cycles+.4)*1000)
        xs, ys, vs = [], [], []
        model_residual_max = 0.
        for k in range(0, total_ms, DECISION_MS):
            x, v = observations(c, previous, (total_ms-k)*.001)
            action, diag = teacher.action(c, k*.001)
            action = bounded_action(action)
            # Landing tail is physical diagnostic only, never a supervised stop target.
            if k < round(CYCLE_S*args.cycles*1000):
                xs.append(x); ys.append(action); vs.append(v)
            model_residual_max = max(model_residual_max, diag['root_balance_residual_norm_over_W'])
            for _ in range(DECISION_MS):
                tr.tick(c, action, SOURCE['teacher'])
                events.extend(observer.measure(c)['events'])
            previous = action.copy()
        path = out / f'episode_{episode:03d}.npz'
        np.savez_compressed(path, X238=np.asarray(xs), Y42=np.asarray(ys), V255=np.asarray(vs))
        report = tr.summary(c, SOURCE['teacher'])
        report.update(cycles=cycle_report(observer, events), model_root_residual_max=model_residual_max,
                      perturbation=pulse, teacher=teacher.metadata)
        report['teacher_episode_condition_passed'] = bool(report['all_gates'] and
            report['cycles']['all_legs_completed'] and model_residual_max <= .02)
        report['interpretation'] = 'new episode diagnostics only; parent decides dataset validity'
        report_path = out / f'episode_{episode:03d}_RESULT.json'
        atomic_json(report_path, report)
        if args.trace:
            trace_info = tr.save(out / f'episode_{episode:03d}_trace.npz')
        else:
            trace_info = None
        verify(ident)
        entries.append(dict(episode=episode, seed=episode_seed,
            split='validation' if episode == args.episodes-1 else 'train',
            path=str(path), sha256=sha256(path), samples=len(xs),
            identity_path=str(ident_path), identity_sha256=sha256(ident_path),
            report_path=str(report_path), report_sha256=sha256(report_path), trace=trace_info,
            perturbation=pulse, teacher_episode_condition_passed=report['teacher_episode_condition_passed']))
        meta=dataset_metadata(out,args,evidence,entries)
        print(json.dumps(dict(episode=episode, samples=len(xs), all_gates=report['all_gates'],
            cycles=report['cycles']['complete_per_leg'], condition=report['teacher_episode_condition_passed'])), flush=True)
        del c, teacher, tr, xs, ys, vs
    validate_dataset(out)
    return meta


def validate_dataset(directory):
    path = Path(directory).resolve()/'DATASET.json'
    meta = json.loads(path.read_text())
    if meta['format_version'] != 1 or not meta['collection_complete']:
        raise ValueError('incomplete dataset')
    for p, digest in meta['source_hashes'].items():
        if sha256(p) != digest:
            raise ValueError('dataset source drift: '+p)
    for p, digest in meta['teacher_reference_evidence']['files'].items():
        if sha256(p) != digest:
            raise ValueError('teacher evidence drift: '+p)
    entries = meta['episodes']
    if len(entries) < 2 or [e['split'] for e in entries] != ['train']*(len(entries)-1)+['validation']:
        raise ValueError('last independent episode must be the sole holdout')
    if len({e['episode'] for e in entries}) != len(entries) or len({e['seed'] for e in entries}) != len(entries):
        raise ValueError('duplicate episode identity')
    if len({e['sha256'] for e in entries}) != len(entries):
        raise ValueError('duplicate exact trajectory across split')
    if meta['all_episode_conditions_passed'] != all(e['teacher_episode_condition_passed'] for e in entries):
        raise ValueError('aggregate teacher condition mismatch')
    for e in entries:
        for path, digest in e.get('reused_original_files',{}).items():
            if not Path(path).is_absolute() or sha256(path) != digest:
                raise ValueError('reused original artifact drift')
        for key, digest in (('path','sha256'), ('identity_path','identity_sha256'), ('report_path','report_sha256')):
            if not Path(e[key]).is_absolute() or sha256(e[key]) != e[digest]:
                raise ValueError('dataset artifact digest mismatch: '+key)
        manifest = json.loads(Path(e['identity_path']).read_text())
        verify(manifest)
        report = json.loads(Path(e['report_path']).read_text())
        if report['teacher_episode_condition_passed'] != e['teacher_episode_condition_passed']:
            raise ValueError('episode condition metadata mismatch')
        if e.get('trace') and sha256(e['trace']['path']) != e['trace']['sha256']:
            raise ValueError('trace digest mismatch')
        with np.load(e['path'], allow_pickle=False) as z:
            n = e['samples']
            if n < 1 or any(z[k].shape != (n,d) for k,d in [('X238',238),('Y42',42),('V255',255)]):
                raise ValueError('dataset dimensions')
            if not all(np.isfinite(z[k]).all() for k in z.files) or np.any(np.abs(z['Y42']) > 1+1e-12):
                raise ValueError('dataset finite/bounded contract')
            if not np.array_equal(z['X238'], z['V255'][:,:238]):
                raise ValueError('critic prefix mismatch')
            if n > 1 and not np.array_equal(z['X238'][1:,-43:-1],z['Y42'][:-1]):
                raise ValueError('previous actual action continuity mismatch')
            if not np.allclose(z['X238'][:,-1],TARGET_SPEED,rtol=0,atol=1e-15):
                raise ValueError('constant target speed mismatch')
            if n != meta['supervision']['samples_per_episode'] or not meta['supervision']['landing_tail_excluded']:
                raise ValueError('supervised gait coverage or tail exclusion mismatch')
    return meta


def bc_mean(model, x):
    """A full contiguous episode starts with explicit zero hidden; no random windows."""
    if model.architecture == 'gru238':
        mean, _ = model.actor.sequence(x[:, None, :], model.actor.initial_state(1))
        return mean[:,0]
    return model.actor(x)


def bounded_bc_loss(model, episodes):
    total = sum(x.shape[0]*42 for x,y in episodes)
    return sum((torch.tanh(bc_mean(model,x))-y).square().sum() for x,y in episodes)/total


def make_trainer(model, config=None):
    if model.architecture == 'gru238':
        return RecurrentPPOTrainer(model, None if config is None else RecurrentPPOConfig(**config))
    return PPOTrainer(model, None if config is None else PPOConfig(**config))


def new_model(architecture, normalization_floor=NORMALIZATION_FLOOR):
    return ActorCritic(architecture, critic_obs_dim=255, dtype=torch.float64,
                       initial_logstd=math.log(.05), normalization_floor=normalization_floor)


def fit_actor_normalization(meta, dataset_path, floor=NORMALIZATION_FLOOR):
    """Population statistics over every TRAIN frame, weighted by sample count.

    Validation arrays are never opened here. std_floor is in raw feature units;
    variance_floor is its square. The final affine transform has no clipping.
    """
    floor = float(floor)
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError('normalization floor must be positive and finite')
    arrays, files = [], []
    for entry in meta['episodes']:
        if entry['split'] != 'train':
            continue
        path = Path(entry['path']).resolve()
        if sha256(path) != entry['sha256']:
            raise ValueError('normalization training file drift')
        with np.load(path, allow_pickle=False) as z:
            x = np.asarray(z['X238'], dtype=np.float64).copy()
        if x.ndim != 2 or x.shape[1] != 238 or len(x) == 0 or not np.isfinite(x).all():
            raise ValueError('normalization requires finite nonempty TRAIN X238')
        arrays.append(x)
        files.append(dict(path=str(path), sha256=entry['sha256'], count=len(x), episode=entry['episode']))
    if not arrays:
        raise ValueError('normalization requires TRAIN data')
    x = np.concatenate(arrays, axis=0)
    mean, std = x.mean(axis=0), x.std(axis=0, ddof=0)
    scale = np.maximum(std, floor)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError('nonfinite normalization statistics')
    stats = dict(actor_input_transform=ACTOR_INPUT_TRANSFORM, dimension=238,
        count=len(x), mean=mean.tolist(), std=std.tolist(), scale=scale.tolist(),
        std_floor=floor, variance_floor=floor**2, variance_ddof=0, clipping=None,
        weighting='each TRAIN frame equally; no holdout', train_files=files,
        dataset_path=str(Path(dataset_path).resolve()), dataset_sha256=sha256(dataset_path),
        arrays_sha256=hashlib.sha256(np.stack([mean, std, scale]).astype('<f8').tobytes()).hexdigest())
    stats['statistics_sha256'] = hashlib.sha256(json.dumps(stats, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return stats


def verify_actor_normalization(model, stats, meta, dataset_path):
    expected = fit_actor_normalization(meta, dataset_path, model.actor.normalization_floor)
    if stats != expected:
        raise ValueError('checkpoint actor normalization provenance mismatch')
    for key, field in (('center','mean'), ('scale','scale')):
        actual = getattr(model.actor, key)
        wanted = torch.as_tensor(expected[field], dtype=actual.dtype, device=actual.device)
        if not torch.equal(actual, wanted):
            raise ValueError('checkpoint actor normalization buffer mismatch: '+key)
    return True


def bc(args):
    if args.steps < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('positive steps/lr required')
    meta = validate_dataset(args.dataset)
    if not meta['all_episode_conditions_passed'] and not args.allow_failed_teacher_episodes:
        raise ValueError('teacher episode condition failed; parent must explicitly choose --allow-failed-teacher-episodes')
    out = exclusive_out(args.out)
    seed_all(args.seed)
    c = Core(adhesion=False)
    dataset_path = Path(args.dataset).resolve()/'DATASET.json'
    extras = SOURCE_FILES + [dataset_path]
    extras.extend(Path(p) for p in meta['teacher_reference_evidence']['files'])
    for e in meta['episodes']:
        extras.extend(Path(e[k]) for k in ('path','identity_path','report_path'))
        extras.extend(Path(p) for p in e.get('reused_original_files',{}))
        if e.get('trace'):
            extras.append(Path(e['trace']['path']))
    ident = capture(c, dict(mode='bc', cli=vars(args), dataset_sha256=sha256(dataset_path)), extra_paths=extras)
    atomic_json(out/'IDENTITY.json', ident)
    episodes = {'train':[], 'validation':[]}
    for e in meta['episodes']:
        with np.load(e['path'], allow_pickle=False) as z:
            episodes[e['split']].append((torch.as_tensor(z['X238'].copy(), dtype=torch.float64),
                                         torch.as_tensor(z['Y42'].copy(), dtype=torch.float64)))
    floor = getattr(args, 'normalization_floor', NORMALIZATION_FLOOR)
    model = new_model(args.architecture, floor)
    normalization = fit_actor_normalization(meta, dataset_path, floor)
    model.actor.set_input_stats(normalization['mean'], normalization['scale'])
    atomic_json(out/'ACTOR_NORMALIZATION.json', normalization)
    # Critic and exploration std remain untouched by supervised learning.
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=args.lr)
    curve=[]
    for step in range(args.steps+1):
        if step:
            optimizer.zero_grad(set_to_none=True)
            loss = bounded_bc_loss(model, episodes['train'])
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite BC loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.actor.parameters()):
                raise FloatingPointError('nonfinite BC gradient')
            optimizer.step()
        with torch.no_grad():
            row = dict(step=step, train_MSE=float(bounded_bc_loss(model, episodes['train'])),
                       heldout_MSE=float(bounded_bc_loss(model, episodes['validation'])))
        curve.append(row)
        atomic_json(out/'CURVE.json', dict(curve=curve, interpretation='supervised bounded action MSE; no closed-loop claim'))
        if step == 0 or step == args.steps or step % 25 == 0:
            print(json.dumps(row), flush=True)
    # Fresh PPO optimizer states, preserving the independent BC Adam in extra.
    trainer = make_trainer(model)
    verify(ident)
    save_checkpoint(out/'final.pt', trainer, ident, extra=dict(kind='wave_bc',
        bc_optimizer=optimizer.state_dict(), bc_config=vars(args), curve=curve,
        dataset_metadata=meta, dataset_path=str(dataset_path), dataset_sha256=sha256(dataset_path),
        collection_identities=[json.loads(Path(e['identity_path']).read_text()) for e in meta['episodes']],
        actor_only_bc=True, std_initial=.05, ppo_optimizers='fresh', actor_normalization=normalization,
        interpretation='MSE does not establish closed-loop gait'))
    payload = torch.load(out/'final.pt', map_location='cpu', weights_only=False)
    check_model = new_model(args.architecture, floor)
    load_checkpoint(out/'final.pt', make_trainer(check_model), ident, restore_rng=False)
    if not all(torch.equal(v,check_model.state_dict()[k]) for k,v in model.state_dict().items()):
        raise ValueError('checkpoint reload mismatch')
    result = dict(checkpoint=str(out/'final.pt'), checkpoint_sha256=sha256(out/'final.pt'),
        steps=args.steps, final=curve[-1], identity_sha256=ident['identity_sha256'],
        checkpoint_reload_exact=True, ppo_actor_state_entries=len(payload['state']['actor']['state']),
        ppo_critic_state_entries=len(payload['state']['critic']['state']), closed_loop='NOT_EVALUATED',
        actor_normalization=normalization,
        normalization_artifact=dict(path=str(out/'ACTOR_NORMALIZATION.json'),
                                    sha256=sha256(out/'ACTOR_NORMALIZATION.json')),
        normalization_buffers_verified=verify_actor_normalization(check_model, normalization, meta, dataset_path))
    atomic_json(out/'RESULT.json', result)
    return result


def load_student(path, core):
    path = Path(path).resolve()
    payload = torch.load(path, map_location='cpu', weights_only=False)
    verify_runtime(payload['identity'], core)
    for manifest in (payload.get('extra') or {}).get('collection_identities', []):
        verify(manifest)
    mi = payload['model_identity']
    if mi['architecture'] not in ('ff238','gru238') or mi['critic_obs_dim'] != 255:
        raise ValueError('deployment requires 238 actor / 255 critic')
    model = ActorCritic(mi['architecture'], critic_obs_dim=255, dtype=torch.float64,
                        logstd_bounds=mi['logstd_bounds'],
                        normalization_floor=mi.get('actor_normalization_floor', NORMALIZATION_FLOOR))
    trainer = make_trainer(model, payload['config'])
    load_checkpoint(path, trainer, payload['identity'], restore_rng=False)
    extra = payload.get('extra') or {}
    if 'dataset_path' in extra:
        if sha256(extra['dataset_path']) != extra['dataset_sha256']:
            raise ValueError('checkpoint dataset metadata changed')
        meta = validate_dataset(Path(extra['dataset_path']).parent)
        verify_actor_normalization(model, extra['actor_normalization'], meta, extra['dataset_path'])
    else:
        raise ValueError('deployment requires TRAIN normalization provenance')
    model.eval()
    return model, payload


def continuity_snapshot(c, previous, state):
    fields = dict(qpos=c.d.qpos, qvel=c.d.qvel, ctrl=c.d.ctrl,
        target=c.actuator.target_rad, history=c.buf, padz_previous=c._padz_prev,
        pad_rate=c._pad_rate, requested_rate=c.requested_rate,
        previous_actual_action=previous, time=np.asarray(c.d.time),
        external_body=c.d.xfrc_applied, external_generalized=c.d.qfrc_applied)
    if state is not None:
        fields['hidden'] = state.detach().cpu().numpy()
    return {k: np.asarray(v).copy() for k,v in fields.items()}


def snapshot_hashes(snapshot):
    return {k: hashlib.sha256(v.tobytes()).hexdigest() for k,v in snapshot.items()}


def continuous_handover(c, teacher, model, trace, warmup_ms, autonomous_ms,
                        control='student', sampled=False):
    """Shared real-run/test path. Teacher calls exist ONLY in declared teacher branches.

    Test may supply short durations and an instrumented teacher. No state/actuator
    reset occurs here. Actor receives only observations() and explicit GRU hidden.
    """
    if warmup_ms < 5 or autonomous_ms < 5 or warmup_ms % 5 or autonomous_ms % 5:
        raise ValueError('durations must be positive multiples of five ms')
    if control not in ('student','zero','teacher') or (control == 'student' and model is None):
        raise ValueError('invalid control/model')
    previous=np.zeros(42)
    state = model.actor.initial_state(1) if model is not None and model.architecture == 'gru238' else None
    warm_observer, warm_events = tracker(c), []
    decisions=[]
    for k in range(0,warmup_ms,5):
        x,v = observations(c,previous,(warmup_ms+autonomous_ms-k)*.001)
        if model is not None:
            primed = model.act(x,v,deterministic=True,state=state)
            state = primed['state']
        action,_ = teacher.action(c,k*.001)
        action=bounded_action(action)
        decisions.append(dict(time=float(c.d.time), X238=x, V255=v, action=action, source=SOURCE['teacher']))
        for _ in range(5):
            trace.tick(c,action,SOURCE['teacher'])
            warm_events.extend(warm_observer.measure(c)['events'])
        previous=action.copy()
    before=continuity_snapshot(c,previous,state)
    # This observer is diagnostic only. Constructing it must not alter any state.
    auto_observer, auto_events=tracker(c), []
    after=continuity_snapshot(c,previous,state)
    exact={key:bool(np.array_equal(before[key],after[key])) for key in before}
    if not all(exact.values()):
        raise RuntimeError('handover state discontinuity')
    transition=dict(time_s=float(c.d.time), exact=exact, all_exact=all(exact.values()),
        before_sha256=snapshot_hashes(before), after_sha256=snapshot_hashes(after),
        previous_actual_action=previous.tolist(), hidden_preserved=state is not None,
        no_reset=True, algorithm_prefix_declared=True)
    for k in range(0,autonomous_ms,5):
        x,v=observations(c,previous,(autonomous_ms-k)*.001)
        if control == 'teacher':
            action,_=teacher.action(c,(warmup_ms+k)*.001)
            source=SOURCE['teacher_replay']
        elif control == 'zero':
            action=np.zeros(42)
            source=SOURCE['zero']
        else:
            output=model.act(x,v,deterministic=not sampled,state=state)
            state=output['state']
            action=output['action'][0].detach().cpu().numpy()
            source=SOURCE['student']
        action=bounded_action(action)
        decisions.append(dict(time=float(c.d.time),X238=x,V255=v,action=action,source=source))
        for _ in range(5):
            trace.tick(c,action,source)
            auto_events.extend(auto_observer.measure(c)['events'])
        previous=action.copy()
    return dict(transition=transition,
        warmup_cycles=cycle_report(warm_observer,warm_events),
        autonomous_cycles=cycle_report(auto_observer,auto_events), decisions=decisions)


def evaluate(args):
    if args.cycles < 1 or (args.control == 'student' and not args.checkpoint):
        raise ValueError('student requires checkpoint and positive cycles')
    evidence=teacher_preflight()
    out=exclusive_out(args.out)
    seed_all(args.seed)
    recorder=NativeRecorder(out/'native') if args.native else None
    c=Core(adhesion=False,recorder=recorder)
    model,payload=load_student(args.checkpoint,c) if args.checkpoint else (None,None)
    extras=SOURCE_FILES+[TEACHER_RESULT,TEACHER_IDENTITY]
    if args.checkpoint:
        extras.append(Path(args.checkpoint).resolve())
    ident=capture(c,dict(mode='evaluate',cli=vars(args),settle_s=1.,teacher_warmup_s=CYCLE_S,
        autonomous_s=CYCLE_S*args.cycles,teacher=teacher_config(args.cycles+1),control_sources=SOURCE),extra_paths=extras)
    atomic_json(out/'IDENTITY.json',ident)
    tr=Trace(full=True)
    try:
        settle(c,tr)
        teacher=StaticTeacher(c,teacher_config(args.cycles+1))
        run=continuous_handover(c,teacher,model,tr,4800,round(CYCLE_S*args.cycles*1000),
                               control=args.control,sampled=args.sampled)
    finally:
        native=recorder.close() if recorder is not None else None
    trace_info=tr.save(out/'trace_1ms.npz')
    decisions=run.pop('decisions')
    decision_path=out/'decisions_5ms.npz'
    np.savez_compressed(decision_path,**{k:np.asarray([r[k] for r in decisions]) for k in decisions[0]})
    phase_source=SOURCE[dict(student='student',zero='zero',teacher='teacher_replay')[args.control]]
    report=dict(control=args.control,sampled=args.sampled,teacher_reference_evidence=evidence,
        algorithm_prefix=dict(settle_s=1.,walk_s=CYCLE_S,teacher_metadata=teacher.metadata),
        settle=tr.summary(c,SOURCE['settle']),teacher_warmup=tr.summary(c,SOURCE['teacher']),
        autonomous=tr.summary(c,phase_source),**run,trace=trace_info,
        decisions=dict(path=str(decision_path),sha256=sha256(decision_path)),
        control_sources=SOURCE,native=native,
        interpretation='stage legacy five gates and strict cycle diagnostics; original acceptance remains UNKNOWN; '
                       'zero is actuator-hold control, teacher replay is algorithm control, student is substitute ANN')
    verify(ident)
    atomic_json(out/'RESULT.json',report)
    print(json.dumps(dict(control=args.control,transition_exact=run['transition']['all_exact'],
        teacher_gates=report['teacher_warmup']['all_gates'],autonomous_gates=report['autonomous']['all_gates'],
        autonomous_cycles=run['autonomous_cycles']['complete_per_leg'])),flush=True)
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='mode',required=True)
    co=sub.add_parser('collect'); co.add_argument('--out',required=True)
    co.add_argument('--episodes',type=int,default=4); co.add_argument('--cycles',type=int,default=2)
    co.add_argument('--seed',type=int,default=20261110); co.add_argument('--trace',action='store_true')
    co.add_argument('--reuse-episode',action='append',help='repeat for independent verified episode directories in train/holdout order')
    bc_p=sub.add_parser('bc'); bc_p.add_argument('--out',required=True);bc_p.add_argument('--dataset',required=True)
    bc_p.add_argument('--architecture',choices=['ff238','gru238'],default='gru238')
    bc_p.add_argument('--steps',type=int,default=300);bc_p.add_argument('--lr',type=float,default=.001)
    bc_p.add_argument('--seed',type=int,default=20261110)
    bc_p.add_argument('--normalization-floor',type=float,default=NORMALIZATION_FLOOR,
                      help='frozen per-feature minimum TRAIN standard deviation in raw units; no clipping')
    bc_p.add_argument('--allow-failed-teacher-episodes',action='store_true',help='explicit parent dataset-validity decision')
    ev=sub.add_parser('evaluate');ev.add_argument('--out',required=True);ev.add_argument('--checkpoint')
    ev.add_argument('--control',choices=['student','zero','teacher'],default='student')
    ev.add_argument('--cycles',type=int,default=2);ev.add_argument('--seed',type=int,default=20261110)
    ev.add_argument('--sampled',action='store_true');ev.add_argument('--native',action='store_true')
    return p


def main():
    args=parser().parse_args()
    student_preflight()
    return dict(collect=collect,bc=bc,evaluate=evaluate)[args.mode](args)


if __name__ == '__main__':
    main()
