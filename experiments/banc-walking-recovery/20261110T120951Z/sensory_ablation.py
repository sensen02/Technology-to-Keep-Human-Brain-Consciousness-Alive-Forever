"""Declared post-handover sensory intervention; NOT a retrained/exported policy.

Only actor raw observation entries [189:195] (foot vertical velocity) are
zeroed after the exact real teacher prefix. Other actor entries and the entire
critic input remain raw. The external one-shot stage boundary is not an actor
feature, gait clock, or periodic controller. No training or execution on import.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import student_wave as sw
from identity import atomic_json, capture, sha256, verify, verify_runtime

HERE = Path(__file__).resolve().parent
CONDITIONS = ('foot_velocity_zero', 'identity')


def source_hashes():
    paths = sw.SOURCE_FILES + [HERE/'sensory_ablation.py', HERE/'check_sensory_ablation.py']
    paths += list((HERE/'sources').glob('*.py'))
    return {str(p.resolve()): sha256(p) for p in paths}


def preflight():
    checks = json.loads((HERE/'ABLATION_CHECKS.json').read_text())
    if not checks['all_passed'] or checks['source_hashes'] != source_hashes():
        raise ValueError('ablation check closure failed or source drift')
    return sw.teacher_preflight()


def weights_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        a = value.detach().cpu().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


class ModelProxy:
    """Original act interface, with one explicit external intervention boundary.

    Call counts correspond exactly to shared continuous_handover's teacher
    priming decisions. No teacher reference, phase, or clock enters model.act.
    """
    def __init__(self, model, core, condition='foot_velocity_zero', prefix_decisions=960,
                 transition_time_s=5.8):
        if condition not in CONDITIONS or prefix_decisions < 1:
            raise ValueError('invalid declared intervention')
        self.model, self.core, self.condition = model, core, condition
        self.architecture, self.actor = model.architecture, model.actor
        self.prefix_decisions = int(prefix_decisions)
        self.transition_time_s = float(transition_time_s)
        self.intervene = False
        self.records = []
        self.cut_continuity = None

    @torch.no_grad()
    def act(self, obs, critic_obs=None, deterministic=False, state=None, reset=None):
        if critic_obs is None:
            raise ValueError('raw critic observations must be supplied explicitly')
        raw = np.asarray(obs).copy()
        critic = np.asarray(critic_obs).copy()
        if raw.shape != (238,) or critic.shape != (255,):
            raise ValueError('requires raw actor238 / critic255')
        index = len(self.records)
        post = index >= self.prefix_decisions
        if index == self.prefix_decisions:
            if abs(float(self.core.d.time)-self.transition_time_s) > 1e-8:
                raise RuntimeError('intervention does not coincide with handover')
            before = sw.continuity_snapshot(self.core, raw[195:237], state)
            self.intervene = True
            after = sw.continuity_snapshot(self.core, raw[195:237], state)
            exact = {k: bool(np.array_equal(before[k], after[k])) for k in before}
            self.cut_continuity = dict(time_s=float(self.core.d.time), exact=exact,
                all_exact=all(exact.values()), before_sha256=sw.snapshot_hashes(before),
                after_sha256=sw.snapshot_hashes(after), no_state_writes=True,
                previous_action_snapshot='raw X238[195:237]', hidden_reset=False)
            if not self.cut_continuity['all_exact']:
                raise RuntimeError('intervention state discontinuity')
        if self.intervene != post:
            raise RuntimeError('stage boundary drift')
        applied = raw.copy()
        if self.intervene and self.condition == 'foot_velocity_zero':
            applied[189:195] = 0.
        state_in = np.empty(0) if state is None else state.detach().cpu().numpy().copy()
        # critic_obs is passed as received, not reconstructed from transformed X.
        output = self.model.act(applied, critic_obs, deterministic=deterministic,
                                state=state, reset=reset)
        if not np.array_equal(np.asarray(critic_obs), critic):
            raise RuntimeError('critic input mutated')
        self.records.append(dict(time=float(self.core.d.time), intervene=self.intervene,
            raw_X238=raw, actor_input_actual=applied.copy(), raw_V255=critic,
            previous_actual_action=raw[195:237].copy(), hidden_in=state_in,
            hidden_out=np.empty(0) if output['state'] is None else output['state'].detach().cpu().numpy().copy(),
            policy_action=output['action'][0].detach().cpu().numpy().copy(),
            policy_mean=output['mean'][0].detach().cpu().numpy().copy()))
        return output


def validate_records(proxy, decisions, warmup_ms, autonomous_ms):
    expected = (warmup_ms+autonomous_ms)//sw.DECISION_MS
    if len(proxy.records) != expected or len(decisions) != expected:
        raise RuntimeError('decision count mismatch')
    for i, (r, d) in enumerate(zip(proxy.records, decisions)):
        post = i >= warmup_ms//sw.DECISION_MS
        expected_x = d['X238'].copy()
        if post and proxy.condition == 'foot_velocity_zero':
            expected_x[189:195] = 0.
        if (r['intervene'] != post or not np.array_equal(r['raw_X238'], d['X238'])
                or not np.array_equal(r['actor_input_actual'], expected_x)
                or not np.array_equal(r['raw_V255'], d['V255'])
                or not np.array_equal(r['previous_actual_action'], d['X238'][195:237])
                or r['time'] != d['time']
                or d['source'] != sw.SOURCE['student' if post else 'teacher']):
            raise RuntimeError('decision input/source mapping mismatch')
        if i and not np.array_equal(r['hidden_in'], proxy.records[i-1]['hidden_out']):
            raise RuntimeError('hidden state discontinuity')
        if post and not np.array_equal(r['policy_action'], d['action']):
            raise RuntimeError('actual student action mismatch')
    if not proxy.cut_continuity or not proxy.cut_continuity['all_exact']:
        raise RuntimeError('missing exact cut continuity')
    return dict(total=expected, teacher=warmup_ms//5, student=autonomous_ms//5,
        mapping_exact=True, raw_critic_preserved=True, hidden_continuous=True,
        changed_actor_indices=[189,190,191,192,193,194] if proxy.condition == 'foot_velocity_zero' else [])


def evaluate(args):
    if args.cycles < 1 or not Path(args.checkpoint).is_absolute() or not Path(args.out).is_absolute():
        raise ValueError('positive cycles and absolute checkpoint/out required')
    if args.condition != 'foot_velocity_zero':
        raise ValueError('identity is helper-check only; no second gait is authorized by this CLI')
    evidence = preflight()
    sw.seed_all(args.seed)
    c = sw.Core(adhesion=False)
    model, payload = sw.load_student(args.checkpoint, c)  # full original closure
    if model.architecture != 'gru238':
        raise ValueError('this declared experiment requires original SlowGRU')
    before_weights = weights_hash(model)
    out = sw.exclusive_out(args.out)
    config = dict(mode='sensory_ablation', cli=vars(args), seed=args.seed,
        condition=args.condition, settle_s=1., teacher_warmup_s=4.8,
        autonomous_s=4.8*args.cycles, teacher=sw.teacher_config(args.cycles+1),
        actor_transform=dict(post_handover_only=True, zero_slice=[189,195],
            trigger='one-shot stage at actor call 960, after actual 4800ms teacher prefix',
            clock_is_actor_feature=False), control_sources=sw.SOURCE)
    ident = capture(c, config, extra_paths=sw.SOURCE_FILES+[
        Path(__file__), HERE/'check_sensory_ablation.py', HERE/'ABLATION_CHECKS.json',
        Path(args.checkpoint), sw.TEACHER_RESULT, sw.TEACHER_IDENTITY])
    atomic_json(out/'IDENTITY.json', ident)
    tr = sw.Trace(full=True)
    sw.settle(c, tr)
    teacher = sw.StaticTeacher(c, sw.teacher_config(args.cycles+1))
    proxy = ModelProxy(model, c, args.condition, 960, float(c.d.time)+4.8)
    auto_ms = round(4800*args.cycles)
    run = sw.continuous_handover(c, teacher, proxy, tr, 4800, auto_ms, control='student')
    decisions = run.pop('decisions')
    mapping = validate_records(proxy, decisions, 4800, auto_ms)
    if not run['transition']['all_exact'] or abs(run['transition']['time_s']-5.8)>1e-8:
        raise RuntimeError('exact 5.8s handover missing')
    after_weights = weights_hash(model)
    if before_weights != after_weights:
        raise RuntimeError('model weights changed')
    verify_runtime(ident,c); verify(ident)
    trace_info = tr.save(out/'trace_1ms.npz')
    dp = out/'decisions_5ms.npz'
    arrays = {k: np.asarray([r[k] for r in decisions]) for k in decisions[0]}
    arrays.update({k: np.asarray([r[k] for r in proxy.records]) for k in proxy.records[0]})
    np.savez_compressed(dp, **arrays)
    report = dict(control='student_sensory_ablation', condition=args.condition,
        experiment='partial sensory input disconnection; not retrained network or exported policy',
        teacher_reference_evidence=evidence, seed=args.seed,
        checkpoint=dict(path=str(Path(args.checkpoint).resolve()), sha256=sha256(args.checkpoint),
                        model_identity=payload['model_identity']),
        algorithm_prefix=dict(settle_s=1.,walk_s=4.8,teacher_metadata=teacher.metadata),
        settle=tr.summary(c,sw.SOURCE['settle']), teacher_warmup=tr.summary(c,sw.SOURCE['teacher']),
        autonomous=tr.summary(c,sw.SOURCE['student']), **run,
        intervention_continuity=proxy.cut_continuity, input_mapping=mapping,
        weights=dict(before_sha256=before_weights,after_sha256=after_weights,unchanged=True),
        compiled_physics_unchanged=True, external_forces='asserted zero at every Trace tick',
        trace=trace_info, decisions=dict(path=str(dp.resolve()),sha256=sha256(dp)),
        control_sources=sw.SOURCE, baseline_comparison=None, native=None,
        interpretation='S2/S5 declared intervention on substitute ANN; legacy five gates and strict '
            'cycles use original source 3 only. Original complete acceptance UNKNOWN. '
            'No claim of original policy success; improved intervention needs new declared export '
            'and revalidation. Only six actor sensory entries are cut; other inputs are retained.')
    atomic_json(out/'RESULT.json',report)
    print(json.dumps(dict(control=report['control'],condition=args.condition,
        transition_exact=run['transition']['all_exact'],
        autonomous={k:report['autonomous'][k] for k in ('all_gates','forward_signed_native_per_s',
            'foot_support_mean','com_height_min','nonfoot_force_max_W','slip_ratio_max')},
        cycles=run['autonomous_cycles']['complete_per_leg'],decisions=mapping['total'])),flush=True)
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--condition',choices=CONDITIONS,default='foot_velocity_zero',
                   help='identity is permitted only through ModelProxy for short parity checks')
    p.add_argument('--cycles',type=int,default=2)
    p.add_argument('--seed',type=int,default=20261110)
    return p

if __name__ == '__main__':
    evaluate(parser().parse_args())
