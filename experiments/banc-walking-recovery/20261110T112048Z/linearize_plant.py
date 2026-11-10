"""Private offline 192D physical small-signal map at ONE actual teacher-prefix state.

The prefix alone is an actual trajectory. Restores and tangent/target/cache writes
are explicitly offline fixtures, never walking or accepted-trajectory evidence.
No helper files are modified. A local linearization is not a stability proof.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import mujoco as mj
import torch
import student_wave as sw
from physics import Core, MAX_RATE
from static_teacher import StaticTeacher
from identity import HERE, atomic_json, capture, verify, verify_runtime, sha256

BLOCKS = {'root_position': (0, 3), 'root_rotation': (3, 6), 'joint_position': (6, 48),
          'qvel': (48, 96), 'target': (96, 138), 'previous_action': (138, 180),
          'pad_previous': (180, 186), 'pad_rate': (186, 192)}
SIMPLE = (str, bool, int, float, np.integer, np.floating, type(None))


def digest(a):
    a = np.asarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()


def simple_state(obj):
    result, refs = {}, {}
    for k, v in vars(obj).items():
        if isinstance(v, np.ndarray):
            result[k] = v.copy()
        elif isinstance(v, SIMPLE):
            result[k] = v
        else:
            refs[k] = v
    return result, refs


def restore_simple(obj, state, refs):
    if set(vars(obj)) != set(state) | set(refs):
        raise RuntimeError('Mutable attribute inventory changed')
    for k, v in refs.items():
        if getattr(obj, k) is not v:
            raise RuntimeError('Non-simple reference changed: ' + k)
    for k, v in state.items():
        setattr(obj, k, v.copy() if isinstance(v, np.ndarray) else v)


def simple_hashes(obj):
    state, _ = simple_state(obj)
    return {k: digest(v) if isinstance(v, np.ndarray) else v for k, v in state.items()}


def data_hashes(d):
    result = {'time': float(d.time), 'warning_number': digest(d.warning.number),
              'warning_lastinfo': digest(d.warning.lastinfo)}
    for k in dir(d):
        if not k.startswith('_'):
            v = getattr(d, k)
            if isinstance(v, np.ndarray):
                result[k] = digest(v)
    # Contact structs and all other physical fields are copied by mj_copyData.
    result['contacts'] = digest(np.array([[x.geom1, x.geom2, x.dist, *x.pos,
                                          *x.frame] for x in d.contact[:d.ncon]]))
    return result


class Snapshot:
    def __init__(self, c, previous):
        self.data = mj.MjData(c.plant)
        mj.mj_copyData(self.data, c.plant, c.d)
        self.core, self.core_refs = simple_state(c)
        self.act, self.act_refs = simple_state(c.actuator)
        self.previous = previous.copy()
        self.hashes = self.fingerprint(c, previous)
        self.restores = 0

    @staticmethod
    def fingerprint(c, previous):
        return dict(data=data_hashes(c.d), core=simple_hashes(c),
                    actuator=simple_hashes(c.actuator), previous=digest(previous))

    def restore(self, c):
        mj.mj_copyData(c.d, c.plant, self.data)
        restore_simple(c, self.core, self.core_refs)
        restore_simple(c.actuator, self.act, self.act_refs)
        previous = self.previous.copy()
        if self.fingerprint(c, previous) != self.hashes:
            raise RuntimeError('Complete restored state fingerprint mismatch')
        self.restores += 1
        return previous


def contact_state(c):
    m = c.measure()
    return dict(ncon=int(c.d.ncon), nefc=int(c.d.nefc),
                geom_pairs=[[int(x.geom1), int(x.geom2)] for x in c.d.contact[:c.d.ncon]],
                pad_contact_counts=m['contact'].tolist(), loads_native=m['loads'].tolist(),
                nonfoot_native=float(m['nonfoot']), com_native=float(m['com']),
                warning_number=c.d.warning.number.tolist(), warning_lastinfo=c.d.warning.lastinfo.tolist())


class ContactRecorder:
    """Record native solved contact/constraint changes, without retaining MjData."""
    def __init__(self):
        self.clear()

    def clear(self):
        self.count = 0
        self.transitions = []
        self.last = None
        self.hasher = hashlib.sha256()
        self.warnings = []

    def before(self, c):
        return None

    def solved(self, c, before):
        d = c.d
        pairs = tuple((int(x.geom1), int(x.geom2)) for x in d.contact[:d.ncon])
        # Active constraint states are included, not just contact pair membership.
        state = (int(d.ncon), int(d.nefc), pairs, tuple(map(int, d.efc_state[:d.nefc])))
        encoded = json.dumps(state, separators=(',', ':')).encode()
        self.hasher.update(encoded)
        if state != self.last:
            self.transitions.append(dict(native_index=self.count, ncon=state[0], nefc=state[1],
                geom_pairs=[list(p) for p in pairs], efc_state=list(state[3])))
            self.last = state
        if d.warning.number.any():
            self.warnings.append(dict(native_index=self.count, number=d.warning.number.tolist(),
                                      lastinfo=d.warning.lastinfo.tolist()))
        self.count += 1
        return None

    def after(self, c, before, solved):
        return None

    def report(self):
        return dict(native_samples=self.count, solved_contact_constraint_sha256=self.hasher.hexdigest(),
                    transitions=self.transitions, warnings=self.warnings)


def action(model, x):
    with torch.no_grad():
        # Exactly the deterministic feed-forward actor path; critic has no action effect.
        a = torch.tanh(model.actor(torch.as_tensor(x, dtype=torch.float64)[None]))[0].numpy()
    return sw.bounded_action(a)


def tail(c, previous):
    return np.r_[c.d.qvel.copy(), c.actuator.target_rad.copy(), previous.copy(),
                 c._padz_prev.copy(), c._pad_rate.copy()]


def perturb(c, previous, col, amount):
    if col < 48:
        tangent = np.zeros(48); tangent[col] = amount
        mj.mj_integratePos(c.plant, c.d.qpos, tangent, 1.)
        mj.mj_forward(c.plant, c.d)
    elif col < 96:
        c.d.qvel[col - 48] += amount
        mj.mj_forward(c.plant, c.d)
    elif col < 138:
        c.actuator.target_rad[col - 96] += amount
    elif col < 180:
        previous[col - 138] += amount
    elif col < 186:
        c._padz_prev[col - 180] += amount
    else:
        c._pad_rate[col - 186] += amount


def spectral(J, scales):
    scaled = J * scales[None, :] / scales[:, None]
    eigenvalues, vectors = np.linalg.eig(scaled)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    modes = []
    for i in order[:12]:
        v = vectors[:, i]
        energy = {k: float(np.sum(np.abs(v[a:b]) ** 2) / np.sum(np.abs(v) ** 2))
                  for k, (a, b) in BLOCKS.items()}
        modes.append(dict(real=float(eigenvalues[i].real), imag=float(eigenvalues[i].imag),
                          magnitude=float(abs(eigenvalues[i])), scaled_block_energy=energy))
    return dict(spectral_radius=float(np.max(np.abs(eigenvalues))),
        largest_abs_real=float(np.max(np.abs(eigenvalues.real))),
        largest_abs_imag=float(np.max(np.abs(eigenvalues.imag))),
        norm2_native_mixed_units=float(np.linalg.norm(J, 2)),
        norm2_scaled=float(np.linalg.norm(scaled, 2)),
        frozen_J_power20_norm2_native=float(np.linalg.norm(np.linalg.matrix_power(J, 20), 2)),
        frozen_J_power20_norm2_scaled=float(np.linalg.norm(np.linalg.matrix_power(scaled, 20), 2)),
        dominant_modes=modes), eigenvalues, scaled


def static_checks():
    # Algebraic tests are diagnostic-critical; no physical trajectory or installation.
    scales = np.linspace(.1, 3., 192)
    matrix = np.diag(np.linspace(.1, .99, 192))
    report, ev, scaled = spectral(matrix, scales)
    assert abs(report['spectral_radius'] - .99) < 1e-14
    assert np.allclose(matrix, scaled)
    assert sum(b-a for a,b in BLOCKS.values()) == 192
    assert len(set(i for a,b in BLOCKS.values() for i in range(a,b))) == 192
    # Confirm API available in installed MuJoCo, not an assumed fallback.
    assert callable(mj.mj_copyData) and callable(mj.mj_integratePos) and callable(mj.mj_differentiatePos)
    result = dict(all_passed=True, checks=['192 coordinates exhaustive and disjoint',
        'similarity scaling and spectral calculation', 'installed MuJoCo full copy/tangent APIs'],
        source_hashes={str(Path(__file__).resolve()): sha256(__file__)},
        protocol='private offline fixtures; strict bitwise restore verified only by actual run')
    atomic_json(HERE/'LINEARIZATION_CHECKS.json', result)
    return result


def run(args):
    started = time.time()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(args.checkpoint).resolve()
    sw.seed_all(args.seed)
    recorder = ContactRecorder()
    c = Core(adhesion=False)
    model, payload = sw.load_student(checkpoint, c)
    if model.architecture != 'ff238':
        raise ValueError('Only stateless FF238 accepted')
    # No additional model construction/reset after actual prefix.
    config = dict(kind='auxiliary offline full physical local FD', seed=args.seed,
        checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), settle_ms=1000,
        teacher_prefix_ms=4800, teacher=sw.teacher_config(3), dt_s=.005,
        state_dimension=192, blocks=BLOCKS, epsilon=1e-5, sensitivity_epsilon=3e-6,
        actual_trajectory='one zero-settle plus teacher prefix only; probes private restored fixtures',
        acceptance='not walking evidence or stability proof', command=vars(args))
    extras = sw.SOURCE_FILES + [Path(__file__).resolve(), checkpoint, HERE/'LINEARIZATION_CHECKS.json']
    if args.prefix_decisions:
        extras.append(Path(args.prefix_decisions).resolve())
    ident = capture(c, config, extra_paths=extras)
    atomic_json(out/'IDENTITY.json', ident)
    prefix_trace = sw.Trace(full=False)
    sw.settle(c, prefix_trace)
    teacher = StaticTeacher(c, sw.teacher_config(3))
    previous = np.zeros(42)
    expected = None
    if args.prefix_decisions:
        with np.load(args.prefix_decisions, allow_pickle=False) as z:
            expected = {k: z[k][:961].copy() for k in ('X238', 'action', 'source', 'time')}
    prefix_obs_maxdiff = 0.; prefix_actions_exact = True
    for k in range(960):
        x = c.obs238(previous, sw.TARGET_SPEED)
        a, _ = teacher.action(c, (k*5)*.001)
        a = sw.bounded_action(a)
        if expected is not None:
            prefix_obs_maxdiff = max(prefix_obs_maxdiff, float(np.max(abs(x - expected['X238'][k]))))
            prefix_actions_exact &= np.array_equal(a, expected['action'][k])
            if expected['source'][k] != sw.SOURCE['teacher'] or float(c.d.time) != expected['time'][k]:
                raise RuntimeError('Expected prefix source/time mismatch')
        for _ in range(5):
            prefix_trace.tick(c, a, sw.SOURCE['teacher'])
        previous = a.copy()
    if expected is not None and (prefix_obs_maxdiff != 0. or not prefix_actions_exact):
        raise RuntimeError('Actual teacher prefix differs from actual evaluation')
    # Teacher can never be queried by probes, even accidentally.
    teacher.action = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('Teacher called after prefix'))
    prefix = dict(settle=prefix_trace.summary(c, sw.SOURCE['settle']),
        teacher=prefix_trace.summary(c, sw.SOURCE['teacher']), time_s=float(c.d.time),
        actual_ticks=5800, native_steps=464000, accepted_motion_state_writes=0,
        expected_prefix_X238_max_abs_diff=prefix_obs_maxdiff if expected else None,
        expected_prefix_actions_bitwise=bool(prefix_actions_exact) if expected else None,
        teacher_calls=960, source_counts={'settle':1000, 'teacher':4800},
        baseline_contact=contact_state(c))
    x_before = c.obs238(previous, sw.TARGET_SPEED).copy()
    first_action = action(model, x_before)
    if expected is not None:
        prefix['handover_X238_bitwise'] = bool(np.array_equal(x_before, expected['X238'][960]))
        prefix['first_student_action_bitwise'] = bool(np.array_equal(first_action, expected['action'][960]))
        if not prefix['handover_X238_bitwise'] or not prefix['first_student_action_bitwise']:
            raise RuntimeError('Handover observation/student command differs from actual evaluation')
    c.recorder = recorder
    snap = Snapshot(c, previous)
    np.savez_compressed(out/'BASELINE_STATE.npz', qpos=c.d.qpos.copy(), qvel=c.d.qvel.copy(),
        target=c.actuator.target_rad.copy(), previous=previous, pad_previous=c._padz_prev.copy(),
        pad_rate=c._pad_rate.copy(), obs238=x_before, first_student_action=first_action)
    atomic_json(out/'PREFIX.json', prefix)
    print(json.dumps(dict(stage='actual_prefix_complete', time_s=c.d.time,
        teacher_gates=prefix['teacher']['all_gates'], expected_X_diff=prefix['expected_prefix_X238_max_abs_diff'])), flush=True)
    reports = []

    def probe(col=None, amount=0., tag='baseline'):
        p = snap.restore(c)
        # Compare exact original before-observation before every perturbation.
        if not np.array_equal(c.obs238(p, sw.TARGET_SPEED), x_before):
            raise RuntimeError('Restored before obs238 differs')
        if col is not None:
            perturb(c, p, col, amount)
        obs = c.obs238(p, sw.TARGET_SPEED)
        a = action(model, obs)
        before = contact_state(c)
        recorder.clear()
        tick_contacts = []
        for _ in range(5):
            sw.no_external(c)
            c.tick(a)
            sw.no_external(c)
            tick_contacts.append(contact_state(c))
        p = a.copy()
        after_obs = c.obs238(p, sw.TARGET_SPEED).copy()
        report = dict(tag=tag, column=col, amount=amount, before=before,
                      after=tick_contacts[-1], tick_contacts=tick_contacts, native=recorder.report(),
                      max_action_abs=float(np.max(abs(a))), angle_clip_count=int(c.actuator.angle_clip_mask.sum()),
                      rate_clip_count=int(c.actuator.rate_clip_mask.sum()))
        reports.append(report)
        return c.d.qpos.copy(), tail(c, p), after_obs, Snapshot.fingerprint(c,p), a.copy()

    base = probe(tag='unperturbed_1')
    repeat = probe(tag='unperturbed_2')
    equal = dict(qpos=np.array_equal(base[0],repeat[0]), tail=np.array_equal(base[1],repeat[1]),
        obs238_after=np.array_equal(base[2],repeat[2]), full_data_core_actuator=base[3]==repeat[3],
        action=np.array_equal(base[4],repeat[4]), native_contacts=reports[0]['native']==reports[1]['native'])
    if not all(equal.values()):
        raise RuntimeError('Unperturbed restore replay is not BITWISE identical: '+str(equal))
    atomic_json(out/'RESTORE_PROOF.json',dict(all_bitwise=True, fields=equal,
        before_obs238_sha256=digest(x_before), after_obs238_sha256=digest(base[2]),
        snapshot_fingerprints=snap.hashes, next_fingerprints=base[3],
        copied='entire physical MjData via mj_copyData into one persistent same-model MjData',
        core_simple_inventory=sorted(snap.core), core_stable_reference_inventory=sorted(snap.core_refs),
        actuator_simple_inventory=sorted(snap.act), actuator_stable_reference_inventory=sorted(snap.act_refs),
        sensory='pure sensory.features; no mutable caches', shim='compiled model views; no mutable caches'))
    print(json.dumps(dict(stage='restore_proved', all_bitwise=True)), flush=True)

    def delta(result):
        dq = np.zeros(48)
        mj.mj_differentiatePos(c.plant, dq, 1., base[0], result[0])
        return np.r_[dq, result[1]-base[1]]

    J = np.zeros((192,192))
    eps = 1e-5
    for col in range(192):
        plus = probe(col,eps,'central_plus')
        minus = probe(col,-eps,'central_minus')
        J[:,col] = (delta(plus)-delta(minus))/(2*eps)
        if (col+1)%24==0:
            print(json.dumps(dict(stage='central_FD', columns_done=col+1, total=192,
                                  elapsed_s=round(time.time()-started,1))),flush=True)
    selected = np.array([0,2,3,6,27,48,69,96,117,138,180,186])
    small = np.zeros((192,len(selected)))
    sensitivity = []
    h = sw.physical_halfspan(c)
    qscale = np.ones(48); qscale[c.a_dof] = h
    scales = np.r_[qscale,np.full(48,10.),h,np.full(42,.1),np.full(6,.05),np.ones(6)]
    for k,col in enumerate(selected):
        plus = probe(int(col),3e-6,'sensitivity_plus')
        minus = probe(int(col),-3e-6,'sensitivity_minus')
        small[:,k] = (delta(plus)-delta(minus))/(6e-6)
        native_diff = small[:,k]-J[:,col]
        sd = native_diff*scales[col]/scales
        sj = J[:,col]*scales[col]/scales
        sensitivity.append(dict(column=int(col), native_difference_norm=float(np.linalg.norm(native_diff)),
            native_relative_difference=float(np.linalg.norm(native_diff)/max(np.linalg.norm(J[:,col]),1e-15)),
            scaled_difference_norm=float(np.linalg.norm(sd)),
            scaled_relative_difference=float(np.linalg.norm(sd)/max(np.linalg.norm(sj),1e-15))))
    if not np.isfinite(J).all():
        raise FloatingPointError('Nonfinite Jacobian')
    full, eig, scaled = spectral(J, scales)
    T = J[96:138,96:138]
    t_eigs = np.linalg.eigvals(T)
    # All post-prefix calls were probes; no teacher action can reach this path.
    for r in reports:
        r['native_contact_constraints_changed_from_unperturbed'] = (
            r['native']['solved_contact_constraint_sha256'] != reports[0]['native']['solved_contact_constraint_sha256'])
        r['post_contact_pairs_changed_from_unperturbed'] = r['after']['geom_pairs'] != reports[0]['after']['geom_pairs']
    np.savez_compressed(out/'JACOBIAN.npz', J=J,J_scaled=scaled,scales=scales,eigenvalues=eig,
        T_memory_only=T,T_memory_eigenvalues=t_eigs,sensitivity_columns=selected,
        sensitivity_J=small,baseline_next_qpos=base[0],baseline_next_tail=base[1],
        baseline_next_obs238=base[2])
    atomic_json(out/'PROBES.json',dict(probes=reports,
        endpoint_loads='native contact forces at actual before/1ms boundaries, separate from native active-set logs',
        native_log='solved native contact pairs, ncon, nefc and all active constraint states; transitions plus full-sequence digest'))
    # Compiled model/source identities MUST remain unchanged, including actuator parameters.
    verify_runtime(ident,c)
    restored = snap.restore(c)
    assert np.array_equal(c.obs238(restored,sw.TARGET_SPEED),x_before)
    verify(ident)
    result = dict(status='COMPLETED_AUXILIARY_DIAGNOSTIC',config=config,prefix=prefix,
        restore_bitwise=equal, full_coupled=full,
        target_memory_only=dict(spectral_radius=float(np.max(abs(t_eigs))),
            interpretation='T next versus T current with qpos/qvel/pad caches/previous fixed; not full physical stability'),
        sensitivity=sensitivity,
        scaling=dict(formula='z=x/scales; J_scaled=diag(1/scales) J diag(scales)',
            values=scales.tolist(),qpos='root translation/rotation 1; actuated hinge tangent native half-range H',
            qvel=10.,target='native actuator half-range H',previous_action=.1,pad_previous=.05,pad_rate=1.,
            caution='norms and modal energies depend on this declared mixed-unit scaling; eigenvalues invariant'),
        contact_changes=dict(native_constraints_changed_probes=sum(r['native_contact_constraints_changed_from_unperturbed'] for r in reports),
            post_pair_changed_probes=sum(r['post_contact_pairs_changed_from_unperturbed'] for r in reports),
            total_probes=len(reports),all_warning_records=[r for r in reports if r['native']['warnings']]),
        execution=dict(actual_prefix_ticks=5800,actual_prefix_native_steps=464000,
            private_probe_holds=len(reports),private_native_steps=len(reports)*400,
            private_aggregate_simulated_s=len(reports)*.005,restore_calls=snap.restores,
            fixture_perturbations=408,teacher_calls_after_prefix=0,
            physical_data_deepcopy_count=1,wall_time_s=time.time()-started,
            actual_prefix_state_writes=0,offline_restore_writes=snap.restores,
            offline_qpos_fixture_probes=sum(r['column'] is not None and r['column']<48 for r in reports),
            offline_qvel_fixture_probes=sum(r['column'] is not None and 48<=r['column']<96 for r in reports),
            source_and_compiled_arrays_verified=True),
        limitations=['ONE nonstationary teacher-prefix operating state; not a fixed point or cycle map',
            'Frozen-J power20 is algebraic local amplification, not 20 actual sequential physical steps',
            'Contact/constraint bifurcations are explicit; central FD may cross hybrid branches',
            'Existing proprioception casts 183 channels to float32; FD sensitivity includes that quantization',
            'Augmented pad previous memory embeds geometric redundancy; no reachability/stability proof',
            'MuJoCo warmstart/solver/internal state restored exactly but excluded from the declared 192 physical coordinates',
            'This auxiliary diagnostic does not establish walking, full stability, falling cause, or neural biological validity'],
        artifacts={str(out/name):sha256(out/name) for name in ('IDENTITY.json','PREFIX.json','RESTORE_PROOF.json','BASELINE_STATE.npz','JACOBIAN.npz','PROBES.json')})
    atomic_json(out/'RESULT.json',result)
    print(json.dumps(dict(stage='done',rho_full=full['spectral_radius'],
        rho_T_only=result['target_memory_only']['spectral_radius'],norm_scaled=full['norm2_scaled'],
        changed_native_probes=result['contact_changes']['native_constraints_changed_probes'],
        wall_time_s=result['execution']['wall_time_s'])),flush=True)
    return result


def validate(directory):
    out = Path(directory).resolve()
    result = json.loads((out/'RESULT.json').read_text())
    for path, old in result['artifacts'].items():
        if sha256(path) != old:
            raise RuntimeError('Output artifact drift: '+path)
    verify(json.loads((out/'IDENTITY.json').read_text()))
    with np.load(out/'JACOBIAN.npz',allow_pickle=False) as z:
        J=z['J']; scales=z['scales']
        report, eig, scaled=spectral(J,scales)
        assert J.shape==(192,192) and np.isfinite(J).all()
        assert np.array_equal(scaled,z['J_scaled'])
        assert report==result['full_coupled']
        assert np.array_equal(np.linalg.eigvals(z['T_memory_only']),z['T_memory_eigenvalues'])
    proof=json.loads((out/'RESTORE_PROOF.json').read_text())
    assert proof['all_bitwise'] and all(proof['fields'].values())
    probes=json.loads((out/'PROBES.json').read_text())['probes']
    assert len(probes)==410 and all(p['native']['native_samples']==400 for p in probes)
    assert not any(p['native']['warnings'] for p in probes)
    validation=dict(all_passed=True, independent_process_artifact_reload=True,
        checks=['all output hashes', 'source identity', '192x192 finite matrix',
                'scaled matrix exact', 'spectral metrics exact reload', 'restore proof',
                '410 probes x 400 native steps', 'no native warnings'],
        result_sha256=sha256(out/'RESULT.json'))
    atomic_json(out/'VALIDATION.json',validation)
    print(json.dumps(validation),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint');p.add_argument('--out');p.add_argument('--seed',type=int,default=20261110)
    p.add_argument('--prefix-decisions');p.add_argument('--check',action='store_true')
    p.add_argument('--validate')
    args=p.parse_args()
    if args.check:
        print(json.dumps(static_checks()),flush=True)
    elif args.validate:
        validate(args.validate)
    else:
        if not args.out or not args.checkpoint:
            p.error('--checkpoint and --out required for run')
        try:
            run(args)
        except Exception:
            if Path(args.out).is_dir():
                atomic_json(Path(args.out)/'TECHNICAL_FAILURE.json',dict(status='STOPPED_TECHNICAL_NO_CLAIM',traceback=traceback.format_exc()))
            raise


if __name__=='__main__':
    main()
