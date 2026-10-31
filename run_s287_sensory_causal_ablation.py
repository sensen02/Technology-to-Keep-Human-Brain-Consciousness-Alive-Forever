"""S287/P3: causal-carry live sensory versus zero submitted sensory current.

Import-safe. No historical runner is executed, modified, or used to generate action.
The live arm must exactly reproduce S286 causal_carry before the zero arm starts.
Successful capture/identity gates are not autonomous support or walking acceptance.
"""
from pathlib import Path
import argparse
import csv
import dataclasses
import hashlib
import json
import traceback
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
ARCHIVE = Path('/home/sensen/Desktop/engine修复验收')
S286 = ROOT / 'run_s286_command_timing_support.py'
S284 = ROOT / 'run_s284_friction_sensitivity.py'
S283 = ROOT / 'run_s283_neural_tau_support.py'
S276 = ARCHIVE / 's276-the-one-cell-that-stops/20261027T010000000000Z/scripts/s276.py'
REFERENCE = ARCHIVE / 's286-command-timing-support/20261026T170200Z'
FROZEN = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
INITIAL = ARCHIVE / 's177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz'
COMMAND_NAMES = ARCHIVE / 's244-chirality-of-the-5-of-6-arm/20261025T205824474993Z/CMD_A3_sym_matched.npz'
GAINS = ARCHIVE / 's255-static-gain-can-it-sustain/20261025T221308522972Z/ARM_VECTORS.json'
POOLS = ROOT / 'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json'
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
ARMS = ('live_sensory', 'sensory_zero')
TARGET = 'RHC_pleural_remotor_and_abductor'
W = 24.446354755878
BUDGET = dict(model_load=2, tier_build=1, network_init=2, live_step=8000,
              neural_tick=1600, live_fwd_position=2, live_forward=0,
              diagnostic_forward=8002)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(str(type(value)))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=serial) + '\n', encoding='utf-8')


def exact_array(x, y):
    """Bit-value identity with NaN sentinels at the same locations allowed."""
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    # Timing records intentionally have NaN initial-source/pending sentinels.
    if np.issubdtype(x.dtype, np.inexact):
        return bool(np.array_equal(x, y, equal_nan=True))
    return bool(np.array_equal(x, y))


def timing_regression():
    """Both arms use the identical causal schedule; no model/neural calls."""
    result = {}
    for arm in ARMS:
        rows = []
        for i in range(400):
            source = i - 1
            valid_from_ms = float(source + 1) if source >= 0 else 0.
            rows.append(dict(application_interval=i, source_interval=source,
                             valid_from_ms=valid_from_ms, applied_start_ms=float(i),
                             applied_end_ms=float(i + 1),
                             anticipatory=bool(i < valid_from_ms)))
        violations = sum(r['anticipatory'] for r in rows)
        result[arm] = dict(violations=violations, expected_violations=0,
                           passed=violations == 0, intervals=rows,
                           final_decoded=dict(source_interval=399, source_start_ms=399.,
                                              source_end_ms=400., valid_from_ms=400.,
                                              actual_applied_start_ms=None,
                                              actual_applied_end_ms=None,
                                              planned_start_ms=400., planned_end_ms=401.,
                                              pending=True))
    result['passed'] = all(result[a]['passed'] for a in ARMS)
    return result


def compare_archive(actual, reference):
    """Compare every historical NPZ field and all state/channel/contact/timing JSON.

    New sensory instrumentation fields may be extra, but no reference field may
    be missing. NaNs are accepted only as identical-position sentinel values.
    """
    checks = []
    for name in ('TRAJECTORY.npz', 'NEURAL.npz', 'CTRL.npz',
                 'COMMAND_TIMING.npz', 'INITIAL_CARRY.npz'):
        with np.load(actual / name, allow_pickle=False) as got, np.load(reference / name, allow_pickle=False) as ref:
            missing = sorted(set(ref.files) - set(got.files))
            checks.append(dict(file=name, field='reference_fields_present', passed=not missing, missing=missing))
            for key in ref.files:
                if key not in got.files:
                    continue
                x, y = got[key], ref[key]
                checks.append(dict(file=name, field=key, passed=exact_array(x, y),
                                   actual_shape=list(x.shape), reference_shape=list(y.shape),
                                   actual_dtype=str(x.dtype), reference_dtype=str(y.dtype)))
    for name in ('STATE.csv', 'CHANNELS.json', 'CONTACTS.json', 'PARAMETERS.json',
                 'DECODED_COMMANDS.json', 'COMMAND_APPLICATIONS.json'):
        checks.append(dict(file=name, field='file_bytes',
                           passed=(actual / name).read_bytes() == (reference / name).read_bytes()))
    passed = all(r['passed'] for r in checks)
    return dict(passed=passed, method='all reference NPZ fields exact dtype/shape/value; matching NaN sentinels; CSV/JSON bytes exact',
                checks=checks, reference=str(reference), candidate_allowed=passed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args(argv).output
    if not out.is_absolute() or out.exists():
        raise ValueError('fresh absolute --output required')
    out = out.resolve()
    # All reads/hashes are before any model/tier/network import or output artifact.
    xml_tree = ET.fromstring(FROZEN.read_bytes())
    ref_prereg = json.loads((REFERENCE / 'PREREG.json').read_text())
    ref_manifest = json.loads((REFERENCE / 'MANIFEST_SHA256.json').read_text())
    assert sha(FROZEN) == ref_prereg['original_xml_sha256']
    assert sha(S286) == ref_prereg['source_hashes'][str(S286)], 'S286 source changed since preregistration'
    assert sha(S286) == ref_manifest[str(S286)], 'S286 source changed since sealed archive'
    # Verify every originally registered runtime/model/mesh source, not just XML.
    assert all(sha(p) == h for p, h in ref_prereg['source_hashes'].items()), 'S286 runtime/model dependency changed'
    paths = [Path(__file__).resolve(), S286, S284, S283, S276, FROZEN, INITIAL, COMMAND_NAMES,
             GAINS, POOLS, ROOT / 'data/flywire/banc_connectome.npz',
             ROOT / 'outputs/neural_rebuild/sensory_channel_lookup.json',
             *sorted((ROOT / 'engine').rglob('*.py')),
             REFERENCE / 'PREREG.json', REFERENCE / 'MANIFEST_SHA256.json']
    paths += [Path(e.attrib['file']) for e in xml_tree.findall('.//asset/mesh')]
    paths += sorted(p for p in (REFERENCE / 'causal_carry').iterdir() if p.is_file())
    hashes = {str(p): sha(p) for p in dict.fromkeys(paths)}
    prereg = dict(round='S287', goal='P3 sensory-current causal ablation, not behavior acceptance',
        arms=list(ARMS), duration_ms=400, seed=0,
        network='all_empty ASSUMED_NETWORK_SCORE', fingerprint='56e0322a33c51b7f03aaae9f7df1ef3f',
        exc_tau_ms=3., inh_tau_ms=3., dt_s=.0001, neural_dt_ms=.5, sensory_ms=1,
        source_hashes=hashes, original_xml=str(FROZEN), original_xml_sha256=sha(FROZEN),
        historical_s286_sha256=sha(S286), reference_arm=str(REFERENCE / 'causal_carry'),
        model_diff='none: same original XML and original friction in both arms',
        parameter_diff={'only_change':'tier.advance(2, submitted): submitted=encoded cur for live_sensory; zeros_like(cur) for sensory_zero; background retained; both causal_carry'},
        initial_carry={'values':[0.] * 90, 'valid_from_s':0., 'source_interval':-1,
                       'source_start_s':None, 'source_end_s':None, 'causal_application_s':[0., .001]},
        time_axes={'body_state_s':'0..0.4 in 0.0001 increments, saved state after each step',
                   'encoder_s':'i*0.001 for i=0..399; d.ctrl set to carry before encoder unchanged',
                   'neural_s':'i*0.001+[0.0005,0.001]; actual_step_current captures argument received by net.step',
                   'decoded_command_s':'source [i,i+1]ms; valid_from=(i+1)ms',
                   'application_s':'both arms: [i,i+1]ms uses source i-1, initial source -1',
                   'last_command':'source [399,400]ms valid400ms, pending; planned [400,401]ms not applied',
                   'ctrl_time_s':'historical NEURAL ctrl_time_s is step END timestamp; application trace contains start/end'},
        current_recording={'sensory_current':'historical encoder output, NOT ablated in logs',
                           'encoded_current':'explicit alias of sensory_current, shape (400,N)',
                           'submitted_current':'actual argument to tier.advance, shape (400,N)',
                           'background_current':'copy of tier.background_nA once per arm, shape (N,)',
                           'actual_step_current':'copies of actual net.step current_nA argument before original bound step, shape (800,N)',
                           'expected_input':'actual_step_current[2*i:2*i+2] == background_current + submitted_current[i] exactly',
                           'zero_condition':'submitted_current identically zero; actual_step_current equals retained background; encoder continues'},
        gate='live all historical NPZ fields (including timing/initial carry) exact dtype/shape/value with equal-position NaNs; state/channels/contacts/parameters/decoded/applications bytes exact vs S286 causal_carry; failure stops BEFORE zero model/network',
        budget=BUDGET, budget_on_gate_fail='model1/tier1/net1/step4000/tick800/fwdPosition1/diagForward4001',
        regression='pure timestamps before model/neural imports; BOTH arms zero anticipation violations',
        sync='unchanged independent diagnostic copyData + forward; no diagnostic feedback',
        sequence='set d.ctrl=carry; applied=carry.copy; live ledger; encode; choose submitted; advance2; decode; apply OLD carry for10 steps; snapshot; state ledger',
        legacy_force='frame @ raw, up=-z; unchanged for live sensation/history',
        diagnostic_force='floor-side sign * frame.T @ raw; unchanged diagnostics',
        contact_raw_columns=['geom1','geom2','dist','pos_x','pos_y','pos_z', *['frame_'+str(i) for i in range(9)], *['force_'+str(i) for i in range(6)]],
        body_velocity='world angular xyz then linear xyz from mj_objectVelocity BODY, flg_local=0',
        limitations='ablation tests dependence on submitted modeled sensory current, not all sensory biology; native physical calibration/support/walking NOT_ESTABLISHED',
        non_neural_controller=False, auxiliary_force=False, training=False, prewarm=False)
    out.mkdir(parents=True, exist_ok=False)
    dump(out / 'PREREG.json', prereg)  # First output artifact.
    regression = timing_regression()
    dump(out / 'TIMESTAMP_REGRESSION.json', regression)
    if not regression['passed']:
        dump(out / 'STOP.json', dict(reason='pure timestamp regression failed', candidate_executed=False))
        return 2
    dump(out / 'CONFIG_DIFF.json', prereg['parameter_diff'])
    calls = {key: 0 for key in BUDGET}
    dump(out / 'CALL_LEDGER.json', calls)
    try:
        import mujoco as mj
        from engine.embodied.adapters import build_neural_tier, TierConfig, NeuralTier, OwnershipLedger
        from engine.embodied.sensory_channels import ChannelSensoryEncoder, ChannelQuantities, ChannelConfig, channel_index_map
        from engine.embodied.motor_output import build_motor_manifest, MotorOutput, MotorOutputConfig
        built = build_neural_tier(TierConfig(sign_fill='all_empty').validate())
        calls['tier_build'] += 1
        assert built['fingerprint'] == prereg['fingerprint']
        N = int(built['n'])
        roots = np.asarray(built['tier']['root_ids'])
        groups, _ = channel_index_map(N, roots)
        with np.load(COMMAND_NAMES, allow_pickle=True) as z:
            names = [str(x) for x in z['names']]
        with np.load(INITIAL) as z:
            qrel = np.asarray(z['qpos'], float).copy()
        gains = np.asarray(json.loads(GAINS.read_text())['v1'], float)
        pools = json.loads(POOLS.read_text())
        man = build_motor_manifest(pools, roots, names)
        resolved = {r['actuator']: r for r in pools['measured']['resolved']}
        cells = [int(np.flatnonzero(roots == int(c))[0]) for c in resolved[TARGET]['pool_root_ids'] if np.any(roots == int(c))]
        assert cells and len(names) == 90
        initial_checks = []
        parameter_reference = None
        background_reference = None

        def run(label):
            nonlocal parameter_reference, background_reference
            assert label in ARMS
            m = mj.MjModel.from_xml_path(str(FROZEN))
            calls['model_load'] += 1
            assert m.opt.timestep == .0001 and (m.nq, m.nv, m.nu, m.nmesh) == (49, 48, 90, 71)
            assert m.npair == 0
            geom_names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom)]
            foot_names = {lg.upper() + 'Tarsus' + str(i) + '_geom' for lg in LEGS for i in range(1, 6)}
            foot = {g: geom_names[g][:2].lower() for g in range(m.ngeom) if geom_names[g] in foot_names}
            assert len(foot) == 30
            floor = geom_names.index('floor')
            expected_friction = np.tile([1., .005, .0001], (m.ngeom, 1))
            assert np.array_equal(m.geom_friction, expected_friction)
            assert np.all(m.geom_priority == 0)
            joints = {}
            for lg in LEGS:
                ids = [j for j in range(m.njnt) if (mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or '').lower().startswith('joint_' + lg)]
                tib = [j for j in ids if 'tibia' in mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j).lower()][0]
                joints[lg] = dict(tib_q=int(m.jnt_qposadr[tib]), tib_v=int(m.jnt_dofadr[tib]), q=[int(m.jnt_qposadr[j]) for j in ids])
            tib0 = np.array([qrel[joints[lg]['tib_q']] for lg in LEGS])
            dest = out / label
            dest.mkdir()
            dump(dest / 'MODEL_METADATA.json', dict(names=names, body_names=[mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)], geom_names=geom_names, geom_leg=foot, floor=floor, geom_bodyid=m.geom_bodyid, geom_friction=m.geom_friction, geom_priority=m.geom_priority, geom_condim=m.geom_condim, integrator=int(m.opt.integrator), timestep=m.opt.timestep, nmesh=m.nmesh, target_index=names.index(TARGET), target_cells=cells, fingerprint=built['fingerprint'], model_xml_sha256=sha(FROZEN)))
            d = mj.MjData(m)
            d.qpos[:] = qrel
            d.qvel[:] = 0
            d.act[:] = 0
            d.ctrl[:] = 0
            mj.mj_fwdPosition(m, d)
            calls['live_fwd_position'] += 1
            q0 = d.qpos.copy()
            diag = mj.MjData(m)
            tier = NeuralTier(built, TierConfig(sign_fill='all_empty', background_seed=0).validate(), OwnershipLedger())
            calls['network_init'] += 1
            tier.net.p = dataclasses.replace(tier.net.p, inh_tau_ms=tier.net.p.exc_tau_ms)
            tier.net.p.validate()
            parameters = dataclasses.asdict(tier.net.p)
            assert parameters['inh_tau_ms'] == parameters['exc_tau_ms'] == 3.
            if parameter_reference is None:
                parameter_reference = parameters
            else:
                assert parameters == parameter_reference
            background_current = np.array(tier.background_nA, dtype=float, copy=True)
            assert background_current.shape == (N,) and np.isfinite(background_current).all()
            if background_reference is None:
                background_reference = background_current.copy()
            else:
                assert np.array_equal(background_reference, background_current)
            dump(dest / 'PARAMETERS.json', parameters)
            initial_checks.append(dict(arm=label, qpos_sha256=hashlib.sha256(d.qpos.tobytes()).hexdigest(), qvel_sha256=hashlib.sha256(d.qvel.tobytes()).hexdigest(), act_sha256=hashlib.sha256(d.act.tobytes()).hexdigest(), ctrl_sha256=hashlib.sha256(d.ctrl.tobytes()).hexdigest(), seed=0, fingerprint=built['fingerprint'], background_sha256=hashlib.sha256(background_current.tobytes()).hexdigest()))
            enc = ChannelSensoryEncoder(groups, N, ChannelConfig())
            dec = MotorOutput(man, MotorOutputConfig(rate_for_full_control_hz=450., per_actuator_gain=gains, filter_tau_s=5e-3))
            carry = np.zeros(90)
            np.savez_compressed(dest / 'INITIAL_CARRY.npz', command=carry.copy(), valid_from_s=np.array(0.), source_interval=np.array(-1), source_start_s=np.array(np.nan), source_end_s=np.array(np.nan))
            traj, ctrls, rows, ticks, sens, rates, rawrates, channels = [], [], [], [], [], [], [], []
            submitted_currents, actual_step_currents = [], []
            decoded, applications = [], []
            step = tier.net.step
            expected_step_drive = None

            def tick(*a, **kw):
                # Capture the REAL current argument delivered to original bound
                # net.step; do not synthesize this record from encoded/submitted logs.
                if a:
                    assert 'current_nA' not in kw
                    actual_drive = np.array(a[0], dtype=float, copy=True)
                else:
                    actual_drive = np.array(kw.get('current_nA', 0.0), dtype=float, copy=True)
                assert actual_drive.shape == (N,) and np.isfinite(actual_drive).all()
                assert expected_step_drive is not None
                assert np.array_equal(actual_drive, expected_step_drive), 'actual net.step argument differs from background+submitted'
                actual_step_currents.append(actual_drive)
                spikes = step(*a, **kw)  # Same arguments, one original neural tick.
                calls['neural_tick'] += 1
                ticks.append(dict(time_ms=tier.net.t_ms, spikes=spikes[cells].copy(), ge=tier.net.ge[cells].copy(), gi=tier.net.gi[cells].copy(), v=tier.net.v[cells].copy()))
                return spikes
            tier.net.step = tick

            def ledger(dd):
                lp = dict(lp=0., lm=0., bp=0., bm=0., np_=0, nm=0)
                per = {lg: dict(touch=0., F_world=np.zeros(3), F_up=0.) for lg in LEGS}
                for ci in range(dd.ncon):
                    c = dd.contact[ci]
                    g1, g2 = int(c.geom1), int(c.geom2)
                    if floor not in (g1, g2):
                        continue
                    other = g2 if g1 == floor else g1
                    fx = np.zeros(6)
                    mj.mj_contactForce(m, dd, ci, fx)
                    fw = np.asarray(c.frame, float).reshape(3, 3) @ fx[:3]
                    up = -float(fw[2])
                    if other in foot:
                        lg = foot[other]
                        per[lg]['touch'] = 1.
                        per[lg]['F_world'] += fw
                        per[lg]['F_up'] += up
                        lp['lp' if c.dist < 0 else 'lm'] += up
                    else:
                        lp['bp' if c.dist < 0 else 'bm'] += up
                        lp['np_' if c.dist < 0 else 'nm'] += 1
                return lp, per

            def snapshot():
                mj.mj_copyData(diag, m, d)
                mj.mj_forward(m, diag)
                calls['diagnostic_forward'] += 1
                mass = m.body_mass
                cv = np.zeros((m.nbody, 6))
                for b in range(m.nbody):
                    mj.mj_objectVelocity(m, diag, mj.mjtObj.mjOBJ_BODY, b, cv[b], 0)
                force = np.zeros((6, 3))
                penetrating = np.zeros(6, int)
                body = 0
                raw = []
                for ci in range(diag.ncon):
                    c = diag.contact[ci]
                    g1, g2 = int(c.geom1), int(c.geom2)
                    fx = np.zeros(6)
                    mj.mj_contactForce(m, diag, ci, fx)
                    raw.append([g1, g2, float(c.dist), *c.pos, *np.asarray(c.frame).ravel(), *fx])
                    if floor not in (g1, g2):
                        continue
                    other = g2 if g1 == floor else g1
                    fw = (1 if g1 == floor else -1) * (np.asarray(c.frame).reshape(3, 3).T @ fx[:3])
                    if other in foot:
                        li = LEGS.index(foot[other])
                        force[li] += fw
                        penetrating[li] += int(c.dist < 0)
                    elif c.dist < 0:
                        body += 1
                limited = np.flatnonzero(m.jnt_limited)
                qa = m.jnt_qposadr[limited]
                limit = np.maximum(np.maximum(m.jnt_range[limited, 0] - diag.qpos[qa], diag.qpos[qa] - m.jnt_range[limited, 1]), 0).max()
                return dict(time=float(d.time), qpos=d.qpos.copy(), qvel=d.qvel.copy(), act=d.act.copy(), ctrl=d.ctrl.copy(), qacc=diag.qacc.copy(), body_xpos=diag.xpos.copy(), body_xipos=diag.xipos.copy(), body_velocity=cv, com=(mass[:, None] * diag.xipos).sum(0) / mass.sum(), com_velocity=(mass[:, None] * cv[:, 3:]).sum(0) / mass.sum(), foot_force_world=force, foot_penetrating_count=penetrating, body_penetrating_contacts=body, max_limit=limit, external_applied=diag.xfrc_applied.copy(), qfrc_applied=diag.qfrc_applied.copy(), contact_raw=raw, qfrc_actuator=diag.qfrc_actuator.copy(), qfrc_constraint=diag.qfrc_constraint.copy())

            traj.append(snapshot())
            for i in range(400):
                d.ctrl[:] = carry  # S286 causal encoder phase unchanged in both arms.
                applied = carry.copy()
                lp, per = ledger(d)
                qv = {lg: ChannelQuantities(tibia_joint_dev_rad=float(d.qpos[joints[lg]['tib_q']] - q0[joints[lg]['tib_q']]), tibia_joint_vel_rad_s=float(d.qvel[joints[lg]['tib_v']]), multi_joint_dev_rad=float(np.mean(d.qpos[joints[lg]['q']] - q0[joints[lg]['q']])), contact_force_norm=float(np.linalg.norm(per[lg]['F_world'])), foot_contact=float(per[lg]['touch'])) for lg in LEGS}
                cur, rec = enc.encode(qv)
                sens.append(cur.copy())  # Historical field always remains encoder output.
                channels.append(rec)
                submitted = cur if label == 'live_sensory' else np.zeros_like(cur)
                submitted_currents.append(submitted.copy())
                assert np.array_equal(tier.background_nA, background_current)
                expected_step_drive = tier.background_nA + submitted
                before_steps = len(actual_step_currents)
                sp = tier.advance(2, submitted)
                assert len(actual_step_currents) == before_steps + 2
                if label == 'sensory_zero':
                    assert not np.any(submitted)
                    assert all(np.array_equal(v, background_current) for v in actual_step_currents[-2:])
                rawrates.append([np.array([sample[idx].mean() / .0005 if len(idx) else 0 for idx in dec.groups]) for sample in sp])
                carry, info = dec.consume(sp, spike_dt_s=.0005, start_s=i * .001, end_s=i * .001 + .001, connected=True)
                rates.append(np.asarray(info['rates_hz']))
                a = np.clip(applied, 0, 1.)
                pending = i == 399
                planned_start = (i + True) * .001  # Exact S286 causal arithmetic.
                planned_end = planned_start + .001
                decoded.append(dict(source_interval=i, source_start_s=info['source_start_s'], source_end_s=info['source_end_s'], valid_from_s=info['valid_from_s'], command=carry.copy(), planned_applied_start_s=planned_start, planned_applied_end_s=planned_end, actual_applied_start_s=None if pending else planned_start, actual_applied_end_s=None if pending else planned_end, pending_at_capture_end=pending))
                src = i - 1
                applications.append(dict(application_interval=i, applied_start_s=i * .001, applied_end_s=(i + 1) * .001, source_interval=src, source_start_s=None if src < 0 else src * .001, source_end_s=None if src < 0 else src * .001 + .001, valid_from_s=0. if src < 0 else src * .001 + .001, command=a.copy(), initial_carry=src < 0, anticipatory=False))
                for j in range(10):
                    d.ctrl[:] = a
                    ctrls.append(a.copy())
                    mj.mj_step(m, d)
                    calls['live_step'] += 1
                    traj.append(snapshot())
                lp, _ = ledger(d)
                tib = np.array([d.qpos[joints[lg]['tib_q']] for lg in LEGS])
                dx, dy = float(d.qpos[0] - q0[0]), float(d.qpos[1] - q0[1])
                rows.append(dict(t_ms=float(i + 1), dx=dx, dy=dy, dz=float(d.qpos[2] - q0[2]), horiz=float(np.hypot(dx, dy)), leg_up_pen_W=lp['lp'] / W, tib_excursion=float(np.mean(abs(tib - tib0))), n_body_ground=int(lp['np_'] + lp['nm']), nbg_pen=int(lp['np_'])))
            counts = np.array([len(r['contact_raw']) for r in traj], dtype=np.int64)
            offsets = np.concatenate(([0], np.cumsum(counts)))
            raw = np.asarray([c for r in traj for c in r['contact_raw']], dtype=float).reshape(-1, 21)
            arrays = {k: np.asarray([r[k] for r in traj]) for k in traj[0] if k != 'contact_raw'}
            np.savez_compressed(dest / 'TRAJECTORY.npz', **arrays, contact_raw=raw, contact_offsets=offsets, contact_count=counts)
            dump(dest / 'CONTACTS.json', [r['contact_raw'] for r in traj])
            encoded_array = np.asarray(sens)
            submitted_array = np.asarray(submitted_currents)
            actual_array = np.asarray(actual_step_currents)
            assert encoded_array.shape == submitted_array.shape == (400, N)
            assert actual_array.shape == (800, N)
            assert np.array_equal(actual_array, np.repeat(background_current[None, :] + submitted_array, 2, axis=0))
            assert np.array_equal(tier.background_nA, background_current)
            if label == 'live_sensory':
                assert np.array_equal(encoded_array, submitted_array)
            else:
                assert not np.any(submitted_array)
            np.savez_compressed(dest / 'NEURAL.npz', time_ms=np.array([r['time_ms'] for r in ticks]), **{k: np.array([r[k] for r in ticks]) for k in ('spikes', 'ge', 'gi', 'v')}, target_cells=np.array(cells), sensory_current=encoded_array, encoded_current=encoded_array, submitted_current=submitted_array, background_current=background_current, actual_step_current=actual_array, sensory_time_ms=np.arange(400, dtype=float), actual_step_time_ms=np.array([r['time_ms'] for r in ticks]), rates_hz=np.asarray(rates), raw_rates_hz=np.asarray(rawrates), ctrl=np.asarray(ctrls), ctrl_time_s=np.asarray(arrays['time'][1:]))
            dump(dest / 'CURRENT_PATH_CHECK.json', dict(passed=True, encoded_shape=list(encoded_array.shape), submitted_shape=list(submitted_array.shape), actual_step_shape=list(actual_array.shape), background_shape=list(background_current.shape), actual_argument_capture='net.step positional current_nA (or keyword), copied before original bound step; exactly one original call', actual_equals_background_plus_submitted=True, background_retained=True, zero_submitted=label == 'sensory_zero', calls_recorded=len(actual_step_currents)))
            np.savez_compressed(dest / 'CTRL.npz', ctrl=np.asarray(ctrls), names=np.asarray(names))
            dump(dest / 'CHANNELS.json', channels)
            dump(dest / 'DECODED_COMMANDS.json', decoded)
            dump(dest / 'COMMAND_APPLICATIONS.json', applications)
            np.savez_compressed(dest / 'COMMAND_TIMING.npz', decoded=np.asarray([r['command'] for r in decoded]), applied=np.asarray([r['command'] for r in applications]), applied_source_index=np.asarray([r['source_interval'] for r in applications]), pending_final=carry.copy(), decoded_command=np.asarray([r['command'] for r in decoded]), source_start_s=np.asarray([r['source_start_s'] for r in decoded]), source_end_s=np.asarray([r['source_end_s'] for r in decoded]), valid_from_s=np.asarray([r['valid_from_s'] for r in decoded]), planned_applied_start_s=np.asarray([r['planned_applied_start_s'] for r in decoded]), planned_applied_end_s=np.asarray([r['planned_applied_end_s'] for r in decoded]), actual_applied_start_s=np.asarray([np.nan if r['actual_applied_start_s'] is None else r['actual_applied_start_s'] for r in decoded]), actual_applied_end_s=np.asarray([np.nan if r['actual_applied_end_s'] is None else r['actual_applied_end_s'] for r in decoded]), pending_at_capture_end=np.asarray([r['pending_at_capture_end'] for r in decoded]), applied_command=np.asarray([r['command'] for r in applications]), applied_start_s=np.asarray([r['applied_start_s'] for r in applications]), applied_end_s=np.asarray([r['applied_end_s'] for r in applications]), applied_source_interval=np.asarray([r['source_interval'] for r in applications]), applied_valid_from_s=np.asarray([r['valid_from_s'] for r in applications]), initial_carry=np.zeros(90))
            with (dest / 'STATE.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            assert len(traj) == 4001 and len(ctrls) == 4000 and len(ticks) == 800
            assert np.asarray(ctrls).shape == (4000, 90)
            assert np.array_equal(arrays['qpos'][0], qrel)
            assert not np.any(arrays['external_applied']) and not np.any(arrays['qfrc_applied'])
            expected_apply = np.concatenate([np.zeros((1, 90)), np.asarray([r['command'] for r in decoded])[:-1]])
            assert np.array_equal(np.asarray([r['command'] for r in applications]), expected_apply)
            assert np.array_equal(np.asarray(ctrls), np.repeat(expected_apply, 10, axis=0))
            assert all(r['applied_start_s'] + 1e-12 >= r['valid_from_s'] for r in applications)
            dump(out / 'CALL_LEDGER.json', calls)

        run('live_sensory')
        gate = compare_archive(out / 'live_sensory', REFERENCE / 'causal_carry')
        dump(out / 'GATE_LIVE_EXACT.json', gate)
        if not gate['passed']:
            dump(out / 'STOP.json', dict(reason='live_sensory exact reproduction failed; sensory_zero NOT executed', candidate_executed=False, calls=calls))
            return 2
        run('sensory_zero')
        assert {k: v for k, v in initial_checks[0].items() if k != 'arm'} == {k: v for k, v in initial_checks[1].items() if k != 'arm'}
        dump(out / 'INITIAL_INVARIANTS.json', dict(passed=True, arms=initial_checks, initial_carry=[0.] * 90, neural_parameters_identical=True, background_identical=True, model_identical=True, causal_schedule_identical=True, decoder='same manifest/V1/rate450/filter5ms', exact_historical_gate=True))
        assert calls == BUDGET
        unchanged = all(sha(p) == h for p, h in hashes.items())
        assert unchanged, 'source changed during run'
        dump(out / 'COMPLETE.json', dict(capture_complete=True, gate_live_exact=True, calls=calls, source_unchanged=unchanged, actual_step_argument_recorded=True, background_retained=True, auxiliary_force=False, training=False, non_neural_controller=False, behavior='NOT_ESTABLISHED'))
    except BaseException:
        dump(out / 'CALL_LEDGER.json', calls)
        dump(out / 'FAILED.json', dict(traceback=traceback.format_exc(), calls=calls))
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
