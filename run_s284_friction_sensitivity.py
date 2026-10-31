"""S284/P1/S2: fixed S283 tau_equal network, foot-ground friction sensitivity.

Import-safe; no historical runner is imported or executed. This file does not run
until main is called. Only the new, absolute --output directory is writable.
"""
from pathlib import Path
import argparse
import csv
import dataclasses
import hashlib
import json
import re
import traceback
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
ARCHIVE = Path('/home/sensen/Desktop/engine修复验收')
S283 = ROOT / 'run_s283_neural_tau_support.py'
S276 = ARCHIVE / 's276-the-one-cell-that-stops/20261027T010000000000Z/scripts/s276.py'
FROZEN = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
INITIAL = ARCHIVE / 's177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz'
COMMAND_NAMES = ARCHIVE / 's244-chirality-of-the-5-of-6-arm/20261025T205824474993Z/CMD_A3_sym_matched.npz'
GAINS = ARCHIVE / 's255-static-gain-can-it-sustain/20261025T221308522972Z/ARM_VECTORS.json'
POOLS = ROOT / 'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json'
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
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


def derive_xml(source):
    """Preserve every source byte except 30 local friction-attribute insertions.

    This deliberately supports the inspected frozen XML only, not arbitrary MJCF
    defaults. Fail closed if defaults/contact rules/assets cease to match it.
    Floor is NOT edited: raising its friction would also change body-floor pairs.
    """
    tree = ET.fromstring(source)
    if tree.findall('.//include') or tree.findall('.//pair'):
        raise ValueError('unexpected includes or explicit contact pairs')
    if any('friction' in e.attrib or 'priority' in e.attrib or 'childclass' in e.attrib
           or (e.tag == 'geom' and 'class' in e.attrib) for e in tree.iter()):
        raise ValueError('unexpected friction/default/priority override; re-review protocol')
    for mesh in tree.findall('.//asset/mesh'):
        if not Path(mesh.attrib['file']).is_absolute():
            raise ValueError('derived XML requires absolute mesh paths')
    expected = {lg.upper() + 'Tarsus' + str(i) + '_geom' for lg in LEGS for i in range(1, 6)}
    edits = []
    def change(match):
        tag = match.group(0)
        name = re.search(r'\bname="([^"]+)"', tag)
        if name is None or name.group(1) not in expected:
            return tag
        if not tag.endswith('/>'):
            raise ValueError('unexpected geom syntax')
        edits.append(dict(geom=name.group(1), old=[1., .005, .0001], new=[2., .005, .0001],
                          old_explicit=False, reason='double sliding only; preserve torsion/rolling'))
        return tag[:-2] + ' friction="2 0.005 0.0001"/>'
    derived = re.sub(r'<geom\b[^>]*>', change, source)
    if len(edits) != 30 or {e['geom'] for e in edits} != expected:
        raise ValueError('expected exactly 30 unique foot geoms')
    if derived.replace(' friction="2 0.005 0.0001"', '') != source:
        raise AssertionError('non-friction XML text changed')
    ET.fromstring(derived)
    return derived, edits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args(argv).output
    if not out.is_absolute() or out.exists():
        raise ValueError('fresh absolute --output required')
    out = out.resolve()
    # Read/derive/hash in memory. PREREG is the first file created, before model
    # loads, neural builds, copied assets, or any other output artifact.
    source_bytes = FROZEN.read_bytes()
    source = source_bytes.decode('utf-8')
    derived, edits = derive_xml(source)
    derived_bytes = derived.encode('utf-8')
    xml_tree = ET.fromstring(source)
    paths = [Path(__file__).resolve(), S283, S276, FROZEN, INITIAL, COMMAND_NAMES,
             GAINS, POOLS, ROOT / 'data/flywire/banc_connectome.npz',
             ROOT / 'outputs/neural_rebuild/sensory_channel_lookup.json',
             *sorted((ROOT / 'engine').rglob('*.py'))]
    paths += [Path(e.attrib['file']) for e in xml_tree.findall('.//asset/mesh')]
    hashes = {str(p): sha(p) for p in dict.fromkeys(paths)}
    derived_path = out / 'MODEL_FOOT_SLIDING_X2.xml'
    prereg = dict(round='S284', goal='P1/S2 friction sensitivity, not behavior acceptance',
        arms=['baseline', 'friction_x2'], duration_ms=400, seed=0,
        network='all_empty ASSUMED_NETWORK_SCORE', fingerprint='56e0322a33c51b7f03aaae9f7df1ef3f',
        exc_tau_ms=3., inh_tau_ms=3., dt_s=.0001, neural_dt_ms=.5, sensory_ms=1,
        source_hashes=hashes, original_xml=str(FROZEN), original_xml_sha256=sha(FROZEN),
        derived_xml=str(derived_path), derived_xml_sha256=hashlib.sha256(derived_bytes).hexdigest(),
        parameter_diff=edits, friction_policy='only 30 foot geom sliding coefficients 1->2; floor unchanged; torsional .005 and rolling .0001 preserved; equal-priority geom friction max makes foot-floor sliding 2, non-foot-floor unchanged',
        frozen_policy='original frozen XML is read-only, never modified; derived copy is explicitly a new model branch',
        gate='NO exact S283 first-200ms trajectory/command gate. Both arms use identical initial qpos/qvel/act/ctrl, seed, network and decoder. Friction acts from t=0; sensory feedback and commands may diverge. No historical outputs are replayed.',
        budget=BUDGET, sync='legacy live loop unchanged; copy live MjData into independent diagnostic MjData, then mj_forward; no feedback from diagnostics',
        sequence='set carry -> legacy contact ledger -> encode -> advance 2 neural ticks -> decode/clip -> 10 live steps with snapshots -> legacy state ledger',
        legacy_force='frame @ raw, up=-z; intentionally preserved for sensation/history',
        diagnostic_force='floor-side sign * frame.T @ raw; physical world diagnostics only',
        contact_raw_columns=['geom1','geom2','dist','pos_x','pos_y','pos_z', *['frame_'+str(i) for i in range(9)], *['force_'+str(i) for i in range(6)]],
        body_velocity='world angular xyz then linear xyz from mj_objectVelocity BODY, flg_local=0',
        limitations='uncalibrated sensitivity; units/real material friction and autonomous support/walking NOT_ESTABLISHED',
        non_neural_controller=False, auxiliary_force=False, training=False, prewarm=False)
    out.mkdir(parents=True, exist_ok=False)
    dump(out / 'PREREG.json', prereg)
    derived_path.write_bytes(derived_bytes)
    dump(out / 'CONFIG_DIFF.json', dict(neural='both arms tau_equal 3/3 ms', parameter_diff=edits))
    calls = {key: 0 for key in BUDGET}
        # neural_tick is a tick count: each scheduler interval advances two 0.5 ms steps.
        # The wrapped net.step below increments by the returned substep count.
    dump(out / 'CALL_LEDGER.json', calls)
    try:
        # Imports are declarations only, never historical runner execution.
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

        def run(label, model_path):
            nonlocal parameter_reference
            m = mj.MjModel.from_xml_path(str(model_path))
            calls['model_load'] += 1
            assert m.opt.timestep == .0001 and (m.nq, m.nv, m.nu, m.nmesh) == (49, 48, 90, 71)
            assert m.npair == 0
            geom_names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom)]
            foot_names = {e['geom'] for e in edits}
            foot = {g: geom_names[g][:2].lower() for g in range(m.ngeom) if geom_names[g] in foot_names}
            assert len(foot) == 30
            floor = geom_names.index('floor')
            expected_friction = np.tile([1., .005, .0001], (m.ngeom, 1))
            if label == 'friction_x2':
                expected_friction[list(foot), 0] = 2.
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
            dump(dest / 'MODEL_METADATA.json', dict(names=names, body_names=[mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)], geom_names=geom_names, geom_leg=foot, floor=floor, geom_bodyid=m.geom_bodyid, geom_friction=m.geom_friction, geom_priority=m.geom_priority, geom_condim=m.geom_condim, integrator=int(m.opt.integrator), timestep=m.opt.timestep, nmesh=m.nmesh, target_index=names.index(TARGET), target_cells=cells, fingerprint=built['fingerprint'], model_xml_sha256=sha(model_path)))
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
            dump(dest / 'PARAMETERS.json', parameters)
            initial_checks.append(dict(arm=label, qpos_sha256=hashlib.sha256(d.qpos.tobytes()).hexdigest(), qvel_sha256=hashlib.sha256(d.qvel.tobytes()).hexdigest(), act_sha256=hashlib.sha256(d.act.tobytes()).hexdigest(), ctrl_sha256=hashlib.sha256(d.ctrl.tobytes()).hexdigest(), seed=0, fingerprint=built['fingerprint']))
            enc = ChannelSensoryEncoder(groups, N, ChannelConfig())
            dec = MotorOutput(man, MotorOutputConfig(rate_for_full_control_hz=450., per_actuator_gain=gains, filter_tau_s=5e-3))
            carry = np.zeros(90)
            traj, ctrls, rows, ticks, sens, rates, rawrates, channels = [], [], [], [], [], [], [], []
            step = tier.net.step
            def tick(*a, **kw):
                spikes = step(*a, **kw)
                calls['neural_tick'] += 1
                ticks.append(dict(time_ms=tier.net.t_ms, spikes=spikes[cells].copy(), ge=tier.net.ge[cells].copy(), gi=tier.net.gi[cells].copy(), v=tier.net.v[cells].copy()))
                return spikes
            tier.net.step = tick

            def ledger(dd):
                # Exactly S276/S283 live sensory convention, including stale
                # post-step derived contacts. Diagnostic forward never touches dd.
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
                d.ctrl[:] = carry
                lp, per = ledger(d)
                qv = {lg: ChannelQuantities(tibia_joint_dev_rad=float(d.qpos[joints[lg]['tib_q']] - q0[joints[lg]['tib_q']]), tibia_joint_vel_rad_s=float(d.qvel[joints[lg]['tib_v']]), multi_joint_dev_rad=float(np.mean(d.qpos[joints[lg]['q']] - q0[joints[lg]['q']])), contact_force_norm=float(np.linalg.norm(per[lg]['F_world'])), foot_contact=float(per[lg]['touch'])) for lg in LEGS}
                cur, rec = enc.encode(qv)
                sens.append(cur.copy())
                channels.append(rec)
                sp = tier.advance(2, cur)
                rawrates.append([np.array([sample[idx].mean() / .0005 if len(idx) else 0 for idx in dec.groups]) for sample in sp])
                carry, info = dec.consume(sp, spike_dt_s=.0005, start_s=i * .001, end_s=i * .001 + .001, connected=True)
                rates.append(np.asarray(info['rates_hz']))
                a = np.clip(carry, 0, 1.)
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
            # Ragged contacts are packed numerically with offsets, no pickle.
            counts = np.array([len(r['contact_raw']) for r in traj], dtype=np.int64)
            offsets = np.concatenate(([0], np.cumsum(counts)))
            raw = np.asarray([c for r in traj for c in r['contact_raw']], dtype=float).reshape(-1, 21)
            arrays = {k: np.asarray([r[k] for r in traj]) for k in traj[0] if k != 'contact_raw'}
            np.savez_compressed(dest / 'TRAJECTORY.npz', **arrays, contact_raw=raw, contact_offsets=offsets, contact_count=counts)
            dump(dest / 'CONTACTS.json', [r['contact_raw'] for r in traj])
            np.savez_compressed(dest / 'NEURAL.npz', time_ms=np.array([r['time_ms'] for r in ticks]), **{k: np.array([r[k] for r in ticks]) for k in ('spikes', 'ge', 'gi', 'v')}, target_cells=np.array(cells), sensory_current=np.asarray(sens), rates_hz=np.asarray(rates), raw_rates_hz=np.asarray(rawrates), ctrl=np.asarray(ctrls), ctrl_time_s=np.asarray(arrays['time'][1:]))
            np.savez_compressed(dest / 'CTRL.npz', ctrl=np.asarray(ctrls), names=np.asarray(names))
            dump(dest / 'CHANNELS.json', channels)
            with (dest / 'STATE.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            assert len(traj) == 4001 and len(ctrls) == 4000 and len(ticks) == 800
            assert np.asarray(ctrls).shape == (4000, 90)
            assert np.array_equal(arrays['qpos'][0], qrel)
            assert not np.any(arrays['external_applied']) and not np.any(arrays['qfrc_applied'])
            dump(out / 'CALL_LEDGER.json', calls)

        run('baseline', FROZEN)
        run('friction_x2', derived_path)
        assert {k: v for k, v in initial_checks[0].items() if k != 'arm'} == {k: v for k, v in initial_checks[1].items() if k != 'arm'}
        dump(out / 'INITIAL_INVARIANTS.json', dict(passed=True, arms=initial_checks, neural_parameters_identical=True, decoder='same manifest/V1/rate450/filter5ms', exact_historical_gate=False))
        assert calls == BUDGET
        unchanged = all(sha(p) == h for p, h in hashes.items())
        assert unchanged, 'source changed during run'
        assert sha(derived_path) == prereg['derived_xml_sha256']
        dump(out / 'COMPLETE.json', dict(capture_complete=True, calls=calls, source_unchanged=unchanged, derived_xml_sha256=sha(derived_path), auxiliary_force=False, training=False, non_neural_controller=False, behavior='NOT_ESTABLISHED'))
    except BaseException:
        dump(out / 'CALL_LEDGER.json', calls)
        dump(out / 'FAILED.json', dict(traceback=traceback.format_exc(), calls=calls))
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
