"""S295/P3 phased open-loop lift probe (diagnostic, not neural control, not gait).

One fixed 120 ms phase sequence per arm:
  0-20 ms   all channels zero (rest)
  20-40 ms  target leg's direction group at 0.5
  40-100 ms target leg's direction group at 1.0 (hold)
  100-120 ms all channels zero (release)
14 declared runs: 1 passive zero control + 6 legs x 2 predicted directions + 1 bitwise repeat.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
ARCH = Path('/home/sensen/Desktop/engine修复验收')
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
S292 = ARCH / 's292-current-sample-two-channel-calibration/20261026T181800Z/original_mask'
S84 = ARCH / 's84-mechanical-mapping/20261023T201631Z'
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
PREFIX = {'lf': 'LF', 'lm': 'LM', 'lh': 'LH', 'rf': 'RF', 'rm': 'RM', 'rh': 'RH'}
PHASES = ((0, 20, 0.0, 'rest'), (20, 40, 0.5, 'ramp'), (40, 100, 1.0, 'hold'), (100, 120, 0.0, 'release'))
STEPS = 1200          # 120 ms at dt = 0.1 ms
THRESH = 1.0          # native lift threshold, identical to S294
SUPPORT_LO, SUPPORT_HI = .9, 1.2
SUPPORT_FRACTION_BAR = .8
LIFT_OBSERVED_BAR = .5
S294_BEST_TARGET_DIFF = .366
S294_BEST_ANY_DIFF = .488
WEIGHT = 24.446354755878


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, v): Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, default=float) + '\n', encoding='utf-8')


def declared_arms(groups):
    arms = [dict(id='passive_zero', leg='all', direction='none', indices=[])]
    for leg in LEGS:
        for d in ('plus', 'minus'):
            arms.append(dict(id=f'{leg}_{d}', leg=leg, direction=d, indices=groups[leg][d]))
    arms.append(dict(id='lf_plus_repeat', leg='lf', direction='plus', indices=groups['lf']['plus'], repeat_of='lf_plus'))
    return arms


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True); ap.add_argument('--prereg-only', action='store_true')
    args = ap.parse_args(); out = args.output
    tree = ET.fromstring(XML.read_bytes())
    names = [e.get('name') for sec in tree.findall('./actuator') for e in sec]
    dg = json.loads((out / 'DIRECTION_GROUPS.json').read_text())
    groups = {leg: dict(plus=dg['legs'][leg]['plus'], minus=dg['legs'][leg]['minus']) for leg in LEGS}
    arms = declared_arms(groups)
    assert len(arms) == 14
    prereg = dict(
        round='S295', mode='phased OPEN-LOOP diagnostic: not neural control, not gait, not periodic movement, no support or walking claim',
        frozen_xml=str(XML), frozen_xml_sha256=sha(XML),
        steps_per_run=STEPS, run_duration_ms=STEPS * .1, phases=[dict(start_ms=a, end_ms=b, command=c, label=l,
                                                                    steps=[a * 10, b * 10]) for a, b, c, l in PHASES],
        phase_semantics='every channel outside the target direction group is exactly 0 in every phase; the group value is the phase constant',
        reset_semantics='one model load for the whole round; each run resets qpos/qvel/act to the saved S292 t=0 state and then applies the phase constants',
        declared_calls=dict(model_load=1, forward_init=14, physics_step=STEPS * 14), run_count=14, run_count_fixed=True,
        arms=[dict(id=a['id'], leg=a['leg'], direction=a['direction'], indices=a['indices'], actuators=[names[i] for i in a['indices']],
                   repeat_of=a.get('repeat_of')) for a in arms],
        direction_group_source=dict(groups=str(out / 'DIRECTION_GROUPS.json'), groups_sha256=sha(out / 'DIRECTION_GROUPS.json'),
                                    moments='S84 same-XML signed moment 90x48 at initial/first_activation/formal_start',
                                    jacobian='first-order dz/dq at the S292 t=0 pose, forward kinematics only',
                                    hypothesis='score = sum_j dz/dq_j * moment[a,j]; positive score predicts raising the tarsus; not a proof, hence both directions are run per leg'),
        criteria=dict(lift=f'differential lift of the target leg vs the passive control at the same index >= {THRESH} native units during the hold phase',
                      support=f'fraction of hold-phase samples with support ratio in [{SUPPORT_LO}, {SUPPORT_HI}] >= {SUPPORT_FRACTION_BAR}',
                      span='differential lift divided by the leg span measured from XML site positions at t=0',
                      lift_observed_bar=LIFT_OBSERVED_BAR, s294_best_target_diff=S294_BEST_TARGET_DIFF, s294_best_any_diff=S294_BEST_ANY_DIFF),
        decision_rules=dict(
            rule1_neural='any leg/direction satisfies lift>=1.0 AND support fraction>=0.8 -> mechanical lifting is sufficient, next round is sourced neural muscle-coverage correction',
            rule2_mechanical='all 12 phased arms < 1.0 -> report missing factors against 1.0 and against the leg span; next round opens a NEW derived model plus ledger for gear/limit/mass, frozen file untouched',
            rule3_incompatible='lift_observed>=0.5 but support fraction<0.8 -> single-leg lift and support retention are incompatible in this geometry; must state whether load transfer or contact geometry is limiting and which legs carry the load',
            rule4_no_improvement='no phased arm exceeds the S294 static best differential lift -> the temporal structure of constant commands is not an improvement and the premise for choosing it is refuted'),
        prohibitions=['no extra levels, legs, directions or phase durations', 'no XML, gear, mass, gravity, timestep or gain edits',
                      'no neural or sensory drive', 'the phase sequence is not a controller and must not be delivered as one'],
        input_hashes={str(p): sha(p) for p in [Path(__file__).resolve(), XML, S292 / 'TRAJECTORY.npz', out / 'DIRECTION_GROUPS.json',
                                              S84 / 'ACTUATOR_MAPPING.csv', S84 / 'STATE_JOINT_FORCES.csv'] +
                                              [S84 / (s + '_MECHANICS.npz') for s in ('initial', 'first_activation', 'formal_start')]},
    )
    if args.prereg_only:
        assert not (out / 'PREREG.json').exists(), 'prereg already written'
        dump(out / 'PREREG.json', prereg); print(json.dumps(dict(prereg=str(out / 'PREREG.json'), arms=len(arms)))); return 0
    existing = json.loads((out / 'PREREG.json').read_text())
    assert existing['run_count'] == 14 and [a['id'] for a in existing['arms']] == [a['id'] for a in arms], 'run list changed'
    assert existing['input_hashes'][str(Path(__file__).resolve())] == sha(__file__), 'runner changed after prereg'
    assert existing['input_hashes'][str(out / 'DIRECTION_GROUPS.json')] == sha(out / 'DIRECTION_GROUPS.json'), 'direction groups changed after prereg'

    import mujoco as mj
    m = mj.MjModel.from_xml_path(str(XML)); d = mj.MjData(m)
    calls = dict(model_load=1, forward_init=0, physics_step=0)
    compiled = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    assert compiled == names and m.nu == 90
    mass = float(m.body_mass.sum()); weight = mass * -float(m.opt.gravity[2])
    assert abs(weight - WEIGHT) < 1e-9
    geom_names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom)]
    floor = geom_names.index('floor')
    leg_of_geom = {g: geom_names[g][:2].lower() for g in range(m.ngeom) if geom_names[g][:2].lower() in LEGS}
    tip = {leg: geom_names.index(PREFIX[leg] + 'Tarsus5_geom') for leg in LEGS}
    z0 = np.load(S292 / 'TRAJECTORY.npz'); q0, v0, a0 = z0['qpos'][0].copy(), z0['qvel'][0].copy(), z0['act'][0].copy()

    def reset():
        d.qpos[:] = q0; d.qvel[:] = v0; d.act[:] = a0; d.ctrl[:] = 0
        mj.mj_forward(m, d); calls['forward_init'] += 1

    def sample():
        fz = np.zeros(6); total = 0.0
        for ci in range(d.ncon):
            c = d.contact[ci]; g1, g2 = int(c.geom1), int(c.geom2)
            if floor not in (g1, g2): continue
            other = g2 if g1 == floor else g1
            fx = np.zeros(6); mj.mj_contactForce(m, d, ci, fx)
            fw = (1 if g1 == floor else -1) * (np.asarray(c.frame, float).reshape(3, 3).T @ fx[:3])
            up = float(fw[2]); total += up
            leg = leg_of_geom.get(other)
            if leg is not None: fz[LEGS.index(leg)] += up
        com = (m.body_mass[:, None] * d.xipos).sum(axis=0) / mass
        lim = np.flatnonzero(m.jnt_limited); qa = m.jnt_qposadr[lim]
        limit = float(np.maximum(np.maximum(m.jnt_range[lim, 0] - d.qpos[qa], d.qpos[qa] - m.jnt_range[lim, 1]), 0).max())
        ang = 2 * float(np.arccos(np.clip(abs(d.qpos[3]), 0, 1)))
        return fz, total, com.copy(), ang, limit, float(d.qpos[2])

    raw = out / 'runs'; raw.mkdir(exist_ok=True); summary = {}
    for arm in arms:
        cmd_of_phase = []; cmds = []
        for a, b, c, label in PHASES:
            cmd = np.zeros(90); cmd[arm['indices']] = c
            for _ in range((b - a) * 10): cmds.append(cmd.copy())
        cmds = np.asarray(cmds[:STEPS], np.float32)
        assert cmds.shape == (STEPS, 90)
        reset(); base = sample(); base_z = np.array([float(d.geom_xpos[tip[l], 2]) for l in LEGS]); base_com = base[2]
        rec = {k: [] for k in ('fz', 'total', 'foot_z', 'com', 'orientation', 'limit', 'root_z')}
        for i in range(STEPS):
            d.ctrl[:] = cmds[i]; mj.mj_step(m, d); calls['physics_step'] += 1
            fz, total, com, ang, lim, rz = sample()
            rec['fz'].append(fz); rec['total'].append(total); rec['com'].append(com)
            rec['orientation'].append(ang); rec['limit'].append(lim); rec['root_z'].append(rz)
            rec['foot_z'].append([float(d.geom_xpos[tip[l], 2]) for l in LEGS])
        arr = {k: np.asarray(v, float) for k, v in rec.items()}
        np.savez_compressed(raw / (arm['id'] + '.npz'), **arr, ctrl=cmds, base_foot_z=base_z, base_com=base_com,
                            base_total=np.array(base[1]), cmd_of_phase=np.array([c for _, _, c, _ in PHASES]))
        lift = arr['foot_z'] - base_z[None, :]
        ratio = arr['total'] / weight
        hold = slice(400, 1000)
        j = LEGS.index(arm['leg']) if arm['leg'] in LEGS else None
        summary[arm['id']] = dict(id=arm['id'], leg=arm['leg'], direction=arm['direction'], n_indices=len(arm['indices']),
                                  commanded_indices=[int(i) for i in np.flatnonzero(cmds[400])],
                                  phases={label: dict(start_ms=a, end_ms=b, command=c,
                                                      target_lift=float(lift[b * 10 - 1, j]) if j is not None else None,
                                                      target_fz=float(arr['fz'][b * 10 - 1, j]) if j is not None else None,
                                                      others_fz=float(arr['fz'][b * 10 - 1].sum() - (arr['fz'][b * 10 - 1, j] if j is not None else 0)),
                                                      support_ratio=float(ratio[b * 10 - 1]),
                                                      com_dz=float(arr['com'][b * 10 - 1, 2] - base_com[2]),
                                                      com_vz=float((arr['com'][b * 10 - 1, 2] - arr['com'][b * 10 - 2, 2]) / .0001),
                                                      orientation=float(arr['orientation'][b * 10 - 1]),
                                                      joint_limit=float(arr['limit'][b * 10 - 1]))
                                              for a, b, c, label in PHASES},
                                  hold=dict(target_lift_max=float(lift[hold, j].max()) if j is not None else None,
                                            target_lift_end=float(lift[999, j]) if j is not None else None,
                                            support_ratio_mean=float(ratio[hold].mean()), support_ratio_min=float(ratio[hold].min()),
                                            support_ratio_max=float(ratio[hold].max()),
                                            support_in_band_fraction=float(np.mean((ratio[hold] >= SUPPORT_LO) & (ratio[hold] <= SUPPORT_HI))),
                                            target_fz_mean=float(arr['fz'][hold, j].mean()) if j is not None else None,
                                            others_fz_mean=float((arr['fz'][hold].sum(axis=1) - (arr['fz'][hold, j] if j is not None else 0)).mean()),
                                            com_dz=float(arr['com'][999, 2] - base_com[2]),
                                            com_vz=float((arr['com'][999, 2] - arr['com'][998, 2]) / .0001),
                                            orientation_max=float(arr['orientation'][hold].max()),
                                            joint_limit_max=float(arr['limit'][hold].max())),
                                  com_dz_total=float(arr['com'][-1, 2] - base_com[2]),
                                  span_native=float(dg['legs'][arm['leg']]['span_native']) if arm['leg'] in LEGS else None)
        dump(out / 'CALL_LEDGER.json', calls)
    dump(out / 'PHASED.json', dict(runs=summary, weight_native=weight, steps_per_run=STEPS,
                                   criteria=prereg['criteria'], decision_rules=prereg['decision_rules']))
    dump(out / 'CALL_LEDGER.json', calls)
    print(json.dumps(dict(runs=len(summary), steps=calls['physics_step'],
                          best_hold_lift=max((v['hold']['target_lift_max'] or 0) for v in summary.values()))))
    return 0


if __name__ == '__main__': raise SystemExit(main())
