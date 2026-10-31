"""S294/P3 bounded open-loop support & lift capability envelope (diagnostic, not neural control).

Purpose: measure whether the existing 90 actuators, at declared constant commands, can
(a) hold vertical support sum(Fz)/(m g) >= 1 and (b) lift a foot >= 1 native unit, from the
saved S292 t=0 state. Frozen XML, bodies, gravity, timestep, gear and joint limits are
untouched. Nothing here is neural control, walking, or evidence of support success.

Two phases:
  --prereg-only   write PREREG.json (groups, levels, fixed run list, criteria, hashes) and exit
  (default)       verify PREREG against this source, then run exactly the declared runs
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
ARCH = Path('/home/sensen/Desktop/engine修复验收')
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
S292 = ARCH / 's292-current-sample-two-channel-calibration/20261026T181800Z'
S289 = ARCH / 's289-decoder-range-support/20261026T173400Z/decoder_original'
S84 = ARCH / 's84-mechanical-mapping/20261023T201631Z'
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
LEG_PREFIX = {'lf': 'LF', 'lm': 'LM', 'lh': 'LH', 'rf': 'RF', 'rm': 'RM', 'rh': 'RH'}
PRESS = ('sternal_adductor', 'sternal_posterior_rotator', 'trochanter_extensor',
         'sterno-tergo-trochanter_extensor_a', 'sterno-tergo-trochanter_extensor_b', 'Tibia_extensor_93932')
LIFT = ('pleural_promotor', 'pleural_remotor_and_abductor', 'trochanter_flexor_a',
        'trochanter_flexor_b', 'accesory_trochanter_flexor', 'Tibia_flex_93434')
LEVELS = (0.5, 1.0)
STEPS = 500           # 50 ms at dt = 0.1 ms
LIFT_THRESHOLD = 1.0  # native length unit; absolute mm calibration NOT_ESTABLISHED
SUPPORT_BAR = 1.0
SUPPORT_WINDOW_MS = 100.0


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, v): Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, default=float) + '\n', encoding='utf-8')


def actuator_names():
    tree = ET.fromstring(XML.read_bytes())
    return [e.get('name') for sec in tree.findall('./actuator') for e in sec]


def group_indices(names):
    groups = {}
    for leg in LEGS:
        pre = LEG_PREFIX[leg]
        idx = [i for i, n in enumerate(names) if n.startswith(pre)]
        assert len(idx) == 15, leg
        press = [i for i in idx if any(names[i].endswith(s) for s in PRESS)]
        lift = [i for i in idx if any(names[i].endswith(s) for s in LIFT)]
        assert len(press) == 6 and len(lift) == 6, (leg, len(press), len(lift))
        groups[leg] = dict(all15=idx, press=press, lift=lift,
                           unassigned=[i for i in idx if i not in press + lift])
    return groups


def declared_runs(groups):
    runs = []
    for leg in LEGS:
        for kind in ('press', 'lift'):
            for level in LEVELS:
                runs.append(dict(id=f'{leg}_{kind}_{level:.1f}', leg=leg, kind=kind, level=level, indices=groups[leg][kind]))
        for level in LEVELS:
            runs.append(dict(id=f'{leg}_all15_{level:.1f}', leg=leg, kind='all15', level=level, indices=groups[leg]['all15']))
    runs.append(dict(id='all90_1.0', leg='all', kind='all90', level=1.0, indices=list(range(90))))
    runs.append(dict(id='zero_control', leg='all', kind='zero', level=0.0, indices=[]))
    return runs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--prereg-only', action='store_true')
    args = ap.parse_args(argv); out = args.output
    names = actuator_names(); groups = group_indices(names); runs = declared_runs(groups)
    base = json.loads((S292 / 'PREREG.json').read_text())
    inputs = [Path(__file__).resolve(), XML, S292 / 'PREREG.json', S292 / 'original_mask' / 'TRAJECTORY.npz',
              S292 / 'original_mask' / 'MODEL_METADATA.json', S289 / 'TRAJECTORY.npz', S289 / 'MODEL_METADATA.json',
              S84 / 'ACTUATOR_MAPPING.csv', S84 / 'STATE_JOINT_FORCES.csv', S84 / 'PREREG.json']
    inputs += [S84 / (s + '_MECHANICS.npz') for s in ('initial', 'first_activation', 'formal_start')]
    prereg = dict(
        round='S294', mode='open-loop envelope diagnostic: NOT neural control, NOT walking, NOT support success evidence',
        frozen_xml=str(XML), frozen_xml_sha256=sha(XML), timestep_s=.0001, steps_per_run=STEPS, run_duration_ms=STEPS * .1,
        reset_source=str(S292 / 'original_mask' / 'TRAJECTORY.npz'), reset_index=0,
        reset_definition='qpos/qvel/act copied from saved t=0 state; ctrl overwritten each step by the declared constant command',
        declared_command_semantics='ctrl[i] = level for declared indices, exactly 0 for all others, held constant for all steps',
        levels=list(LEVELS), lift_threshold_native=LIFT_THRESHOLD,
        lift_threshold_note='native length unit; absolute mm calibration NOT_ESTABLISHED (XML uses mm-consistent gravity/mass but no calibrated length)',
        groups={leg: dict(press=[names[i] for i in v['press']], lift=[names[i] for i in v['lift']],
                          all15=[names[i] for i in v['all15']], unassigned=[names[i] for i in v['unassigned']]) for leg, v in groups.items()},
        role_label_caveat='press/lift labels follow the user-declared muscle-name sets; the mechanical up/down role is documented from S84 same-XML signed moments where available and otherwise marked NOT_ESTABLISHED, never inferred from names alone',
        runs=[dict(id=r['id'], leg=r['leg'], kind=r['kind'], level=r['level'], indices=r['indices'], actuators=[names[i] for i in r['indices']]) for r in runs],
        run_count=len(runs), run_count_fixed=True,
        prohibitions=['no additional levels, groups or runs', 'no XML, gear, gain, timestep, gravity or mass edits', 'no per-step pose or command correction', 'no neural or sensory drive in this probe'],
        criteria=dict(support_ok=f'sum(Fz)/(m g) >= {SUPPORT_BAR} sustained through {SUPPORT_WINDOW_MS:.0f} ms',
                      lift_ok=f'any foot height rise >= {LIFT_THRESHOLD} native unit',
                      blocking_rule='if no declared run sustains support >= 1 for 100 ms, or no run lifts a foot >= threshold, the blocking point is mechanical/actuator capability and a concrete missing factor must be reported',
                      sufficient_rule='if the envelope clearly supports and lifts, the gap is neural drive distribution rather than mechanics'),
        input_hashes={str(p): sha(p) for p in inputs if Path(p).is_file()},
        unreadable_inputs=[str(p) for p in inputs if not Path(p).is_file()],
        calls=dict(model_load=len(runs), forward_init=len(runs), physics_step=len(runs) * STEPS),
    )
    if args.prereg_only:
        assert not out.exists(), 'fresh output required'
        out.mkdir(parents=True); dump(out / 'PREREG.json', prereg); print(json.dumps(dict(prereg=str(out / 'PREREG.json'), runs=len(runs)))); return 0
    existing = json.loads((out / 'PREREG.json').read_text())
    assert existing['run_count'] == len(runs) and existing['run_count_fixed'] is True, 'run list changed'
    assert [r['id'] for r in existing['runs']] == [r['id'] for r in runs], 'run ids changed'
    assert existing['input_hashes'][str(Path(__file__).resolve())] == sha(__file__), 'runner changed after prereg'
    assert existing['frozen_xml_sha256'] == sha(XML), 'frozen XML changed'

    import mujoco as mj
    m = mj.MjModel.from_xml_path(str(XML)); d = mj.MjData(m)
    calls = dict(model_load=1, forward_init=0, physics_step=0, diagnostic_forward=0)
    compiled = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    assert compiled == names and m.nu == 90 and (m.nq, m.nv) == (49, 48), 'compiled actuator order/nq/nv mismatch'
    mass = float(m.body_mass.sum()); gravity = float(-m.opt.gravity[2]); weight = mass * gravity
    geom_names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom)]
    floor = geom_names.index('floor')
    leg_of_geom = {g: geom_names[g][:2].lower() for g in range(m.ngeom) if geom_names[g][:2].lower() in LEGS}
    tip = {}
    for leg in LEGS:
        cand = [g for g in range(m.ngeom) if geom_names[g] == LEG_PREFIX[leg] + 'Tarsus5_geom']
        assert len(cand) == 1, (leg, cand)
        tip[leg] = cand[0]
    z = np.load(S292 / 'original_mask' / 'TRAJECTORY.npz')
    q0, v0, a0 = z['qpos'][0].copy(), z['qvel'][0].copy(), z['act'][0].copy()
    meta = json.loads((S292 / 'original_mask' / 'MODEL_METADATA.json').read_text())
    info = dict(compiled_order_exact=True, mass=mass, gravity=gravity, weight_native=weight, nq=m.nq, nv=m.nv, nu=m.nu,
                tip_geoms={leg: geom_names[g] for leg, g in tip.items()}, floor_geom=geom_names[floor],
                model_xml_sha256=sha(XML), target_legs_geoms=len(leg_of_geom))
    dump(out / 'MODEL_INFO.json', info)
    assert abs(weight - 24.446354755878) < 1e-9, 'weight mismatch against recorded model weight'

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
            up = float(fw[2])
            leg = leg_of_geom.get(other)
            if leg is not None: fz[LEGS.index(leg)] += up
            total += up
        com = (m.body_mass[:, None] * d.xipos).sum(axis=0) / mass
        limited = np.flatnonzero(m.jnt_limited); qa = m.jnt_qposadr[limited]
        lim = float(np.maximum(np.maximum(m.jnt_range[limited, 0] - d.qpos[qa], d.qpos[qa] - m.jnt_range[limited, 1]), 0).max())
        quat = d.qpos[3:7]
        ang = 2 * float(np.arccos(np.clip(abs(quat[0]), 0, 1)))
        return fz, total, com.copy(), ang, lim, float(d.qpos[2])

    summary = {}
    raw_dir = out / 'runs'; raw_dir.mkdir(exist_ok=True)
    for run in runs:
        cmd = np.zeros(90); cmd[run['indices']] = run['level']
        reset(); base = sample(); base_z = {leg: float(d.geom_xpos[tip[leg], 2]) for leg in LEGS}
        rec = {k: [] for k in ('fz', 'total', 'foot_z', 'com', 'orientation', 'limit', 'root_z')}
        ctrl_logged = []
        for _ in range(STEPS):
            d.ctrl[:] = cmd; mj.mj_step(m, d); calls['physics_step'] += 1
            fz, total, com, ang, lim, rz = sample()
            rec['fz'].append(fz); rec['total'].append(total); rec['com'].append(com)
            rec['orientation'].append(ang); rec['limit'].append(lim); rec['root_z'].append(rz)
            rec['foot_z'].append([float(d.geom_xpos[tip[leg], 2]) for leg in LEGS])
            ctrl_logged.append(cmd.copy())
        arr = {k: np.asarray(v, float) for k, v in rec.items()}
        ctrl = np.asarray(ctrl_logged, np.float32)
        np.savez_compressed(raw_dir / (run['id'] + '.npz'), **arr, ctrl=ctrl, base_foot_z=np.array([base_z[l] for l in LEGS]),
                            base_total=np.array(base[1]), base_com=base[2], cmd=cmd)
        lift = arr['foot_z'] - np.array([base_z[l] for l in LEGS])[None, :]
        ratio = arr['total'] / weight
        windows = {}
        for k in (10, 25, 50):
            windows[k] = dict(min=float(ratio[:k * 10].min()), mean=float(ratio[:k * 10].mean()),
                              max=float(ratio[:k * 10].max()), frac_ge1=float(np.mean(ratio[:k * 10] >= SUPPORT_BAR)))
        summary[run['id']] = dict(
            id=run['id'], leg=run['leg'], kind=run['kind'], level=run['level'], n_indices=len(run['indices']),
            actuator_names=[names[i] for i in run['indices']],
            commanded_nonzero_indices=[int(i) for i in np.flatnonzero(cmd)],
            commanded_value=float(run['level']),
            support_ratio=dict(t0=float(base[1] / weight), final=float(ratio[-1]), min=float(ratio.min()), max=float(ratio.max()),
                               mean=float(ratio.mean()), frac_ge1=float(np.mean(ratio >= SUPPORT_BAR)), windows=windows),
            lift=dict(peak_per_leg={l: float(lift[:, j].max()) for j, l in enumerate(LEGS)},
                      final_per_leg={l: float(lift[-1, j]) for j, l in enumerate(LEGS)},
                      target_leg=run['leg'] if run['leg'] in LEGS else None,
                      target_peak=float(lift[:, LEGS.index(run['leg'])].max()) if run['leg'] in LEGS else float(lift.max()),
                      any_ge_threshold=bool((lift >= LIFT_THRESHOLD).any()),
                      legs_ge_threshold=[l for j, l in enumerate(LEGS) if (lift[:, j] >= LIFT_THRESHOLD).any()]),
            contact_fz=dict(final_per_leg={l: float(arr['fz'][-1, j]) for j, l in enumerate(LEGS)},
                            max_per_leg={l: float(arr['fz'][:, j].max()) for j, l in enumerate(LEGS)}),
            com=dict(dz=float(arr['com'][-1, 2] - base[2][2]), z0=float(base[2][2]), z_final=float(arr['com'][-1, 2]),
                     vz_final=float((arr['com'][-1, 2] - arr['com'][-2, 2]) / .0001),
                     x_disp=float(arr['com'][-1, 0] - base[2][0]), y_disp=float(arr['com'][-1, 1] - base[2][1])),
            orientation_max_rad=float(arr['orientation'].max()), joint_limit_max=float(arr['limit'].max()),
            root_dz=float(arr['root_z'][-1] - arr['root_z'][0]),
        )
        dump(out / 'CALL_LEDGER.json', calls)
    dump(out / 'ENVELOPE.json', dict(model_info=info, runs=summary, weight_native=weight, support_bar=SUPPORT_BAR,
                                     lift_threshold_native=LIFT_THRESHOLD, steps_per_run=STEPS,
                                     verdict_pending='see OUTCOMES.json written by the verifier/report'))
    dump(out / 'CALL_LEDGER.json', calls)
    print(json.dumps(dict(runs=len(summary), steps=calls['physics_step'],
                          best_support=max(v['support_ratio']['mean'] for v in summary.values()),
                          lift_runs=[k for k, v in summary.items() if v['lift']['any_ge_threshold']])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
