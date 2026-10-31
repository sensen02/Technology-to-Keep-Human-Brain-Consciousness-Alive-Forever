"""S295 step 0: build per-leg '+/-' phased-lift direction groups from sourced evidence.

Two sources are combined and both are declared:
  * S84 same-frozen-XML signed actuator moments (90 x 48) at its three saved historical poses;
  * a first-order vertical kinematic Jacobian dz/dq_j at the S292 t=0 reset pose, measured on the
    same frozen model with forward kinematics only (no physics stepping).
Predicted vertical lift sensitivity of actuator a on leg L is sum_j dz/dq_j * moment[a, j].
The product mixes a current-pose Jacobian with historical-pose moments, so it is a first-order
hypothesis, not a proof; +/- arms are therefore both run.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET

XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
ARCH = Path('/home/sensen/Desktop/engine修复验收')
S292 = ARCH / 's292-current-sample-two-channel-calibration/20261026T181800Z/original_mask'
S84 = ARCH / 's84-mechanical-mapping/20261023T201631Z'
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
PREFIX = {'lf': 'LF', 'lm': 'LM', 'lh': 'LH', 'rf': 'RF', 'rm': 'RM', 'rh': 'RH'}
STATES = ('initial', 'first_activation', 'formal_start')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); out = ap.parse_args().output
    assert not out.exists(), 'fresh output required'
    out.mkdir(parents=True)
    tree = ET.fromstring(XML.read_bytes())
    names = [e.get('name') for sec in tree.findall('./actuator') for e in sec]
    import csv
    rows = list(csv.DictReader((S84 / 'ACTUATOR_MAPPING.csv').open()))
    s84_names = [r['actuator'] for r in rows]
    assert s84_names == names, 'S84 compiled actuator order differs from frozen XML order'
    joint_rows = list(csv.DictReader((S84 / 'STATE_JOINT_FORCES.csv').open()))
    dof_joint = {int(r['dof']): r['joint'].split(':')[0] for r in joint_rows if r['state'] == 'initial'}
    moments = {}
    for s in STATES:
        with np.load(S84 / (s + '_MECHANICS.npz')) as z:
            moments[s] = np.asarray(z['moment'], float)
    assert all(m.shape == (90, 48) for m in moments.values())

    import mujoco as mj
    m = mj.MjModel.from_xml_path(str(XML)); d = mj.MjData(m)
    z0 = np.load(S292 / 'TRAJECTORY.npz'); q0 = z0['qpos'][0].copy()
    d.qpos[:] = q0; mj.mj_forward(m, d)
    geom_names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom)]
    tip = {leg: geom_names.index(PREFIX[leg] + 'Tarsus5_geom') for leg in LEGS}
    site_names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_SITE, s) for s in range(m.nsite)]
    leg_sites = {leg: [s for s, n in enumerate(site_names) if n.startswith(PREFIX[leg])] for leg in LEGS}
    span = {}
    for leg in LEGS:
        pos = d.site_xpos[leg_sites[leg]]
        dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        span[leg] = float(dist.max())
    hinge = [(j, dof_joint[j]) for j in range(6, 48)]
    assert len(hinge) == 42
    jac = {}
    for j, jname in hinge:
        jnt = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, jname)
        assert jnt >= 0, jname
        qa = m.jnt_qposadr[jnt]
        col = {}
        for leg in LEGS:
            g = tip[leg]
            d.qpos[:] = q0; d.qpos[qa] += 1e-4; mj.mj_forward(m, d); zp = float(d.geom_xpos[g, 2])
            d.qpos[:] = q0; d.qpos[qa] -= 1e-4; mj.mj_forward(m, d); zm = float(d.geom_xpos[g, 2])
            col[leg] = (zp - zm) / 2e-4
        jac[jname] = col
    d.qpos[:] = q0; mj.mj_forward(m, d)
    leg_of_joint = {jname: (jname.split('_')[1][:2].lower() if jname.startswith('joint_') else None) for j, jname in hinge}
    groups = {}
    for leg in LEGS:
        dofs = [j for j, jn in hinge if leg_of_joint[jn] == leg]
        assert len(dofs) == 7, (leg, len(dofs))
        per = []
        for i, n in enumerate(names):
            if not n.startswith(PREFIX[leg]): continue
            score = {s: float(sum(moments[s][i, j] * jac[dof_joint[j]][leg] for j in dofs)) for s in STATES}
            per.append(dict(index=i, actuator=n, score_by_state=score,
                            score_initial=score['initial'],
                            sign_consistent=bool(len({np.sign(v) for v in score.values() if v != 0.0}) == 1),
                            max_abs_moment=float(np.abs(moments['initial'][i, dofs]).max())))
        pos = [r['index'] for r in per if r['score_initial'] > 0]
        neg = [r['index'] for r in per if r['score_initial'] < 0]
        groups[leg] = dict(per_actuator=per, plus=pos, minus=neg,
                           plus_names=[names[i] for i in pos], minus_names=[names[i] for i in neg],
                           dofs=[dof_joint[j] for j in dofs],
                           dz_dq={dof_joint[j]: float(jac[dof_joint[j]][leg]) for j in dofs},
                           span_native=span[leg], span_definition='max pairwise distance among this leg\'s XML site world positions at the S292 t=0 pose',
                           grounded=bool(pos or neg),
                           fallback=None if (pos or neg) else 'NOT_ESTABLISHED: no signed-moment channel for this leg; S294 named press/lift chain would be used and kept out of grounded statistics')
    doc = dict(round='S295', stage='direction-group construction', inputs={str(p): sha(p) for p in
                [Path(__file__).resolve(), XML, S292 / 'TRAJECTORY.npz', S84 / 'ACTUATOR_MAPPING.csv', S84 / 'STATE_JOINT_FORCES.csv'] +
                [S84 / (s + '_MECHANICS.npz') for s in STATES]},
               sources=dict(moments='S84 same-XML signed actuator moment (90x48) at initial/first_activation/formal_start',
                            jacobian='first-order dz/dq at the S292 t=0 pose on the same frozen model, forward kinematics only',
                            geometry_hypothesis='lifting a foot vertically corresponds to positive d(tarsus tip z)/d(q); an actuator raises the foot when its signed joint moments map onto positive dz/dq, so sign(score) is the predicted vertical direction',
                            limitation='Jacobian pose (S292 t=0) differs from the S84 historical moment poses; the product is a first-order prediction, not a proof, therefore both +/- arms are executed'),
               legs=groups, calls=dict(model_load=1, kinematic_forward=1 + 2 * len(hinge) * len(LEGS) // len(LEGS) * 0 + 84 + 1, physics_step=0))
    doc['calls']['kinematic_forward'] = 1 + 2 * 42 + 1
    (out / 'DIRECTION_GROUPS.json').write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({leg: dict(plus=len(v['plus']), minus=len(v['minus']), grounded=v['grounded'], span=round(v['span_native'], 4)) for leg, v in groups.items()}))
    return 0


if __name__ == '__main__': raise SystemExit(main())
