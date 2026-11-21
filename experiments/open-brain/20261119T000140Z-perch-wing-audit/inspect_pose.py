"""Compare the actual late perch pose with the original ground folded reference."""
import argparse
import json
from pathlib import Path
import sys
import hashlib

import mujoco as mj
import numpy as np

HOME = Path(__file__).resolve().parent
FLIGHT = HOME.parent / '20261118T234622Z-flight-fruit'
sys.path.insert(0, str(FLIGHT))
from perching import build_scene


def pose_facts(physics, qpos, wing_ids, abdomen_ids):
    m, d = physics.model, physics.data
    physics.data.qpos[:] = qpos
    physics.forward()
    thorax = m.name2id('walker/thorax', 'body')
    basis = d.xmat[thorax].reshape(3, 3)
    center = d.xpos[thorax].copy()
    rad = d.qpos[m.jnt_qposadr[wing_ids]]
    result = dict(wing_joint_rad=rad.tolist(), abdomen_joint_rad=d.qpos[m.jnt_qposadr[abdomen_ids]].tolist(),
                  thorax_forward_world=basis[:, 0].tolist(), thorax_up_world=basis[:, 2].tolist())
    for side in ('left', 'right'):
        gid = m.name2id(f'walker/wing_{side}_membrane_collision', 'geom')
        anc = m.name2id(f'walker/wing_{side}', 'body')
        result[f'{side}_root_thorax_mm'] = ((d.xpos[anc] - center) @ basis * 10).tolist()
        result[f'{side}_wing_centre_thorax_mm'] = ((d.geom_xpos[gid] - center) @ basis * 10).tolist()
        gaps = []
        for i in range(1, 8):
            prefix = f'abdomen_{i}' if i > 1 else 'abdomen'
            abdomen = m.name2id(f'walker/{prefix}_collision', 'geom')
            gap = mj.mj_geomDistance(m.ptr, d.ptr, gid, abdomen, .5, None)
            gaps.append(float(gap * 10.))
        result[f'{side}_abdomen_gaps_mm'] = gaps
    abdomen = m.name2id('walker/abdomen_7_collision', 'geom')
    result['abdomen_tip_thorax_mm'] = ((d.geom_xpos[abdomen] - center) @ basis * 10).tolist()
    contacts = []
    for contact_id, c in enumerate(d.contact[:d.ncon]):
        names = [m.id2name(int(g), 'geom') for g in c.geom]
        if any(n.startswith('walker/wing_') for n in names) and any(
                n.startswith('walker/abdomen') for n in names):
            wrench = np.zeros(6)
            mj.mj_contactForce(m.ptr, d.ptr, contact_id, wrench)
            contacts.append(dict(geoms=names, distance_mm=float(c.dist * 10.),
                                 normal_force_native=float(wrench[0])))
    result['wing_abdomen_contacts'] = contacts
    return result


def audit(case):
    meta = json.loads((case / 'RESULT.json').read_text())
    for name in ('TRACE.npy', 'ACTION.npy', 'QPOS.npy', 'INITIAL.npy'):
        assert hashlib.sha256((case / name).read_bytes()).hexdigest() == meta['files'][name]
    trace = np.load(case / 'TRACE.npy', allow_pickle=False)
    action = np.load(case / 'ACTION.npy', allow_pickle=False)
    qpos = np.load(case / 'QPOS.npy', allow_pickle=False)
    before = np.load(case / 'INITIAL.npy', allow_pickle=False)
    trace = dict(zip(meta['columns'], trace.T, strict=True))
    env = build_scene(meta['geometry']['x_cm'], meta['geometry']['y_cm'])
    env.reset()
    phy = env.physics
    m = phy.model
    names = [f'wing_{a}_{s}' for s in ('left', 'right') for a in ('yaw', 'roll', 'pitch')]
    wings = [m.name2id('walker/' + j, 'joint') for j in names]
    wing_actions = [env.action_spec().name.split('\t').index(j) for j in names]
    abdomens = [i for i in range(m.njnt)
                if (m.id2name(i, 'joint') or '').startswith('walker/') and 'abdomen' in m.id2name(i, 'joint')]
    samples = {}
    for t in (0, .07, .28, meta['fold_s'], *(meta['fold_s'] + dt for dt in (.004, .008, .012, .016, .024, .034, .06)), (meta['first_perch_s'] or .86) + .05, 1.36, 1.42):
        idx = min(len(qpos)-1, max(0, round(t / .0002) - 1))
        facts = pose_facts(phy, qpos[idx] if t else before[:m.nq], wings, abdomens)
        at = idx * 4 + 3
        facts['wing_actuator_action'] = action[idx, wing_actions].tolist()
        for name in ('wing_mix', 'wing_force', 'landing_phase', 'fruit_feet', 'perch_ready',
                     'pitch_rad', 'roll_rad', 'yaw_rad'):
            facts[name] = float(trace[name][at])
        samples[str(t)] = facts
    late = qpos[-1000:, m.jnt_qposadr[wings]]
    spring = m.qpos_spring[m.jnt_qposadr[wings]]
    result = dict(case=case.name, native_steps=len(trace['time_s']), joints=names, initial_matches_xml=bool(np.array_equal(before[m.jnt_qposadr[wings]], spring)),
                  spring_rad=spring.tolist(), final_rad=late[-1].tolist(),
                  last_200ms_joint_range_deg=np.rad2deg(np.ptp(late, axis=0)).tolist(),
                  last_200ms_max_wing_force_native=float(np.max(trace['wing_force'][-4000:])),
                  max_late_stow_error_deg=float(np.rad2deg(np.abs(late - spring).max())),
                  samples=samples)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('cases', type=Path, nargs='*')
    args = parser.parse_args()
    result = [audit(case) for case in (args.cases or (FLIGHT / 'NEAR', FLIGHT / 'FAR'))]
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for facts in result:
        print(facts['case'], 'wing stow max errors (deg)', facts['max_late_stow_error_deg'],
              'oscillation (deg)', facts['last_200ms_joint_range_deg'])
        for t in ('0', '.07', '0.07', '1.36', '1.42'):
            if t not in facts['samples']:
                continue
            s = facts['samples'][t]
            print(t, np.round(s['wing_joint_rad'], 5), 'abdomen tip', np.round(s['abdomen_tip_thorax_mm'], 4),
                  'wing centre', np.round(s['left_wing_centre_thorax_mm'], 4),
                  'abdomen gaps', np.round(s['left_abdomen_gaps_mm'], 4))
