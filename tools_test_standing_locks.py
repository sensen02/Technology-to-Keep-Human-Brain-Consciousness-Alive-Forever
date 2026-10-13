"""Guard against inherited immobilisation in a six-leg MuJoCo standing model."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import mujoco
from flygym.compose.fly.musculoskeletal import _load_mjcf

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / 'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml'
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')


def inspect_model(m):
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or '' for i in range(m.njnt)]
    legs = [i for i, n in enumerate(names) if any(n.startswith('joint_' + leg) for leg in LEGS)]
    locks = []
    for i in range(m.neq):
        if int(m.eq_type[i]) != int(mujoco.mjtEq.mjEQ_JOINT):
            continue
        j1, j2 = int(m.eq_obj1id[i]), int(m.eq_obj2id[i])
        if j1 in legs and j2 == -1:
            locks.append({'name': mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, i),
                          'joint': names[j1], 'active_by_default': bool(m.eq_active0[i])})
    checks = {'42_leg_joints': len(legs) == 42,
              'one_free_root': int(np.sum(m.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE))) == 1,
              'no_single_joint_leg_equalities_even_if_inactive': not locks}
    return {'status': 'PASS' if all(checks.values()) else 'FAIL',
            'checks': checks, 'inherited_locks': locks, 'nq': m.nq, 'nv': m.nv}


def require_unlocked(m):
    report = inspect_model(m)
    if report['status'] != 'PASS':
        raise ValueError('Standing lock guard failed: ' + json.dumps(report))
    return report


def selftest():
    # A name-independent lock, including an initially inactive one, must be caught.
    joints = ''.join(f'<body name="b{leg}{j}" pos="0 0 {j + 1}"><joint name="joint_{leg}test{j}"/><geom size=".01"/></body>'
                     for leg in LEGS for j in range(7))
    base = '<mujoco><worldbody><body><freejoint/><geom size=".1"/>' + joints + '</body></worldbody>{}</mujoco>'
    assert inspect_model(mujoco.MjModel.from_xml_string(base.format('')))['status'] == 'PASS'
    for active in ('true', 'false'):
        eq = '<equality><joint name="unexpected_name" joint1="joint_RFtest0" active="' + active + '"/></equality>'
        assert inspect_model(mujoco.MjModel.from_xml_string(base.format(eq)))['status'] == 'FAIL'
    print('PASS: nonempty valid model accepted; active and inactive unnamed-pattern locks rejected')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xml', type=Path, default=DEFAULT)
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs/standing_locks_check.json')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    selftest()
    if args.selftest:
        return 0
    m = _load_mjcf(str(args.xml.resolve())).compile()
    report = inspect_model(m)
    report.update(xml=str(args.xml.resolve()), xml_sha256=hashlib.sha256(args.xml.read_bytes()).hexdigest(),
                  limitation='Structural build guard only; passing does not establish standing.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
