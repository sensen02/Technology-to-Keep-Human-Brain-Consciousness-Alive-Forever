"""S179 engine-free regressions (no engine import)."""
import argparse, json, sys
from pathlib import Path
import numpy as np
from s179_core import (build_row_dof_map, validate_identity, IdentityError, classify_root,
                       select_best, row_of_dof, leg_of_name)


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    t = []

    def check(name, cond):
        assert cond, name
        t.append(dict(test=name, passed=True))

    # dof names are indexed by DOF: 6 root DOFs then the 42 leg DOFs in model order
    names = ['root_t%d' % i for i in range(6)]
    for leg in ('LF', 'LH', 'LM', 'RF', 'RH', 'RM'):
        names += [f'{leg}J{j}' for j in range(7)]
    rm = build_row_dof_map(names)
    check('row map covers rows 0..41 and dofs 6..47',
          [v['joint_row'] for v in rm] == list(range(42)) and [v['dof'] for v in rm] == list(range(6, 48)))
    check('joint_ prefix accepted by leg_of_name', leg_of_name('joint_RMCoxa_yaw') == 'rm' and leg_of_name('rm') == 'rm' and leg_of_name('LH') == 'lh')
    check('leg names derived from dof names', rm[0]['leg'] == 'lf' and rm[7]['leg'] == 'lh' and rm[21]['leg'] == 'rf' and rm[35]['leg'] == 'rm')
    check('row_of_dof round trip', all(row_of_dof(rm, v['dof']) == v['joint_row'] for v in rm))

    # misaligned row/dof must raise (the exact S178 defect this guards)
    try:
        validate_identity(rm, rows=[13, 16], dofs=[13, 16], context='deliberate misalignment')
        raise AssertionError('misalignment not detected')
    except IdentityError:
        t.append(dict(test='misaligned row/dof raises', passed=True))
    check('aligned row/dof passes', validate_identity(rm, rows=[13, 16], dofs=[19, 22], context='aligned'))

    check('balanced class', classify_root([1e-3, 1e-9], 1e-12) == 'ROOT_BALANCED')
    check('diverge class on increase', classify_root([1.0, 5.0, 9.0], 9.0) == 'ROOT_DIVERGE')
    check('diverge class on nonfinite', classify_root([1.0, np.inf], np.inf, nonfinite=True) == 'ROOT_DIVERGE')
    check('contact-set class', classify_root([30.0, 4.0, 1.0], 1.0, contact_changed=True) == 'ROOT_CONTACT_SET_CHANGE')
    check('floor class', classify_root([30.0, 4.0, 0.7], 0.7) == 'ROOT_RESIDUAL_FLOOR')

    cands = [dict(name='root_failed_low_minimax', root_ok=False, geom_ok=False, minimax=0.01),
             dict(name='root_ok_geom_fail', root_ok=True, geom_ok=False, minimax=0.02),
             dict(name='a', root_ok=True, geom_ok=True, minimax=5.0),
             dict(name='b', root_ok=True, geom_ok=True, minimax=3.0)]
    sel = select_best(cands)
    check('best never crosses classes', sel['best']['name'] == 'b' and sel['n_valid'] == 2)
    check('empty valid set reported explicitly',
          select_best([dict(name='x', root_ok=False, geom_ok=True, minimax=0.0)])['selection'] == 'NO_ROOT_BALANCED_GEOM_VALID_POINT')
    check('no engine imported', 'mujoco' not in sys.modules)
    (out/'TEST_RESULTS.json').write_text(json.dumps(dict(passed=True, engineering_only=True, model_load=0, forward=0, tests=t), indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True)
    run(ap.parse_args().output); print('S179 core regressions passed')
