"""Parent opt-in: exactly one Core, private FK only, ZERO integration."""
import argparse
import sys
from common import *


def main():
    p = argparse.ArgumentParser(); p.add_argument('--execute-parent-preflight', action='store_true'); a = p.parse_args()
    if not a.execute_parent_preflight: p.error('Parent preflight flag required')
    cpu1(); sources = frozen(); verify(sources)
    if (ROOT/'PREFLIGHT.json').exists(): raise FileExistsError(ROOT/'PREFLIGHT.json')
    sw, physics, identity, static, Peak, ast_ids = physical_stack()
    import numpy as np
    import mujoco as mj
    import teacher
    checks = {}; peak = Peak(); c = physics.Core(adhesion=False, recorder=peak)
    start = snapshot(c); t0 = float(c.d.time)
    mesh = c.plant.geom_type == mj.mjtGeom.mjGEOM_MESH
    checks['full71opaque42finitePD'] = bool(mesh.sum() == 71 and np.all(c.plant.geom_rgba[mesh, 3] == 1.) and c.plant.nu == 42 and np.all(c.actuator.kp > 0) and np.all(c.actuator.torque_limit > 0) and np.isfinite(c.actuator.torque_limit).all() and np.isfinite(c.actuator.target_bounds_rad).all())
    checks['noadhesion_zeroextra'] = not c.adhesion_on and not np.any(c.d.xfrc_applied) and not np.any(c.d.qfrc_applied)
    length = teacher.body_length_from_compiled_mesh(c)
    checks['bodylength7namedmeshes'] = len(length['segments']) == 7 and length['length_native'] > 0
    warm = static.StaticTeacher(c, sw.teacher_config(3))
    order = warm.reference.order_indices
    action, diag = warm.action(c, 0.)
    checks['warm_finite42_private_only'] = action.shape == (42,) and np.isfinite(action).all() and np.max(np.abs(action)) <= 1.
    assert_same(start, snapshot(c))
    candidate = teacher.make_teacher(c, 'left_pivot', order_indices=order)
    action, diag = candidate.action(c, 0.)
    checks['candidate_initial_target_matches_actual_rest'] = bool(np.max(np.abs(np.asarray(diag['desired_foot_xy_bottom_z'])[:, :2]-candidate.rest[:, :2])) < 1e-10)
    checks['candidate_private_qdes_nq_act42'] = candidate.q_des.shape == c.d.qpos.shape and candidate.q_des[candidate.qa].shape == (42,) and np.isfinite(candidate.q_des).all()
    action2, diag2 = candidate.action(c, .005)
    checks['candidate42finite_no_actual_write'] = bool(action2.shape == (42,) and np.isfinite(action2).all() and np.max(np.abs(action2)) <= 1.)
    assert_same(start, snapshot(c))
    # Central differences of one actual per-leg joint, in the same private FK frame.
    errors = []
    original = candidate.q_des.copy()
    for li in range(6):
        candidate._fk(); pos, jac = candidate._leg_kinematics(c, li)
        qi = candidate.qa[candidate.leg_act[li][0]]; eps = 1e-7
        candidate.q_des[qi] += eps; candidate._fk(); plus = candidate._leg_kinematics(c, li, False)
        candidate.q_des[qi] -= 2*eps; candidate._fk(); minus = candidate._leg_kinematics(c, li, False)
        candidate.q_des[:] = original
        errors.append(float(np.max(np.abs((plus[:2]-minus[:2])/(2*eps)-jac[:2, 0]))))
    candidate._fk()
    checks['private_actualframe_jacobian_central_difference'] = max(errors) < 1e-4
    checks['zero_integration'] = float(c.d.time) == t0 and c.t == 0 and peak.count == 0
    checks['no_torch_or_optimizer'] = 'torch' not in sys.modules
    manifest = identity.capture(c, dict(kind='PIVOT_TEACHER_ZERO_INTEGRATION_PREFLIGHT', source_hashes=sources), extra_paths=list(sources))
    identity.verify_runtime(manifest, c); assert_same(start, snapshot(c)); verify(sources)
    result = dict(all_passed=bool(all(checks.values())), checks=checks, physical_integration_steps=0, Core_constructions=1, neural_forwards=0, optimizer_updates=0, initial_body_length=length, jacobian_errors=errors, order_indices=list(order), actual_snapshot_unchanged=True, source_hashes=sources, selected_legacy_ast=ast_ids, identity=manifest, warning='One unintegrated Core/private FK probe only; no physical teacher feasibility or stop/pivot success established. Actual runner creates fresh Core and teachers per case.')
    save(ROOT/'PREFLIGHT.json', result)
    if not result['all_passed']: raise ValueError('Preflight failed; evidence retained')
    print(json.dumps(dict(preflight=str(ROOT/'PREFLIGHT.json'), all_passed=True, integration_steps=0)))


if __name__ == '__main__': main()
