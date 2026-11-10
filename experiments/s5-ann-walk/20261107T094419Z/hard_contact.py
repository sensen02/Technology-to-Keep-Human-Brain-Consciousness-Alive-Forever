"""One-mechanism teacher comparison: hard zero loaded-contact acceleration.

Uses the frozen original092817 torque conversion, weights, bounds and fallback.
Rigid contact is a declared QP assumption; MuJoCo contact remains soft and unchanged.
Only parent-reviewed execution may step physics. Import/mechanism checks do not.
"""
import hashlib
import json
from pathlib import Path
import types

SOURCE = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T092817Z/wbc_teacher.py')
SOURCE_SHA256 = 'd32b450e7907df40283c25308fc53f54491d5487ac81086b10fac32f104ce4d5'
OUT = Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T094419Z')
original = SOURCE.read_text()
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA256, 'Frozen baseline changed'
patches = [
    ("HERE = Path(__file__).resolve().parent", "HERE = Path('"+str(SOURCE.parent)+"')"),
    ("20261107T092817Z')\nb.OUT = OUT", "20261107T094419Z')\nb.OUT = OUT"),
    ('constraint_tolerance=1e-4)',
     'constraint_tolerance=1e-4, hard_contact_acceleration_tolerance_native=1e-3)'),
    ('self.nc = 120  # dynamics48, torque42, normal6, pyramid24',
     'self.nc = 138  # dynamics48, torque42, normal6, pyramid24, hard_contact18'),
    ("        H = np.diag(np.r_[np.full(48, P['acceleration_regularizer']),",
     """        # Rigid no-acceleration assumption at actual loaded stance material points.
        # Jdotv has no anchor/velocity restoration term. Inactive rows are 0=0.
        lo[120:138] = 0.; hi[120:138] = 0.
        for li in range(6):
            if active[li]:
                rows = slice(120+3*li, 123+3*li)
                contact_scaled = Jc[li]*self.scale[:48]
                row_norm = np.maximum(np.linalg.norm(contact_scaled, axis=1), 1e-8)
                A[rows, :48] = contact_scaled/row_norm[:, None]
                lo[rows] = hi[rows] = -jdc[li]/row_norm
        H = np.diag(np.r_[np.full(48, P['acceleration_regularizer']),"""),
    ("                    and violation <= P['constraint_tolerance']",
     """                    and violation <= P['constraint_tolerance']
                    and np.max(abs((Jc@solution[:48]+jdc)*active[:, None])) <= P['hard_contact_acceleration_tolerance_native']"""),
    ("""        for li in range(6):
            if active[li]:
                acc = -P['stance_kp']*(points[li]-self.contact_anchor[li])-P['stance_kd']*(Jc[li]@d.qvel)
                cost(Jc[li], acc-jdc[li], P['contact_weight'])""",
     """        contact_soft_desired = np.zeros((6, 3))
        for li in range(6):
            if active[li]:
                acc = -P['stance_kp']*(points[li]-self.contact_anchor[li])-P['stance_kd']*(Jc[li]@d.qvel)
                contact_soft_desired[li] = acc
                cost(Jc[li], acc-jdc[li], P['contact_weight'])"""),
    ("            dynamics_rhs=-bias.copy(),",
     """            dynamics_rhs=-bias.copy(),
            soft_contact_desired_acceleration=contact_soft_desired.copy(),
            hard_contact_acceleration_residual=(Jc@solution[:48]+jdc)*active[:, None],"""),
    ("default='wbc_screen'", "default='hard_contact_screen'"),
]
effective = original
for old, new in patches:
    assert effective.count(old) == 1, ('Expected exactly one frozen source match', old)
    effective = effective.replace(old, new, 1)
EFFECTIVE_SHA256 = hashlib.sha256(effective.encode()).hexdigest()
runtime = types.ModuleType('hard_contact_effective_runtime')
runtime.__file__ = str(SOURCE)
exec(compile(effective, str(SOURCE), 'exec'), runtime.__dict__)
original_prereg = runtime.b.prereg


def prereg(folder, args):
    # New run directory only; archive the exact count-checked executable code
    # before any actual physical integration, preserving the frozen baseline.
    effective_path = folder/'EFFECTIVE_WBC.py'
    if effective_path.exists():
        raise FileExistsError(effective_path)
    effective_path.write_text(effective)
    original_prereg(folder, args)
    path = folder/'PREREGISTRATION.json'
    protocol = json.loads(path.read_text())
    protocol.update(stage='S5 hard-contact dynamics QP teacher feasibility only',
        single_mechanism_change='Hard loaded nominal stance Jc*qacc+Jdotv=0; inactive extra rows0=0. This changes desired material acceleration from the baseline nonzero centroid-anchor/velocity-PD target to zero in the hard constraint, rather than merely hardening that old target. The old soft PD term remains in the objective; its desired acceleration is logged. Original092817 instantaneous torque conversion, weights/blend/nativebounds unchanged. No094008 lookahead.',
        hard_contact=dict(rows=[120, 138], total_constraints=138, variables=108,
            native_acceleration_tolerance=runtime.P['hard_contact_acceleration_tolerance_native'],
            assumptions='Rigid zero contact acceleration is a teacher planning approximation. Actual MuJoCo contact compliance/friction/forces and real root state unchanged. No anchor restoration PD in hard equality.',
            diagnostic='QP.hard_contact_acceleration_residual and soft_contact_desired_acceleration have native6x3 vectors; inactive rows0. Old forceweighted contact centroid can change as contact patches roll; it is not a persistent material anchor. Infeasible/status/dynamics/contact-residual failures use nominal fallback and cannot count as hard-QP success.'),
        baseline=dict(source=str(SOURCE), sha256=SOURCE_SHA256),
        adaptation=dict(source=str(Path(__file__).resolve()), sha256=runtime.n.sha(__file__),
            effective_source=str(effective_path), effective_sha256=EFFECTIVE_SHA256,
            patch_count=len(patches), patches=[dict(old=old, new=new, required_matches=1) for old, new in patches]),
        exclusions='No physical qpos/qvel/root writes, auxiliary force assignment, torque/friction increase, constraint relaxation or neural claim.')
    protocol['solver']['fixed_constraints'] = 138
    runtime.n.dump(path, protocol)


runtime.b.prereg = prereg


if __name__ == '__main__':
    runtime.main()
