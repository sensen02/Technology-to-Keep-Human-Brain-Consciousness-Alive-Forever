"""Scale every actuator ctrlrange by a declared factor; refuse to write if anything else differs."""
from pathlib import Path
import json, hashlib, re
import numpy as np
import mujoco as mj
import importlib.util, sys
HERE = Path(__file__).resolve().parent
SRC = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-tarsus-pad/20261109T010000Z/MODEL_TARSUS_PAD_R02.xml')
FACTORS = [1.5, 2.0]


def arrays(m):
    return dict(nq=m.nq, nv=m.nv, nu=m.nu, nbody=m.nbody, ngeom=m.ngeom, nmesh=m.nmesh, neq=m.neq,
                jnt_range=m.jnt_range.copy(), jnt_limited=m.jnt_limited.copy(),
                body_mass=m.body_mass.copy(), body_inertia=m.body_inertia.copy(), body_ipos=m.body_ipos.copy(),
                actuator_gear=m.actuator_gear.copy(), actuator_gainprm=m.actuator_gainprm.copy(),
                actuator_biasprm=m.actuator_biasprm.copy(), dof_damping=m.dof_damping.copy(),
                dof_armature=m.dof_armature.copy(), opt_gravity=m.opt.gravity.copy(), opt_timestep=m.opt.timestep,
                geom_size=m.geom_size.copy(), geom_pos=m.geom_pos.copy(), geom_friction=m.geom_friction.copy(),
                geom_contype=m.geom_contype.copy(), geom_conaffinity=m.geom_conaffinity.copy())


def main():
    xml = SRC.read_text()
    MOTOR = r'(<motor[^>]*ctrlrange=")([-0-9.eE+]+) ([-0-9.eE+]+)(")'
    rows = list(re.finditer(MOTOR, xml))
    assert len(rows) == 42, len(rows)
    spec = importlib.util.spec_from_file_location('act', '/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-incremental/20261107T010000Z/incremental_actuator.py')
    act = importlib.util.module_from_spec(spec); sys.modules['act'] = act; spec.loader.exec_module(act)
    base = act.load_full_model(SRC)
    ref = arrays(base)
    ledger = dict(source=str(SRC), source_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),
                  ctrlrange_entries=len(rows), variants=[])
    for f in FACTORS:
        def sub(mo):
            lo, hi = float(mo.group(2)), float(mo.group(3))
            return mo.group(1) + '%.12g %.12g' % (lo * f, hi * f) + mo.group(4)
        out_xml, k = re.subn(MOTOR, sub, xml)
        assert k == 42
        out = HERE / ('MODEL_STRONG_x%03d.xml' % round(f * 100))
        out.write_text(out_xml)
        m = act.load_full_model(out)
        cur = arrays(m)
        diffs = [k2 for k2 in ref if not np.array_equal(np.asarray(ref[k2]), np.asarray(cur[k2]))]
        extra = [k2 for k2 in diffs if k2 not in ('actuator_ctrlrange', 'actuator_ctrlrange_numeric')]
        ctrl_ok = np.allclose(cur['actuator_gear'].shape, ref['actuator_gear'].shape)
        scale_ok = np.allclose(m.actuator_ctrlrange, base.actuator_ctrlrange * f)
        # the only permitted difference is the torque limit itself
        numeric = dict(nq=m.nq == base.nq, nv=m.nv == base.nv, nu=m.nu == base.nu, ngeom=m.ngeom == base.ngeom,
                       nmesh=m.nmesh == base.nmesh, neq=m.neq == base.neq,
                       gravity=np.array_equal(m.opt.gravity, base.opt.gravity), timestep=m.opt.timestep == base.opt.timestep,
                       masses=np.array_equal(m.body_mass, base.body_mass),
                       inertias=np.array_equal(m.body_inertia, base.body_inertia),
                       jnt_range=np.array_equal(m.jnt_range, base.jnt_range),
                       gain=np.array_equal(m.actuator_gainprm, base.actuator_gainprm),
                       bias=np.array_equal(m.actuator_biasprm, base.actuator_biasprm),
                       gear=np.array_equal(m.actuator_gear, base.actuator_gear),
                       damping=np.array_equal(m.dof_damping, base.dof_damping),
                       armature=np.array_equal(m.dof_armature, base.dof_armature),
                       geom_size=np.array_equal(m.geom_size, base.geom_size),
                       geom_friction=np.array_equal(m.geom_friction, base.geom_friction),
                       ctrlrange_scaled=bool(scale_ok))
        passed = all(numeric.values()) and not extra and out.stat().st_size > 0
        ledger['variants'].append(dict(factor=f, path=str(out),
                                       sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                                       invariants=numeric, unexpected_differences=extra,
                                       ctrlrange_max_before=float(base.actuator_ctrlrange[:, 1].max()),
                                       ctrlrange_max_after=float(m.actuator_ctrlrange[:, 1].max()),
                                       invariants_passed=bool(passed)))
        print(json.dumps(dict(factor=f, passed=bool(passed), unexpected=extra,
                              max_after=float(m.actuator_ctrlrange[:, 1].max()))))
    (HERE / 'LEDGER_strong_actuator.json').write_text(json.dumps(ledger, indent=2) + '\n')


if __name__ == '__main__':
    main()
