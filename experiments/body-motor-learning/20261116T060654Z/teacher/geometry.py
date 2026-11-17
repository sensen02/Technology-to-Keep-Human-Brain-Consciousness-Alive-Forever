"""Static ground-mounted obstacle added to the full frozen fly plant."""
from pathlib import Path
import sys

import mujoco as mj
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / '20261116T000536Z'))
from walking_bench import stand

LEG = 'RF'
SEGMENTS = tuple(LEG + name + '_geom' for name in ('Tibia', 'Tarsus1', 'Tarsus2'))
OBSTACLE = 'stumble_box'


def make_obstacle_model(scenario):
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    original = stand.physics.act_mod.load_full_model(stand.physics.PAD_XML)
    if scenario is None:
        return original
    xyz = np.array(scenario['center'], dtype=float)
    half = np.array(scenario['half_size'], dtype=float)
    if not np.isfinite(np.r_[xyz, half]).all() or np.any(half <= 0.) or xyz[2] != half[2]:
        raise ValueError('Expected a finite upright box with its base on the ground')
    spec = _load_mjcf(stand.physics.PAD_XML)
    spec.worldbody.add_geom(name=OBSTACLE, type=mj.mjtGeom.mjGEOM_BOX, size=half,
                            pos=xyz, rgba=[.88, .26, .09, 1.], contype=1, conaffinity=2,
                            condim=3, solref=[.002, 1.])
    model = spec.compile()
    stand.physics.act_mod.validate_full_model(model)
    for material in range(model.nmat):
        model.mat_rgba[material] = original.mat_rgba[material]
    for g in range(original.ngeom):
        old = original.geom(g)
        new = model.geom(old.name)
        model.geom_rgba[new.id] = original.geom_rgba[old.id]
        for attr in ('type','contype','conaffinity','condim','size','pos','quat',
                     'friction','solref','solimp','margin','gap','priority'):
            np.testing.assert_array_equal(getattr(new, attr), getattr(old, attr), err_msg=f'{old.name} {attr}')
    for attr in ('jnt_type','jnt_range','jnt_qposadr','jnt_dofadr','body_mass','body_inertia','body_ipos','body_iquat',
                 'actuator_trnid','actuator_gear','dof_damping','dof_armature','jnt_stiffness'):
        np.testing.assert_array_equal(getattr(model, attr), getattr(original, attr), err_msg=attr)
    return model
