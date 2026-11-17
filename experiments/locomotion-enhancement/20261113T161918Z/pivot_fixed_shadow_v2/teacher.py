"""ONLYCODE two pivot cases, fixed private FK factor; v1 trajectories unchanged."""
from pathlib import Path
import importlib.util
import sys
import types

OLD=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/pivot_teacher_design')
spec=importlib.util.spec_from_file_location('_pivotfixed_frozen_teacher_v1',OLD/'teacher.py')
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
pin_baseline=old.pin_baseline
body_length_from_compiled_mesh=old.body_length_from_compiled_mesh
CONDITIONS=old.CONDITIONS


def make_teacher(c,condition,*,order_indices):
    if condition not in ('left_pivot','right_pivot'):raise ValueError('Only two new fixed-shadow pivot cases; no stop rerun')
    result=old.make_teacher(c,condition,order_indices=order_indices)
    from static_teacher import StaticTeacher
    import numpy as np
    fixed_private_root=result.q_des[:7].copy()
    fixed_rotation=result.initial_rot.copy()
    fixed_origin=result.initial_origin.copy()

    def fixed_action(self,c,t):
        before={name:getattr(c.d,name).copy() for name in ('qpos','qvel','xfrc_applied','qfrc_applied')}
        action,diag=StaticTeacher.action(self,c,t)
        if not (np.array_equal(self.q_des[:7],fixed_private_root) and np.array_equal(self.initial_rot,fixed_rotation) and np.array_equal(self.initial_origin,fixed_origin)):
            raise RuntimeError('Private handover root must stay fixed')
        if any(not np.array_equal(getattr(c.d,name),value) for name,value in before.items()):
            raise RuntimeError('Teacher changed actual generalized state or external forces')
        diag.update(teacher_reference=self.reference.last_snapshot,actual_pose_shadow=False,fixed_handover_private_shadow=True,desired_root_reference_only=True,no_actual_root_or_external_force_write=True)
        return action,diag

    result.action=types.MethodType(fixed_action,result)
    result.metadata.update(shadow='PRIVATE FK fixed at actual handover pose; continuous42desiredjoints; no actualrootwrite',single_factor='skip actualprivate-root/initial_rot/initial_origin synchronization, validated as supportfactor in archived stopcontrol',matched_control='same-direction archived v1 actual-shadow pivot, not rerun')
    return result
