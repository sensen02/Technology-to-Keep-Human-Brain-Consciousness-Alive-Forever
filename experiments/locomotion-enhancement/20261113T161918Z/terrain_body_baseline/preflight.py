"""Parent CPU-only bounded full-body contact preflight; no NN or full trial.

Exactly four Core instances and 6400 total native steps if complete: original
plane and derived plane in lockstep, then derived flat box and seed17 blocks.
This admits measurement/identity integration only, never locomotion capability.
"""
from pathlib import Path
import argparse
import json
import os

from prepare import HERE, BASE, CASES, verify_prepared, save
import terrain_core
import terrain_metrics

TICKS = 20


def jsonable(value):
    if isinstance(value, dict):
        return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [jsonable(v) for v in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def checks_model(np, mj, original, derived):
    check = dict(full_body=(derived.plant.nq, derived.plant.nv, derived.plant.nu,
        derived.plant.nmesh, derived.plant.neq) == (49,48,42,71,0),
        all_71_mesh_opaque=bool(np.all(derived.plant.geom_rgba[derived.plant.geom_type == mj.mjtGeom.mjGEOM_MESH,3] == 1.)),
        zero_external=not bool(np.any(derived.d.xfrc_applied) or np.any(derived.d.qfrc_applied)),
        free_root_count=int(np.sum(derived.plant.jnt_type == mj.mjtJoint.mjJNT_FREE)) == 1,
        terrain_mask_exact=bool(np.array_equal(derived.planes, np.asarray([derived.plant.geom(i).name.startswith('terrain_') for i in range(derived.plant.ngeom)]))))
    for name in ('body_mass','body_inertia','body_ipos','body_iquat','jnt_type','jnt_range',
                 'jnt_qposadr','jnt_dofadr','dof_damping','dof_armature','qpos0',
                 'actuator_trnid','actuator_ctrlrange','actuator_gainprm','actuator_biasprm'):
        check[name+'_unchanged'] = bool(np.array_equal(getattr(original.plant,name), getattr(derived.plant,name)))
    original_fly = [i for i in range(original.plant.ngeom) if original.plant.geom(i).name != 'floor']
    for field in ('geom_type','geom_size','geom_pos','geom_quat','geom_bodyid','geom_contype','geom_conaffinity','geom_condim','geom_friction'):
        check['all_original_fly_'+field+'_unchanged_by_name'] = all(np.array_equal(
            getattr(original.plant,field)[i], getattr(derived.plant,field)[derived.plant.geom(original.plant.geom(i).name).id]) for i in original_fly)
    check['body_count_unchanged'] = original.plant.nbody == derived.plant.nbody
    for method in ('__init__','_pose','reset','update','torque','state_dict'):
        check['actuator_'+method+'_bytecode_unchanged'] = getattr(type(original.actuator),method).__code__.co_code == getattr(type(derived.actuator),method).__code__.co_code
    for name in ('kp','kd','torque_limit','max_rate_rad_s','target_bounds_rad'):
        check['finite_PD_'+name+'_unchanged'] = bool(np.array_equal(getattr(original.actuator,name),getattr(derived.actuator,name)))
    check['gravity_unchanged'] = bool(np.array_equal(original.plant.opt.gravity, derived.plant.opt.gravity))
    check['dt_unchanged'] = float(original.plant.opt.timestep) == float(derived.plant.opt.timestep)
    return check


def flat_closure(np, old, new):
    check = {}
    for name in ('qpos','qvel','ctrl','actuator_force'):
        check[name] = bool(np.array_equal(getattr(old.d,name),getattr(new.d,name)))
    for name in ('buf','_pad_rate','_padz_prev'):
        check[name] = bool(np.array_equal(getattr(old,name),getattr(new,name)))
    check['actuator_target'] = bool(np.array_equal(old.actuator.target_rad,new.actuator.target_rad))
    a,b = old.measure(),new.measure()
    for name in ('loads','nonfoot','speed','raw_speed','padz','foot','com','contact'):
        check['measurement_'+name] = bool(np.array_equal(a[name],b[name]))
    check['X238'] = bool(np.array_equal(old.obs238(np.zeros(42),.075), new.obs238(np.zeros(42),.075)))
    return check


def method_paths(base, core):
    np = base.np
    value = core.measure()
    contact_count = base.rc.Core.contact_count(core)
    native = base.NativeRecorder.solved(None,core,{'qvel_before':core.d.qvel.copy()})
    # Exercise exact inherited contact methods with the same registered mask.
    return dict(contact_count_matches_measure=bool(np.array_equal(contact_count,value['contact'])),
        native_contact_count_matches=bool(np.array_equal(native['contact_count'],value['contact'])),
        native_loads_match=bool(np.array_equal(native['loads_native'],value['loads'])),
        native_nonfoot_matches=bool(native['nonfoot_force_native'] == value['nonfoot']),
        observation_shape=core.obs238(np.zeros(42),.075).shape == (238,),
        no_hidden_force=not bool(np.any(core.d.xfrc_applied) or np.any(core.d.qfrc_applied)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-resource-released', action='store_true')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if not args.parent_resource_released:
        print('SOURCE_ONLY no Core construction, model load, NN or integration')
        return
    if args.out is None:
        parser.error('--out required for parent numerical preflight')
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(key) != '1':
            raise ValueError(key+'=1 required before imports')
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    frozen = verify_prepared()
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True,exist_ok=False)
    completed = {}
    native_steps = 0
    try:
        base = terrain_core.load_base_physics()
        np,mj = base.np,base.mj
        Derived = terrain_core.core_type(base)
        original = base.Core(adhesion=False)
        plane_recorder = terrain_metrics.TerrainPeakRecorder()
        plane = Derived(case='plane_control',recorder=plane_recorder)
        model_check = checks_model(np,mj,original,plane)
        closure = [flat_closure(np,original,plane)]
        for _ in range(TICKS):
            original.tick(np.zeros(42))
            plane.tick(np.zeros(42))
            native_steps += 160
            closure.append(flat_closure(np,original,plane))
        plane_paths = method_paths(base,plane)
        terrain_core_record = dict(identity=terrain_core.actual_identity(plane), model_checks=model_check,
            all_checks=[*model_check.values(),*plane_paths.values(),*(v for row in closure for v in row.values())],
            flat_closure_by_tick=closure, inherited_method_checks=plane_paths,
            native=plane_recorder.close(), independent_claim='flat plane measurement/state/X238 zero-action bitexact; no NN tested')
        completed['plane_control'] = terrain_core_record
        save(out/'plane_control.json',jsonable(terrain_core_record))
        for case in CASES[1:]:
            recorder = terrain_metrics.TerrainPeakRecorder()
            core = Derived(case=case,recorder=recorder)
            model_check = checks_model(np,mj,original,core)
            ray_checks = []
            for item in core.terrain_ledger['terrain']:
                footprint = item['footprint']
                xy = [0.,0.] if isinstance(footprint,str) else [(footprint[0]+footprint[1])*.5,(footprint[2]+footprint[3])*.5]
                ground, hit = terrain_metrics.height(core,xy)
                ray_checks.append(dict(terrain=item['name'],xy=xy,height=ground,
                    expected=item['top_z'],hit_name=core.plant.geom(hit).name,
                    correct=abs(ground-item['top_z']) < 1e-10))
            for _ in range(TICKS):
                core.tick(np.zeros(42))
                native_steps += 80
            paths = method_paths(base,core)
            value = core.measure()
            native = recorder.close()
            gate = dict(model_check)
            gate.update(paths)
            gate['terrain_height_queries'] = all(row['correct'] for row in ray_checks)
            gate['all_native_steps_recorded'] = native['actual_native_steps'] == TICKS*80
            if case == 'flat_box_control':
                gate['registered_box_contact_exists'] = native['registered_contact_count'] > 0
                gate['old_plane_zero_new_box_positive'] = native['legacy_plane_contact_count'] == 0 and native['registered_contact_count'] > 0
            completed[case] = dict(identity=terrain_core.actual_identity(core), checks=gate,
                all_checks=list(gate.values()), ray_checks=ray_checks, native=native,
                final_measurement={k:value[k] for k in ('loads','normal_loads','world_z_force_signed','padz','padz_world','com','com_world','nonfoot','nonfoot_normal','terrain_contact_count','legacy_plane_contact_count')},
                forward_body_velocity=float(core.forward_body_velocity()),
                obstacle_traversal_status='NOT_TESTED; short zero-action contact integration, no walking')
            save(out/(case+'.json'),jsonable(completed[case]))
        if 'student_wave' in __import__('sys').modules:
            raise ValueError('Contact-only stage unexpectedly loaded the student-wave runner')
        # FlyGym may import optional Torch-backed features at package import.
        # Their incidental presence is recorded; this probe loads no weights and
        # calls no NN forward/trainer, independently of package import side effects.
        verify_prepared()
        result = dict(kind='BOUNDED_FULL_BODY_TERRAIN_CORE_PREFLIGHT_V1',mujoco_version=mj.__version__,
            actual_Core_constructions=4, NN_forwards=0, actual_native_steps=native_steps,
            fixed_total_expected=6400, source_hashes=frozen['source_hashes'],
            cases={k:dict(all_passed=all(v['all_checks'])) for k,v in completed.items()},
            all_passed=native_steps == 6400 and all(all(v['all_checks']) for v in completed.values()),
            full_3case_neural_runner_ready=False, full_body_locomotion_or_hook_claim=False)
        save(out/'RESULT.json',result)
        print(json.dumps(result))
        if not result['all_passed']:
            raise SystemExit(2)
    except BaseException as error:
        if not isinstance(error,SystemExit):
            save(out/'FAILURE.json',dict(error=repr(error),actual_native_steps=native_steps,completed_cases=list(completed)))
        raise


if __name__ == '__main__':
    main()
