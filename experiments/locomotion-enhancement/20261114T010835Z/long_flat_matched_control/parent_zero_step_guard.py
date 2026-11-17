"""Future parent-only compiled-model guard. Importing this file uses stdlib only.

This file has deliberately NOT been executed during ONLYCODE preparation.
Run only after the parent releases resources, with --parent-resource-released.
The orchestrator must require all_passed, steps == 0 and Core == 1 in the
exclusive-create PARENT_ZERO_STEP_GUARD.json before admitting any physical run.
"""
from pathlib import Path
import argparse
import contextlib
import hashlib
import importlib.util
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent


def _load_runtime():
    spec = importlib.util.spec_from_file_location('_flat_parent_zero_step_runtime', HERE/'runtime.py')
    r = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = r
    spec.loader.exec_module(r)
    return r


@contextlib.contextmanager
def _prohibit_execution(mj, np, counters):
    """Block all three native integration entrypoints and array/checkpoint loads.

    The profile hook also refuses Python neural forward calls, including aliases
    already imported by dependencies. No policy loader is called by this guard.
    These are process-local tripwires, not an adversarial security sandbox.
    """
    originals = []
    previous_profile = sys.getprofile()
    physical_objects = set()

    def step_forbidden(*args, **kwargs):
        counters['forbidden_step_attempts'] += 1
        raise RuntimeError('Zero-step guard prohibits mj_step/mj_step1/mj_step2')

    def payload_forbidden(*args, **kwargs):
        counters['payload_attempts'] += 1
        raise RuntimeError('Zero-step guard prohibits array/checkpoint payload loading')

    def profile(frame, event, arg):
        if event == 'call':
            name = frame.f_code.co_name
            if name in ('forward', '_call_impl', '_wrapped_call_impl', 'load_policies'):
                counters['neural_attempts'] += 1
                raise RuntimeError('Zero-step guard prohibits neural forwards/policy loading')
            if name == '__init__' and frame.f_code.co_qualname.endswith('Core.__init__'):
                # Count the base physical constructor, not the TerrainCore wrapper.
                if frame.f_code.co_qualname.split('.')[-2] == 'Core':
                    physical_objects.add(id(frame.f_locals['self']))
                    counters['base_Core_constructors'] = len(physical_objects)
                    if len(physical_objects) > 1:
                        raise RuntimeError('Exactly one physical Core is permitted')
        if previous_profile is not None:
            previous_profile(frame, event, arg)

    def replace(owner, name, replacement):
        originals.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    try:
        for name in ('mj_step', 'mj_step1', 'mj_step2'):
            replace(mj, name, step_forbidden)
        for name in ('load', 'loadtxt', 'genfromtxt', 'fromfile'):
            replace(np, name, payload_forbidden)
        sys.setprofile(profile)
        yield
    finally:
        sys.setprofile(previous_profile)
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)


def _payload_audit(counters):
    """Return an audit callback that is inactive after this guard finishes."""
    active = [True]
    forbidden = {'.npy', '.npz', '.pt', '.pth', '.pkl', '.pickle', '.safetensors'}

    def audit(event, args):
        if active[0] and event == 'open' and isinstance(args[0], (str, bytes)):
            value = args[0].decode() if isinstance(args[0], bytes) else args[0]
            if Path(value).suffix.lower() in forbidden:
                counters['payload_attempts'] += 1
                raise RuntimeError('Zero-step guard prohibits payload file access: '+value)
    return active, audit


def _named_model_comparison(np, mj, old, new):
    """Compare compiled values, remapping references and ragged mesh slices.

    Address arrays are storage offsets, not physical values. They are checked by
    comparing the named objects or slices they address. body_geomadr/body_geomnum
    are checked against mapped body geom membership after the six deletions.
    """
    checks, details = {}, {}
    groups = {
        'body': (mj.mjtObj.mjOBJ_BODY, old.nbody, new.nbody),
        'jnt': (mj.mjtObj.mjOBJ_JOINT, old.njnt, new.njnt),
        'actuator': (mj.mjtObj.mjOBJ_ACTUATOR, old.nu, new.nu),
        'mesh': (mj.mjtObj.mjOBJ_MESH, old.nmesh, new.nmesh),
        'geom': (mj.mjtObj.mjOBJ_GEOM, old.ngeom, new.ngeom),
    }
    maps = {}
    for group, (kind, old_count, new_count) in groups.items():
        def names(model, count):
            values = [mj.mj_id2name(model, kind, i) for i in range(count)]
            if any(not name for name in values) or len(set(values)) != count:
                raise ValueError('All '+group+' objects must have unique names')
            return {name: i for i, name in enumerate(values)}
        left, right = names(old, old_count), names(new, new_count)
        maps[group] = (left, right)
        checks[group+'_name_set'] = (set(right) <= set(left) if group == 'geom' else set(left) == set(right))
    removed = sorted(set(maps['geom'][0])-set(maps['geom'][1]))
    checks['exact_six_deleted_block_names'] = removed == ['terrain_block_%02d'%i for i in range(6)]
    checks['no_added_geoms'] = not (set(maps['geom'][1])-set(maps['geom'][0]))
    checks['deleted_geoms_static'] = all(int(old.geom_bodyid[maps['geom'][0][name]]) == 0 for name in removed)
    details['deleted_geoms'] = removed

    def ref(group, model, value):
        kind = groups[group][0]
        def one(index):
            index = int(index)
            return '<none>' if index < 0 else mj.mj_id2name(model, kind, index)
        return [one(v) for v in np.asarray(value).reshape(-1)]

    # qpos and dofs have no independent names; identity is (joint name, local slot).
    def joint_slots(model, names, address, total):
        result = {}
        for name, index in names.items():
            start = int(getattr(model, address)[index])
            stop = int(getattr(model, address)[index+1]) if index+1 < model.njnt else total
            for slot, item in enumerate(range(start, stop)):
                result[(name, slot)] = item
        return result
    dofs = (joint_slots(old, maps['jnt'][0], 'jnt_dofadr', old.nv),
            joint_slots(new, maps['jnt'][1], 'jnt_dofadr', new.nv))
    qposs = (joint_slots(old, maps['jnt'][0], 'jnt_qposadr', old.nq),
             joint_slots(new, maps['jnt'][1], 'jnt_qposadr', new.nq))
    checks['dof_named_slots'] = set(dofs[0]) == set(dofs[1])
    checks['qpos_named_slots'] = set(qposs[0]) == set(qposs[1])
    for field, mapping in (('qpos0', qposs), ('qpos_spring', qposs)):
        checks[field] = all(np.array_equal(getattr(old, field)[mapping[0][key]],
                                          getattr(new, field)[mapping[1][key]]) for key in mapping[1])

    references = {
        'body_parentid': 'body', 'body_rootid': 'body', 'body_weldid': 'body',
        'jnt_bodyid': 'body', 'geom_bodyid': 'body', 'geom_matid': None,
        'dof_bodyid': 'body', 'dof_jntid': 'jnt',
    }
    addressed = {'body_geomadr', 'body_geomnum', 'body_jntadr', 'body_jntnum',
                 'body_dofadr', 'body_dofnum', 'jnt_qposadr', 'jnt_dofadr',
                 'body_bvhadr', 'body_bvhnum', 'dof_Madr', 'dof_simplenum'}
    # Mesh addresses are handled with the physical slices below, including normals,
    # texture coordinates and convex hull graph data where exposed by MuJoCo.
    mesh_addresses = {name for name in dir(old) if name.startswith('mesh_') and name.endswith('adr')}
    compared, addressed_fields = [], []
    for group in ('body', 'jnt', 'dof', 'actuator', 'mesh', 'geom'):
        left, right = dofs if group == 'dof' else maps[group]
        for field in sorted(name for name in dir(old) if name.startswith(group+'_')):
            a, b = getattr(old, field), getattr(new, field)
            if not isinstance(a, np.ndarray):
                continue
            if field in addressed or field in mesh_addresses:
                addressed_fields.append(field)
                continue
            if group == 'mesh' and (a.ndim == 0 or a.shape[0] != old.nmesh):
                continue
            if a.ndim == 0 or a.shape[0] != len(left) or b.shape[0] != len(right):
                if group != 'mesh':
                    raise ValueError('Unclassified compiled array '+field)
                continue
            matched = True
            for name, j in right.items():
                i = left[name]
                if field in references and references[field] is not None:
                    equal = ref(references[field], old, a[i]) == ref(references[field], new, b[j])
                elif field == 'geom_dataid':
                    if int(old.geom_type[i]) == int(mj.mjtGeom.mjGEOM_MESH):
                        equal = ref('mesh', old, a[i]) == ref('mesh', new, b[j])
                    else:
                        equal = np.array_equal(a[i], b[j])
                elif field == 'actuator_trnid':
                    # This scene must use joint transmission only, as frozen finite PD does.
                    equal = (int(old.actuator_trntype[i]) == int(mj.mjtTrn.mjTRN_JOINT)
                             and int(new.actuator_trntype[j]) == int(mj.mjtTrn.mjTRN_JOINT)
                             and ref('jnt', old, a[i][:1]) == ref('jnt', new, b[j][:1])
                             and int(a[i][1]) == int(b[j][1]))
                elif field == 'dof_parentid':
                    inverse_old = {v:k for k,v in dofs[0].items()}
                    inverse_new = {v:k for k,v in dofs[1].items()}
                    equal = inverse_old.get(int(a[i]), '<none>') == inverse_new.get(int(b[j]), '<none>')
                else:
                    equal = np.array_equal(a[i], b[j])
                matched = matched and bool(equal)
            checks['compiled_'+field] = matched
            compared.append(field)

    for body_name, ni in maps['body'][1].items():
        oi = maps['body'][0][body_name]
        def members(model, index, address, count, mapping):
            start, length = int(getattr(model, address)[index]), int(getattr(model, count)[index])
            inverse = {v:k for k,v in mapping.items()}
            return {inverse[k] for k in range(start, start+length)}
        old_geoms = members(old, oi, 'body_geomadr', 'body_geomnum', maps['geom'][0])-set(removed)
        new_geoms = members(new, ni, 'body_geomadr', 'body_geomnum', maps['geom'][1])
        checks['body_geom_membership_'+body_name] = old_geoms == new_geoms
        for label, address, count, mapping in (
            ('joint', 'body_jntadr', 'body_jntnum', maps['jnt']),
            ('dof', 'body_dofadr', 'body_dofnum', dofs)):
            checks['body_'+label+'_membership_'+body_name] = (
                members(old, oi, address, count, mapping[0]) == members(new, ni, address, count, mapping[1]))

    # Global mesh buffers are all unchanged: terrain removal deletes no mesh.
    for field in sorted(name for name in dir(old) if name.startswith('mesh_')):
        a, b = getattr(old, field), getattr(new, field)
        if isinstance(a, np.ndarray) and field not in mesh_addresses and field not in compared:
            checks['mesh_buffer_'+field] = np.array_equal(a, b)
    for field in sorted(mesh_addresses):
        if field == 'mesh_bvhadr':
            # Global BVH addresses can move when static terrain geoms disappear;
            # the actual named mesh vertices/faces/poses are compared above.
            addressed_fields.append(field)
            continue
        checks['mesh_storage_'+field] = np.array_equal(getattr(old, field), getattr(new, field))
    # All global options are compared, including gravity, integrator, contact solver.
    for field in sorted(name for name in dir(old.opt) if not name.startswith('_')):
        a, b = getattr(old.opt, field), getattr(new.opt, field)
        if isinstance(a, (np.ndarray, int, float)):
            checks['option_'+field] = bool(np.array_equal(a, b))
    details.update(compared_named_arrays=compared, storage_address_fields=addressed_fields,
                   mapping='body/joint/actuator/mesh/geom names; dof and qpos by joint name plus local slot',
                   comparison='exact compiled values; geom_dataid and joint transmissions mapped by name')
    return checks, details


def run():
    # This function is called only behind the explicit CLI flag below.
    import numpy as np
    import mujoco as mj

    counters = dict(steps=0, Core=0, old_Core=0, old_xml_compiles=0,
                    base_Core_constructors=0, forbidden_step_attempts=0,
                    neural_attempts=0, payload_attempts=0)
    active, audit = _payload_audit(counters)
    sys.addaudithook(audit)
    checks, details = {}, {}
    r = None
    error = None
    try:
        with _prohibit_execution(mj, np, counters):
            r = _load_runtime()
            r.verify()
            checks['frozen_sources_before'] = True
            proof = r.read(HERE/'XML_DIFF_PROOF.json')
            old_xml = Path(proof['source_xml']).resolve()
            scene = r.scene_metadata()
            new_xml = Path(scene['new_scene_xml']).resolve()
            ns, state = r.context(audit=True)
            sw, identity = ns['pin']()
            counters['Core'] += 1
            core = sw.Core(adhesion=False, recorder=None)
            if counters['Core'] != 1:
                raise RuntimeError('Exactly one flat Core required')
            new = core.plant
            checks['case_matches_local_runtime'] = core.terrain_case == r.CASE
            checks['new_xml_matches_metadata'] = Path(core.derived_xml).resolve() == new_xml
            checks['new_xml_hash_matches_metadata'] = r.sha(new_xml) == scene['new_scene_xml_sha256']
            checks['rough_and_flat_paths_differ'] = old_xml != new_xml
            # Compile the rough model directly: never instantiate an old Core/data.
            old = mj.MjModel.from_xml_path(str(old_xml))
            counters['old_xml_compiles'] += 1
            checks['dimensions'] = (old.ngeom, new.ngeom, old.nq, new.nq,
                                    old.nv, new.nv, old.nu, new.nu) == (96,90,49,49,48,48,42,42)
            mesh_ids = np.flatnonzero(new.geom_type == mj.mjtGeom.mjGEOM_MESH)
            checks['full_71_mesh_opaque'] = (new.nmesh == 71 and len(mesh_ids) == 71
                and bool(np.all(new.geom_rgba[mesh_ids,3] == 1)))
            checks['eighteen_foot_geoms'] = len(core.foot_ids) == 18
            terrain = set(core.terrain_ids)
            checks['terrain_only_plane'] = (len(terrain) == 1 and all(
                new.geom(i).name == 'terrain_plane' and int(new.geom_bodyid[i]) == 0
                and int(new.geom_type[i]) == int(mj.mjtGeom.mjGEOM_PLANE)
                for i in terrain))
            checks['only_one_world_geom'] = set(map(int, np.flatnonzero(new.geom_bodyid == 0))) == terrain
            checks['no_adhesion_or_external_force'] = (not core.adhesion_on
                and not np.any(core.d.xfrc_applied) and not np.any(core.d.qfrc_applied)
                and not np.any(core._adhesion_wrench))
            checks['no_adhesion_actuator'] = not np.any(new.actuator_dyntype == mj.mjtDyn.mjDYN_MUSCLE) and all(
                'adhesion' not in (new.actuator(i).name or '').lower() for i in range(new.nu))
            checks['no_xml_adhesion_element'] = all(
                not list(ET.parse(path).getroot().iter('adhesion')) for path in (old_xml,new_xml))
            checks['time_zero'] = float(core.d.time) == 0.0
            compiled, comparison = _named_model_comparison(np, mj, old, new)
            checks.update(compiled)
            details.update(comparison)
            # Capture and verify actual identity using the returned identity namespace.
            ident = identity.capture(core, dict(parent_zero_step_guard=True, adhesion=False))
            checks['identity_verify'] = bool(identity.verify(ident))
            checks['identity_runtime_verify'] = bool(identity.verify_runtime(ident,core))
            r.verify()
            checks['frozen_sources_after'] = True
            checks['exact_one_base_Core_constructor'] = counters['base_Core_constructors'] == 1
            checks['zero_integration_and_no_payload_or_NN'] = (
                counters['steps'] == 0 and counters['forbidden_step_attempts'] == 0
                and counters['neural_attempts'] == 0 and counters['payload_attempts'] == 0)
            checks['exact_one_flat_Core_zero_old_Core'] = counters['Core'] == 1 and counters['old_Core'] == 0
            details.update(case=r.CASE, source_xml=str(old_xml), source_xml_sha256=r.sha(old_xml),
                           flat_xml=str(new_xml), flat_xml_sha256=r.sha(new_xml),
                           identity_sha256=ident['identity_sha256'])
    except Exception as exc:
        error = type(exc).__name__+': '+str(exc)
        checks['guard_completed_without_error'] = False
    finally:
        active[0] = False
    if r is None:
        raise RuntimeError(error or 'Could not load local runtime')
    report = dict(kind='PARENT_FLAT_MATCHED_ZERO_STEP_GUARD_V1', checks=checks,
                  all_passed=bool(checks) and all(checks.values()) and error is None,
                  **counters, Core_constructions=counters['Core'],
                  physics_steps=counters['steps'], freeze_sha256=r.sha(HERE/'FROZEN.json'),
                  details=details, error=error,
                  limitations=[
                      'Compiled model parity and constructor-only evidence; no trajectory or behavior acceptance.',
                      'No neural forwards, checkpoint loads, saved array payload loads, old Core or integration.',
                      'Tripwires cover named Python entrypoints and payload file suffixes; not a security sandbox.',
                      'Constructor mj_forward/kinematics is permitted; native integration is prohibited.',
                      'Compiler storage offsets/BVH layouts are not claimed physical parameters.',
                  ])
    r.save(HERE/'PARENT_ZERO_STEP_GUARD.json', report)
    if not report['all_passed']:
        raise RuntimeError('Parent zero-step guard FAILED: '+str(error or [k for k,v in checks.items() if not v]))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-resource-released', action='store_true')
    args = parser.parse_args()
    if not args.parent_resource_released:
        parser.error('Explicit --parent-resource-released is required before scientific imports or model construction')
    run()


if __name__ == '__main__':
    main()
