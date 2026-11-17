"""Parent-only tiny single-sphere fixture. Import is stdlib only; no full fly/NN.

Run explicitly in the existing physical environment with --run-parent-authorized.
No worker has run this numerical probe. All changes are confined to a fresh out.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math

import terrain_contacts as measure

HERE = Path(__file__).resolve().parent
CASES = ('plane', 'box', 'tilted_box_10deg', 'no_contact')
DT = 0.0000125
STEPS = 1600  # 20 ms, fixed; never adapt duration to obtain a pass.


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def exclusive_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def xml_for(case):
    if case not in CASES:
        raise ValueError(case)
    tilted = case == 'tilted_box_10deg'
    angle = math.radians(10.) if tilted else 0.
    normal = [math.sin(angle), 0., math.cos(angle)]
    if case in ('plane', 'no_contact'):
        terrain = '<geom name="terrain_plane" type="plane" size="0.1 0.1 0.001"/>'
        surface = [0., 0., 0.]
    else:
        quat = f'{math.cos(angle / 2.):.17g} 0 {math.sin(angle / 2.):.17g} 0'
        terrain = (f'<geom name="terrain_{case}" type="box" pos="0 0 0.002" '
                   f'size="0.1 0.1 0.002" quat="{quat}"/>')
        surface = [0.002 * normal[0], 0., 0.002 + 0.002 * normal[2]]
    separation = 0.058 if case == 'no_contact' else 0.008 - 0.00005
    center = [s + separation * n for s, n in zip(surface, normal)]
    pos = ' '.join(f'{v:.17g}' for v in center)
    xml = f'''<mujoco model="registered_terrain_{case}">
 <compiler angle="radian"/>
 <option timestep="0.0000125" gravity="0 0 -9.81" integrator="Euler"/>
 <default><geom condim="3" friction="1.5 0.005 0.0001" margin="0" solref="0.002 1"/></default>
 <worldbody>{terrain}
  <body name="free_footprobe" pos="{pos}"><freejoint name="probe_free"/>
   <geom name="footprobe" type="sphere" size="0.008" mass="0.001"/>
  </body>
 </worldbody>
</mujoco>'''
    return xml, normal


def run_case(mj, np, out, case):
    xml, expected_normal = xml_for(case)
    xmlpath = out / (case + '.xml')
    with xmlpath.open('x') as stream:
        stream.write(xml + '\n')
    model = mj.MjModel.from_xml_string(xml)
    data = mj.MjData(model)
    # qpos/qvel remain the model's initial state; no pose updates during integration.
    if (model.nq, model.nv, model.nu, model.neq) != (7, 6, 0, 0):
        raise ValueError('Requires one free sphere, no actuator or equality')
    if float(model.opt.timestep) != DT:
        raise ValueError('Native timestep mismatch')
    terrain_ids = {i for i in range(model.ngeom) if model.geom(i).name.startswith('terrain_')}
    foot_ids = {model.geom('footprobe').id}
    if len(terrain_ids) != 1:
        raise ValueError('Expected exactly one explicitly registered terrain')
    mj.mj_forward(model, data)
    initial = measure.collect(mj, np, model, data, terrain_ids, foot_ids, qvel_before=data.qvel.copy())
    records = []
    for index in range(STEPS):
        before_time, before_vel = float(data.time), data.qvel.copy()
        if np.any(data.xfrc_applied) or np.any(data.qfrc_applied):
            raise ValueError('Fixture forbids external forces')
        mj.mj_step(model, data)
        # Euler mj_step leaves the contact solve and xpos at the beginning-of-step
        # geometry. qvel_before above is paired to that solve, before integration.
        contacts = measure.collect(mj, np, model, data, terrain_ids, foot_ids, qvel_before=before_vel)
        if np.any(data.warning.number):
            raise RuntimeError('MuJoCo warning in static fixture')
        if np.any(data.xfrc_applied) or np.any(data.qfrc_applied):
            raise ValueError('Unexpected external forces after integration')
        records.append(dict(step=index, time_before_s=before_time, time_after_s=float(data.time),
                            contacts=contacts, foot_normal_force=sum(c['compressive_normal_force'] for c in contacts if c['is_foot']),
                            nonfoot_normal_force=sum(c['compressive_normal_force'] for c in contacts if not c['is_foot'])))
    contacts = [contact for record in records for contact in record['contacts']]
    unit_velocity = [1., 0., 0.]
    projected = measure.tangent(unit_velocity, expected_normal)
    # Pure coordinate diagnostic; this velocity is never imposed on the sphere.
    coordinate = dict(normal=expected_normal, hypothetical_world_velocity=unit_velocity,
                      projected_tangent=projected, tangent_speed=measure.norm(projected),
                      legacy_xy_speed=1., tangent_normal_dot=measure.dot(projected, expected_normal))
    normal_error = max((measure.norm([a-b for a,b in zip(c['normal_terrain_to_body'], expected_normal)]) for c in contacts), default=0.)
    gates = dict(
        initial_contact_expected=bool(initial) == (case != 'no_contact'),
        integrated_contact_expected=bool(contacts) == (case != 'no_contact'),
        old_plane_recognition_expected=all(c['legacy_plane_recognized'] == (case == 'plane') for c in contacts),
        foot_classification_all=all(c['is_foot'] for c in contacts),
        positive_support_if_contact=(sum(c['compressive_normal_force'] for c in contacts) > 0.) if contacts else case == 'no_contact',
        unit_normals=all(c['frame_normal_length_error'] < 1e-10 for c in contacts),
        analytic_surface_normal=normal_error < 1e-8,
        force_frame_conversion=all(c['force_transform_error'] < 1e-10 for c in contacts),
        tangent_projection_orthogonal=all(abs(c['tangent_normal_dot']) < 1e-10 for c in contacts),
        synthetic_coordinate_projection=abs(coordinate['tangent_normal_dot']) < 1e-10,
        tilted_projection_differs=(abs(coordinate['tangent_speed'] - 1.) > .01) if case == 'tilted_box_10deg' else abs(coordinate['tangent_speed']-1.) < 1e-12,
        nonfoot_zero=all(r['nonfoot_normal_force'] == 0. for r in records),
    )
    metadata = {}
    for name in ('geom_type', 'geom_contype', 'geom_conaffinity', 'geom_condim', 'geom_friction', 'body_mass', 'body_inertia'):
        values = np.asarray(getattr(model, name))
        metadata[name] = dict(shape=list(values.shape), dtype=str(values.dtype), sha256=hashlib.sha256(values.tobytes()).hexdigest(), values=values.tolist())
    mj.mj_saveModel(model, str(out / (case + '.mjb')), None)
    result = dict(case=case, geometry_scope='single free sphere engineering fixture; no claw/pulvilli/full-fly validation',
                  native_steps=STEPS, duration_s=float(data.time), dt_s=DT, nq=model.nq, nv=model.nv, nu=model.nu, neq=model.neq,
                  xml_path=str(xmlpath), xml_sha256=sha(xmlpath), compiled_mjb_sha256=sha(out / (case + '.mjb')),
                  compiled_metadata=metadata, registered_terrain_names=[model.geom(i).name for i in sorted(terrain_ids)],
                  initial_contacts=initial, all_native_contact_count=len(contacts),
                  old_plane_only_count=sum(c['legacy_plane_recognized'] for c in contacts),
                  new_registered_terrain_count=len(contacts),
                  legacy_missed_count=sum(not c['legacy_plane_recognized'] for c in contacts),
                  peak_normal_force=max((r['foot_normal_force'] for r in records), default=0.),
                  peak_nonfoot_normal_force=max(r['nonfoot_normal_force'] for r in records),
                  analytic_normal_max_error=normal_error, coordinate_diagnostic=coordinate,
                  extra_force_max=0., adhesion=False, state_writes_after_initialization=0,
                  gates=gates, all_passed=all(gates.values()), native_records=records)
    exclusive_json(out / (case + '.json'), result)
    return {k:v for k,v in result.items() if k not in ('native_records', 'initial_contacts', 'compiled_metadata')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-parent-authorized', action='store_true', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    # Heavy imports only after the parent's explicit run flag.
    import mujoco as mj
    import numpy as np
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = {str(p):sha(p) for p in (Path(__file__).resolve(), HERE/'terrain_contacts.py', HERE/'PREREG.json')}
    cases = []
    try:
        for case in CASES:
            cases.append(run_case(mj, np, out, case))
        if any(sha(p) != h for p,h in sources.items()):
            raise ValueError('Probe source drift during execution')
        summary = dict(kind='REGISTERED_TERRAIN_SINGLE_SPHERE_CONTACT_FIXTURE_V1',
                       mujoco_version=mj.__version__, source_hashes=sources, cases=cases,
                       native_steps=sum(c['native_steps'] for c in cases), all_passed=all(c['all_passed'] for c in cases),
                       no_fullfly_no_nn_no_claw_claim=True)
        exclusive_json(out/'RESULT.json', summary)
        print(json.dumps(summary, allow_nan=False))
        if not summary['all_passed']:
            raise SystemExit(2)
    except Exception as error:
        exclusive_json(out/'FAILURE.json', dict(type=type(error).__name__, message=str(error), source_hashes=sources, completed_cases=cases))
        raise


if __name__ == '__main__':
    main()
