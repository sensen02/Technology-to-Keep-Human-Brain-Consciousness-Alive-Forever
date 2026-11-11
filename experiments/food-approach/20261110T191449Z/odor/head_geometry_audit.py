"""Read-only baseline geometry audit; no mj_step, tick, reset or state writes.

Creates HEAD_GEOMETRY_AUDIT.json and an optional clearly labelled illustration.
Geometry derived from compiled mesh vertices and current geom transforms.
"""
from pathlib import Path
import hashlib
import json
import sys
import math
import numpy as np

HERE = Path(__file__).resolve().parent
BASELINE = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
sys.path.insert(0, str(BASELINE))
import physics
import mujoco as mj
from food_field import bilateral_positions, FoodSource, BilateralOdorSensory


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bounds(points):
    return dict(min=points.min(axis=0).tolist(), max=points.max(axis=0).tolist(),
                extent=np.ptp(points, axis=0).tolist())


def capture(core):
    arrays = {name: getattr(core.d, name).copy() for name in
              ('qpos', 'qvel', 'qacc', 'act', 'ctrl', 'xpos', 'xmat', 'geom_xpos',
               'geom_xmat', 'xfrc_applied', 'qfrc_applied')}
    arrays['actuator_target_rad'] = core.actuator.target_rad.copy()
    arrays['command_history'] = core.buf.copy()
    return float(core.d.time), int(core.t), arrays


def same(before, after):
    return before[0:2] == after[0:2] and all(np.array_equal(a, after[2][key]) for key, a in before[2].items())


def illustrate(audit, polygons, head_position, head_rotation, offsets, xy, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Circle

    length = audit['body_model_length_excluding_appendages']
    heading = np.asarray(audit['thorax_horizontal_heading_xy'])
    left = np.array([-heading[1], heading[0]])
    center = head_position[:2] + .6 * length * heading + .10 * length * left
    field = FoodSource(tuple(center), patch_radius=.12 * length, amplitude=1,
                       sigma=.40 * length, onset_s=1, rise_s=.2)
    audit['illustration_source'] = dict(center_xy=center.tolist(),
        patch_radius=field.patch_radius, amplitude=field.amplitude,
        sigma=field.sigma, onset_s=field.onset_s, rise_s=field.rise_s,
        relation='head origin +0.6 trunk model lengths forward +0.10 lengths left',
        diagnostic_only=True)
    mesh_xy = np.concatenate([p.reshape(-1, 2) for p in polygons])
    low = np.minimum(mesh_xy.min(axis=0), center - .5 * length)
    high = np.maximum(mesh_xy.max(axis=0), center + .5 * length)
    axis_x = np.linspace(low[0] - .1 * length, high[0] + .1 * length, 360)
    axis_y = np.linspace(low[1] - .1 * length, high[1] + .1 * length, 300)
    xx, yy = np.meshgrid(axis_x, axis_y)
    cc = np.exp(-((xx-center[0])**2 + (yy-center[1])**2)/(2*field.sigma**2))
    fig, (ax, response) = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={'width_ratios': [1.25, 1]})
    image = ax.contourf(xx, yy, cc, levels=np.linspace(0, 1, 21), cmap='YlOrRd', alpha=.65)
    # Every mesh appears in this static top-down projection; no pose animation.
    ax.add_collection(PolyCollection(np.concatenate(polygons), facecolor='#283746', edgecolor='none', alpha=.32, zorder=3))
    ax.add_patch(Circle(center, field.patch_radius, fill=False, edgecolor='#a52020', linewidth=1.8))
    ax.scatter(*center, marker='*', s=170, c='#8b1a1a', zorder=5, label='Illustrative food source')
    for i, (color, label) in enumerate((('#1579c4', 'Left engineering sensor'), ('#1b8a56', 'Right engineering sensor'))):
        ax.plot([head_position[0], xy[i, 0]], [head_position[1], xy[i, 1]], color=color, linewidth=1.4, zorder=6)
        ax.scatter(*xy[i], s=55, color=color, edgecolor='white', zorder=7, label=label)
    ax.arrow(head_position[0], head_position[1], .25*length*heading[0], .25*length*heading[1],
             width=.004*length, color='#192939', length_includes_head=True, zorder=5)
    ax.set(xlim=(axis_x[0], axis_x[-1]), ylim=(axis_y[0], axis_y[-1]), aspect='equal',
           xlabel='World x (native model units)', ylabel='World y (native model units)',
           title='Actual initial geometry: top view, all 71 meshes')
    ax.legend(loc='lower left', fontsize=8, framealpha=.95)
    fig.colorbar(image, ax=ax, label='Gaussian concentration / amplitude', fraction=.045, pad=.03)
    sensory = BilateralOdorSensory(field)
    starts = np.arange(0, 2, .005)
    readings = np.asarray([sensory.sample(xy, float(t), .005) for t in starts])
    for i, color in enumerate(('#1579c4', '#1b8a56')):
        response.plot(starts + .005, readings[:, i], color=color, label=('Left receptor', 'Right receptor')[i], linewidth=2)
    response.axvline(1, color='gray', linestyle='--', linewidth=1, label='Source onset')
    response.axvspan(1, 1.2, color='gray', alpha=.10, label='0.2 s smooth rise')
    response.set(xlabel='Sensory model time (s), head held fixed', ylabel='Bounded receptor output',
                 xlim=(0, 2), ylim=(0, .6), title='Mechanistic response only: 5 ms / tau = 50 ms')
    response.grid(alpha=.2)
    response.legend(loc='upper left', fontsize=8)
    fig.suptitle('ENGINEERING MODEL ILLUSTRATION — NO PHYSICAL APPROACH BEHAVIOR ESTABLISHED', fontsize=12, weight='bold')
    fig.text(.5, .02, f'Actual audited baseline Head pose | forward offset {offsets[0]:.4g}, lateral offset {offsets[1]:.4g} native units | offsets not anatomical; SI calibration NOT_ESTABLISHED', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    audit['illustration'] = dict(path=str(destination), sha256=sha(destination),
        explanation='static audited initial pose; model sensory-time trace only; no simulation steps or behavior claim')


def main():
    assert Path(physics.__file__).resolve() == BASELINE/'physics.py'
    core = physics.Core(adhesion=False)
    model, data = core.plant, core.d
    before = capture(core)
    head_candidates = [dict(id=i, name=model.body(i).name) for i in range(model.nbody)
                       if 'head' in (model.body(i).name or '').lower()]
    assert head_candidates == [dict(id=model.body('Head').id, name='Head')]
    head_id = model.body('Head').id
    head_position = data.xpos[head_id].copy()
    head_rotation = data.xmat[head_id].reshape(3, 3).copy()
    thorax_rotation = data.xmat[core.th].reshape(3, 3).copy()
    heading = thorax_rotation[:2, 0].copy()
    heading /= np.linalg.norm(heading)
    thorax_left = np.array([-heading[1], heading[0]])
    mesh_records, polygons = [], []
    head_world, head_local, trunk_world = [], [], []
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] != mj.mjtGeom.mjGEOM_MESH:
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        start, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        vertices = model.mesh_vert[start:start+count].astype(np.float64)
        world = vertices @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
        face_start, face_count = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
        faces = model.mesh_face[face_start:face_start+face_count]
        polygons.append(world[faces, :2])
        body_id = int(model.geom_bodyid[geom_id])
        body_name = model.body(body_id).name
        record = dict(geom_id=geom_id, geom_name=model.geom(geom_id).name,
                      body_id=body_id, body_name=body_name, mesh_id=mesh_id,
                      mesh_name=model.mesh(mesh_id).name, vertex_count=count,
                      world_aabb=bounds(world))
        mesh_records.append(record)
        if body_id == head_id:
            head_world.append(world)
            head_local.append((world-head_position) @ head_rotation)
        if body_name == 'Thorax' or body_name == 'Head' or body_name.startswith('Abdomen'):
            trunk_world.append(world)
    assert len(mesh_records) == 71 and len(head_world) > 0 and trunk_world
    head_local = np.concatenate(head_local)
    head_world = np.concatenate(head_world)
    all_world = np.concatenate([p.reshape(-1, 2) for p in polygons])
    local_min, local_max = head_local.min(axis=0), head_local.max(axis=0)
    extent = np.ptp(head_local, axis=0)
    # Both sensors lie ahead of the exact head local x maximum, and lateral
    # offsets exceed both absolute head local-y bounds. These are deliberate
    # engineering proxy locations, no anatomical antenna claim.
    forward = float(local_max[0] + .10 * extent[0])
    lateral = float(max(abs(local_min[1]), abs(local_max[1])) + .10 * extent[1])
    sensor_xy = bilateral_positions(head_position, head_rotation, forward, lateral)
    sensor_local = np.array([[forward, lateral, 0], [forward, -lateral, 0]])
    sensor_world = sensor_local @ head_rotation.T + head_position
    nearest_vertices = [float(np.linalg.norm(head_world-point, axis=1).min()) for point in sensor_world]
    heading_coords = np.concatenate(trunk_world)[:, :2] @ heading
    trunk_length = float(np.ptp(heading_coords))
    assert forward > local_max[0] and lateral > max(abs(local_min[1]), abs(local_max[1]))
    assert np.dot((sensor_xy[0]-sensor_xy[1]), thorax_left) > 0
    audit = dict(status='PASS', scope='S4 read-only actual baseline geometry; engineering sensor proxies only',
        baseline=str(BASELINE), imported_physics=str(Path(physics.__file__).resolve()),
        command=[sys.executable, '-B', str(Path(__file__).resolve())],
        native_timestep_s=float(model.opt.timestep), adhesion_enabled=core.adhesion_on,
        head_candidates=head_candidates, head=dict(body_id=head_id, name='Head',
        xpos=head_position.tolist(), xmat=head_rotation.tolist(),
        local_mesh_aabb=bounds(head_local), world_mesh_aabb=bounds(head_world)),
        thorax=dict(body_id=core.th, name=model.body(core.th).name,
        xpos=data.xpos[core.th].tolist(), xmat=thorax_rotation.tolist()),
        thorax_horizontal_heading_xy=heading.tolist(),
        frame=dict(local_x_forward_heading_dot=float(head_rotation[:2, 0] @ heading),
                   local_y_thorax_left_dot=float(head_rotation[:2, 1] @ thorax_left),
                   local_z_world_up_dot=float(head_rotation[2, 2]),
                   head_to_thorax_rotation=(thorax_rotation.T@head_rotation).tolist(),
                   conclusion='Actual initial Head local +x aligns thorax horizontal forward; +y is left; +z up. Always transform live head xmat before XY projection.'),
        sensor_offsets=dict(forward_offset=forward, lateral_offset=lateral,
            selection='head local max x + 0.10*x extent; max(abs(local y bounds)) +0.10*y extent; local z=0',
            anatomical_calibration='NOT_ESTABLISHED', local_xyz=sensor_local.tolist(),
            world_xyz=sensor_world.tolist(), world_xy=sensor_xy.tolist()),
        gaps=dict(forward_clearance_from_head_local_aabb=forward-float(local_max[0]),
            lateral_clearance_from_head_local_aabb=[lateral-float(local_max[1]), -float(local_min[1])+lateral if False else -float(local_min[1]+lateral)*-1],
            sensor_separation_world=float(np.linalg.norm(sensor_world[0]-sensor_world[1])),
            sensor_to_nearest_head_mesh_vertex=nearest_vertices,
            nearest_vertex_caveat='Vertex distance is an upper bound on nearest surface distance, not exact mesh-surface clearance; positive AABB-axis margins prove external engineering points.'),
        mesh_count=int(model.nmesh), visible_mesh_geometry_count=len(mesh_records),
        model_size=dict(nq=int(model.nq), nv=int(model.nv), nu=int(model.nu),
                        mesh_world_xy_aabb=bounds(all_world)),
        body_model_length_excluding_appendages=trunk_length,
        model_length_definition='Projection extent along initial thorax horizontal forward of Head, Thorax and named Abdomen mesh vertices; legs/wings excluded',
        mesh_geometry=mesh_records,
        time_before_s=before[0], core_tick_before=before[1],
        source_hashes={str(p):sha(p) for p in (Path(__file__).resolve(), HERE/'food_field.py', BASELINE/'physics.py', BASELINE/'sources/repair_core.py', physics.PAD_XML, physics.POSE_FILE)},
        limitations=['No physical integration steps after Core construction; constructor performs its standard initial reset/mj_forward.',
                     'No evidence for physical food approach, anatomical receptor position, original connectome or independent SI calibration.'])
    png = HERE/'FIELD_RECEPTOR_ILLUSTRATION.png'
    illustrate(audit, polygons, head_position, head_rotation, (forward,lateral), sensor_xy, png)
    after = capture(core)
    audit.update(no_state_or_time_change=same(before, after), time_after_s=after[0],
        core_tick_after=after[1], physics_steps_executed_by_audit=0,
        state_arrays_exact_unchanged={key:bool(np.array_equal(value, after[2][key])) for key,value in before[2].items()},
        state_before_sha256={key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in before[2].items()},
        state_after_sha256={key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in after[2].items()})
    assert audit['no_state_or_time_change'] and before[0] == 0 and not core.adhesion_on
    output = HERE/'HEAD_GEOMETRY_AUDIT.json'
    output.write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=audit['status'], path=str(output), head_id=head_id,
        forward_offset=forward, lateral_offset=lateral, trunk_length=trunk_length,
        state_unchanged=audit['no_state_or_time_change'], meshes=len(mesh_records),
        illustration=str(png)), indent=2))


if __name__ == '__main__':
    main()
