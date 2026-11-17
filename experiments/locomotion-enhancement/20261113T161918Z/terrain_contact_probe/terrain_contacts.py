"""Registered-terrain contact measurement only; no controller, adhesion or state writes.

Caller supplies MuJoCo/NumPy modules; importing this utility imports stdlib only.
Forces act on the non-terrain geom. Normal points terrain -> body. The caller
must pass the velocity at the same pre-integration geometry as the contact solve.
This utility does not change any frozen Core or historical acceptance metric.
"""
import math


def dot(a, b):
    return sum(float(x) * float(y) for x, y in zip(a, b))


def tangent(vector, unit_normal):
    normal_component = dot(vector, unit_normal)
    return [float(v) - normal_component * float(n) for v, n in zip(vector, unit_normal)]


def norm(vector):
    return math.sqrt(dot(vector, vector))


def collect(mj, np, model, data, terrain_ids, foot_ids, *, qvel_before):
    """Read every registered-terrain contact, including nonfoot contacts.

    Reject ambiguous terrain-terrain pairs. Terrain can have any geom type.
    Local force is MuJoCo's contact force on geom2. Rotation uses frame.T;
    reversing terrain/foot geom order reverses world force and normal together.
    Relative point velocity subtracts terrain velocity and retains all 3 axes.
    """
    terrain_ids, foot_ids = set(terrain_ids), set(foot_ids)
    if terrain_ids & foot_ids:
        raise ValueError('Terrain and foot geom sets must be disjoint')
    rows = []
    for ci in range(data.ncon):
        contact = data.contact[ci]
        ga, gb = int(contact.geom1), int(contact.geom2)
        a_terrain, b_terrain = ga in terrain_ids, gb in terrain_ids
        if not (a_terrain or b_terrain):
            continue
        if a_terrain and b_terrain:
            raise ValueError('Ambiguous terrain-terrain contact')
        tid, gid, sign = (ga, gb, 1.) if a_terrain else (gb, ga, -1.)
        local_force = np.zeros(6)
        mj.mj_contactForce(model, data, ci, local_force)
        frame = np.asarray(contact.frame).reshape(3, 3)
        force = sign * (frame.T @ local_force[:3])
        normal = sign * frame[0]
        jac_body, jac_terrain = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        position = np.asarray(contact.pos).copy()
        mj.mj_jac(model, data, jac_body, None, position, int(model.geom_bodyid[gid]))
        mj.mj_jac(model, data, jac_terrain, None, position, int(model.geom_bodyid[tid]))
        velocity = (jac_body - jac_terrain) @ qvel_before
        tangent_velocity = tangent(velocity, normal)
        fn = dot(force, normal)
        rows.append(dict(
            contact_index=ci, geom1=ga, geom2=gb, terrain_id=tid, body_geom_id=gid,
            terrain_name=model.geom(tid).name, body_geom_name=model.geom(gid).name,
            is_foot=gid in foot_ids, legacy_plane_recognized=bool(
                model.geom_type[ga] == mj.mjtGeom.mjGEOM_PLANE or
                model.geom_type[gb] == mj.mjtGeom.mjGEOM_PLANE),
            point_world=position.tolist(), frame=frame.tolist(), distance=float(contact.dist),
            normal_terrain_to_body=normal.tolist(), local_wrench_on_geom2=local_force.tolist(),
            force_world_on_body=force.tolist(), normal_force=fn,
            compressive_normal_force=max(0., fn), tangent_force=tangent(force, normal),
            relative_point_velocity_world=velocity.tolist(), tangent_velocity=tangent_velocity,
            tangent_speed=norm(tangent_velocity), legacy_xy_speed=norm(velocity[:2]),
            frame_normal_length_error=abs(norm(normal) - 1.),
            force_transform_error=abs(fn - float(local_force[0])),
            tangent_normal_dot=dot(tangent_velocity, normal),
            legacy_world_z_load=max(0., float(force[2])),
        ))
    return rows
