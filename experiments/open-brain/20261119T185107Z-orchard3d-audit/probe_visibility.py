"""Audit the Blender orchard's actual compound-eye samples around one fallen apple."""
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector
import numpy as np


source, output = map(Path, sys.argv[sys.argv.index('--') + 1:])
sys.dont_write_bytecode = True
sys.path.insert(0, str(source))
from orchard import BODY_RADIUS, OrchardEnv, Pose


def inside_apple(origin, props):
    """Exclude starts inside closed apple meshes as well as close to a surface."""
    for obj, world_to_local, low, high in props:
        point = world_to_local @ origin
        if any(value <= lower or value >= upper for value, lower, upper in zip(point, low, high)):
            continue
        direction = Vector((1., .127, .311)).normalized()
        crossings = 0
        while True:
            hit, position, _, _ = obj.ray_cast(point, direction)
            if not hit:
                break
            crossings += 1
            point = position + direction * 1e-6
        if crossings % 2:
            return obj
    return None


def run():
    env = OrchardEnv(lighting='sun', wind=0.65)
    scene = env.orchard
    soil, = [obj for obj in scene.object_ids if obj['orchard_class'] == 'soil']
    target, = [obj for obj in scene.object_ids if obj.get('fruit_id') == 'ground-001']
    target_id = scene.object_ids[target]
    apples = [obj for obj in scene.object_ids if obj['orchard_class'] == 'apple']
    apple_ids = [scene.object_ids[obj] for obj in apples]
    apple_shapes = []
    for obj in apples:
        bounds = np.asarray(obj.bound_box)
        apple_shapes.append((obj, obj.matrix_world.inverted(), bounds.min(axis=0), bounds.max(axis=0)))
    world_to_soil = soil.matrix_world.inverted()
    down = world_to_soil.to_3x3() @ Vector((0., 0., -1.))
    center = sum((target.matrix_world @ Vector(p) for p in target.bound_box), Vector()) / 8
    rows = []
    for radius_mm in (100, 250, 500):
        for angle_deg in range(0, 360, 15):
            angle = math.radians(angle_deg)
            x = center.x + radius_mm / 1000 * math.cos(angle)
            y = center.y + radius_mm / 1000 * math.sin(angle)
            hit, ground, _, _ = soil.ray_cast(world_to_soil @ Vector((x, y, 1.)), down)
            assert hit
            ground_z = (soil.matrix_world @ ground).z
            yaw = math.atan2(center.y - y, center.x - x)
            for height_mm in (2, 40):
                origin = Vector((x, y, ground_z + height_mm / 1000))
                clearance, obstruction = scene.nearest(origin, BODY_RADIUS)
                row = dict(radius_mm=radius_mm, angle_deg=angle_deg, height_mm=height_mm,
                           position_m=tuple(origin), ground_m=ground_z, yaw_rad=yaw)
                if obstruction:
                    row.update(start_obstruction=scene.object_labels[obstruction],
                               clearance_mm=1000 * clearance)
                else:
                    embedded = inside_apple(origin, apple_shapes)
                    if embedded is not None:
                        row.update(start_obstruction=scene.object_labels[scene.object_ids[embedded]],
                                   inside_apple=True)
                    else:
                        ids = env.observe(Pose(tuple(origin), yaw))['object_id']
                        ray = scene.ray(origin, (center - origin).normalized())
                        first_id = int(ray[4].ids[ray[2]]) if ray else 0
                        row.update(target_receptors=np.sum(ids == target_id, axis=1).tolist(),
                                   any_apple_receptors=np.sum(np.isin(ids, apple_ids), axis=1).tolist(),
                                   center_ray_first_hit=scene.object_labels.get(first_id))
                rows.append(row)
    summary = []
    for radius_mm in (100, 250, 500):
        for height_mm in (2, 40):
            group = [r for r in rows if r['height_mm'] == height_mm and r['radius_mm'] == radius_mm]
            free = [r for r in group if 'target_receptors' in r]
            summary.append(dict(radius_mm=radius_mm, height_mm=height_mm, probes=len(group),
                                free=len(free), target_hit=sum(sum(r['target_receptors']) > 0 for r in free),
                                no_apple_hit=sum(sum(r['any_apple_receptors']) == 0 for r in free),
                                target_in_both_eyes=sum(min(r['target_receptors']) > 0 for r in free)))
    print(json.dumps(summary))
    output.write_text(json.dumps(dict(source=str(source), blender=bpy.app.version_string,
                                      scene=bpy.data.filepath, target_name=target.name, target_id=target_id,
                                      target_center_m=list(center), light=env.lighting, wind=env.wind,
                                      summary=summary, rows=rows), indent=2) + '\n')


if __name__ == '__main__':
    run()
