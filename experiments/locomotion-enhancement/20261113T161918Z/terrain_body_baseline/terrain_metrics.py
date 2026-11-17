"""Versioned terrain metrics, stdlib-only until parent explicitly constructs Core.

World-z support and normal support are distinct. Original XY/world-z formulas
are retained as legacy diagnostics; no new rough-terrain behavioral PASS gate.
"""
import importlib.util
import sys
from prepare import VALIDATED_UTILITY, VALIDATED_UTILITY_SHA, sha

VERSION = 'REGISTERED_TERRAIN_NORMAL_TANGENT_V1'


def contact_module():
    if sha(VALIDATED_UTILITY) != VALIDATED_UTILITY_SHA:
        raise ValueError('Parent-validated contact utility drift')
    name = '_parent_validated_terrain_contact_measurement'
    spec = importlib.util.spec_from_file_location(name, VALIDATED_UTILITY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def contacts(core, qvel=None):
    np, mj = core._terrain_np, core._terrain_mj
    rows = core._terrain_contact_module.collect(mj, np, core.plant, core.d,
        core.terrain_ids, core.foot_ids,
        qvel_before=core.d.qvel if qvel is None else qvel)
    normal_load = np.zeros(6)
    signed_z = np.zeros(6)
    positive_z = np.zeros(6)
    tangent_weighted = np.zeros(6)
    old_plane_z = np.zeros(6)
    old_plane_xy_weighted = np.zeros(6)
    nf_z, nf_normal, nf_signed_z = 0., 0., 0.
    old_nf = 0.
    for row in rows:
        li = int(core.geom_leg[row['body_geom_id']])
        zforce = float(row['force_world_on_body'][2])
        fn = row['compressive_normal_force']
        if li < 0:
            nf_z += max(0., zforce)
            nf_normal += fn
            nf_signed_z += zforce
            if row['legacy_plane_recognized']:
                old_nf += max(0., zforce)
            continue
        normal_load[li] += fn
        signed_z[li] += zforce
        positive_z[li] += max(0., zforce)
        tangent_weighted[li] += fn * row['tangent_speed']
        if row['legacy_plane_recognized']:
            old_plane_z[li] += max(0., zforce)
            old_plane_xy_weighted[li] += max(0., zforce) * row['legacy_xy_speed']
    return dict(version=VERSION, contacts=rows, normal_loads=normal_load,
        world_z_force_signed=signed_z, world_z_loads_positive=positive_z,
        tangent_speed=tangent_weighted / np.maximum(normal_load, 1e-12),
        legacy_plane_loads=old_plane_z,
        legacy_plane_xy_speed=old_plane_xy_weighted / np.maximum(old_plane_z, 1e-12),
        nonfoot_world_z_positive=nf_z, nonfoot_normal=nf_normal,
        nonfoot_world_z_signed=nf_signed_z, legacy_plane_nonfoot_world_z_positive=old_nf,
        old_plane_contact_count=sum(row['legacy_plane_recognized'] for row in rows),
        registered_contact_count=len(rows))


def height(core, xy):
    """Vertical ground height at XY using ONLY static registered terrain group5.

    This is vertical clearance, not shortest signed distance near side walls.
    Plane/huge box control top is analytically zero; snap only those cases to
    zero after verifying ray agreement, preserving original plane observations.
    """
    np, mj = core._terrain_np, core._terrain_mj
    start = np.array([float(xy[0]), float(xy[1]), core.ray_origin_z], dtype=float)
    downward = np.array([0., 0., -1.])
    geomid, normal = np.array([-1], dtype=np.int32), np.zeros(3)
    distance = float(mj.mj_ray(core.plant, core.d, start, downward,
        core.ray_group_mask, True, -1, geomid, normal))
    if distance < 0 or int(geomid[0]) not in core.terrain_ids:
        raise ValueError('No registered terrain ray hit; clearance UNKNOWN')
    ground = core.ray_origin_z - distance
    if core.terrain_case in ('plane_control', 'flat_box_control'):
        if abs(ground) > 1e-10:
            raise ValueError('Flat control terrain ray height is not zero')
        ground = 0.
    return ground, int(geomid[0])


class TerrainPeakRecorder:
    """Every native solve, no full native state arrays; before-step velocity pairing.

    Inherits old peak summary key meanings (positive WORLD-Z nonfoot force).
    Extra normal/tangent statistics and legacy-plane counts are explicitly V1.
    This is a measurement recorder, not a control or acceptance implementation.
    """
    def __init__(self):
        self.count = self.post_count = 0
        self.peak = self.post_peak = self.normal_peak = 0.
        self.extra_max = 0.
        self.registered_count = self.old_count = 0
        self.nonfoot_count = 0
        self.com_min = float('inf')
        self.rows_by_terrain = {}

    def before(self, core):
        np = core._terrain_np
        extra = max(float(np.abs(core.d.xfrc_applied).max()), float(np.abs(core.d.qfrc_applied).max()))
        self.extra_max = max(self.extra_max, extra)
        if core.adhesion_on or extra != 0.:
            raise ValueError('Terrain baseline forbids adhesion and external forces')
        return float(core.d.time), core.d.qvel.copy()

    def solved(self, core, before):
        return contacts(core, before[1])

    def after(self, core, before, solved):
        self.count += 1
        self.peak = max(self.peak, solved['nonfoot_world_z_positive'])
        self.normal_peak = max(self.normal_peak, solved['nonfoot_normal'])
        self.registered_count += solved['registered_contact_count']
        self.old_count += solved['old_plane_contact_count']
        for row in solved['contacts']:
            name = row['terrain_name']
            self.rows_by_terrain[name] = self.rows_by_terrain.get(name, 0) + 1
            self.nonfoot_count += not row['is_foot']
        if before[0] >= 5.8 - 1e-8:
            self.post_count += 1
            self.post_peak = max(self.post_peak, solved['nonfoot_world_z_positive'])
            self.com_min = min(self.com_min, float(core.d.subtree_com[core.th][2]))

    def close(self):
        return dict(actual_native_steps=self.count, post_native_steps=self.post_count,
            nonfoot_peak_native=self.peak, post_nonfoot_peak_native=self.post_peak,
            post_COM_min=None if self.com_min == float('inf') else self.com_min,
            extra_force_max=self.extra_max, full_arrays_retained=False,
            metric_version=VERSION, nonfoot_normal_peak_native=self.normal_peak,
            registered_contact_count=self.registered_count, legacy_plane_contact_count=self.old_count,
            nonfoot_contact_count=self.nonfoot_count, native_contacts_by_terrain=self.rows_by_terrain,
            scope='Every native registered-terrain solved NF; no rough-locomotion acceptance')
