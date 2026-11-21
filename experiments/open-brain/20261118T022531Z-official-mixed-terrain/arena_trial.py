"""Evaluate the same frozen body in FlyGym's installed official world."""
from pathlib import Path
from unittest.mock import patch
import sys

import numpy as np
import mujoco

TRIAL = Path(__file__).resolve().parent
BASE = TRIAL.parent
ORIGINAL = BASE / "20261118T002600Z-ascending-behavior-comparison"
TIMING = BASE / "20261118T020425Z-front-touch-timing-repair"
sys.path[:0] = [str(ORIGINAL), str(TIMING)]
import live
import body_model
from flygym.compose.world import MixedTerrainWorld

SPAWN_X = 3.20
EXIT_X = 11.0
RAISED_ENTER_X = 4.935
GAP_EDGES = ((9.4, 9.7), (10.7, 11.0))
TERRAIN_CLASS_NAMES = ("initial_flat", "raised_block", "other_block",
                       "bridge", "gap_bottom", "exit_flat", "other")


class MixedBody(body_model.Body):
    def __init__(self, seed):
        # Reuse the canonical body constructor and change only the world it attaches to.
        with patch.object(body_model, "Terrain", lambda _height: MixedTerrainWorld()):
            super().__init__(height=0., adhesion=True, seed=seed)
        self.restore(seed)
        self.d.qpos[0] += SPAWN_X
        mujoco.mj_forward(self.m, self.d)
        self.classes = np.full(self.m.ngeom, -1, dtype=int)
        self.tiles = []
        for geom in self.ground_geoms:
            name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, geom)
            pos = self.m.geom_pos[geom]
            size = self.m.geom_size[geom]
            height = pos[2] + size[2]
            if name == "ground_flat0":
                cls = 0
            elif name.startswith("ground_mixed_block1_"):
                cls = 1 if height > -.01 else 2
            elif name in ("ground_gap_pre1", "ground_gap_post1"):
                cls = 3
            elif name.startswith("ground_base"):
                cls = 4
            elif name == "ground_flat1":
                cls = 5
            else:
                cls = 6
            self.classes[geom] = cls
            self.tiles.append(dict(id=geom, name=name, terrain_class=TERRAIN_CLASS_NAMES[cls],
                type=int(self.m.geom_type[geom]), pos=pos.tolist(), half_size=size.tolist()))
        names = {row["name"]: row for row in self.tiles}
        bounds = {}
        for name in ("ground_flat0", "ground_gap_pre1", "ground_gap_post1", "ground_flat1"):
            tile = names[name]
            x, half = tile["pos"][0], tile["half_size"][0]
            bounds[name] = (x - half, x + half)
        assert np.allclose(bounds["ground_flat0"], (2., 5.))
        assert np.allclose([bounds["ground_gap_pre1"][1], bounds["ground_gap_post1"][0]], GAP_EDGES[0])
        assert np.allclose([bounds["ground_gap_post1"][1], bounds["ground_flat1"][0]], GAP_EDGES[1])
        assert abs(bounds["ground_flat1"][0] - EXIT_X) < 1e-9
        self.boundaries = bounds

    def detail(self):
        """Use the same native-step normal load as the archived contact model."""
        load = np.zeros(6)
        ground = np.zeros((len(TERRAIN_CLASS_NAMES), 6))
        bad_force = 0.
        force = np.zeros(6)
        for i, contact in enumerate(self.d.contact[:self.d.ncon]):
            if contact.exclude:
                continue
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if self.classes[g1] >= 0:
                ground_geom, other = g1, g2
            elif self.classes[g2] >= 0:
                ground_geom, other = g2, g1
            else:
                continue
            mujoco.mj_contactForce(self.m, self.d, i, force)
            leg = self.geom_leg.get(other)
            if leg is None:
                if self.m.geom_bodyid[other] in self.axis_bodies:
                    bad_force += float(np.linalg.norm(force[:3]))
                continue
            load[leg] += force[0]
            ground[self.classes[ground_geom], leg] += force[0]
        return load, bad_force, ground

    def initial_receipt(self):
        return dict(qpos=self.d.qpos.tolist(), qvel=self.d.qvel.tolist(),
                    thorax=self.d.xpos[self.thorax].tolist(), stance_feet=self.d.xpos[self.feet].tolist(),
                    boundary=self.boundaries, bodyweight_native=self.weight,
                    ground_geom_count=len(self.tiles), native_pairs=int(self.m.npair),
                    body_nq=int(self.m.nq), body_nv=int(self.m.nv), actuator_count=int(self.m.nu))


def measure(arrays, body):
    row = live.measure(arrays, body)
    position = arrays["position"]
    ground = arrays["terrain_normal"]
    event_epsilon = 1e-6
    raised = (ground[:, 1, :].max(axis=1) > event_epsilon)
    after_exit = position[:, 0] >= EXIT_X - 1e-9
    exit_supported = ground[after_exit, 5, :].max(axis=0) > event_epsilon if after_exit.any() else np.zeros(6, dtype=bool)
    def time_first(flag):
        ticks = np.flatnonzero(flag)
        return float((ticks[0] + 1) * live.DT) if ticks.size else None
    crossed = bool(after_exit.any() and exit_supported[:3].any() and exit_supported[3:].any())
    row.update(first_raised_contact_s=time_first(raised), raised_contact_pass=bool(raised.any()),
               first_gap_start_s=time_first(position[:, 0] >= GAP_EDGES[0][0] - 1e-9),
               first_gap_exit_s=time_first(position[:, 0] >= GAP_EDGES[0][1] - 1e-9),
               first_far_edge_s=time_first(after_exit), bilateral_exit_pass=crossed,
               exit_support_legs=[leg for leg, flag in zip(body.legs, exit_supported, strict=True) if flag],
               deepest_ground_contact_s=time_first(ground[:, 4, :].max(axis=1)>event_epsilon),
               max_x_mm=float(position[:, 0].max()), min_start_relative_height_mm=float(position[:, 2].min()-arrays["initial_position"][2]),
               near_fall_first_s=time_first((arrays["upright"] < .8) | (position[:, 2] < .7*arrays["initial_position"][2])),
               terminal_x_pass=bool(position[-1, 0] >= EXIT_X - 1e-9))
    row["course_pass"] = bool(row["safety_pass"] and row["advance_pass"] and row["final_stand_pass"]
                              and row["raised_contact_pass"] and row["bilateral_exit_pass"]
                              and row["terminal_x_pass"])
    return row
