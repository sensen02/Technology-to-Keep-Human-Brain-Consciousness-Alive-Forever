"""Render and publish head-mounted eye images without changing the physics."""
import json
from pathlib import Path
import sys

import mujoco as mj
import numpy as np

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent / "20261112T045316Z"
sys.path.insert(0, str(ARCHIVE))
from camera import add_decor, eye_frames, render_eye_pair

CONFIG = json.loads((ARCHIVE / "PREREG_CAMERA_PROBE.json").read_text())
GEOMETRY = json.loads(Path(CONFIG["head_geometry_audit"]).read_text())


class Eyes:
    def __init__(self, body):
        self.body = body
        self.head = body.plant.body("Head").id
        self.renderer = mj.Renderer(body.plant, height=32, width=32)
        self.pending = None
        self.released = None

    def pose(self):
        return self.body.d.xpos[self.head], self.body.d.xmat[self.head].reshape(3, 3)

    def place(self, angle, distance):
        pos, rot = self.pose()
        heading = np.arctan2(rot[1, 0], rot[0, 0]) + np.deg2rad(angle)
        target = pos + distance * np.array([np.cos(heading), np.sin(heading), 0.])
        target[2] = 1.25
        return target

    def bearing(self, target):
        pos, rot = self.pose()
        dx, dy = (target - pos)[:2]
        heading = np.arctan2(rot[1, 0], rot[0, 0])
        angle = np.arctan2(dy, dx) - heading
        return float(np.rad2deg(np.arctan2(np.sin(angle), np.cos(angle))))

    def image(self, target, width, color, background, cover=None):
        pos, rot = self.pose()
        frames = eye_frames(pos, rot, GEOMETRY, CONFIG["eyes"])
        plant, data = self.body.plant, self.body.d

        def decorate(scene):
            if background is not None:
                origin, shift, level = background
                for offset in (-3.1, -1.9, -.5, .9, 2.2, 3.4):
                    p = [origin[0]+7., origin[1]+offset+shift, 1.3]
                    add_decor(scene, mj, mj.mjtGeom.mjGEOM_BOX, [.045, .15, 1.3], p,
                              np.eye(3), [level, level, level, 1.])
            if target is not None:
                add_decor(scene, mj, mj.mjtGeom.mjGEOM_CYLINDER, [width, 1.09, 0.], target,
                          np.eye(3), color)
            if cover is not None:
                center, rotation = cover
                add_decor(scene, mj, mj.mjtGeom.mjGEOM_BOX, [.04, .70, 1.4], center,
                          rotation, [.19, .19, .19, 1.])
        before = (data.qpos.copy(), data.qvel.copy(), float(data.time))
        image, _ = render_eye_pair(self.renderer, plant, data, frames, CONFIG["eyes"], decorate)
        assert np.array_equal(data.qpos, before[0]) and np.array_equal(data.qvel, before[1])
        assert data.time == before[2]
        return image

    def capture(self, tick, target, width, color, background, cover=None):
        if self.pending is not None:
            assert tick == self.pending[0] + 20
            self.released = self.pending
        self.pending = (tick, self.image(target, width, color, background, cover))
        return self.released

    def close(self):
        self.renderer.close()
