"""Audit sampled clearance of the physical meshes and all 18 pads."""
import mujoco as mj
import numpy as np


class Envelope:
    def __init__(self, body):
        self.model = body.plant
        self.data = mj.MjData(self.model)
        self.obstacle = self.model.geom("navigation_obstacle")
        self.data.qpos[:] = body.d.qpos
        mj.mj_forward(self.model, self.data)
        self.near = float(self.data.geom_xpos[self.obstacle.id, 0] - self.obstacle.size[0])
        self.far = float(self.data.geom_xpos[self.obstacle.id, 0] + self.obstacle.size[0])
        self.meshes, self.pads = [], []
        for geom in range(self.model.ngeom):
            if self.model.geom_bodyid[geom] in (0, int(self.obstacle.bodyid[0])):
                continue
            if self.model.geom_type[geom] == mj.mjtGeom.mjGEOM_MESH:
                mesh = self.model.geom_dataid[geom]
                start = self.model.mesh_vertadr[mesh]
                end = start + self.model.mesh_vertnum[mesh]
                self.meshes.append((geom, self.model.mesh_vert[start:end]))
            elif self.model.geom_type[geom] == mj.mjtGeom.mjGEOM_SPHERE:
                self.pads.append(geom)
            else:
                raise ValueError(f"Unsupported moving geometry: {self.model.geom(geom).name}")
        assert len(self.meshes) == 71 and len(self.pads) == 18

    def used_bounds(self):
        data, model = self.data, self.model
        ids = self.pads
        r = model.geom_size[ids, 0]
        pad = data.geom_xpos[ids, 0]
        front, rear = pad + r, pad - r
        low, high = float(rear.min()), float(front.max())
        for geom, mesh in self.meshes:
            along_x = mesh @ data.geom_xmat[geom, :3] + data.geom_xpos[geom, 0]
            low = min(low, float(along_x.min()))
            high = max(high, float(along_x.max()))
        return low, high, float(rear.min()), float(front.max())

    def set_pose(self, pose):
        self.data.qpos[:] = pose
        mj.mj_kinematics(self.model, self.data)

    def path(self, qpos, first, last, stride=20):
        rows = []
        for tick in sorted(set(range(first, last + 1, stride)) | {last}):
            self.set_pose(qpos[tick])
            rows.append((tick, *self.used_bounds()))
        return np.asarray(rows)
