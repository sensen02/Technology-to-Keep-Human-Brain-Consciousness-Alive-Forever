"""Same normal force reduction as Body.contact, with one native call per step."""
import ctypes
import os
from pathlib import Path

import mujoco
import numpy as np

LIBRARY = Path(__file__).with_name('foot_load.so')
force = ctypes.CDLL(str(LIBRARY)).per_foot_normal_load
force.argtypes = [ctypes.c_void_p] * 5
force.restype = None


class FootLoad:
    def __init__(self, body):
        self.body = body
        self.ground = np.zeros(body.m.ngeom, dtype=np.intc)
        self.ground[list(body.ground_geoms)] = 1
        self.leg = np.full(body.m.ngeom, -1, dtype=np.intc)
        for g, leg in body.geom_leg.items():
            self.leg[g] = leg
        self.args = (body.m._address, body.d._address, self.ground.ctypes.data, self.leg.ctypes.data)
        self.verify = os.environ.get('VERIFY_FOOT_LOAD') == '1'

    def record(self, out):
        force(*self.args, out.ctypes.data)
        if self.verify:
            np.testing.assert_array_equal(out, self.body.contact()[0])
