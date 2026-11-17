"""Controlled tactile fixture, source-identified receiver and frozen body adapter."""
from collections import defaultdict
import ctypes
from pathlib import Path
import json
import sys

import mujoco as mj
import numpy as np

HERE = Path(__file__).resolve().parent
VISUAL = HERE.parent / "20261116T170000Z"
sys.path.insert(0,str(VISUAL))
from walk_pilot import turn, visual_encoder, WalkHead, Eyes
from bench_runtime import stand_window
from forward_link import ForwardLink, SourceTransport, sha
from control_body import load_body_policy

BINDING = HERE / "TOUCH_BINDING.json"
GRAPH = HERE / "AN_GRAPH.json"
BODY = turn.stand.physics
W = BODY.W
POSITION = np.array([0.,0.,4.])
RADIUS = .038
START = turn.STARTS / "nominal.npz"
TOUCH_W = .0001
NATIVE_PER_MS = 80


def build(output):
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    loader = BODY.act_mod.load_full_model
    original = loader(BODY.PAD_XML)
    spec = _load_mjcf(BODY.PAD_XML)
    actuator = spec.worldbody.add_body(name="controlled_touch_probe",mocap=True,pos=POSITION)
    actuator.add_geom(name="touch_tip",type=mj.mjtGeom.mjGEOM_SPHERE,size=[RADIUS,0.,0.],
                     rgba=[0.,0.,0.,0.],friction=[.25,.005,.0001])
    model = spec.compile()
    for attr in ("jnt_type","jnt_qposadr","jnt_dofadr","jnt_range","actuator_trnid","actuator_gear","dof_damping"):
        np.testing.assert_array_equal(getattr(model,attr),getattr(original,attr))
    for item in range(original.ngeom):
        source = original.geom(item)
        target = model.geom(source.name)
        for attr in ("type","pos","quat","size","contype","conaffinity"):
            np.testing.assert_array_equal(getattr(source,attr),getattr(target,attr))
    model.mat_rgba[:original.nmat] = original.mat_rgba
    BODY.act_mod.load_full_model = lambda *args, **kwargs: model
    try:
        bench = turn.stand.Bench(output)
    finally:
        BODY.act_mod.load_full_model = loader
    body = bench.body
    body.physics_order = BODY.rc.SENS_IDX
    assert all(original.body(i).name == model.body(i).name for i in range(original.nbody))
    assert sha(START) == turn.read(turn.STARTS/"BANK.json")["nominal"]["sha256"]
    with np.load(START,allow_pickle=False) as snapshot:
        offset, fields = 0, []
        for index in range(int(bench.spec).bit_length()):
            field = 1 << index
            if not int(bench.spec) & field:
                continue
            old_size = mj.mj_stateSize(original,field)
            values = np.empty(mj.mj_stateSize(model,field))
            mj.mj_getState(model,body.d,values,field)
            values[:old_size] = snapshot["state"][offset:offset+old_size]
            offset += old_size
            fields.append(values)
        assert offset == len(snapshot["state"])
        bench.state = np.concatenate(fields)
        bench.t = int(snapshot["t"])
        bench.command_ticks = int(snapshot["ticks"])
        bench.center = snapshot["observation"]
        bench.fields = {name: snapshot[f"core_{name}"] for name in bench.fields}
        bench.actuator_fields = {name: snapshot[f"act_{name}"] for name in bench.actuator_fields}
        bench.reset()
        np.testing.assert_array_equal(body.d.qpos,snapshot["qpos"])
        np.testing.assert_array_equal(body.d.qvel,snapshot["qvel"])
    return body


class Fixture:
    def __init__(self, body, side, onset, depth, duration):
        self.body = body
        self.side = side
        self.onset = onset
        self.depth = depth
        self.duration = duration
        self.target = max(body.pads[("LF","RF")[side]],key=lambda i: (1-2*side)*body.d.geom_xpos[i,1])
        self.offset = RADIUS + body.plant.geom_size[self.target,0]

    def move(self, tick):
        age = tick-self.onset
        if age < -25 or age >= self.duration+25:
            self.body.d.mocap_pos[0] = POSITION
            return
        scale = min(1.,(age+25)/25.,(self.duration+25-age)/25.)
        center = self.body.d.geom_xpos[self.target].copy()
        center[1] += (1-2*self.side)*(self.offset + .075*(1.-scale) - self.depth*scale)
        self.body.d.mocap_pos[0] = center


class Recording:
    def __init__(self, body, finish):
        self.count = 0
        self.raw = np.zeros((finish*NATIVE_PER_MS, 6))
        self.probe = body.plant.geom("touch_tip").id
        names = [body.plant.geom(i).name for i in range(body.plant.ngeom)]
        self.leg = np.asarray([0 if s.startswith("LFTarsus") else 1 if s.startswith("RFTarsus") else -1
                               for s in names],dtype=np.int32)
        self.env = np.asarray(body.planes,dtype=np.int32)
        self.env[self.probe] = 1
        library = ctypes.CDLL(str(HERE/"probe_contact.so"))
        self.inspect = library.inspect_contact
        self.inspect.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int,ctypes.c_void_p]
        self.inspect.restype = None

    def before(self, body):
        return None

    def solved(self, body, _):
        self.inspect(body.plant._address,body.d._address,self.leg.ctypes.data,
                     self.env.ctypes.data,self.probe,self.raw[self.count].ctypes.data)
        self.count += 1

    def after(self, body, before, solved):
        pass

    def activity(self):
        if self.count==0:
            return [0.,0.]
        local_force = self.raw[self.count-NATIVE_PER_MS:self.count,:2].mean(0)/W
        return [float(v) for v in np.clip(local_force/.02,0.,1.)]


class TouchLink(ForwardLink):
    def __init__(self):
        super().__init__()
        self.transport = SourceTransport(BINDING,sha(BINDING))
        graph = json.loads(GRAPH.read_text())
        self.an_ids = self.transport.ports["AN"]
        assert self.an_ids == graph["roots"] and self.root_ids == self.transport.ports["DN"]
        totals = defaultdict(lambda: [0.,0.])
        for index, root in enumerate(self.an_ids):
            for edge in graph["brain_outputs"][root]:
                totals[edge["post"]][index] += edge["count"]
        self.matrix = np.asarray(list(totals.values()))
        self.matrix /= self.matrix.max(1)[:,None]
        # Exclusive target pools preserve an interpretable side at engineering readout.
        self.pool = [np.flatnonzero((self.matrix[:,j]>0.) & (self.matrix[:,1-j]==0.)) for j in range(2)]
        assert all(len(p)>0 for p in self.pool)
        self.an_cut = False
        self.an_time = None
        self.an = np.zeros(2)

    def cut_an(self, cut):
        self.an_cut = cut
        self.transport.set_cut("AN",cut)
        self.an_time = None
        self.an[:] = 0.

    def sense(self, values):
        if not self.an_cut:
            self.transport.send(dict(direction="AN",root_ids=self.an_ids,values=values,
                 capture_step=self.transport.step,source_bindings_sha256=self.transport.digest,
                 **{"from":"other","to":"neck"}))

    def advance(self):
        for packet in self.transport.advance():
            values = packet["values"]
            step = packet["release_step"]
            if packet["direction"]=="AN":
                target = self.matrix @ values
                self.an = np.array([target[p].mean() for p in self.pool])
                self.an_time = step
            else:
                self.latest = values
                self.delivered_step = step
                # Bilateral DNg100 pilot sends matched amplitudes.
                self.decoded = int(sum(a*values[0]+b*values[1] for a,b in self.weights) >
                                   .5*sum(a+b for a,b in self.weights))

    def observed(self):
        return self.an.copy() if self.an_time is not None and self.transport.step-self.an_time < 10 else None
