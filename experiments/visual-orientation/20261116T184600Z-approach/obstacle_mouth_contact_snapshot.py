"""Single solid visibly green obstacle. Qualification measures actual native hits."""
import argparse
import ctypes
from pathlib import Path

import mujoco as mj
import numpy as np

import approach
from approach_runtime import HERE,turn

original_prepare=approach.prepare
original_score=approach.score
original_record=approach.Record
BODY=turn.stand.physics
LIB=HERE.parent/"20261116T180500Z-touch/probe_contact.so"


def head_tip(body):
    geom=body.plant.geom("Head_geom")
    mesh=int(geom.dataid[0])
    start=int(body.plant.mesh_vertadr[mesh])
    count=int(body.plant.mesh_vertnum[mesh])
    vertices=body.plant.mesh_vert[start:start+count]
    world=vertices@body.d.geom_xmat[geom.id].reshape(3,3).T+body.d.geom_xpos[geom.id]
    return world[np.argmax(world[:,0])]


def prepare(path,start):
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    measure,_ = original_prepare(path.with_name(path.name+"_FIXTURE"),start)
    tip=head_tip(measure)
    center=np.r_[tip[0]+.37,measure.d.xpos[measure.plant.body("Head").id,1],.50]
    loader=BODY.act_mod.load_full_model
    original=loader(BODY.PAD_XML)
    spec=_load_mjcf(BODY.PAD_XML)
    station = spec.worldbody.add_body(name="navigation_fixture",pos=center)
    station.add_geom(name="navigation_obstacle",type=mj.mjtGeom.mjGEOM_BOX,
                     size=[.045,.19,.50],rgba=[.03,.90,.08,1.])
    model=spec.compile()
    for i in range(original.ngeom):
        geom=original.geom(i)
        model.geom_rgba[model.geom(geom.name).id]=geom.rgba
    model.mat_rgba[:original.nmat]=original.mat_rgba
    path.mkdir()
    BODY.act_mod.load_full_model=lambda *a,**kw:model
    try:
        body=BODY.Core(adhesion=False)
    finally:
        BODY.act_mod.load_full_model=loader
    body.physics_order=BODY.rc.SENS_IDX
    assert all(original.body(i).name==model.body(i).name for i in range(original.nbody))
    initial=turn.STARTS/f"{start}.npz"
    assert turn.stand.sha(initial)==turn.read(turn.STARTS/"BANK.json")[start]["sha256"]
    with np.load(initial,allow_pickle=False) as state:
        flags=int(mj.mjtState.mjSTATE_INTEGRATION)
        index=0
        for bit in range(flags.bit_length()):
            field=1<<bit
            if not (flags & field):
                continue
            length=mj.mj_stateSize(original,field)
            values=np.zeros(mj.mj_stateSize(model,field))
            values[:length]=state["state"][index:index+length]
            if len(values):
                mj.mj_setState(model,body.d,values,field)
            index+=length
        assert index==len(state["state"])
        mj.mj_forward(model,body.d)
        body.t=int(state["t"])
        body.actuator.command_ticks=int(state["ticks"])
        for item in state.files:
            if item.startswith("core_"):
                setattr(body,item[5:],state[item].copy())
            if item.startswith("act_"):
                setattr(body.actuator,item[4:],state[item].copy())
        np.testing.assert_array_equal(body.d.qpos,measure.d.qpos)
        np.testing.assert_array_equal(body.d.qvel,measure.d.qvel)
        delta = turn.stand.observe(body)-state["observation"]
        box = body.plant.geom("navigation_obstacle").id
        hits = [(body.plant.geom(int(c.geom1)).name,body.plant.geom(int(c.geom2)).name,float(c.dist))
                for c in body.d.contact[:body.d.ncon] if box in c.geom]
        print("fixture start", center, "tip",tip, "initial contacts",hits, "obs delta",list(zip(np.flatnonzero(abs(delta)>1.e-12),delta[abs(delta)>1.e-12])),flush=True)
        assert not hits
        np.testing.assert_allclose(turn.stand.observe(body),state["observation"],rtol=0.,atol=1.e-12)
    motor=approach.Motion(body)
    turn.stand.put(path/"OBSTACLE.json",dict(center=center.tolist(),half_extent=[.045,.19,.50],
                      head_tip_at_start=tip.tolist(),native_obstacle_source="solid static MuJoCo box"))
    return body,motor


class CollisionRecord(original_record):
    def __init__(self,body,duration):
        super().__init__(body,duration)
        lib=ctypes.CDLL(str(LIB))
        self.check=lib.inspect_contact
        self.check.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int,ctypes.c_void_p]
        self.check.restype=None
        self.box=body.plant.geom("navigation_obstacle").id
        self.env=np.asarray(body.planes,dtype=np.int32)
        self.env[self.box]=1
        self.leg=np.full(body.plant.ngeom,-1,dtype=np.int32)
        self.extra=np.zeros(6)
        self.block=np.zeros(duration*80)

    def solved(self,c,before):
        super().solved(c,before)
        self.check(c.plant._address,c.d._address,self.leg.ctypes.data,self.env.ctypes.data,
                   self.box,self.extra.ctypes.data)
        self.block[self.count]=self.extra[4]


class GreenStop(approach.Policy):
    def __init__(self):
        super().__init__()
        self.hazard=False

    def act(self,image):
        intended,estimate=super().act(image)
        rgb=image.astype(float)
        green=(rgb[...,1]>75.)&(rgb[...,1]>2.*rgb[...,0])&(rgb[...,1]>2.*rgb[...,2])
        self.hazard = self.hazard or (int(green.sum())>=6 and intended[0]>0)
        if self.hazard:
            self.phase="done"
            return (0.,0),estimate
        return intended,estimate


def score(trace,decisions,goal,actual,angle,mode,record):
    result=original_score(trace,decisions,goal,actual,angle,mode,record)
    result["approach_ordinary_checks"]=result["checks"]
    force=record.block[:record.count]
    hit=np.flatnonzero(force>1e-6)
    result["impact_count_native"]=int(len(hit))
    result["first_obstacle_force_ms"]=float(hit[0]/80) if len(hit) else None
    result["peak_obstacle_force_W"]=float(force.max()/BODY.W)
    result["approach_course_gate_applies"]=False
    result["passed"]=bool(not len(hit) and result["standing"]["passed"])
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    p.add_argument("--blind",action="store_true")
    a=p.parse_args()
    approach.prepare=prepare
    approach.Record=CollisionRecord
    approach.score=score
    if not a.blind:
        approach.Policy=GreenStop
    answer=approach.trial(a.out.resolve(),0.,2.68,"intact",.12,11500)
    raise SystemExit(0 if answer["passed"] else 1)
