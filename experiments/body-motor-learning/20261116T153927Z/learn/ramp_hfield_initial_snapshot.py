"""Repeat the finite grade using one continuous heightfield collision surface."""
import argparse
import ctypes
from pathlib import Path

import mujoco as mj
import numpy as np

import ramp_course as bench
from verify_native import Record

HERE=Path(__file__).resolve().parent
SURFACE='terrain_heightfield'
BINDING=HERE/'hfield_contacts.so'


def scene(plan):
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    stand=bench.stan
    loader=stand.physics.act_mod.load_full_model
    original=loader(stand.physics.PAD_XML)
    x0=plan['lower_x']
    x1=plan['upper_x']
    span=10.
    center=x0-1.
    xs=np.linspace(center-span,center+span,201)
    profile=bench.surface(xs,plan)
    max_height=max(.02,float(profile.max()))
    data=np.tile(profile/max_height,(3,1)).ravel()
    spec=_load_mjcf(stand.physics.PAD_XML)
    floor=spec.geom('floor')
    base=original.geom('floor')
    floor.contype=0
    floor.conaffinity=0
    floor.rgba=np.zeros(4)
    floor.material=''
    spec.add_hfield(name='finite_graded_floor',size=[span,plan['width_half'],max_height,.2],
                    nrow=3,ncol=len(xs),userdata=data.tolist())
    spec.worldbody.add_geom(name=SURFACE,type=mj.mjtGeom.mjGEOM_HFIELD,hfieldname='finite_graded_floor',
                            pos=[center,0.,0.],rgba=[.24,.32,.27,1.],contype=int(base.contype[0]),
                            conaffinity=int(base.conaffinity[0]),condim=int(base.condim[0]),
                            friction=base.friction,solref=base.solref,solimp=base.solimp,
                            margin=float(base.margin[0]),gap=float(base.gap[0]),
                            solmix=float(base.solmix[0]),priority=int(base.priority[0]))
    model=spec.compile()
    stand.physics.act_mod.validate_full_model(model)
    model.mat_rgba[:original.nmat]=original.mat_rgba
    for i in range(original.ngeom):
        g=original.geom(i)
        if g.name=='floor':
            continue
        n=model.geom(g.name)
        model.geom_rgba[n.id]=original.geom_rgba[g.id]
        for k in ('type','pos','quat','size','friction','solref','solimp','condim','contype','conaffinity'):
            np.testing.assert_array_equal(getattr(n,k),getattr(g,k),err_msg=g.name+' '+k)
    for slot in ('body_mass','body_inertia','body_ipos','body_iquat','jnt_type','jnt_range','jnt_qposadr',
                 'jnt_dofadr','actuator_trnid','actuator_gear','dof_damping','dof_armature'):
        np.testing.assert_array_equal(getattr(model,slot),getattr(original,slot),err_msg=slot)
    np.testing.assert_array_equal(model.opt.gravity,original.opt.gravity)
    stand.physics.act_mod.load_full_model=lambda *a,**kw:model
    try:
        body=stand.physics.Core(adhesion=False)
    finally:
        stand.physics.act_mod.load_full_model=loader
    field=model.geom(SURFACE).id
    role=np.zeros(model.ngeom,dtype=np.int32)
    role[field]=1
    body.planes=role>0
    body.physics_order=stand.physics.rc.SENS_IDX
    start=bench.INITIAL
    sha=bench.turn.read(bench.turn.STARTS/'BANK.json')['nominal']['sha256']
    assert stand.sha(start)==sha
    with np.load(start,allow_pickle=False) as z:
        mj.mj_setState(model,body.d,z['state'],mj.mjtState.mjSTATE_INTEGRATION)
        mj.mj_forward(model,body.d)
        body.t=int(z['t'])
        body.actuator.command_ticks=int(z['ticks'])
        for key in z.files:
            if key.startswith('core_'):
                setattr(body,key[5:],z[key].copy())
            if key.startswith('act_'):
                setattr(body.actuator,key[4:],z[key].copy())
        np.testing.assert_array_equal(body.d.qpos,z['qpos'])
        np.testing.assert_array_equal(body.d.qvel,z['qvel'])
        delta=float(np.max(np.abs(stand.observe(body)-z['observation'])))
    ray=np.array([0.,0.,-1.])
    queried=np.r_[x0-.15,x0+.05,.5*(x0+x1),x1-.05,x1+.15]
    expected=bench.surface(queried,plan)
    obtained=np.array([1.-mj.mj_rayHfield(model,body.d,field,np.array([x,.25,1.]),ray) for x in queried])
    np.testing.assert_allclose(obtained,expected,atol=1.e-8,rtol=0.)
    print('Heightfield elevation equality',list(zip(queried.tolist(),obtained.tolist())),
          'snapshot observation difference',delta,flush=True)
    return body,role,sha,delta


class Capture(Record):
    def __init__(self,body,finish,mask):
        super().__init__(body,finish)
        field=np.flatnonzero(mask)
        assert len(field)==1
        self.field=int(field[0])
        plan=bench.turn.read(bench.PLAN)
        self.lo=plan['lower_x']+.01
        self.hi=plan['upper_x']-.01
        self.block=np.zeros((finish*80,19))
        self.slope_values=np.zeros((finish*80,13))
        self.last_native=np.zeros(32)
        library=ctypes.CDLL(str(BINDING))
        self.contacts=library.hfield_contacts
        self.contacts.argtypes=[ctypes.c_void_p]*2+[ctypes.c_int]+[ctypes.c_void_p]*2+[ctypes.c_double]*2+[ctypes.c_void_p]*3
        self.contacts.restype=None

    def solved(self,c,before):
        result=self.forces[self.count]
        score=self.slope_values[self.count]
        block=self.block[self.count]
        self.contacts(c.plant._address,c.d._address,self.field,self.foot_map.ctypes.data,self.velocity.ctypes.data,
                      self.lo,self.hi,result.ctypes.data,score.ctypes.data,block.ctypes.data)
        self.last_native[13:]=block
        if self.count%4031==0:
            old=bench.stan.physics.NativeRecorder.solved(None,c,{'qvel_before':self.velocity})
            expected=np.r_[old['loads_native'],old['loads_native']*old['contact_speed_native_per_s'],old['nonfoot_force_native']]
            np.testing.assert_allclose(result,expected,rtol=1.e-12,atol=1.e-9)
            self.cross_checks+=1


def run(output,flat):
    old_scene,old_record=bench.scene,bench.RampRecord
    old_capture=bench.stan.identity.capture
    old_put=bench.stan.put

    def capture(core,scope,*,extra_paths):
        return old_capture(core,{**scope,'ground':'single heightfield, no contact seam'},
                           extra_paths=[*extra_paths,Path(__file__),HERE/'hfield_contacts.cpp',BINDING])

    def output_result(path,data):
        if path==output/'RESULT.json':
            data.update(surface_type='heightfield',terrain_geom=SURFACE,
                        slope_specific_ground='contacts inside incline interval excluding 0.01 native length at both ends')
        return old_put(path,data)

    bench.scene=scene
    bench.RampRecord=Capture
    bench.stan.identity.capture=capture
    bench.stan.put=output_result
    try:
        bench.run(output,flat)
    finally:
        bench.stan.put=old_put
        bench.stan.identity.capture=old_capture
        bench.scene=old_scene
        bench.RampRecord=old_record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path, nargs='?')
    p.add_argument('--flat',action='store_true')
    p.add_argument('--compile-only',action='store_true')
    a=p.parse_args()
    if a.compile_only:
        plan=bench.turn.read(bench.PLAN)
        if a.flat:
            plan['slope_degrees']=0.
        scene(plan)
    else:
        run(a.output.resolve(),a.flat)
