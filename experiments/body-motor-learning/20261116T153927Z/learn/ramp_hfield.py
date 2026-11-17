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


def scene(plan,overlay=False):
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    stand=bench.stan
    loader=stand.physics.act_mod.load_full_model
    original=loader(stand.physics.PAD_XML)
    x0=plan['lower_x']
    x1=plan['upper_x']
    span=10.
    center=x0-1.
    xs=np.linspace(center-span,center+span,401 if overlay else 201)
    profile=bench.surface(xs,plan)
    burial=.12 if overlay else 0.
    if overlay:
        if plan['slope_degrees']>0.:
            profile += burial*(np.clip((xs-(x0-.10))/.10,0.,1.)-1.)
        else:
            profile-=burial
    max_height=max(.02,float(profile.max()+burial))
    values=np.clip((profile+burial)/max_height,0.,1.)
    data=np.tile(values,(3,1)).ravel()
    spec=_load_mjcf(stand.physics.PAD_XML)
    floor=spec.geom('floor')
    base=original.geom('floor')
    if not overlay:
        floor.contype=0
        floor.conaffinity=0
        floor.rgba=np.zeros(4)
        floor.material=''
    spec.add_hfield(name='finite_graded_floor',size=[span,plan['width_half'],max_height,.2],
                    nrow=3,ncol=len(xs),userdata=data.tolist())
    spec.worldbody.add_geom(name=SURFACE,type=mj.mjtGeom.mjGEOM_HFIELD,hfieldname='finite_graded_floor',
                            pos=[center,0.,-burial],rgba=[.24,.32,.27,1.],contype=int(base.contype[0]),
                            conaffinity=int(base.conaffinity[0]),condim=int(base.condim[0]),
                            friction=base.friction,solref=base.solref,solimp=base.solimp,
                            margin=float(base.margin[0]),gap=float(base.gap[0]),
                            solmix=float(base.solmix[0]),priority=int(base.priority[0]))
    model=spec.compile()
    stand.physics.act_mod.validate_full_model(model)
    model.mat_rgba[:original.nmat]=original.mat_rgba
    for i in range(original.ngeom):
        g=original.geom(i)
        if g.name=='floor' and not overlay:
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
    if overlay:
        role[model.geom('floor').id]=1
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
    if overlay:
        obtained=np.maximum(0.,obtained)
        initial_top=body.d.geom_xpos[np.asarray([body.pads[leg] for leg in bench.LEGS]),0]
        pref=max(initial_top.ravel())
        ahead=float(1.-mj.mj_rayHfield(model,body.d,field,np.array([pref+.015,.25,1.]),ray))
        initial_env_contacts=[i for i in range(body.d.ncon) if field in body.d.contact[i].geom]
        if ahead>-.040 or initial_env_contacts or delta!=0.:
            raise RuntimeError(f'Start must preserve all frozen physical sensors, got {(ahead,len(initial_env_contacts),delta)}')
    np.testing.assert_allclose(obtained,expected,atol=1.e-8,rtol=0.)
    print('Heightfield elevation equality',list(zip(queried.tolist(),obtained.tolist())),
          'snapshot observation difference',delta,flush=True)
    if overlay:
        print('Original plane and sole settled contacts intact; upload starts after x',pref,flush=True)
    return body,role,sha,delta


class Capture(Record):
    def __init__(self,body,finish,mask):
        super().__init__(body,finish)
        self.field=body.plant.geom(SURFACE).id
        assert mask[self.field]
        self.floor=body.plant.geom('floor').id
        if not mask[self.floor]:
            self.floor=None
        plan=bench.turn.read(bench.PLAN)
        self.lo=plan['lower_x']+.01
        self.hi=plan['upper_x']-.01
        self.block=np.zeros((finish*80,19))
        self.slope_values=np.zeros((finish*80,13))
        self.last_native=np.zeros(32)
        self.lower=np.zeros(13)
        self.unused_slope=np.zeros(13)
        self.unused_flags=np.zeros(19)
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
        if self.floor is not None:
            self.contacts(c.plant._address,c.d._address,self.floor,self.foot_map.ctypes.data,self.velocity.ctypes.data,
                          1.,-1.,self.lower.ctypes.data,self.unused_slope.ctypes.data,self.unused_flags.ctypes.data)
            result+=self.lower
        self.last_native[13:]=block
        if self.count%4031==0:
            old=bench.stan.physics.NativeRecorder.solved(None,c,{'qvel_before':self.velocity})
            expected=np.r_[old['loads_native'],old['loads_native']*old['contact_speed_native_per_s'],old['nonfoot_force_native']]
            np.testing.assert_allclose(result,expected,rtol=1.e-12,atol=1.e-9)
            self.cross_checks+=1


def run(output,flat,overlay=False,plan_source=None):
    old_scene,old_record=bench.scene,bench.RampRecord
    old_capture=bench.stan.identity.capture
    old_put=bench.stan.put
    original_plan=bench.PLAN
    if plan_source is not None:
        bench.PLAN=plan_source

    def capture(core,scope,*,extra_paths):
        kind='original plane and buried slope shelf' if overlay else 'single heightfield, no contact seam'
        return old_capture(core,{**scope,'ground':kind},
                           extra_paths=[*extra_paths,Path(__file__),HERE/'hfield_contacts.cpp',BINDING])

    def output_result(path,data):
        if path==output/'RESULT.json':
            data.update(surface_type='heightfield_overlay' if overlay else 'heightfield',terrain_geom=SURFACE,
                        slope_specific_ground='contacts inside incline interval excluding 0.01 native length at both ends')
            if overlay and flat and data['complete']:
                checks=data['checks']
                checks.pop('incline_load')
                checks['buried_control_without_ramp_support']=not data['named_ramp_support']
                data['passed']=all(checks.values())
        return old_put(path,data)

    bench.scene=lambda plan:scene(plan,overlay)
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
        bench.PLAN=original_plan


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path, nargs='?')
    p.add_argument('--flat',action='store_true')
    p.add_argument('--compile-only',action='store_true')
    p.add_argument('--over-original-plane',action='store_true')
    p.add_argument('--plan',type=Path)
    a=p.parse_args()
    if a.compile_only:
        plan=bench.turn.read(a.plan or bench.PLAN)
        if a.flat:
            plan['slope_degrees']=0.
        scene(plan,a.over_original_plane)
    else:
        run(a.output.resolve(),a.flat,a.over_original_plane,a.plan.resolve() if a.plan else None)
