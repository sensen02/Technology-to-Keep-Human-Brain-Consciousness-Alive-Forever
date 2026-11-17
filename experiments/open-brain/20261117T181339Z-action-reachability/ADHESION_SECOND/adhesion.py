"""48-output derivative qualification and an explicitly engineering single-pad bench."""
import time
from context import *
from flygym.compose.world import FlatGroundWorld
from flygym.compose import ActuatorType
from flygym.simulation import Simulation
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain.common import make_locomotion_fly


def run():
    old=body_run.Body();old.at_start(119)
    gain=.5*old.weight
    config=dict(status='ENGINEERING_FUNCTIONAL_BENCH',gain_per_pad_native=gain,gain_per_pad_bodyweight=.5,
        bench_load_fraction_bodyweight=1/6,pad_radius_native=.03,pad_geometry='effective sphere; not anatomical claw/pad',
        friction=1.,gravity_native=float(old.m.opt.gravity[2]),timesteps=[.0001,.00005],
        duration_s=.1,release_s=.05,off_displacement_min_native=1.,on_displacement_max_native=.02,
        release_displacement_min_native=.1,no_contact_force_max_native=1e-12,
        full_body_zero_adhesion_steps=300,derivative_qpos_tolerance=1e-12,neural_ticks=0,training_updates=0,
        source_sha256={str(Path(__file__).resolve()):sha(__file__),str(HERE/'context.py'):sha(HERE/'context.py')})
    write(HERE/'ADHESION_CONFIG.json',config)
    fly=make_locomotion_fly(add_adhesion=True,adhesion_gain=gain)
    world=FlatGroundWorld();world.add_fly(fly,(0,0,.5),Rotation3D('quat',(1,0,0,0)))
    sim=Simulation(world,timestep=body_run.DT);m,d=sim.mj_model,sim.mj_data
    assert (m.nq,m.nv,m.nu)==(old.m.nq,old.m.nv,48)
    identical={}
    # Adding body-targeted actuators changes compiler static-body fusion.
    # Compare the actual joint/geometry dynamics rather than raw body indices.
    for attr in ('jnt_range','jnt_axis','geom_type','geom_size','geom_contype','geom_conaffinity','qpos0'):
        identical[attr]=bool(np.array_equal(getattr(old.m,attr),getattr(m,attr)))
    assert all(identical.values())
    assert not np.any(old.d.xfrc_applied)
    for attr in ('qpos','qvel','act','qacc_warmstart','qfrc_applied','eq_active','userdata','mocap_pos','mocap_quat'):
        getattr(d,attr)[:]=getattr(old.d,attr)
    d.time=old.d.time
    new_pos=sim._intern_actuatorids_by_type_by_fly[ActuatorType.POSITION][fly.name]
    assert len(new_pos)==42
    d.ctrl[new_pos]=old.d.ctrl[old.actuators]
    for a in range(m.nu):
        if a not in new_pos:d.ctrl[a]=0.
    mujoco.mj_forward(m,d)
    mass_old=np.empty((old.m.nv,old.m.nv));mass_new=np.empty((m.nv,m.nv))
    mujoco.mj_fullM(old.m,old.d,mass_old);mujoco.mj_fullM(m,d,mass_new)
    extra=dict(original_nbody=old.m.nbody,derived_nbody=m.nbody,
               initial_mass_matrix_max_error=float(np.max(np.abs(mass_old-mass_new))),
               initial_geom_world_max_error=float(np.max(np.abs(old.d.geom_xpos-d.geom_xpos))),
               total_mass_error=float(abs(old.m.body_mass.sum()-m.body_mass.sum())))
    state=np.empty(mujoco.mj_stateSize(m,body_run.FLAGS));mujoco.mj_getState(m,d,state,body_run.FLAGS)
    np.savez_compressed(HERE/'BODY48_INIT.npz',integration=state,source_qpos=old.d.qpos.copy())
    mujoco.mj_saveModel(m,str(HERE/'BODY48.mjb'),None)
    pairs=[]
    for tick in range(300):
        mujoco.mj_step(old.m,old.d);mujoco.mj_step(m,d)
        pairs.append([np.max(np.abs(old.d.qpos-d.qpos)),np.max(np.abs(old.d.qvel-d.qvel))])
    pairs=np.asarray(pairs);np.savez_compressed(HERE/'BODY48_ZERO_PAIR.npz',error=pairs,original_qpos=old.d.qpos.copy(),derived_qpos=d.qpos.copy())
    derivative=dict(nq=m.nq,nv=m.nv,nu=m.nu,physical_arrays_identical=identical,compiler_fusion=extra,
       zero_adhesion_max_qpos_error=float(pairs[:,0].max()),zero_adhesion_max_qvel_error=float(pairs[:,1].max()),
       zero_adhesion_pass=bool(pairs.max()<=1e-12),physical_steps=600,body_sha256=sha(HERE/'BODY48.mjb'),
       actuator_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(m.nu)])
    write(HERE/'BODY48_RESULT.json',derivative)
    results=[]
    for dt in config['timesteps']:
        for mode in ('off','on','release','no_contact'):
            mass=float(old.m.body_mass.sum()/6)
            xml=f'''<mujoco><option timestep="{dt}" gravity="0 0 {old.m.opt.gravity[2]}"/>
            <default><geom friction="1 .005 .0001" solref=".002 1" solimp=".9 .95 .001"/></default>
            <worldbody><geom name="wall" type="plane" size="2 2 .1" quat=".7071067811865476 0 .7071067811865476 0" rgba=".3 .4 .6 1"/>
            <body name="pad" pos=".0299 0 1"><joint name="normal" type="slide" axis="1 0 0"/><joint name="vertical" type="slide" axis="0 0 1"/>
            <geom name="padgeom" type="sphere" size=".03" mass="{mass}" rgba=".8 .4 .1 1"/></body></worldbody>
            <actuator><adhesion name="pad_adhesion" body="pad" gain="{gain}" ctrlrange="0 1"/></actuator></mujoco>'''
            bm=mujoco.MjModel.from_xml_string(xml);bd=mujoco.MjData(bm)
            if mode=='no_contact':bd.qpos[0]=.3
            mujoco.mj_forward(bm,bd)
            steps=round(.1/dt);rows=[];wrench=np.zeros(6)
            for tick in range(steps):
                bd.ctrl[0]=float(mode in ('on','no_contact') or (mode=='release' and tick*dt<.05))
                mujoco.mj_step(bm,bd)
                normal=tangent=0.
                for index in range(bd.ncon):
                    mujoco.mj_contactForce(bm,bd,index,wrench)
                    normal+=wrench[0];tangent+=np.linalg.norm(wrench[1:3])
                rows.append([bd.time,*bd.qpos,*bd.qvel,bd.ctrl[0],bd.actuator_force[0],*bd.qfrc_actuator,normal,tangent,bd.ncon])
            trace=np.asarray(rows)
            name=f'pad_{mode}_dt{dt:g}';np.savez_compressed(HERE/(name+'.npz'),trace=trace)
            if mode=='on':passed=abs(trace[-1,2])<=.02
            elif mode=='off':passed=-trace[-1,2]>=1.
            elif mode=='release':passed=abs(trace[round(.05/dt)-1,2])<=.02 and -(trace[-1,2]-trace[round(.05/dt)-1,2])>=.1
            else:passed=np.max(np.abs(trace[:,7:9]))<=1e-12 and trace[:,-1].max()==0
            result=dict(mode=mode,dt=dt,steps=steps,pass_=bool(passed),vertical_displacement_native=float(trace[-1,2]),
                generalized_adhesion_force_max_native=float(np.max(np.abs(trace[:,7:9]))),
                warning=bd.warning.number.tolist(),trace_sha256=sha(HERE/(name+'.npz')))
            assert not bd.warning.number.any() and np.isfinite(trace).all()
            results.append(result);print('pad',mode,dt,result['pass_'],round(result['vertical_displacement_native'],6),flush=True)
    write(HERE/'ADHESION_RESULT.json',dict(results=results,all_bench_pass=all(r['pass_'] for r in results),
        physical_steps=600+sum(r['steps'] for r in results),derivative=derivative,
        note='Engineering active pad only; no anatomical claw, no neural learning and no whole-fly slope acceptance.'))


if __name__=='__main__':run()
