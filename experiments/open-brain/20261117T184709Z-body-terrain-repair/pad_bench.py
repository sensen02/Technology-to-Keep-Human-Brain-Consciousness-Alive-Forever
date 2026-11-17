"""Single effective pad with the canonical body's actual contact parameters."""
from body_model import *

m=mujoco.MjModel.from_binary_path(str(HERE/'BODY48.mjb'))
weight=float(m.body_mass.sum()*abs(m.opt.gravity[2]));mass=float(m.body_mass.sum()/6)
solref=m.pair_solref[0];solimp=m.pair_solimp[0];friction=m.pair_friction[0,[0,2,3]]
f=lambda a:' '.join(str(float(x)) for x in a)
config=dict(source_body_sha256=sha(HERE/'BODY48.mjb'),solref=solref.tolist(),solimp=solimp.tolist(),
 friction=friction.tolist(),condim=int(m.pair_dim[0]),noslip_iterations=int(m.opt.noslip_iterations),
 gain=GAIN,mass_native=mass,gravity=m.opt.gravity.tolist(),stepsizes=[.0001,.00005],
 duration_s=.1,release_s=.05,on_displacement_max=.02,off_displacement_min=1.,release_drop_min=.1,
 no_contact_qforce_max=1e-12,geometry='Declared engineering effective spherical pad; not real pad microstructure.',
 source_sha256={str(Path(__file__).resolve()):sha(__file__)})
write(HERE/'PAD_BENCH_CONFIG.json',config)
results=[]
for dt in config['stepsizes']:
 for mode in ('off','on','release','no_contact'):
    xml=f'''<mujoco><option timestep="{dt}" gravity="{f(m.opt.gravity)}" iterations="100" noslip_iterations="{m.opt.noslip_iterations}"/>
    <default><geom friction="{f(friction)}" condim="{m.pair_dim[0]}" solref="{f(solref)}" solimp="{f(solimp)}"/></default>
    <worldbody><geom name="wall" type="plane" size="2 2 .1" quat=".7071067811865476 0 .7071067811865476 0"/>
    <body name="pad" pos=".0299 0 1"><joint name="normal" type="slide" axis="1 0 0"/><joint name="vertical" type="slide" axis="0 0 1"/>
    <geom name="padgeom" type="sphere" size=".03" mass="{mass}"/></body></worldbody>
    <actuator><adhesion name="pad_adhesion" body="pad" gain="{GAIN}" ctrlrange="0 1"/></actuator></mujoco>'''
    bm=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(bm)
    if mode=='no_contact':d.qpos[0]=.3
    mujoco.mj_forward(bm,d);rows=[]
    for tick in range(round(.1/dt)):
        d.ctrl[0]=float(mode in ('on','no_contact') or (mode=='release' and tick*dt<.05));mujoco.mj_step(bm,d)
        rows.append([d.time,*d.qpos,*d.qvel,d.ctrl[0],*d.qfrc_actuator,d.ncon])
    a=np.asarray(rows);name=f'{mode}_{dt:g}';np.savez_compressed(HERE/f'PAD_{name}.npz',trace=a)
    if mode=='off':passed=-a[-1,2]>=1.
    elif mode=='on':passed=abs(a[-1,2])<=.02
    elif mode=='release':passed=abs(a[round(.05/dt)-1,2])<=.02 and -(a[-1,2]-a[round(.05/dt)-1,2])>=.1
    else:passed=bool(np.max(np.abs(a[:,6:8]))<=1e-12 and a[:,-1].max()==0)
    result=dict(mode=mode,dt=dt,pass_=bool(passed),displacement=float(a[-1,2]),steps=len(a),warning=d.warning.number.tolist())
    assert np.isfinite(a).all() and not d.warning.number.any()
    results.append(result);print('pad',mode,dt,result['pass_'],result['displacement'],flush=True)
write(HERE/'PAD_BENCH_RESULT.json',dict(results=results,all_pass=all(r['pass_'] for r in results),physics_steps=sum(r['steps'] for r in results)))
