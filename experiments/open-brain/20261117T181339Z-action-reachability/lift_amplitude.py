"""Finite force-limited amplitude probes along the measured IK lifting direction."""
from context import *

source=HERE/'LIFT_RESULT.json'
data=json.loads(source.read_text())
directions={r['leg']:np.array(r['delta_rad']) for r in data['results'] if r['bound_rad']==.15}
config=dict(source_ik_sha256=sha(source),max_per_joint_delta_rad=[.015,.06,.15],physical_steps_per_case=2000,
     planned_physics_steps=36000,ramp_s=.03,hold_end_s=.12,return_end_s=.15,
     physical_lift_pass_native=.15,min_upright=.8,nonfoot_peak_max_bodyweight=.2,
     note='Uniform prespecified amplitude sweep on same IK direction; no brain command or ANN claim; native joint angle limits remain absent.',
     source_sha256={str(Path(__file__).resolve()):sha(__file__),str(HERE/'context.py'):sha(HERE/'context.py')})
write(HERE/'AMPLITUDE_CONFIG.json',config)
results=[]
for leg,direction in directions.items():
    direction=direction/np.max(np.abs(direction))
    for bound in config['max_per_joint_delta_rad']:
        b=body_run.Body();b.at_start(119)
        selected=[i for i,dof in enumerate(b.dofs) if dof.child.pos==leg]
        foot=b.sim._internal_bodyids_by_fly[b.name][b.fly.get_bodysegs_order().index(BodySegment(f'{leg}_tarsus5'))]
        p0=b.d.xpos[foot].copy();c0=b.d.xpos[b.thorax].copy();rows=[];poses=[];wrench=np.zeros(6)
        for tick in range(2000):
            t=tick*.0001
            if t<.03:scale=.5-.5*np.cos(np.pi*t/.03)
            elif t<.12:scale=1.
            elif t<.15:scale=.5+.5*np.cos(np.pi*(t-.12)/.03)
            else:scale=0.
            ctrl=b.neutral.copy();ctrl[selected]+=scale*direction*bound;b.apply(ctrl)
            lift=(b.d.xpos[foot,2]-p0[2])-(b.d.xpos[b.thorax,2]-c0[2])
            nf=0.
            for index,c in enumerate(b.d.contact[:b.d.ncon]):
                if c.geom1 in b.ground_geoms:other=c.geom2
                elif c.geom2 in b.ground_geoms:other=c.geom1
                else:continue
                if b.m.geom_bodyid[other] in b.axis_bodies:
                    mujoco.mj_contactForce(b.m,b.d,index,wrench);nf+=np.linalg.norm(wrench[:3])
            rows.append([b.d.time,lift,b.d.xmat[b.thorax,8],np.max(np.abs(b.d.actuator_force[b.actuators])),nf])
            if tick%10==0:poses.append(b.d.qpos.copy())
        a=np.asarray(rows);name=f'{leg}_bound{bound:g}'
        np.savez_compressed(HERE/f'AMPLITUDE_{name}.npz',trace=a,qpos=np.array(poses),delta=direction*bound)
        result=dict(leg=leg,bound_rad=bound,physical_steps=2000,hold_lift_mean_native=float(a[600:1200,1].mean()),
             peak_lift_native=float(a[:,1].max()),min_upright=float(a[:,2].min()),force_peak_native=float(a[:,3].max()),
             nonfoot_peak_bodyweight=float(a[:,4].max()/b.weight),
             pass_=bool(a[600:1200,1].mean()>=.15 and a[:,2].min()>=.8 and a[:,4].max()/b.weight<=.2))
        assert np.isfinite(a).all() and not b.d.warning.number.any()
        results.append(result);print('amplitude',leg,bound,round(result['hold_lift_mean_native'],4),result['pass_'],flush=True)
write(HERE/'AMPLITUDE_RESULT.json',dict(results=results,physical_steps=36000,training_updates=0,
     scope='Local actuator feasibility only; whole-body navigation and biological angle compliance unproven.'))
