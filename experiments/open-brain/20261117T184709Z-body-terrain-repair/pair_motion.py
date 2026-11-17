"""Zero-pad dynamic control equivalence beyond the static standing comparison."""
from body_model import *
a=Body(adhesion=False);b=Body(adhesion=True);a.restore(119);b.restore(119)
config=dict(steps_per_arm=3000,control='neutral + .08*sin(2*pi*8*t + arange(42)*.2), clipped to fixed joint target bounds',
        tolerance=1e-12,source_sha256={str(Path(__file__).resolve()):sha(__file__),str(HERE/'body_model.py'):sha(HERE/'body_model.py')})
write(HERE/'PAIR_MOTION_CONFIG.json',config)
rows=[]
for tick in range(3000):
    ctrl=np.clip(a.neutral+.08*np.sin(2*np.pi*8*tick*DT+np.arange(42)*.2),a.minimum,a.maximum)
    a.apply(ctrl);b.apply(ctrl,np.zeros(6))
    rows.append([np.max(np.abs(a.d.qpos-b.d.qpos)),np.max(np.abs(a.d.qvel-b.d.qvel)),
                 np.max(np.abs(a.d.actuator_force[a.actuators]-b.d.actuator_force[b.actuators]))])
x=np.asarray(rows);np.savez_compressed(HERE/'PAIR_MOTION.npz',error=x)
result=dict(pass_=bool(x.max()<=1e-12),qpos_max=float(x[:,0].max()),qvel_max=float(x[:,1].max()),
        joint_force_max=float(x[:,2].max()),physics_steps=6000,warning_a=a.d.warning.number.tolist(),warning_b=b.d.warning.number.tolist())
write(HERE/'PAIR_MOTION_RESULT.json',result);print(json.dumps(result))
