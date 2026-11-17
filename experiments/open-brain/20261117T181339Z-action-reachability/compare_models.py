"""Read-only comparison of compiled baseline and adhesion derivative."""
from context import *
a=mujoco.MjModel.from_binary_path(str(HERE/'RUNS/flat_teacher_119/BODY.mjb'))
b=mujoco.MjModel.from_binary_path(str(HERE/'BODY48.mjb'))
differences={}
for attr in ('dof_armature','dof_damping','jnt_stiffness','qpos_spring','dof_M0','dof_invweight0'):
    x=getattr(a,attr);y=getattr(b,attr)
    differences[attr]=dict(original_shape=list(x.shape),derived_shape=list(y.shape),max_error=float(np.max(np.abs(x-y))))
for typ,count,label in ((mujoco.mjtObj.mjOBJ_JOINT,a.njnt,'joint_names'),(mujoco.mjtObj.mjOBJ_GEOM,a.ngeom,'geom_names')):
    differences[label]=[(i,mujoco.mj_id2name(a,typ,i),mujoco.mj_id2name(b,typ,i)) for i in range(count)
                       if mujoco.mj_id2name(a,typ,i)!=mujoco.mj_id2name(b,typ,i)]
differences['baseline_pair_solref']=a.pair_solref[:3].tolist()
differences['baseline_pair_friction']=a.pair_friction[:3].tolist()
write(HERE/'MODEL_DIFFERENCES.json',differences)
print(json.dumps(differences,ensure_ascii=False))
