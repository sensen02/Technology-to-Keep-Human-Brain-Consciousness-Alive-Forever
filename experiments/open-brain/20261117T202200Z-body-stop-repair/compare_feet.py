from common import *
results=[]
for label,folder in [('old',PARENT/'HELD/block015_1723'),('routing_only',HERE/'ROUTING_PROBE/0.15_1723'),('fine_tuned',HERE/'FIT_PROBE/0.15_1723')]:
 if not (folder/'TRACE.npz').exists():continue
 with np.load(folder/'TRACE.npz',allow_pickle=False) as z:poses=z['qpos'].copy()
 m=mujoco.MjModel.from_binary_path(str(folder/'BODY.mjb'));d=mujoco.MjData(m)
 feet=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f'nmf/{leg}_tarsus5') for leg in ('lf','lm','lh','rf','rm','rh')]
 xyz=[]
 for q in poses[13000:]:d.qpos[:]=q;mujoco.mj_kinematics(m,d);xyz.append(d.xpos[feet].copy())
 xyz=np.array(xyz);row=dict(label=label,folder=str(folder),zero_command_window_s=[1.3,1.7],peak_foot_z=xyz[:,:,2].max(0).tolist(),left_middle_peak=float(xyz[:,1,2].max()))
 results.append(row);print(label,'LM peak',row['left_middle_peak'])
write(HERE/'FOOT_HEIGHT_COMPARISON.json',results)
