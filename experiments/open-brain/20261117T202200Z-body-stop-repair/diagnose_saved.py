from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'20261117T192322Z-terrain-body-learning'
sys.path.insert(0,str(PARENT))
from shared import *
from policy_plus import read
HERE=Path(__file__).resolve().parent
actor=read(PARENT/'FIT2/epoch030.pt');torch.set_num_threads(1)
results=[]
for group,name in [('HELD','block015_1723'),('DEV2','block030_263')]:
 folder=PARENT/group/name
 with np.load(folder/'TRACE.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
 m=mujoco.MjModel.from_binary_path(str(folder/'BODY.mjb'));d=mujoco.MjData(m)
 feet=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f'nmf/{leg}_tarsus5') for leg in ('lf','lm','lh','rf','rm','rh')]
 thorax=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'nmf/c_thorax')
 rows=[]
 for row,(x,p) in enumerate(zip(a['x'],a['phase'])):
  q,pad,raw=actor.act(x,p);base=actor.rhythmic(p[None])[0]
  gate=min(np.max(np.abs(p[6:]))/.05,1)
  drive=np.clip(raw*actor.lambda_scale.numpy()*gate,0,80)
  d.qpos[:]=a['qpos'][row*10];mujoco.mj_kinematics(m,d)
  rows.append(dict(t=row*.001,command=x[:2].tolist(),mag=p[6:].tolist(),raw=raw.tolist(),drive=drive.tolist(),
    max_correction=float(abs(q-base).max()),foot_height=d.xpos[feet,2].tolist(),thorax_height=float(d.xpos[thorax,2]),
    base=base.tolist(),control=q.tolist(),pad=pad.tolist()))
 sub=[r for r in rows if (1.20<=r['t']<=1.41 if group=='HELD' else .90<=r['t']<=1.12)]
 result=dict(group=group,name=name,lambda_scale=actor.lambda_scale.tolist(),rows=rows)
 results.append(result)
 for r in sub[::10]:print(group,round(r['t'],3),'cmd',np.round(r['command'],3),'mag',round(max(r['mag']),3),'drive',np.round(r['drive'],1),'feetZ',np.round(r['foot_height'],3),'corr',round(r['max_correction'],3))
write(HERE/'SAVED_DIAGNOSIS.json',results)
