from common import *
from shadow import run
results=[]
for height,seed in ((.15,1723),(.30,263)):
 result=run(height,seed,'shadow',HERE/'SHADOW'/f'{height}_{seed}',CHECKPOINT)
 original=PARENT/('HELD/block015_1723' if seed==1723 else 'DEV2/block030_263')/'TRACE.npz'
 with np.load(original,allow_pickle=False) as a,np.load(HERE/'SHADOW'/f'{height}_{seed}'/'TRACE.npz',allow_pickle=False) as b:
  assert all(np.array_equal(a[k],b[k]) for k in ('qpos','qvel','control','pads','x','phase'))
  for t in ((1.32,1.34,1.35,1.36) if seed==1723 else (1.04,1.08)):
   row=round(t*1000);p=b['phase'][row];pred,pad,raw=legacy_read(CHECKPOINT).act(b['x'][row],p)
   drive=np.clip(raw*legacy_read(CHECKPOINT).lambda_scale.numpy()*min(max(p[6:])/.05,1),0,80)
   print(seed,t,'pred',np.round(drive,2),'teacher_same_state',np.round(b['lambda'][row],2),flush=True)
 result['shadow_original_body_exact']=True;results.append(result)
write(HERE/'SHADOW_RESULT.json',results)
