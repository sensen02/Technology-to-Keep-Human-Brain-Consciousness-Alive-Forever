"""Saved policy, full physical metrics, identities and independent replay verification."""
from scope import *
from metrics_v3 import compare
import hashlib

choice=json.loads((HERE/'SELECTION.json').read_text());checkpoint=Path(choice['checkpoint']);actor=read(checkpoint);torch.set_num_threads(1)
records=[];replays=[];sources={}
for group in ('CALIBRATED','HELD','HELD_A01','REPLAY'):
 for folder in sorted((HERE/group).glob('*_student')):
  r=json.loads((folder/'RESULT.json').read_text());inputs=json.loads((folder/'INPUTS.json').read_text())
  verify_sources(inputs['source_sha256']);sources.update(inputs['source_sha256']);assert r['weight_sha256']==sha(checkpoint) and r['teacher_calls']==0 and not any(r['warning'])
  assert sha(folder/'TRACE.npz')==r['trace_sha256']
  with np.load(folder/'TRACE.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
  for row,(x,p) in enumerate(zip(a['x'],a['phase'])):
   control,pad,_=actor.act(x,p)
   assert np.array_equal(control,a['applied'][row]) and np.array_equal(control,a['control'][row*10]) and np.array_equal(pad,a['pads'][row*10])
  assert np.array_equal(a['control'],np.repeat(a['applied'],10,axis=0))
  m=mujoco.MjModel.from_binary_path(str(folder/'BODY.mjb'));d=mujoco.MjData(m);j=np.flatnonzero(m.jnt_limited);q=a['qpos'][:,m.jnt_qposadr[j]]
  ex=np.maximum(0,np.maximum(m.jnt_range[j,0]-q,q-m.jnt_range[j,1]))
  rms=float(np.sqrt(np.mean(np.sum((np.diff(a['position'][-2000:,:2],axis=0)/DT)**2,axis=1))))
  assert float(ex.max())==r['joint_excess'] and rms==r['stand_rms']
  assert float(a['position'][:,2].min())==r['min_height'] and float(a['upright'].min())==r['min_upright']
  if group=='CALIBRATED':teacher=HERE/('DEV_A01' if r['family'].startswith('a01') else 'DEV')/folder.name.replace('_student','_teacher')
  elif group=='REPLAY':teacher=HERE/('HELD_A01' if r['family'].startswith('a01') else 'HELD')/folder.name.replace('_student','_teacher')
  else:teacher=folder.parent/folder.name.replace('_student','_teacher')
  t=json.loads((teacher/'RESULT.json').read_text());assert not any(t['warning'])
  assert sha(teacher/'BODY.mjb')==sha(folder/'BODY.mjb')
  command_dir=HERE/'COMMANDS'/f"{r['family']}_{r['variant']}"
  cmd=command_dir/'TRACE.npz';assert inputs['source_sha256'][str(cmd)]==json.loads((teacher/'INPUTS.json').read_text())['source_sha256'][str(cmd)]
  meta=json.loads((command_dir/'RESULT.json').read_text());first=meta['action']+1800 if r['family'].startswith(('arc','stride','a01')) else meta['start']+1800;last=meta['off']-300
  pos=a['position'][first:last,:2];yaw=a['yaw'][first:last];v=float(np.sum(np.diff(pos,axis=0)*np.c_[np.cos(yaw[:-1]),np.sin(yaw[:-1])])/(DT*(len(pos)-1)))
  assert v==r['forward_speed']
  geometry_error=0.;thorax=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'nmf/c_thorax')
  with np.load(TEACHER/'CANONICAL_INIT.npz',allow_pickle=False) as z:initial=z['qpos'].copy()
  for i in (0,1000,6000,12000,19000):
   d.qpos[:]=initial if i==0 else a['qpos'][i-1];mujoco.mj_kinematics(m,d);geometry_error=max(geometry_error,float(abs(d.xpos[thorax]-a['position'][i]).max()))
  assert geometry_error<1e-10
  pair=compare(t,r);pair.update(folder=str(folder),group=group,exact_outputs=True,body_identical=True,command_identical=True,geometry_error=geometry_error)
  records.append(pair)
  if group=='REPLAY':
   ref=HERE/('HELD_A01' if r['family'].startswith('a01') else 'HELD')/folder.name
   with np.load(ref/'TRACE.npz',allow_pickle=False) as z:checks={k:np.array_equal(z[k],v) for k,v in a.items()};assert all(checks.values())
   replays.append(dict(folder=str(folder),original=str(ref),arrays_exact=checks))

if (HERE/'FIT/RESULT.json').exists():
 load=lambda p:torch.load(p,map_location='cpu',weights_only=True)
 parent=load(WEIGHT);start=load(HERE/'FIT/epoch000.pt')
 def same(a,b):
  if isinstance(a,torch.Tensor):return torch.equal(a,b)
  if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
  if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
  return a==b
 assert all(same(parent[k],start[k]) for k in ('state','optimizer','rng','torch_rng','update'))
 logs=[json.loads(l) for l in (HERE/'FIT/UPDATES.jsonl').read_text().splitlines()]
 assert len(logs)==json.loads((HERE/'FIT/RESULT.json').read_text())['actual_updates']
 for row in logs:
  assert sha(row['checkpoint'])==row['sha256'];saved=load(row['checkpoint']);assert saved['update']==row['update'];h=hashlib.sha256()
  for key,v in saved['state'].items():
   if key.startswith('reflex.'):h.update(key.encode());h.update(v.numpy().tobytes())
  assert h.hexdigest()==row['parameter_sha256']
 assert len(set(r['parameter_sha256'] for r in logs))==len(logs)
 end=load(checkpoint)
 for k in parent['state']:
  if not k.startswith('reflex.'):assert torch.equal(parent['state'][k],end['state'][k])
 assert json.loads((HERE/'FIT/GRADIENT_CHECK.json').read_text())['pass_']
 verify_sources(json.loads((HERE/'FIT/INPUTS.json').read_text())['source_sha256'])
write(HERE/'INDEPENDENT.json',dict(records=records,replays=replays,source_sha256=sources,
  all_actual_policy_updates_verified=True,optimizer_rng_resume_verified=True,physical_steps=0,neural_ticks=0))
print('Independent outputs/metrics/command/model identity/replays verified:',len(records),len(replays))
