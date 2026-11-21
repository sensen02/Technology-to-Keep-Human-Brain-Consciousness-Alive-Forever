"""Independent same-run output, physical metrics, recovery identity and replay audit."""
from common import *
from policy_repair import read
import hashlib

torch.set_num_threads(1);selected=json.loads((HERE/'SELECTION.json').read_text());checkpoint=Path(selected['checkpoint']);actor=read(checkpoint)
records=[];replays=[];identities={}
for group in (selected['group'],'HELD','REPLAY'):
 for folder in sorted((HERE/group).glob('*_*')):
  if not (folder/'RESULT.json').exists():continue
  r=json.loads((folder/'RESULT.json').read_text());assert r['teacher_calls']==0
  assert sha(folder/'TRACE.npz')==r['trace_sha256'] and r['weight_sha256']==sha(checkpoint)
  meta=json.loads((folder/'INPUTS.json').read_text());verify_sources(meta['source_sha256']);identities.update(meta['source_sha256'])
  with np.load(folder/'TRACE.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
  predictions=[];pads=[]
  for x,phase in zip(a['x'],a['phase']):
   pred,pad,_=actor.act(x,phase);predictions.append(pred);pads.append(pad)
  assert np.array_equal(predictions,a['applied'])
  assert np.array_equal(np.repeat(predictions,10,axis=0),a['control']) and np.array_equal(np.repeat(pads,10,axis=0),a['pads'])
  m=mujoco.MjModel.from_binary_path(str(folder/'BODY.mjb'));d=mujoco.MjData(m)
  feet=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f'nmf/{leg}_tarsus5') for leg in ('lf','lm','lh','rf','rm','rh')]
  initial=a['initial_position'];pos=a['position'];j=np.flatnonzero(m.jnt_limited);q=a['qpos'][:,m.jnt_qposadr[j]]
  excess=np.maximum(0,np.maximum(m.jnt_range[j,0]-q,q-m.jnt_range[j,1]))
  rms=float(np.sqrt(np.mean(np.sum((np.diff(pos[-2000:,:2],axis=0)/DT)**2,axis=1))))
  weight=float(m.body_mass.sum()*abs(m.opt.gravity[2]))
  gates=dict(completed=len(pos)==17000,finite=all(np.isfinite(v).all() for v in a.values()),
    no_warning=r['gates']['no_warning'],upright=bool(a['upright'].min()>=.8),height=bool(pos[:,2].min()>=.7*initial[2]),
    nonfoot=bool(a['nonfoot'].max()/weight<=.2),joint_limit=bool(excess.max()<=.02),
    target_limits=bool((a['control']>=actor.minimum.numpy()-1e-12).all() and (a['control']<=actor.maximum.numpy()+1e-12).all()),stand=rms<=.30)
  if r['height']:gates.update(block_contact=bool(a['block_force'].max()>1e-6),cross=bool(pos[-1,0]>=3.15))
  else:gates['advance']=bool(pos[-1,0]-initial[0]>=2.)
  assert gates==r['gates'] and bool(all(gates.values()))==r['pass_']
  assert rms==r['stand_rms'] and float(excess.max())==r['joint_excess']
  d.qpos[:]=a['qpos'][-1];mujoco.mj_kinematics(m,d);feet_cross=bool((d.xpos[feet,0]>2.65).all()) if r['height'] else True
  peak=np.zeros(6)
  for q in a['qpos'][13000:]:d.qpos[:]=q;mujoco.mj_kinematics(m,d);peak=np.maximum(peak,d.xpos[feet,2])
  records.append(dict(group=group,folder=str(folder),pass_=r['pass_'] and feet_cross,original_pass=r['pass_'],feet_cross=feet_cross,
   stop_peak_foot_z=peak.tolist(),stand_rms=rms,joint_excess=float(excess.max()),dx=r['dx'],height=r['height'],seed=r['seed'],
   teacher_calls=0,network_outputs_exact=True,nonfoot_peak=a['nonfoot'].max()/weight))
  if group=='REPLAY':
   original=HERE/selected['group']/folder.name
   with np.load(original/'TRACE.npz',allow_pickle=False) as z:
    checks={k:np.array_equal(z[k],a[k]) for k in a};assert all(checks.values())
   replays.append(dict(original=str(original),replay=str(folder),arrays_exact=checks))
  original_body=PARENT/'HELD'/('flat_1151' if not r['height'] else 'block015_1151' if r['height']==.15 else 'block030_1151')/'BODY.mjb'
  assert sha(original_body)==sha(folder/'BODY.mjb'),'compiled body changed'

def same(a,b):
 if isinstance(a,torch.Tensor):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b
load=lambda p:torch.load(p,map_location='cpu',weights_only=True)
parent=load(HERE/'ROUTING_FIXED.pt');start=load(HERE/'FIT_REPAIR/epoch000.pt')
assert all(same(parent[k],start[k]) for k in ('state','optimizer','rng','torch_rng','update'))
meta=json.loads((HERE/'FIT_REPAIR/INPUTS.json').read_text());verify_sources(meta['source_sha256']);identities.update(meta['source_sha256'])
logs=[json.loads(s) for s in (HERE/'FIT_REPAIR/UPDATES.jsonl').read_text().splitlines()]
assert len(logs)==json.loads((HERE/'FIT_REPAIR/RESULT.json').read_text())['actual_updates']
for row in logs:
 assert sha(row['checkpoint'])==row['checkpoint_sha256']
 weights=load(row['checkpoint']);assert weights['update']==row['update']
 h=hashlib.sha256()
 for name,value in weights['state'].items():
  if name.startswith('reflex.'):h.update(name.encode());h.update(value.numpy().tobytes())
 assert h.hexdigest()==row['parameter_sha256']
assert len({r['parameter_sha256'] for r in logs})==len(logs)
end=load(HERE/'FIT_REPAIR/epoch040.pt')
for k in parent['state']:
 if not k.startswith('reflex.'):assert torch.equal(parent['state'][k],end['state'][k])
assert json.loads((HERE/'FIT_REPAIR/GRADIENT_CHECK.json').read_text())['pass_']
write(HERE/'INDEPENDENT.json',dict(records=records,replays=replays,actual_updates=len(logs),
    optimizer_rng_resume_exact=True,all_update_checkpoints_verified=True,compiled_body_byte_identical=True,
    source_sha256=identities,learning_changes='Experimental sensory fine-tune only; rejected for physical regressions. Selected routing-only checkpoint has unchanged sensory parameters.',new_physical_steps=0))
print('Independent audit completed',len(records),'episodes,',len(logs),'updates,',len(replays),'exact replays')
