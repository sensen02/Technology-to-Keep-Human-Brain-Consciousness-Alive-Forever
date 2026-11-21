"""Independent saved-output, physical metric, learning and replay audit."""
from shared import *
from policy_plus import read,migrate

torch.set_num_threads(1)
checkpoint=HERE/'FIT2/epoch030.pt';actor=read(checkpoint)
records=[];identities={};replays=[]
for group in ('DEV2','HELD'):
 for folder in sorted((HERE/group).glob('*_*')):
    if not folder.is_dir():continue
    r=json.loads((folder/'RESULT.json').read_text())
    assert r['teacher_calls']==0 and r['weight_sha256']==sha(checkpoint)
    inputs=json.loads((folder/'INPUTS.json').read_text())
    for path,digest in inputs['source_sha256'].items():assert sha(path)==digest;identities[path]=digest
    m=mujoco.MjModel.from_binary_path(str(folder/'BODY.mjb'));d=mujoco.MjData(m)
    thorax=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'nmf/c_thorax')
    feet=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f'nmf/{leg}_tarsus5') for leg in ('lf','lm','lh','rf','rm','rh')]
    with np.load(folder/'TRACE.npz',allow_pickle=False) as z:
        a={k:z[k].copy() for k in z.files}
    predictions=[];pads=[]
    for x,phase in zip(a['x'],a['phase']):
        q,p,_=actor.act(x,phase);predictions.append(q);pads.append(p)
    predictions=np.array(predictions);pads=np.array(pads)
    assert np.array_equal(predictions,a['applied'])
    assert np.array_equal(np.repeat(predictions,HOLD,axis=0),a['control'])
    assert np.array_equal(np.repeat(pads,HOLD,axis=0),a['pads'])
    j=np.flatnonzero(m.jnt_limited);q=a['qpos'][:,m.jnt_qposadr[j]]
    ex=np.maximum(0,np.maximum(m.jnt_range[j,0]-q,q-m.jnt_range[j,1]))
    index=np.unravel_index(np.argmax(ex),ex.shape)
    rms=float(np.sqrt(np.mean(np.sum((np.diff(a['position'][-2000:,:2],axis=0)/DT)**2,axis=1))))
    dx=float(a['position'][-1,0]-a['initial_position'][0])
    assert dx==r['dx'] and rms==r['stand_rms'] and float(ex.max())==r['joint_excess']
    assert a['position'][:,2].min()==r['min_height'] and a['upright'].min()==r['min_upright']
    weight=m.body_mass.sum()*abs(m.opt.gravity[2])
    assert a['nonfoot'].max()/weight==r['nonfoot_peak_bodyweight']
    assert bool(ex.max()<=.02)==r['gates']['joint_limit']
    assert bool(rms<=.30)==r['gates']['stand']
    assert bool(a['position'][:,2].min()>=.7*a['initial_position'][2])==r['gates']['height']
    assert bool(a['upright'].min()>=.8)==r['gates']['upright']
    assert bool(a['nonfoot'].max()/weight<=.2)==r['gates']['nonfoot']
    assert len(a['qpos'])==17000 and all(np.isfinite(v).all() for v in a.values())
    assert (a['control']>=actor.minimum.numpy()-1e-12).all() and (a['control']<=actor.maximum.numpy()+1e-12).all()
    if r['height']:
        assert bool(a['block_force'].max()>1e-6)==r['gates']['block_contact']
        assert bool(a['position'][-1,0]>=3.15)==r['gates']['cross']
    else:assert bool(dx>=2)==r['gates']['advance']
    d.qpos[:]=a['qpos'][-1];mujoco.mj_kinematics(m,d)
    feet_cross=bool((d.xpos[feet,0]>2.65).all()) if r['height'] else True
    record=dict(folder=str(folder),pass_original=r['pass_'],complete_cross_and_original=r['pass_'] and feet_cross,
       feet_cross=feet_cross,final_feet_x=d.xpos[feet,0].tolist(),stand_rms=rms,dx=dx,joint_excess=float(ex.max()),
       worst_joint=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j[index[1]]),worst_joint_time_s=(index[0]+1)*DT,
       network_outputs_exact=True,teacher_calls=0)
    records.append(record)
for path in sorted((HERE/'REPLAY').glob('*/TRACE.npz')):
    original=HERE/'HELD'/path.parent.name/'TRACE.npz'
    with np.load(original,allow_pickle=False) as a,np.load(path,allow_pickle=False) as b:
        assert set(a.files)==set(b.files)
        checks={key:np.array_equal(a[key],b[key]) for key in a.files};assert all(checks.values())
    replays.append(dict(original=str(original),replay=str(path),arrays_exact=checks))
assert len(replays)==3
def same(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b
load=lambda p:torch.load(HERE/p,map_location='cpu',weights_only=True)
p0=load('FIT0/epoch030.pt');s1=load('FIT1/epoch030.pt')
assert same(p0,s1),'resume state must exactly match parent including Adam/RNG'
p1=load('FIT1/epoch050.pt');s2=load('FIT2/epoch000.pt')
expected=migrate(p1['state']).state_dict()
for name,value in expected.items():
    if name in ('center','scale'):
        assert torch.equal(value[:230],s2['state'][name][:230]) and torch.equal(value[254:],s2['state'][name][254:])
    else:assert torch.equal(value,s2['state'][name])
assert not s2['optimizer']['state'] and s2['update']==0
total_updates=0;learning=[]
for name,last in (('FIT0','epoch060.pt'),('FIT1','epoch070.pt'),('FIT2','epoch040.pt')):
    out=HERE/name;meta=json.loads((out/'RESULT.json').read_text());logs=[json.loads(s) for s in (out/'UPDATES.jsonl').read_text().splitlines()]
    assert len(logs)==meta['actual_updates']
    assert len({r['parameter_sha256'] for r in logs})==len(logs)
    for path,digest in json.loads((out/'INPUTS.json').read_text())['source_sha256'].items():assert sha(path)==digest;identities[path]=digest
    start=load(f'{name}/'+('epoch030.pt' if name=='FIT1' else 'epoch000.pt'));end=load(f'{name}/{last}')
    assert any(not torch.equal(v,end['state'][k]) for k,v in start['state'].items() if k.startswith('reflex.'))
    for key in ('fourier','order','neutral','synergy','pad_weights','minimum','maximum'):
        assert torch.equal(start['state'][key],end['state'][key])
    total_updates+=len(logs);learning.append(dict(stage=name,actual_updates=len(logs),frames=meta['frames']))
for name in ('FIT0','FIT2'):assert json.loads((HERE/name/'GRADIENT_CHECK.json').read_text())['pass_']
write(HERE/'INDEPENDENT.json',dict(records=records,replays=replays,learning=learning,total_updates=total_updates,
  source_sha256=identities,resume_state_optimizer_rng_exact=True,
  resume_parent_binding='FIT1 parent not prehashed in INPUTS; checkpoint exact equality verified after run',
  observation_migration_exact=True,new_optimizer_explicitly_reset=True,
  held_pass=sum(r['complete_cross_and_original'] for r in records if '/HELD/' in r['folder']),
  dev_pass=sum(r['complete_cross_and_original'] for r in records if '/DEV2/' in r['folder']),
  limitations='Limited body course with learned readouts and external local CPG; not strict connectome control. No original 69-channel regression or new brain-neck simulation.',
  new_physical_steps=0))
print('independent saved-output/metrics/learning/replay audit complete',total_updates)
