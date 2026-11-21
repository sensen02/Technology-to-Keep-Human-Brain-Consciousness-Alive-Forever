from scope import *
from metrics_v2 import measure,compare,window
records=[];replays=[];sources={};torch.set_num_threads(1);actor=read(WEIGHT)
for group in ('DEV','RATE','HELD','REPLAY'):
 for folder in sorted((HERE/group).glob('*_student')):
  r=json.loads((folder/'RESULT.json').read_text());assert r['teacher_calls']==0 and not any(r['warning']) and r['weight_sha256']==sha(WEIGHT)
  identity=json.loads((folder/'INPUTS.json').read_text())['source_sha256'];verify_sources(identity);sources.update(identity)
  assert sha(folder/'TRACE.npz')==r['trace_sha256']
  with np.load(folder/'TRACE.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
  predictions=[];pads=[]
  for x,p in zip(a['x'],a['phase']):
   q,pad,_=actor.act(x,p);predictions.append(q);pads.append(pad)
  assert np.array_equal(predictions,a['applied']) and np.array_equal(np.repeat(predictions,10,axis=0),a['control']) and np.array_equal(np.repeat(pads,10,axis=0),a['pads'])
  m=mujoco.MjModel.from_binary_path(str(folder/'BODY.mjb'));d=mujoco.MjData(m);command=HERE/'COMMANDS'/f"{r['family']}_{r['variant']}"
  meta=json.loads((command/'RESULT.json').read_text());fresh=measure(a,m,meta)
  for k in ('stand_rms','min_height','min_upright','joint_excess','peak_excursion'):assert fresh[k]==r[k]
  if 'active_yaw_degrees' in r:assert fresh['active_yaw_degrees']==r['active_yaw_degrees'] and fresh['yaw_degrees']==r['yaw_degrees']
  else:assert fresh['active_yaw_degrees']==r['yaw_degrees']
  expected_yaw=np.rad2deg(np.unwrap(a['yaw'])[-1]-np.unwrap(a['yaw'])[meta['start']])
  if r['family'].startswith('pivot'):assert float(expected_yaw)==fresh['yaw_degrees']
  teacher=(HERE/'HELD' if group=='REPLAY' else folder.parent)/folder.name.replace('_student','_teacher')
  t=json.loads((teacher/'RESULT.json').read_text());assert not any(t['warning'])
  assert sha(teacher/'BODY.mjb')==sha(folder/'BODY.mjb')
  ts=json.loads((teacher/'INPUTS.json').read_text())['source_sha256'];assert ts[str(command/'TRACE.npz')]==identity[str(command/'TRACE.npz')]
  pair=compare(t,r);pair.update(group=group,folder=str(folder),outputs_exact=True,body_same=True,commands_same=True,
    original_active_yaw=r['yaw_degrees'],completed_yaw=fresh['yaw_degrees'])
  records.append(pair)
  if group=='REPLAY':
   ref=HERE/'HELD'/folder.name
   with np.load(ref/'TRACE.npz',allow_pickle=False) as z:checks={k:np.array_equal(z[k],v) for k,v in a.items()};assert all(checks.values())
   replays.append(dict(original=str(ref),replay=str(folder),arrays_exact=checks))
  if r['family'].startswith('stop'):
   hold=window(a,meta['action']+2500,meta['off']-500,float(m.body_mass.sum()*abs(m.opt.gravity[2])))
   assert hold==r['hold']
   with np.load(command/'TRACE.npz',allow_pickle=False) as z:
    ids=[i for i,p in enumerate(meta['ports']) if p['group']=='forward'];arrivals=int(z['arrived'][meta['action']+2500:meta['off']-500,ids].sum());assert arrivals==r['forward_arrived_during_hold'] and arrivals>0
   assert r['max_hold_command']<=.035
assert len(replays)==5
write(HERE/'INDEPENDENT.json',dict(records=records,replays=replays,source_sha256=sources,physical_steps=0,neural_ticks=0,training_updates=0,
 note='Old pivot RESULT active angle retained; completed angle recomputed separately. Formal pivot uses completed-angle v2 prospectively.'))
print('Audited',len(records),'student episodes;',len(replays),'exact native replays, no new physics')
