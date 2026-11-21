from scope import *
from metrics_v2 import compare
actor=read(WEIGHT);torch.set_num_threads(1);rows=[];replays=[];sources={}
for f in ('stop60','stop74a','stop74b'):
 cmd=HERE/'COMMANDS'/f'{f}_7';meta=json.loads((cmd/'RESULT.json').read_text());verify_sources(json.loads((cmd/'INPUTS.json').read_text())['source_sha256'])
 with np.load(cmd/'TRACE.npz',allow_pickle=False) as z:
  maximum=float(abs(z['command'][meta['off']:]).max());ids=[i for i,p in enumerate(meta['ports']) if p['group']=='forward'];hold_arrived=int(z['arrived'][meta['action']+2500:meta['off']-500,ids].sum());assert hold_arrived>0
 pair=[];stats=[]
 for mode in ('teacher','student'):
  folder=HERE/'SEQUENCED'/f'{f}_7_{mode}';r=json.loads((folder/'RESULT.json').read_text());assert not any(r['warning']);identity=json.loads((folder/'INPUTS.json').read_text())['source_sha256'];verify_sources(identity);sources.update(identity)
  assert sha(folder/'TRACE.npz')==r['trace_sha256']
  with np.load(folder/'TRACE.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
  p=a['position'][meta['off']:,:2];distance=float(np.linalg.norm(p-p[0],axis=1).max());stats.append(dict(mode=mode,max_post_forward_off_command=maximum,max_post_forward_off_distance=distance))
  if mode=='student':
   assert r['teacher_calls']==0
   for i,(x,phase) in enumerate(zip(a['x'],a['phase'])):
    q,pad,_=actor.act(x,phase);assert np.array_equal(q,a['applied'][i]) and np.array_equal(pad,a['pads'][10*i])
   assert np.array_equal(np.repeat(a['applied'],10,axis=0),a['control'])
   replay=HERE/'SEQUENCE_REPLAY'/f'{f}_7_student'
   with np.load(replay/'TRACE.npz',allow_pickle=False) as z:checks={k:np.array_equal(z[k],v) for k,v in a.items()};assert all(checks.values())
   replays.append(dict(family=f,arrays_exact=checks))
  pair.append(r)
 assert sha(HERE/'SEQUENCED'/f'{f}_7_teacher/BODY.mjb')==sha(HERE/'SEQUENCED'/f'{f}_7_student/BODY.mjb')
 relation=compare(*pair);gates=dict(original_teacher=pair[0]['absolute_pass'],original_student=pair[1]['absolute_pass'],teacher_relative=relation['relative_pass'],post_command=maximum<=.035,post_movement=stats[1]['max_post_forward_off_distance']<=.01)
 rows.append(dict(family=f,gates=gates,pass_=all(gates.values()),teacher=pair[0],student=pair[1],release_stats=stats,forward_arrived_hold=hold_arrived))
catalog=BASE/'20261117T051249Z-bidirectional/CATALOG.json'
reference=json.loads((HERE/'COMMANDS/stop60_2/INPUTS.json').read_text())['source_sha256'][str(catalog)]
assert sha(catalog)==reference;sources[str(catalog)]=reference
write(HERE/'SEQUENCE_ACCEPTANCE.json',dict(rows=rows,all_pass=all(r['pass_'] for r in rows),replays=replays,source_sha256=sources,new_physical_steps=180000,new_neural_ticks=60000,training_updates=0))
print('release sequence verified',[(r['family'],r['pass_'],r['release_stats'][1]['max_post_forward_off_distance']) for r in rows])
