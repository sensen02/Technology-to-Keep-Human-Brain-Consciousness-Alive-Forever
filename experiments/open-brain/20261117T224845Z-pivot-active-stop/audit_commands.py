from scope import *
rows=[]
for p in sorted((HERE/'COMMANDS').glob('*/TRACE.npz')):
 meta=json.loads((p.parent/'RESULT.json').read_text());assert sha(p)==meta['sha256'];verify_sources(json.loads((p.parent/'INPUTS.json').read_text())['source_sha256'])
 with np.load(p,allow_pickle=False) as z:
  em=z['emitted'];ar=z['arrived'];cmd=z['command'];active=[]
  for i,port in enumerate(meta['ports']):
   if em[:,i].any() or ar[:,i].any():active.append(dict(**port,emitted=int(em[:,i].sum()),arrived=int(ar[:,i].sum())))
  group='walking_steer' if meta['family'].startswith('pivot') else {'stop60':'halt_bluebell','stop74a':'halt_web_a','stop74b':'halt_web_b'}[meta['family']]
  assert any(port['group']==group and port['arrived']>0 for port in active)
  if not meta['family'].startswith('pivot'):
   ids=[i for i,port in enumerate(meta['ports']) if port['group']=='forward'];assert ar[meta['action']+2500:meta['off']-500,ids].sum()>0
 rows.append(dict(family=meta['family'],variant=meta['variant'],period=meta['period'],trace_sha256=meta['sha256'],ports=active,command_min=cmd.min(0).tolist(),command_max=cmd.max(0).tolist()))
write(HERE/'COMMAND_AUDIT.json',dict(rows=rows,new_neural_ticks=0,new_physical_steps=0))
print('Verified actual named DN emission/arrival traces',len(rows))
