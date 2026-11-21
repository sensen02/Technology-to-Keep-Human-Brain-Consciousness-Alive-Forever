from scope import *
cat=json.loads((BASE/'20261117T051249Z-bidirectional/CATALOG.json').read_text())['ports']
rows=[]
for path in sorted((HERE/'COMMANDS').glob('*/TRACE.npz')):
 meta=json.loads((path.parent/'RESULT.json').read_text());assert sha(path)==meta['sha256'];verify_sources(json.loads((path.parent/'INPUTS.json').read_text())['source_sha256'])
 with np.load(path,allow_pickle=False) as z:
  roots=[str(r) for r in z['roots']];em=z['emitted'];ar=z['arrived'];cmd=z['command'];ports=[]
  assert len(cmd)==20000 and np.isfinite(cmd).all()
  for i,root in enumerate(roots):
   if em[:,i].any() or ar[:,i].any():
    p=cat[root];ports.append(dict(root=root,cell_type=p['cell_type'],group=p['group'],side=p['side'],emitted=int(em[:,i].sum()),arrived=int(ar[:,i].sum())))
  if meta['family'].startswith(('arc','stride')):
   group='walking_steer' if meta['family'].startswith('arc') else 'walking_outer_stride'
   side=meta['family'].split('_')[1];assert any(p['group']==group and p['side']==side and p['arrived']>0 for p in ports)
  elif meta['family']=='back':assert any(p['group']=='backward_moonwalker' and p['arrived']>0 for p in ports)
  else:assert any(p['group']=='forward_speed' and p['arrived']>0 for p in ports)
 rows.append(dict(family=meta['family'],variant=meta['variant'],sha256=meta['sha256'],active_ports=ports,command_min=cmd.min(0).tolist(),command_max=cmd.max(0).tolist()))
write(HERE/'COMMAND_SOURCE_AUDIT.json',dict(rows=rows,new_physical_steps=0,new_neural_ticks=0,neck='actual generation20000 ticks per complete command, fixed nonplastic structure/parameters'))
print('Audited',len(rows),'actual postneck command traces')
