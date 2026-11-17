"""Actualnewposecoverage/masses/720pair bookkeeping checks, no diagnosticpolicyforward."""
from data import *
def extra_checks(m,parent):
 rgb,manifest=library();d=arrays('train');meta=read(HERE/'DATASET.json')['splits']['train']['rows'];checks={}
 checks['strict79dballinitialweights']=exact(m.state_dict(),parent['state_dict']);checks['same14paramarchitecture']=len(list(m.parameters()))==14 and m.identity()==parent['identity'];table=pair_table();checks['360pairs720once']=len(table)==360 and sorted(i for pair in table for i in pair)==list(range(720))
 counts=np.zeros(720,dtype=int)
 for a in range(360):
  for b in range(a+1,360):counts[expand([a,b])]+=1
 checks['equal_episode_inclusion1of180']=np.all(counts==359) and abs(359/(360*359/2)-1/180)<1e-15
 lookup={r['frame_id']:r for r in manifest['rows']}
 for j,row in enumerate(meta):
  expected={(row['config_id'],row['pose_id'])};actual={(lookup[int(i)]['config_id'],lookup[int(i)]['pose_id']) for i in d['frame_ids'][:,j]};checks['configpose_unique_lookup'+str(j)]=actual==expected
  if row['motif'] in ('left_pulse','right_pulse','left_occluded','right_occluded'):
   group=masks(row['released_on_s'],row['released_off_s']);checks['mutuallyexclusive_mass'+str(j)]=np.all(group.sum(0)==1) and np.max(abs(np.array([d['weights'][g,j].sum() for g in group])-np.array([.3,.2,.4,.1])))<1e-12
  if row['motif']=='occluder_only':
   scene=np.array([lookup[int(i)]['stimulus'] for i in d['frame_ids'][d['used_capture_index'],j]]);off=scene=='off';box=scene=='occluder_only';checks['boxpreoff_halfmass'+str(j)]=off[0] and box.any() and abs(d['weights'][off,j].sum()-.5)<1e-12 and abs(d['weights'][box,j].sum()-.5)<1e-12 and np.max(abs(d['target'][:,j]))==0
 for a,b in table[:320]:checks['paired_targetweight'+str(a)]=np.array_equal(d['target'][:,a],-d['target'][:,b]) and np.array_equal(d['weights'][:,a],d['weights'][:,b])
 configs={};poses={}
 for name in ('train','validation','controls'):
  a=arrays(name);rows=read(HERE/'DATASET.json')['splits'][name]['rows'];configs[name]={r['config_id'] for r in rows};poses[name]={r['pose_id'] for r in rows}
 checks['train80_actualconfigposes']=len({(r['config_id'],r['pose_id']) for r in meta})==80 and len(poses['train'])==20 and configs['train']=={'T0','T1','T2','T3'}
 checks['newval8_actualconfigposes']=len({(r['config_id'],r['pose_id']) for r in read(HERE/'DATASET.json')['splits']['validation']['rows']})==8 and configs['validation']=={'V8','V9'} and len(poses['validation'])==4
 if not all(bool(v) for v in checks.values()):raise ValueError('coverage preflight '+str({k:bool(v) for k,v in checks.items() if not bool(v)}))
 return {k:bool(v) for k,v in checks.items()}
