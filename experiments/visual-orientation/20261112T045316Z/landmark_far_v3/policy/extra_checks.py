"""Far42 actualRGB/36schedule/neutralmass/lineage checks, runtime only."""
from data import *
def extra_checks(m,parent):
 rgb,manifest=library();d=arrays('train');meta=read(HERE/'DATASET.json')['splits']['train']['rows'];checks={}
 checks['same79dballinitialweights']=exact(m.state_dict(),parent['state_dict']);checks['same14paramarchitecture']=len(list(m.parameters()))==14 and m.identity()==parent['identity'];table=pair_table();checks['18pairs36unique']=len(table)==18 and sorted(v for pair in table for v in pair)==list(range(36))
 count=np.zeros(36,dtype=int)
 for a in range(18):
  for b in range(a+1,18):count[expand([a,b])]+=1
 checks['uniform_episode_inclusion1of9']=np.all(count==17) and abs(17/153-1/9)<1e-15
 for j in list(range(8,16))+list(range(24,32)):
  group=masks(meta[j]['released_on_s'],meta[j]['released_off_s']);checks['fourgroup_cover'+str(j)]=np.all(group.sum(0)==1);checks['fourmass'+str(j)]=np.max(abs(np.array([d['weights'][g,j].sum() for g in group])-np.array([.3,.2,.4,.1])))<1e-12
 for j in range(32,36):
  scenes=np.array([manifest['rows'][int(i)]['stimulus'] for i in d['frame_ids'][d['used_capture_index'],j]]);box=scenes=='occluder_only';off=scenes=='off';checks['boxpreoff'+str(j)]=off[0] and box.any() and np.all(off[:np.flatnonzero(box)[0]]) and np.all(box[np.flatnonzero(box)[0]:]);checks['boxmass'+str(j)]=abs(d['weights'][off,j].sum()-.5)<1e-12 and abs(d['weights'][box,j].sum()-.5)<1e-12 and np.max(abs(d['target'][:,j]))==0
 for a,b in table[:8]+table[12:16]:checks['paired_targetweights'+str(a)]=np.array_equal(d['target'][:,a],-d['target'][:,b]) and np.array_equal(d['weights'][:,a],d['weights'][:,b])
 for name in ('train','validation','controls'):
  a=arrays(name);configs={manifest['rows'][int(i)]['config_id'] for i in a['frame_ids'].ravel()};checks[name+'_only_allowed_configs']=configs<=({'T0','T1','T2','T3'} if name=='train' else {'V4','V5'})
 if not all(bool(v) for v in checks.values()):raise ValueError('far preflight '+str({k:bool(v) for k,v in checks.items() if not bool(v)}))
 return {k:bool(v) for k,v in checks.items()}
