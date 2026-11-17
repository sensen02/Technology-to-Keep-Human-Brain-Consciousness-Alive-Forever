"""Newdomain/inputreuse/neutralmass checks, actualruntime only."""
from data import *
def extra_checks(m,parent):
 rgb,manifest=library();d=arrays('train');meta=read(HERE/'DATASET.json')['splits']['train']['rows'];checks={}
 with np.load(OLD/'train.npz',allow_pickle=False) as z:
  for k in z.files:checks['old24_bitexact_'+k]=np.array_equal(d[k][:,:24],z[k]) if k in ('frame_ids','target','weights') else np.array_equal(d[k],z[k])
 checks['samef1aallinitialweights']=exact(m.state_dict(),parent['state_dict']);checks['same14parameterarchitecture']=len(list(m.parameters()))==14 and m.identity()==parent['identity'];table=pair_table();checks['18pairs_cover36once']=len(table)==18 and sorted(v for p in table for v in p)==list(range(36))
 counts=np.zeros(36,dtype=int)
 for i in range(18):
  for j in range(i+1,18):counts[expand([i,j])]+=1
 checks['uniform_inclusion1of9']=np.all(counts==17) and abs(17/153-4/36)<1e-15
 for j in range(24,32):
  groups=masks(meta[j]['released_on_s'],meta[j]['released_off_s']);checks['occlusion'+str(j)+'_disjointcover']=np.all(groups.sum(0)==1)
  checks['occlusion'+str(j)+'_fourmass']=np.max(abs(np.asarray([d['weights'][g,j].sum() for g in groups])-np.array([.3,.2,.4,.1])))<1e-12
  ids=d['frame_ids'][d['used_capture_index'],j];scenes=[manifest['rows'][int(i)]['stimulus'] for i in ids];wall=np.array([s.startswith('occluded') for s in scenes]);checks['occlusion'+str(j)+'_realwallafteroff']=wall.any() and np.max(abs(d['target'][wall,j]))==0
 for j in range(32,36):
  scenes=np.array([manifest['rows'][int(i)]['stimulus'] for i in d['frame_ids'][d['used_capture_index'],j]]);wall=scenes=='occluder_only';off=scenes=='off'
  checks['purewall'+str(j)+'_preoff_then_actualwall']=off[0] and wall.any() and np.all(off[:np.flatnonzero(wall)[0]]) and np.all(wall[np.flatnonzero(wall)[0]:])
  checks['purewall'+str(j)+'_equal_scene_mass']=abs(d['weights'][off,j].sum()-.5)<1e-12 and abs(d['weights'][wall,j].sum()-.5)<1e-12
  checks['purewall'+str(j)+'_targetzero']=np.max(abs(d['target'][:,j]))==0
 for a,b in table[12:16]:checks['newpairedtargets_'+str(a)]=np.array_equal(d['target'][:,a],-d['target'][:,b]) and np.array_equal(d['weights'][:,a],d['weights'][:,b])
 if not all(bool(v) for v in checks.values()):raise ValueError('newwallpreflight '+str({k:bool(v) for k,v in checks.items() if not bool(v)}))
 return {k:bool(v) for k,v in checks.items()}
