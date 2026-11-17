"""Actual reflection/inputidentity/pairloss-preservation checks; no physics."""
from data import *
def extra_checks(m,parent):
 rgb,manifest=library();d=arrays('train');checks={};before=hashlib.sha256(rgb.tobytes()).hexdigest()
 with np.load(OLD/'train.npz',allow_pickle=False) as z:
  for k in z.files:checks['original36exact_'+k]=np.array_equal(d[k][:,:36],z[k]) if k in ('frame_ids','target','weights') else np.array_equal(d[k],z[k])
 checks['strict69cinitial']=exact(m.state_dict(),parent['state_dict']);checks['same14params']=len(list(m.parameters()))==14 and m.identity()==parent['identity'];checks['mirrorflags36plus36']=np.array_equal(d['mirror_flags'],np.r_[np.zeros(36,bool),np.ones(36,bool)])
 checks['sameframeIDs']=np.array_equal(d['frame_ids'][:,:36],d['frame_ids'][:,36:]);checks['oppositetargets']=np.array_equal(d['target'][:,:36],-d['target'][:,36:]);checks['sameweights']=np.array_equal(d['weights'][:,:36],d['weights'][:,36:])
 table=pair_table();checks['36pairs72once']=len(table)==36 and sorted(i for p in table for i in p)==list(range(72));count=np.zeros(72,dtype=int)
 for a in range(36):
  for b in range(a+1,36):count[expand([a,b])]+=1
 checks['marginal1of18']=np.all(count==35) and abs(35/630-1/18)<1e-15
 # Each old originalsequence reflection is a digital tensor operation, not renderer exposure.
 x,y,w=batch(rgb,d,[8,44]);original=torch.from_numpy(rgb[d['frame_ids'][:,[8]]]);checkmirror=reflect(original)
 checks['actual_batch_original_RGB_exact']=torch.equal(x[:,0:1],original);checks['actual_batch_mirror_eye_and_width_exact']=torch.equal(x[:,1:2],checkmirror);checks['R_squared_exact']=torch.equal(reflect(checkmirror),original)
 checks['targetneg_weightssame']=torch.equal(y[:,0],-y[:,1]) and torch.equal(w[:,0],w[:,1]);zero=np.max(abs(d['target'][:,:36]),axis=0)==0;checks['zero_labels_remain_zero']=np.max(abs(d['target'][:,36:][:,zero]))==0
 checks['raw_library_array_unchanged']=hashlib.sha256(rgb.tobytes()).hexdigest()==before;checks['raw_exposure_count56']=len(rgb)==56 and manifest['digital_augmented_exposures']==0
 for name in ('validation','controls'):
  a=arrays(name);configs={manifest['rows'][int(i)]['config_id'] for i in a['frame_ids'].ravel()};checks[name+'_newactualV6V7only']=configs=={'V6','V7'} and not np.any(a['mirror_flags'])
 if not all(bool(v) for v in checks.values()):raise ValueError('mirrorpreflight '+str({k:bool(v) for k,v in checks.items() if not bool(v)}))
 return {k:bool(v) for k,v in checks.items()}
