"""Exact old720historyandpointweights; explicitscaledobjective meaning andlineage."""
from data import *
def extra_checks(m,parent):
 rgb,manifest=library();d=arrays('train');meta=read(HERE/'DATASET.json')['splits']['train']['rows'];checks={}
 with np.load(OLD/'train.npz',allow_pickle=False) as z:
  for key in z.files:checks['original720_bitexact_'+key]=np.array_equal(d[key],z[key])
 expected=np.array([4. if r['motif'] in ('empty','center','occluder_only') else 1. for r in meta]);checks['scale4_neutral240_scale1_other480']=np.array_equal(d['episode_scale'],expected) and np.sum(expected==4)==240 and np.sum(expected==1)==480
 checks['pointweights_still_sum1']=np.max(abs(d['weights'].sum(0)-1))<1e-12;checks['strict812finitialallweights']=exact(m.state_dict(),parent['state_dict']);checks['same14paramarchitecture']=len(list(m.parameters()))==14 and m.identity()==parent['identity'];checks['360pairs720once']=len(pair_table())==360 and sorted(i for p in pair_table() for i in p)==list(range(720))
 checks['mixedscale_preflight_pair0_160']=d['episode_scale'][expand([0,160])].tolist()==[1.,1.,4.,4.]
 # Confirm scaling changes expected neutralobjective; never renormalize multipliedpointweights.
 checks['neutral_objective_mass_explicit']=float(expected.sum())==1440. and float(expected[expected==4].sum())==960.
 lookup={r['frame_id']:r for r in manifest['rows']}
 for name in ('validation','controls'):
  a=arrays(name);configset={lookup[int(i)]['config_id'] for i in a['frame_ids'].ravel()};checks[name+'_only_actual_newV10V11']=configset=={'V10','V11'} and np.all(a['episode_scale']==1)
 if not all(bool(v) for v in checks.values()):raise ValueError('neutralobjectivepreflight '+str({k:bool(v) for k,v in checks.items() if not bool(v)}))
 return {k:bool(v) for k,v in checks.items()}
