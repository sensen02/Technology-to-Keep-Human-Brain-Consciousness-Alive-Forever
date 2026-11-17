"""Actual pair-array, objective and8gradient-equivalence checks; no physics."""
from copy import deepcopy
from common import *
from data import arrays,pair_table,expand_pairs,training_meta,training_data
from engine import make,objective

def extra_checks(model):
 checks={};errors={};oldarrays=arrays(OLD/'train.npz');newarrays=arrays(HERE/'train.npz');table=pair_table();meta=training_meta()
 for name in oldarrays:checks['unchanged_train_'+name]=np.array_equal(oldarrays[name],newarrays[name])
 checks['twelvepairs_cover24_once']=len(table)==12 and sorted(e for pair in table for e in pair)==list(range(24))
 for j,(a,b) in enumerate(table):
  checks['pair'+str(j)+'_samevariant']=meta[a]['id']==meta[b]['id'] and all(meta[a][k]==meta[b][k] for k in ('amplitude','on_s','off_s','switch_s','seed'))
  if j<8:
   checks['pair'+str(j)+'_receptors_mirrored_exact']=np.array_equal(newarrays['receptors'][:,a],newarrays['receptors'][:,b,::-1])
   checks['pair'+str(j)+'_targets_opposite_exact']=np.array_equal(newarrays['targets'][:,a],-newarrays['targets'][:,b])
   checks['pair'+str(j)+'_weights_exact']=np.array_equal(newarrays['weights'][:,a],newarrays['weights'][:,b])
  else:checks['neutralpair'+str(j)+'_zero_targets']=np.max(np.abs(newarrays['targets'][:,[a,b]]))==0
 # Mathematical inclusion count over all unordered choices of2pairs, no alteration of expectedloss.
 counts=np.zeros(24,dtype=int)
 for p in range(12):
  for q in range(p+1,12):counts[expand_pairs([p,q],table)]+=1
 checks['episode_marginal_inclusion_uniform']=np.all(counts==11) and abs(11/66-4/24)<1e-15
 gen=np.random.Generator(np.random.PCG64(20261111));schedule=[]
 for _ in range(500):
  ids=gen.choice(12,2,replace=False);expanded=expand_pairs(ids,table);schedule.append(ids.tolist());assert len(set(ids.tolist()))==2 and len(set(expanded.tolist()))==4
 checks['actual500_schedule_pair_complete_fourunique']=len(schedule)==500
 data=training_data();idx=expand_pairs([0,4],table);a=deepcopy(model);b=deepcopy(model)
 out,latent,_=make(a)(data['receptors'][:,idx]);whole=objective(out,latent,data['targets'][:,idx],data['weights'][:,idx]);whole.backward();parts=[]
 for pairid in (0,4):
  ix=table[pairid];out,latent,_=make(b)(data['receptors'][:,ix]);value=objective(out,latent,data['targets'][:,ix],data['weights'][:,ix])/2;parts.append(float(value.detach()));value.backward()
 err=abs(float(whole.detach())-sum(parts));errors['batch4_loss_vs_meanof2pairloss']=err;checks['batch4_loss_vs_meanof2pairloss']=err<=1e-12
 for (name,x),(other,y) in zip(a.named_parameters(),b.named_parameters()):
  assert name==other;err=float((x.grad-y.grad).abs().max());errors['pairgradient_'+name]=err;checks['pairgradient_'+name]=bool(err<=1e-12 and torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all())
 if not all(checks.values()):raise ValueError('pairpreflightfailed '+str({k:v for k,v in checks.items() if not v}))
 save(HERE/'PAIR_PREFLIGHT.json',dict(all_passed=True,checks=checks,max_errors=errors,tolerance=1e-12,pair_table=table,neutral_pairs_not_mirror_inputs=True,scope='Exactoldarrays/mapping/unbiasedinclusion/pairobjectivegradient equivalence; NO claim that paired gradients are negatives or bias is provedcause',Core_constructions=0,optimizer_updates=0))
 return {**checks,'all_pair_gradient_errors_tol1e12':max(errors.values())<=1e-12}
