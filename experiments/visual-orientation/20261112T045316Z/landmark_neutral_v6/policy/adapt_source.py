"""One frozenv1 adapter; explicitseparateepisodescale loss and1000budget, not runtimechange."""
from pathlib import Path
import re
HERE=Path(__file__).resolve().parent;V1=HERE.parents[1]/'visual_policy_v1'
def once(src,old,new):
 if src.count(old)!=1:raise ValueError('neutral source shape '+old)
 return src.replace(old,new)
def build(name):
 src=(V1/name).read_text()
 if name in ('train.py','verify_saved.py'):
  src=re.sub(r'\b(501|500|250)\b',lambda m:{'501':'1001','500':'1000','250':'500'}[m.group(0)],src);src=src.replace('checkpoint_000500.pt','checkpoint_001000.pt')
 if name=='preflight.py':
  src=once(src,"check('actual112RGB_library',rgb.shape==(112,2,32,32,3))","check('actual672RGB_library',rgb.shape==(672,2,32,32,3))")
  src=once(src,"check('12pairs24unique',sorted(v for pair in table for v in pair)==list(range(24)))","check('360pairs720unique',sorted(v for pair in table for v in pair)==list(range(720)))")
  src=once(src,'torch.manual_seed(20261112);m=VisualPolicy();a=deepcopy(m);','torch.manual_seed(20261112);m,parent=initialize();a=deepcopy(m);')
  src=once(src,'x,y,w=batch(rgb,d,expand([0,4]))','x,y,w,s=batch(rgb,d,expand([0,160]))')
  src=once(src,'loss1=objective(pa,la,y,w);loss2=objective(pb,lb,y,w)','loss1=objective(pa,la,y,w,s);loss2=objective(pb,lb,y,w,s)')
  src=once(src,'objective(pb[:,:2],lb[:,:2],y[:,:2],w[:,:2])+objective(pb[:,2:],lb[:,2:],y[:,2:],w[:,2:])','objective(pb[:,:2],lb[:,:2],y[:,:2],w[:,:2],s[:2])+objective(pb[:,2:],lb[:,2:],y[:,2:],w[:,2:],s[2:])')
  marker="  pairmean="
  # Independent explicitpersequence expression checks scale before mean, no cancellation.
  manual="manual_scaled=((((pb-y).square()+.01*(lb-torch.atanh(y.clamp(-.95,.95))).square())*w).sum(0)*s).mean();close('manual_scaled_objective',loss2,manual_scaled);"
  src=once(src,' pairmean=', ' '+manual+'pairmean=')
  key=' hashes={str(p):sha(p) for p in sources()+'
  src=once(src,key,' from extra_checks import extra_checks\n checks.update(extra_checks(m,parent))\n'+key)
 elif name=='train.py':
  src=once(src,'torch.manual_seed(20261112);m=VisualPolicy();opt=','torch.manual_seed(20261112);m,parent=initialize();opt=')
  src=once(src,'payload=dict(format_version=FORMAT,','payload=dict(format_version=FORMAT,parent_sha256=PARENT_SHA,pair_coverage=pair_coverage(pairs),objective_group_scale="neutral4_other1_separate",')
  src=once(src,'gen.choice(12,2,replace=False)','gen.choice(360,2,replace=False)')
  src=once(src,'ix=expand([0,4]);x,y,w=batch(rgb,data,ix)','ix=expand([0,160]);x,y,w,s=batch(rgb,data,ix)')
  src=once(src,'batches.append(ix);x,y,w=batch(rgb,data,ix)','batches.append(ix);x,y,w,s=batch(rgb,data,ix)')
  src=once(src,'objective(turn,latent,y,w)','objective(turn,latent,y,w,s)')
  src=once(src,'del turn,latent,h,loss,x,y,w','del turn,latent,h,loss,x,y,w,s')
  src=once(src,"status='INITIAL_RANDOM_VISUAL_POLICY_NOT_BEHAVIOR_QUALIFIED'","status='EXACT_PARENT812f_NEW_NEUTRAL_OBJECTIVE_UNQUALIFIED'")
  src=once(src,"status='TRAINED_NOT_VALIDATED_OR_PHYSICAL',","status='TRAINED_NOT_VALIDATED_OR_PHYSICAL',pair_coverage=pair_coverage(pairs),objective_group_scale='neutral4_other1_separate',")
 elif name=='verify_saved.py':
  src=once(src,'x,_,_=batch(rgb,d,p[\'replay_indices\'])','x,_,_,_=batch(rgb,d,p[\'replay_indices\'])')
  src=once(src,'generator.choice(12,2,replace=False)','generator.choice(360,2,replace=False)')
  src=once(src,"checks['pair_table_exact']=p['pair_table']==pair_table();","checks['pair_table_exact']=p['pair_table']==pair_table();checks['actual_pair_coverage_counts_exact']=p['pair_coverage']==pair_coverage(ids);")
 elif name=='evaluate.py':
  src=src.replace('checkpoint_000500.pt','checkpoint_001000.pt').replace('VERIFY_000500.json','VERIFY_001000.json').replace('final500','final1000');src=once(src,'x,_,_=batch(rgb,d,[j])','x,_,_,_=batch(rgb,d,[j])')
 elif name=='audit_validation.py':src=once(src,'x,_,_=batch(rgb,d,[j])','x,_,_,_=batch(rgb,d,[j])')
 elif name=='run_candidate.py':src=once(src,"('verify_saved.py',['--step','250'],'VERIFY_250'),('verify_saved.py',['--step','500'],'VERIFY_500')","('verify_saved.py',['--step','500'],'VERIFY_500'),('verify_saved.py',['--step','1000'],'VERIFY_1000')")
 else:raise ValueError('unregisteredneutralentry')
 return src
def run(name):
 ns={'__name__':'_neutral_frozen_'+name.replace('.','_'),'__file__':str(HERE/name)};exec(compile(build(name),str(HERE/name)+'::explicit_neutral_objective','exec'),ns);ns['main']()
