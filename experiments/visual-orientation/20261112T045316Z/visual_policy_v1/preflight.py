"""Actual fullbatch RGB/scan/gradient/Adam/RSS checks, parent execution only."""
from copy import deepcopy
import resource,time,gc
from data import *
from model import VisualPolicy,make,production,objective
def main():
 cpu1();start=time.monotonic()
 if not (HERE/'DATASET.json').exists():
  from generate_data import main as generate
  generate()
 if (HERE/'FROZEN.json').exists():raise FileExistsError('already frozen; no blindrerun')
 rgb,manifest=library();d=arrays('train');checks={};errors={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('visual preflight '+k)
 def close(k,a,b):
  err=float((a-b).abs().max());errors[k]=err;check(k,err<=1e-10)
 check('actual112RGB_library',rgb.shape==(112,2,32,32,3));check('strict_motor_rawSHA_no_load',sha(MOTOR)==MOTOR_SHA)
 release=d['capture_time'][d['used_capture_index']]+.020
 check('exact_frame_delay_release',np.max(np.abs(release-np.arange(1920)//4*.020))<1e-12 and np.all(release<=d['decision_time']+1e-12))
 check('capture_atleast20ms_old',np.all(d['capture_time'][d['used_capture_index']]<=d['decision_time']-.020+1e-12))
 check('four_decisions_share_frame',np.array_equal(d['used_capture_index'],np.repeat(np.arange(480),4)))
 check('preroll_actual_off',all(manifest['rows'][int(i)]['stimulus']=='off' for i in d['frame_ids'][0]))
 table=pair_table();check('12pairs24unique',sorted(v for pair in table for v in pair)==list(range(24)))
 check('all_weights_per_episode1',np.max(abs(d['weights'].sum(0)-1))<1e-12)
 for a,b in table[:8]:check('paired_target_sign_'+str(a),np.array_equal(d['target'][:,a],-d['target'][:,b]))
 torch.manual_seed(20261112);m=VisualPolicy();a=deepcopy(m);b=deepcopy(m);x,y,w=batch(rgb,d,expand([0,4]))
 pa,la,ha=production(a,x);pb,lb,hb=make(b)(x);close('full1920_turn',pa,pb);close('full1920_latent',la,lb);close('full1920_memory',ha,hb)
 loss1=objective(pa,la,y,w);loss2=objective(pb,lb,y,w);close('actual_loss',loss1,loss2)
 pairmean=(objective(pb[:,:2],lb[:,:2],y[:,:2],w[:,:2])+objective(pb[:,2:],lb[:,2:],y[:,2:],w[:,2:]))/2.;close('batch4_objective_equals_mean2pairs',loss2,pairmean);pairgrads=torch.autograd.grad(pairmean,tuple(b.parameters()),retain_graph=True)
 loss1.backward();loss2.backward()
 for (name,p),gradient in zip(b.named_parameters(),pairgrads):close('batch4_gradient_equals_mean2pairs_'+name,p.grad,gradient)
 for (name,p),(other,q) in zip(a.named_parameters(),b.named_parameters()):
  check('same_parameter_order_'+name,name==other);close('gradient_'+name,p.grad,q.grad);check('finite_gradient_'+name,torch.isfinite(p.grad).all())
 check('CNN_receives_fullhistory_gradient',a.conv1.weight.grad.abs().max()>0 and a.conv2.weight.grad.abs().max()>0)
 oa=torch.optim.Adam(a.parameters(),lr=1e-4);ob=torch.optim.Adam(b.parameters(),lr=1e-4);torch.nn.utils.clip_grad_norm_(a.parameters(),1.);torch.nn.utils.clip_grad_norm_(b.parameters(),1.);oa.step();ob.step()
 for k,v in a.state_dict().items():close('Adam_model_'+k,v,b.state_dict()[k])
 for i,(u,v) in enumerate(zip(oa.state.values(),ob.state.values())):
  for k in u:close('Adam_moment_'+str(i)+'_'+k,u[k],v[k])
 # Both full batch graphs coexist above: peak RSS is deliberately conservative.
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.;check('fullbatch_peak_RSS_le1800MiB',rss<=1800.)
 for path in (HERE/'train.npz',HERE/'validation.npz',HERE/'controls.npz'):
  meta=read(HERE/'DATASET.json')['splits'][path.stem];check('schedule_hash_'+path.stem,sha(path)==meta['sha256'])
 hashes={str(p):sha(p) for p in sources()+[HERE/'DATASET.json',HERE/'train.npz',HERE/'validation.npz',HERE/'controls.npz']};hashes.update(manifest['source_hashes']);verify(hashes)
 save(HERE/'FROZEN.json',dict(kind=FORMAT,source_hashes=hashes,before_any_training_update=True,physics_steps=0))
 save(HERE/'PREFLIGHT.json',dict(all_passed=True,checks=checks,max_errors=errors,tolerance=1e-10,peak_RSS_MiB=rss,fullbatch=4,fullhistory1920=True,temporary_unitAdam_steps=1,training_updates=0,Core_constructions=0,motor_model_loads=0,seconds=time.monotonic()-start,freeze_sha256=sha(HERE/'FROZEN.json')))
 print(json.dumps(dict(all_passed=True,checks=len(checks),max_error=max(errors.values()),peak_RSS_MiB=rss,seconds=time.monotonic()-start)))
if __name__=='__main__':main()
