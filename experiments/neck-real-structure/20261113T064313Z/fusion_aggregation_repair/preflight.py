"""Parent full24vs3micro8 loss/gradient/Adam andactualqueue preflight; no body."""
from common import *
def main():
 torch,np=backend()
 if (HERE/'FROZEN.json').exists():raise FileExistsError('Inspect priorpreflight')
 from data import main as prepare,arrays,tensors
 from model import python_rollout,make_scan,objective,accumulated_backward
 from copy import deepcopy
 sentinel=object();prior={k:sys.modules.get(k,sentinel) for k in ('model','common','steering_behavior')};prepare();d=arrays('train');x,u,y,w=tensors(d);torch.manual_seed(20261113);m,parent=initialize();a=deepcopy(m);fixed=frozen_tensors(m);ref=anchors(m);checks={};errors={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('Fusionaggpreflight '+k)
 def close(k,l,r,tol=1e-10):
  e=float((l-r).abs().max().detach());errors[k]=e;check(k,e<=tol)
 check('cross_namespace_sensory_loader_restores_allaliases',all(sys.modules.get(k,sentinel) is v for k,v in prior.items()));check('24full1920_shapes',x.shape==(1920,24,111) and u.shape==(1920,24,2) and y.shape==w.shape==(1920,24,1));check('db25allstateinitialexact',all(torch.equal(v,parent['state_dict'][k]) for k,v in m.state_dict().items()));check('10trainable8frozen',len(trainable(m))==10 and sum(not p.requires_grad for p in m.parameters())==8);check('actual14+6+4_sourcecounts',read(HERE/'DATASET.json')['source_counts']==dict(actual_latest14=14,priorvisual6=6,explicitneuralzero4=4,total24=24));replay=read(HERE/'SENSORY_REPLAY.json');check('visionoldproof_exact_newodor_input_lineage',replay['all_actual_visual_bitexact'] and replay['all_old_actor_receptor_inputs_preserved'] and replay['newodor_output_expected_different_from_old66f'] and replay['frozen_modal_parameters_unchanged']);check('same111maskandnorm',m.allowed_raw_indices.tolist()==MASK and torch.equal(m.observation_mean,parent['state_dict']['observation_mean']) and torch.equal(m.observation_std,parent['state_dict']['observation_std']));check('causaltargetcorrect1920',torch.equal(y[0],torch.zeros_like(y[0])) and torch.equal(y[1:],torch.from_numpy(d['current_desired'][:-1])));check('jointboundonlyFdata',bool(y[:,9:12].abs().max()<=.78));check('explicitaux_neuralzero',bool((u[:,20:]==0).all() and (y[:,20:]==0).all()))
 for j in range(24):check('weightmass_'+str(j),abs(float(w[:,j].sum())-1)<1e-12)
 raw=torch.zeros((1,238),dtype=torch.float64,requires_grad=True);an=m.feedback(m.normalize(raw.index_select(1,m.allowed_raw_indices)));grad=torch.autograd.grad(an.sum(),raw)[0];excluded=sorted(set(range(238))-set(MASK));check('forbiddenraw_zeroJacobian',bool((grad[:,excluded]==0).all()) and 237 in excluded)
 original=module('_fusionagg_parent_compute_arch',OLDROOT/'sensory_fusion_calibration/model.py');parentm=original.NeuralBridge(parent['state_dict']['observation_mean'],parent['state_dict']['observation_std']);parentm.load_state_dict(parent['state_dict'],strict=True)
 with torch.no_grad():
  expect=original.python_rollout(parentm,x[:,:1],u[:,:1]);actual=python_rollout(m,x[:,:1],u[:,:1]);script=make_scan(a)(x[:,:1],u[:,:1])
  for k,e,q,s in zip(('turn','DN','AN','recon','brainhidden'),expect,actual,script):close('db25initial_compute_'+k,e,q,1e-12);close('python_jit_'+k,q,s)
 del expect,actual,script,parentm
 full=python_rollout(m,x,u);loss,terms=objective(full,y,w,m,ref);manual=((full[0]-y).square()*w).sum(0).mean()+.1*m.decoder(x.new_zeros((24,2))).square().mean()+.001*full[1].square().mean()+.001*sum((dict(m.named_parameters())[k]-v).square().sum() for k,v in ref.items())/sum(v.numel() for v in ref.values());close('manualfull24objective',loss,manual);lv=float(loss.detach());loss.backward();fg={k:p.grad.detach().clone() for k,p in trainable(m)};check('all10_fullgradfinite',all(torch.isfinite(v).all() for v in fg.values()));del full,loss,terms,manual
 acc,stats=accumulated_backward(make_scan(a),a,x,u,y,w,ref);close('full24_micro8loss',x.new_tensor(lv),x.new_tensor(acc));norms={}
 for k,p in trainable(a):check('microgradfinite_'+k,p.grad is not None and torch.isfinite(p.grad).all());close('full24_micro8gradient_'+k,fg[k],p.grad);norms[k]=float(p.grad.norm())
 check('frozenparam_gradabsent',all(p.grad is None for model in (m,a) for p in model.parameters() if not p.requires_grad));torch.nn.utils.clip_grad_norm_([v for k,v in trainable(m)],1.);torch.nn.utils.clip_grad_norm_([v for k,v in trainable(a)],1.);o1=torch.optim.Adam([v for k,v in trainable(m)],lr=3e-4);o2=torch.optim.Adam([v for k,v in trainable(a)],lr=3e-4);o1.step();o2.step()
 for (k,p),(other,q) in zip(trainable(m),trainable(a)):
  check('sameparameterorder_'+k,k==other);close('Adamparameter_'+k,p,q)
  for key in ('exp_avg','exp_avg_sq','step'):close('Adamstate_'+key+'_'+k,o1.state[p][key],o2.state[q][key])
 check('three_microbatches_oneAdam',len(o1.state)==len(o2.state)==10 and all(float(v['step'])==1 for opt in (o1,o2) for v in opt.state.values()));check_frozen(m,fixed);check_frozen(a,fixed);check('frozenalltensorsexact',True);del fg,a,o1,o2
 bp,b,digest=binding()
 with torch.no_grad():
  expect=python_rollout(m,x[:,9:10],u[:,9:10]);actual,proof=real_queue_rollout(m,x[:,9:10],u[:,9:10],bp,digest)
  for k,l,r in zip(('turn','DN','AN','recon','brainhidden'),expect,actual):close('actual1msqueue_'+k,l,r)
 check('actualsource9600and3838causal',proof['actual_steps']==9600 and len(proof['neural_consumption_events'])==3838 and all(e['queue_release_step']==e['capture_step']+2 and e['neural_consume_step']==e['capture_step']+5 for e in proof['neural_consumption_events']));save(HERE/'PREFLIGHT_QUEUE.json',proof)
 import resource
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.;check('RSSunder1800_nobatchchange',rss<1800.);ds=read(HERE/'DATASET.json');sh={str(p):sha(p) for p in sources()};sh.update(ds['source_hashes']);dh={v['path']:v['sha256'] for v in ds['outputs'].values()};dh[str(HERE/'DATASET.json')]=sha(HERE/'DATASET.json');verify(sh);verify(dh);ob=odor_binding();save(HERE/'PREFLIGHT.json',dict(all_passed=True,checks=checks,numerical_errors=errors,maximum_error=max(errors.values()),gradient_norms=norms,peak_RSS_MiB=rss,effective_batch=24,microbatch=8,accumulation=3,history_steps=1920,parent_sha256=PARENT_SHA,odor_sha256=ob['sha256'],training_updates=0,Core_constructions=0,new_physics_steps=0));sh[str(HERE/'PREFLIGHT.json')]=sha(HERE/'PREFLIGHT.json');sh[str(HERE/'PREFLIGHT_QUEUE.json')]=sha(HERE/'PREFLIGHT_QUEUE.json');save(HERE/'FROZEN.json',dict(source_hashes=sh,data_hashes=dh,parent_sha256=PARENT_SHA,odor_sha256=ob['sha256'],visual_sha256=VISION_SHA,source_binding_sha256=digest,full_neck_training_allowed=False,scope='Actual14feedbackaggregationandnewodor memory signal engineeringrepair,notphysicalvalidated'))
 print(json.dumps(dict(all_passed=True,checks=len(checks),max_error=max(errors.values()),RSS_MiB=rss,effective24_micro8=True)))
if __name__=='__main__':raise SystemExit('Use explicitflag run.py')
