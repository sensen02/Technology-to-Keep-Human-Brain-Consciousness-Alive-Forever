"""Parent real full1920 batch4 gradient/Adam and sourcequeue production preflight."""
from common import *
def main():
 torch,np=backend()
 if (HERE/'FROZEN.json').exists():raise FileExistsError('No blind preflight replay')
 from data import main as prepare,arrays,tensors
 prepare();d=arrays('train');x,cue,y=tensors(d);from model import NeuralBridge,make_scan,python_rollout,objective
 from copy import deepcopy
 torch.manual_seed(20261112);m=NeuralBridge(torch.from_numpy(d['mean']),torch.from_numpy(d['std']));a=deepcopy(m);checks={};errors={}
 def check(key,value):
  checks[key]=bool(value)
  if not checks[key]:raise ValueError('Preflight '+key)
 def close(key,l,r,tol=1e-10):
  err=float((l-r).abs().max().detach());errors[key]=err;check(key,err<=tol)
 rawmask=torch.tensor(MASK);check('allowed111mask_exact',len(MASK)==111 and MASK==list(m.allowed_raw_indices.tolist()));raw=torch.arange(238,dtype=torch.float64);changed=raw.clone();excluded=sorted(set(range(238))-set(MASK));changed[excluded]+=100.;check('excluded_target_error_action_speed_invariance',torch.equal(raw.index_select(0,rawmask),changed.index_select(0,rawmask)))
 check('brain_only_AN1_cue1',m.brain.input_size==2 and m.brain.hidden_size==16 and m.brain_out.out_features==2);check('decoder_DNonly_no_cue',m.decoder[0].in_features==2);check('encoder_only_allowed111',m.feedback[0].in_features==111);check('no42output_head',m.decoder[2].out_features==1)
 out1=python_rollout(m,x,cue);out2=make_scan(a)(x,cue)
 for key,left,right in zip(('turn','DN','AN','recon','brainhidden'),out1,out2):close('production_scan_'+key,left,right)
 loss1,_=objective(out1,y,m.normalize(x));loss2,_=objective(out2,y,a.normalize(x));close('loss',loss1,loss2);loss1.backward();loss2.backward()
 gradients={}
 for (name,p),(other,q) in zip(m.named_parameters(),a.named_parameters()):
  check('parameter_order_'+name,name==other);check('finite_gradient_'+name,p.grad is not None and q.grad is not None and torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all());close('gradient_'+name,p.grad,q.grad);gradients[name]=float(p.grad.norm())
 torch.nn.utils.clip_grad_norm_(m.parameters(),1.);torch.nn.utils.clip_grad_norm_(a.parameters(),1.);o1=torch.optim.Adam(m.parameters(),lr=.001);o2=torch.optim.Adam(a.parameters(),lr=.001);o1.step();o2.step()
 for (name,p),(other,q) in zip(m.named_parameters(),a.named_parameters()):
  close('Adam_param_'+name,p,q)
  for key in ('exp_avg','exp_avg_sq','step'):close('Adam_'+key+'_'+name,o1.state[p][key],o2.state[q][key])
 check('all_parameter_Adam_states',len(o1.state)==len(list(m.parameters())) and all(float(v['step'])==1 for v in o1.state.values()))
 del out1,out2,loss1,loss2,o1,o2,a
 bp,b,digest=binding();m.eval()
 with torch.no_grad():
  expected=python_rollout(m,x[:,2:3],cue[:,2:3]);actual,queueproof=real_queue_rollout(m,x[:,2:3],cue[:,2:3],bp,digest)
  for key,l,r in zip(('turn','DN','AN','recon','brainhidden'),expected,actual):close('actual_sourcequeue1ms_'+key,l,r)
 check('actual_sourcequeue9600steps',queueproof['actual_steps']==9600);check('two_direction_events',len(queueproof['neural_consumption_events'])==3838);check('release2ms_neuralconsume5ms',all(e['queue_release_step']==e['capture_step']+2 and e['neural_consume_step']==e['capture_step']+5 for e in queueproof['neural_consumption_events']))
 check('both_initial_delivery_zero',queueproof['initial_delivered_AN_DN_zero']);save(HERE/'PREFLIGHT_QUEUE.json',queueproof)
 import resource
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.;check('RSS_under1800MiB',rss<1800.)
 ds=read(HERE/'DATASET.json');sh={str(p):sha(p) for p in sources()};sh.update(ds['source_hashes']);dh={row['path']:row['sha256'] for row in ds['outputs'].values()};dh[str(HERE/'DATASET.json')]=sha(HERE/'DATASET.json');verify(sh);verify(dh)
 save(HERE/'PREFLIGHT.json',dict(all_passed=True,checks=checks,numerical_errors=errors,maximum_error=max(errors.values()),gradient_norms=gradients,peak_RSS_MiB=rss,production_batch=4,time_steps=1920,actual_sourcequeue_steps=9600,training_updates=0,Core_constructions=0,motor_model_loads=0,source_binding_sha256=digest))
 sh[str(HERE/'PREFLIGHT.json')]=sha(HERE/'PREFLIGHT.json');sh[str(HERE/'PREFLIGHT_QUEUE.json')]=sha(HERE/'PREFLIGHT_QUEUE.json');save(HERE/'FROZEN.json',dict(source_hashes=sh,data_hashes=dh,source_binding_sha256=digest,full_neck_training_allowed=False,scope='Parent-authorized offline engineering calibration only'))
 print(json.dumps(dict(all_passed=True,checks=len(checks),max_error=max(errors.values()),peak_RSS_MiB=rss)))
if __name__=='__main__':raise SystemExit('Use run.py --execute-parent-calibration')
