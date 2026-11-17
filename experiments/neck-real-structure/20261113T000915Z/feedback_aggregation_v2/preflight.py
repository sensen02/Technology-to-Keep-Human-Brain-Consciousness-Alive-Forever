"""Parent real full1920 batch7 preflight before fixed400, no automatic minibatch fallback."""
from common import *
def main():
 torch,np=backend()
 if (HERE/'FROZEN.json').exists():raise FileExistsError('No blind preflight repeat')
 from data import main as prepare,arrays,tensors
 from model import make_scan,python_rollout,objective
 from copy import deepcopy
 prepare();d=arrays('train');x,cue,y,w=tensors(d);torch.manual_seed(20261112);m,parent=initialize();a=deepcopy(m);checks={};errors={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('Aggregationpreflight '+k)
 def close(k,l,r):
  err=float((l-r).abs().max().detach());errors[k]=err;check(k,err<=1e-10)
 check('parent_all_tensors_exact',all(torch.equal(v,parent['state_dict'][k]) for k,v in m.state_dict().items()));check('same_architecture',m.identity()==parent['identity']);check('batch7full1920',x.shape==(1920,7,111) and y.shape==cue.shape==w.shape==(1920,7,1));check('cue_instruction_teacher_exact',torch.equal(y,.8*cue))
 with np.load(OLD/'train.npz',allow_pickle=False) as olddata:check('original4inputs_exact',np.array_equal(d['body111'][:,:4],olddata['body111']) and np.array_equal(d['cue'][:,:4],olddata['cue']))
 check('same111normalizer',torch.equal(m.observation_mean,torch.from_numpy(d['mean'])) and torch.equal(m.observation_std,torch.from_numpy(d['std'])));check('same_allowedmask',m.allowed_raw_indices.tolist()==MASK and len(MASK)==111)
 for j in range(7):
  on=cue[:,j,0].abs()>.5;off=~on;check('weight_mass_'+str(j),abs(float(w[:,j].sum())-1)<1e-12)
  if bool(on.any()) and bool(off.any()):check('balanced_onneutral_'+str(j),abs(float(w[on,j].sum())-.5)<1e-12 and abs(float(w[off,j].sum())-.5)<1e-12)
 check('brain_onlyANcue',m.brain.input_size==2 and m.brain.hidden_size==16);check('decoder_onlyDN',m.decoder[0].in_features==2 and m.decoder[2].out_features==1);check('feedback111only',m.feedback[0].in_features==111)
 o1=python_rollout(m,x,cue);o2=make_scan(a)(x,cue)
 for key,l,r in zip(('turn','DN','AN','recon','brainhidden'),o1,o2):close('production_scan_'+key,l,r)
 loss1,terms1=objective(o1,y,m.normalize(x),w,m);loss2,terms2=objective(o2,y,a.normalize(x),w,a);close('weighted_loss',loss1,loss2)
 manual=(((o2[0]-y).square()*w).sum(0)).mean()+.01*(o2[3]-a.normalize(x)).square().mean()+.001*o2[1].square().mean()+.1*a.decoder(x.new_zeros((7,2))).square().mean();close('independent_manual_all_loss_terms',loss2,manual)
 loss1.backward();loss2.backward();gradients={}
 for (name,p),(other,q) in zip(m.named_parameters(),a.named_parameters()):check('finitegrad_'+name,p.grad is not None and q.grad is not None and torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all());close('gradient_'+name,p.grad,q.grad);gradients[name]=float(p.grad.norm())
 torch.nn.utils.clip_grad_norm_(m.parameters(),1.);torch.nn.utils.clip_grad_norm_(a.parameters(),1.);op1=torch.optim.Adam(m.parameters(),lr=.001);op2=torch.optim.Adam(a.parameters(),lr=.001);op1.step();op2.step()
 for (name,p),(other,q) in zip(m.named_parameters(),a.named_parameters()):
  close('Adamparameter_'+name,p,q)
  for key in ('exp_avg','exp_avg_sq','step'):close('Adamstate_'+key+'_'+name,op1.state[p][key],op2.state[q][key])
 check('all_Adam_counters_one',len(op1.state)==len(list(m.parameters())) and all(float(v['step'])==1 for v in op1.state.values()));del o1,o2,loss1,loss2,manual,op1,op2,a
 bp,b,digest=binding();m.eval()
 with torch.no_grad():
  expected=python_rollout(m,x[:,5:6],cue[:,5:6]);actual,proof=real_queue_rollout(m,x[:,5:6],cue[:,5:6],bp,digest)
  for k,l,r in zip(('turn','DN','AN','recon','brainhidden'),expected,actual):close('real_onpolicy_sourcequeue_'+k,l,r)
 check('actual_sourcequeue9600steps',proof['actual_steps']==9600);check('all3838_neuralconsume_delays',len(proof['neural_consumption_events'])==3838 and all(e['queue_release_step']==e['capture_step']+2 and e['neural_consume_step']==e['capture_step']+5 for e in proof['neural_consumption_events']));save(HERE/'PREFLIGHT_QUEUE.json',proof)
 import resource
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.;check('RSSunder1800MiB_without_batchchange',rss<1800.)
 ds=read(HERE/'DATASET.json');sh={str(p):sha(p) for p in sources()};sh.update(ds['source_hashes']);dh={v['path']:v['sha256'] for v in ds['outputs'].values()};dh[str(HERE/'DATASET.json')]=sha(HERE/'DATASET.json');verify(sh);verify(dh)
 save(HERE/'PREFLIGHT.json',dict(all_passed=True,checks=checks,numerical_errors=errors,maximum_error=max(errors.values()),gradient_norms=gradients,peak_RSS_MiB=rss,production_batch=7,history_steps=1920,source_binding_sha256=digest,training_updates=0,Core_constructions=0,physics_steps=0))
 sh[str(HERE/'PREFLIGHT.json')]=sha(HERE/'PREFLIGHT.json');sh[str(HERE/'PREFLIGHT_QUEUE.json')]=sha(HERE/'PREFLIGHT_QUEUE.json');save(HERE/'FROZEN.json',dict(source_hashes=sh,data_hashes=dh,source_binding_sha256=digest,parent_sha256=PARENT_SHA,full_neck_training_allowed=False,scope='Parent-authorized data aggregation and objective joint engineering repair only'))
 print(json.dumps(dict(all_passed=True,checks=len(checks),max_error=max(errors.values()),peak_RSS_MiB=rss,batch=7)))
if __name__=='__main__':raise SystemExit('Use run.py --execute-parent-aggregation')
