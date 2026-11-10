"""Regression-protected translator: neural-only residual with known good-state preservation."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MUJOCO_GL','egl')
import sys,time,copy,pickle,hashlib,json
from pathlib import Path
import numpy as np, torch
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-neural-translator/20261105T200021Z');TIMING=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-neural-translator/20261105T234531Z');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(TIMING))
from translator import Translator,NumpyTranslator
import timing as old
rt=old.rt;PREV=old.O;DATA_SRC=old.previous.O;O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-translator/20261106T002653Z');rt.O=O;rt.b.O=O
torch.set_num_threads(1)
DEV=[1.225,1.275,1.3,1.325,1.375];FINAL=[1.206,1.256,1.306,1.356]
MODEL=Path(old.js(PREV/'SUMMARY.json')['best_nonzero']['model'])
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def protected():return [PREV/'MANIFEST_SHA256.json',PREV/'SUMMARY.json',PREV/'DATA/CACHE.pkl',MODEL,Path(old.__file__),Path(rt.__file__),Path(__file__)]+rt.b.sources()
def prereg():
 O.mkdir(parents=True,exist_ok=False);(O/'DATA').mkdir();(O/'MODELS').mkdir();(O/'CONFIGS').mkdir()
 dump(O/'PREREGISTRATION.json',dict(stage='S5 regression-protected neural translator',frozen='same selected double-gain MLP weights as initialization; SNN, sensory, RH0.375, old readout, plant and limits frozen',inputs='only204-cell causal neural history; no body state/time/phase/target input',data='closed-loop eligible correction labels from 20261105T231405Z plus 1.300s warmup neural features with zero-residual preservation target; prior final test excluded',lambdas=[.1,.25,.5,1.,2.,5.],loss='closed correction MSE + lambda*1.300s warm residual MSE; normalized by joint limits; warm target residual exactly zero',selection='first require 1.300s J/foot support/COM/first nonfoot no worse than old baseline; then full gate count, worst dev J, mean dev J',phases=dict(dev=DEV,final=FINAL),controls='zero old branch, selected model repeat, all motor cut, sensory cut, new branch cut',intent='walk/stop intent NOT_ESTABLISHED; standing is not walking',units='NOT_ESTABLISHED',importance='核心必需'))
 dump(O/'INPUT_HASHES_BEFORE.json',{str(p):sha(p) for p in protected()});dump(O/'CHANGE_LEDGER.json',dict(old_file=str(PREV/'PREREGISTRATION.json'),old_sha256=sha(PREV/'PREREGISTRATION.json'),new_file=str(O/'PREREGISTRATION.json'),new_sha256=sha(O/'PREREGISTRATION.json'),reason='Approved regression constraint to protect known1.300s behavior while testing neural-only corrective mapping.'));dump(O/'CLEANUP_LEDGER.json',dict(importance='重要辅助',deleted=[]))
def arr(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def model_path(lam):
 d=arr(MODEL);stamp=time.strftime('%Y%m%dT%H%M%S',time.gmtime())+str(time.time_ns()%1000000000).zfill(9)+'Z';folder=O/'CONFIGS'/(stamp+'_lambda_'+str(lam));folder.mkdir();p=folder/'MODEL.npz';np.savez_compressed(p,**d,regression_lambda=lam);dump(folder/'PREREGISTRATION.json',dict(lambda_regression=lam,source=str(MODEL),source_sha256=sha(MODEL),new_sha256=sha(p)));return p
def fit(t,lam):
 warm=arr(DATA_SRC/'DATA/warm_dev.npz');closed=arr(DATA_SRC/'DATA/closed_train.npz');dev=arr(DATA_SRC/'DATA/closed_dev.npz');limit=t.limits.astype(np.float32)
 # preserve 1.3 warm behavior: residual target zero; use warm train only from archived warm_dev/teacher feature data
 warmfit=arr(PREV/'DATA/warm_train.npz');wx=torch.tensor(warmfit['x'],dtype=torch.float32);wy=torch.zeros((len(wx),42));cx=torch.tensor(closed['x'][closed['eligible']],dtype=torch.float32);co=torch.tensor(closed['old'][closed['eligible']]/limit,dtype=torch.float32);cy=torch.tensor(closed['y'][closed['eligible']]/limit,dtype=torch.float32);vx=torch.tensor(dev['x'][dev['eligible']],dtype=torch.float32);vo=torch.tensor(dev['old'][dev['eligible']]/limit,dtype=torch.float32);vy=torch.tensor(dev['y'][dev['eligible']]/limit,dtype=torch.float32)
 torch.manual_seed(20261106);m=Translator();m.restore_export(MODEL);opt=torch.optim.Adam(m.parameters(),lr=.0002,weight_decay=.0001);best=1e9;state=None;pat=0;rng=np.random.default_rng(20261106);hist=[]
 for epoch in range(120):
  for _ in range(max(1,int(np.ceil(max(len(cx),len(wx))/256)))):
   ci=rng.integers(len(cx),size=256);wi=rng.integers(len(wx),size=256);pred=torch.clamp(co[ci]+2*torch.tanh(m(cx[ci])),-1,1);loss=((pred-cy[ci])**2).mean()+lam*(m(wx[wi])**2).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
  with torch.no_grad():v=((torch.clamp(vo+2*torch.tanh(m(vx)),-1,1)-vy)**2).mean().item()+lam*(m(wx)**2).mean().item()
  hist.append(dict(epoch=epoch,loss=v));
  if v<best-1e-6:best=v;state=copy.deepcopy(m.state_dict());pat=0
  else:pat+=1
  if pat>=15:break
 m.load_state_dict(state);p=model_path(lam);m.export(p);dump(Path(p).parent/'FIT.json',dict(lambda_regression=lam,best_loss=best,history=hist,model_sha256=sha(p)));return p
def actor(p):
 m=NumpyTranslator(p);gain=float(arr(p)['residual_gain']);return lambda x,stochastic:(np.arctanh(gain*np.tanh(m(x))) if gain not in [0,1] else (np.zeros(42,np.float32) if gain==0 else m(x)),0.,0.)
def main():
 mode=sys.argv[1]
 if mode=='prereg':prereg();return
 before=js(O/'INPUT_HASHES_BEFORE.json');assert before=={str(p):sha(p) for p in protected()}
 t=old.runtime(True);base=t.cache;records=[]
 def evalp(p,name,phases):
  rs=[]
  for q in phases:
   r,_=t.episode(name+'_'+str(q),q,actor(p),p);rs.append(r);records.append(r)
  return rs
 zero=old.make_variant('regression_zero',0.,old.TAUS['original']);baseline=evalp(zero,'zero',DEV);models=[]
 for lam in [.1,.25,.5,1.,2.,5.]:
  p=fit(t,lam);rs=evalp(p,'lambda_'+str(lam),DEV);guard=next(r for r in rs if r['release_time_s']==1.3);base13=next(r for r in baseline if r['release_time_s']==1.3);ok=guard['J']<=base13['J']+1e-9 and guard['parts']['foot_support_mean']>=base13['parts']['foot_support_mean']-1e-9 and guard['parts']['com_z_min']>=base13['parts']['com_z_min']-1e-9 and guard['parts']['first_nonfoot_s']>=base13['parts']['first_nonfoot_s']-1e-9;models.append(dict(lambda_regression=lam,model=str(p),results=rs,guard_pass=ok,rank=old.rank(rs)));dump(O/'CANDIDATES.json',models)
 safe=[m for m in models if m['guard_pass']];selected=max(safe,key=lambda m:m['rank']) if safe else dict(lambda_regression=0,model=str(zero),results=baseline,guard_pass=True,rank=old.rank(baseline));final=[]
 for q in FINAL:
  oldr=evalp(zero,'final_zero',[q])[0];newr=evalp(selected['model'],'final_selected',[q])[0];final.append(dict(phase=q,old=oldr,selected=newr))
 rep=evalp(selected['model'],'representative',[1.3])[0];repeat=evalp(selected['model'],'repeat',[1.3])[0];motor=evalp(selected['model'],'motor_cut',[1.3])[0];sens=evalp(selected['model'],'sensory_cut',[1.3])[0];branch=evalp(selected['model'],'new_branch_cut',[1.3])[0]
 dump(O/'SUMMARY.json',dict(candidates=models,selected=selected,final=final,representative=rep,repeat=repeat,motor_cut=motor,sensory_cut=sens,translator_cut=branch,total=len(records),walking_certified=False,intent_preserved='NOT_ESTABLISHED'))
 after={str(p):sha(p) for p in protected()};dump(O/'INPUT_HASHES_AFTER.json',after);assert before==after;print('COMPLETE',len(records),'selected',selected['lambda_regression'],flush=True)
if __name__=='__main__':main()
