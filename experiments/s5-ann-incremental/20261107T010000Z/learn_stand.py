"""Full-runtime brain -> causal temporal ANN -> incremental joint targets.
Teachers are used only for supervised collection; deployed action is exported ANN.
Actuator holding and learned/artificial versus original brain contributions stay separate.
"""
import os,sys,json,time,copy,hashlib
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import numpy as np
from scipy import sparse
import mujoco as mj
from plant_probe import Plant,O,R,dump,sha,dsm
sys.path.insert(0,str(R))
from engine.neural_cond import ConductanceNetwork,ConductanceParams
from engine.embodied.sensory_channels import ChannelSensoryEncoder,ChannelConfig
from run_neural_muscle_closure import channel_index_map,observe_quantities
from run_integrated_repaired_closure import identities
H=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
LOOK=Path('/home/sensen/Desktop/engine修复验收/s3-banc888-runtime-admission/20261101T220018Z/SENSORY_LOOKUP_SNAPSHOT888.json')
W=Path('/home/sensen/Desktop/engine修复验收/s5-route-b-ablation/20261108T060000Z/ARMS/A')
DT=.0000125;CDT=.001;PDT=.005;RATE=4.;SCALE=2.

def protocol():
 dest=O/'learning';dest.mkdir(exist_ok=False)
 src=[Path(__file__),Path(__file__).with_name('ann_translator.py'),Path(__file__).with_name('incremental_actuator.py'),Path(__file__).with_name('plant_probe.py'),R/'engine/neural_cond.py',R/'engine/embodied/sensory_channels.py',W/'ACTUAL_We.npz',W/'ACTUAL_Wi.npz',LOOK,H/'IDENTITIES.npz']
 p=dict(importance='核心必需',user_authorization='incremental finite PD and artificial neural translator or replacement brain explicitly authorized',brain='all 12000 cells of current runtime, NOT full153962 bank',network='shared12000->32 projection + causal Conv1d on4 past5ms rate snapshots + own42 actuator target state ->64->42 tanh',brain_dt_ms=.125,holdout_brain_dt_ms=.0625,brain_rate_tau_ms=20,policy_dt_s=PDT,actuator_dt_s=CDT,physics_dt_s=DT,actuator_gain_scale=SCALE,max_increment_rad_s=RATE,teacher='training only: clip(-5*(ref-q0)+2*(q0-q)-.05*qvel, +/-4) rad/s; finite incremental PD; no body state overwrite or external forces',data='3 independent train rollouts +1 dev, each3s, deterministic target command disturbances; supervised target is correction action excluding disturbance',deployment='ANN sees only brain rate history and own target state, no body q/qvel/time/teacher; initial1s same teacher warmup then2s ANN continuation; no neural/physical reset',warmup_s=1,release_s=2,training=dict(epochs=100,batch_size=32,lr=.001,seed=20261107),assessment='all D gates and finite/no warnings; full3s run with final2s scored, report hold/brain_cut/motor_cut and cold control; brain-cut success means original brain necessity not established',perturbation='declared command input disturbances during training; one initial100ms command disturbance during evaluation, no hidden trajectory writes',hashes={str(s):sha(s) for s in src},units='NOT_ESTABLISHED')
 dump(dest/'PREREGISTRATION.json',p)
 return dest

class Runtime:
 def __init__(self,ndt=.125):
  self.p=Plant();self.m=self.p.m;self.ndt=ndt;self.joints,self.geom=identities(self.m)
  roots=np.load(H/'IDENTITIES.npz')['root_ids'];self.groups,_=channel_index_map(12000,roots,lookup_path=LOOK)
  self.we=sparse.load_npz(W/'ACTUAL_We.npz');self.wi=sparse.load_npz(W/'ACTUAL_Wi.npz')
 def start(self):
  d=self.p.fresh();a=self.p.actuator(d,SCALE);n=ConductanceNetwork(12000,ConductanceParams(),dt_ms=self.ndt);n.set_connectivity(self.we,self.wi)
  return dict(d=d,a=a,n=n,enc=ChannelSensoryEncoder(self.groups,12000,ChannelConfig()),rates=np.zeros(12000),history=np.zeros((4,12000),np.float32))
 def teacher(self,s):
  d=s['d'];a=s['a'];return np.clip(-5*(a.target_rad-self.p.q0[self.p.qa])+2*(self.p.q0[self.p.qa]-d.qpos[self.p.qa])-.05*d.qvel[self.p.va],-RATE,RATE)
 def tick(self,s,rate,motor=True):
  d=s['d'];a=s['a'];a.update(rate,CDT)
  current=np.asarray(s['enc'].encode(observe_quantities(self.m,d,self.p.q0,self.joints,self.geom))[0]);decay=np.exp(-self.ndt/20);counts=np.zeros(12000,np.uint16)
  for _ in range(round(1/self.ndt)):
   sp=s['n'].step(current_nA=current);counts+=sp;s['rates']=decay*s['rates']+(1-decay)*sp/(self.ndt/1000)/100
  rows=[]
  for _ in range(80):
   d.ctrl[:]=a.torque(d) if motor else 0
   assert not d.qfrc_applied.any() and not d.xfrc_applied.any()
   mj.mj_step(self.m,d);mj.mj_forward(self.m,d);rows.append(self.p.contact(d))
  assert np.isfinite(d.qpos).all() and not d.warning.number.any()
  return rows,counts,float(np.linalg.norm(current))
 def snapshot(self,s):
  return dict(integration=self.p.snapshot(s['d']),target=s['a'].target_rad.copy(),rates=s['rates'].copy(),history=s['history'].copy(),**{'net_'+k:np.array(getattr(s['n'],k)) for k in ['v','ge','gi','adaptation','refractory_until_ms','spike_count','_ring','_ring_i','t_ms','steps']})
 def run(self,tag,mode,seed=0,model=None,seconds=3.,disturb=True):
  dest=O/'learning'/tag;dest.mkdir(exist_ok=False);s=self.start();d=s['d'];a=s['a'];rng=np.random.default_rng(seed);rows=[];body=[];frames=[];states=[];labels=[];actions=[];counts=[];norms=[];targets=[];start=time.monotonic();total=round(seconds/CDT)
  # Command disturbances excite actual visited states; all42 joints retained, avoid joints at soft limits.
  margin=np.minimum(a.target_rad-a.target_bounds_rad[:,0],a.target_bounds_rad[:,1]-a.target_rad)
  mask=margin>.15;directions=[]
  for _ in range(3):directions.append(rng.choice([-1.,1.],42)*mask*rng.uniform(.4,1.2,42))
  pending=np.zeros(42);ncount=np.zeros(12000,np.uint16)
  for k in range(total):
   if k==1000:np.savez_compressed(dest/'HANDOVER.npz',**self.snapshot(s))
   if k%5==0:
    s['history'][:-1]=s['history'][1:];s['history'][-1]=s['rates']
    target_state=(a.target_rad-self.p.q0[self.p.qa]).astype(np.float32)
    wanted=self.teacher(s)/RATE
    if mode=='teacher' or k<1000:pending=wanted*RATE
    elif mode in ['hold','motor_cut']:pending=np.zeros(42)
    else:pending=RATE*model((np.zeros_like(s['history']) if mode=='brain_cut' else s['history'])[None],target_state[None])[0]
    frames.append(s['rates'].astype(np.float16));states.append(target_state);labels.append(wanted.astype(np.float32));actions.append(pending.copy());counts.append(ncount.copy());ncount[:]=0
   perturb=np.zeros(42)
   if disturb:
    for i,t0 in enumerate([1100,1800,2400] if mode=='teacher' else [1100]):
     if t0<=k<t0+100:perturb=directions[i]
   targets.append(np.r_[d.time,a.target_rad,pending,perturb])
   rr,cc,norm=self.tick(s,pending+perturb,motor=not(mode=='motor_cut' and k>=1000));rows.extend(rr);ncount+=cc;norms.append(norm)
   if k%5==4:body.append(self.p.bodyrow(d))
   if (k+1)%1000==0:print(json.dumps(dict(tag=tag,sim_s=(k+1)/1000,wall_s=round(time.monotonic()-start,1))),flush=True)
  arr=np.asarray(rows);assist=arr[(arr[:,0]>=.8)&(arr[:,0]<1)];release=arr[arr[:,0]>=1]
  ref=dict(foot_support_mean=float(assist[:,1:7].sum(1).mean()/dsm.WEIGHT),com_z_mean=float(assist[:,10].mean()))
  gate=dsm.evaluate_native(ref,release);gate['checks']['D1_no_nonfoot_ground_force']=bool(np.max(release[:,7])==0);gate['passed']=all(gate['checks'].values())
  np.savez_compressed(dest/'DATA.npz',frames=frames,target_states=states,teacher_actions=labels,actions=actions,counts=counts,sensory_norm=norms)
  np.savez_compressed(dest/'NATIVE.npz',rows=arr);np.savez_compressed(dest/'COMMANDS.npz',rows=targets,kp=a.kp,kd=a.kd,limits=a.torque_limit,q0=self.p.q0[self.p.qa])
  np.savez_compressed(dest/'BODY.npz',**{key:np.array([r[key] for r in body]) for key in body[0]});np.savez_compressed(dest/'END.npz',**self.snapshot(s))
  result=dict(tag=tag,mode=mode,seed=seed,brain_dt_ms=self.ndt,validation=gate,warnings=int(d.warning.number.sum()),finite=True,seconds=seconds,folder=str(dest),wall_s=time.monotonic()-start,controller='teacher' if mode=='teacher' else 'ANN' if mode in ['ann','brain_cut'] else mode,brain_necessity='NOT_ESTABLISHED',walking=False)
  dump(dest/'RESULT.json',result);print(json.dumps(result),flush=True);return result

def collect():
 protocol();rt=Runtime();results=[]
 for tag,seed in [('train0',11),('train1',22),('train2',33),('dev0',44)]:results.append(rt.run(tag,'teacher',seed))
 dump(O/'learning/COLLECTION.json',results)

def train():
 import torch
 from ann_translator import FullBrainTranslator,NumpyFullBrainTranslator
 torch.set_num_threads(1);torch.manual_seed(20261107);rng=np.random.default_rng(20261107)
 root=O/'learning';trainsets=[];devsets=[]
 for name in ['train0','train1','train2','dev0']:
  with np.load(root/name/'DATA.npz') as z:dataset={k:z[k].copy() for k in ['frames','target_states','teacher_actions']}
  (devsets if name.startswith('dev') else trainsets).append(dataset)
 def batch(data,ids):
  hs=[];ts=[];ys=[]
  for ep,i in ids:
   ds=data[ep];hs.append(ds['frames'][np.maximum(0,np.arange(i-3,i+1))]);ts.append(ds['target_states'][i]);ys.append(ds['teacher_actions'][i])
  return torch.tensor(np.array(hs),dtype=torch.float32),torch.tensor(np.array(ts)),torch.tensor(np.array(ys))
 ids=np.array([(e,i) for e,ds in enumerate(trainsets) for i in range(40,len(ds['frames']))]);dev=np.array([(e,i) for e,ds in enumerate(devsets) for i in range(40,len(ds['frames']))])
 model=FullBrainTranslator();optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-5);best=float('inf');history=[];best_state=None
 for epoch in range(100):
  model.train();losses=[]
  for start in range(0,len(ids),32):
   if start==0:rng.shuffle(ids)
   x,t,y=batch(trainsets,ids[start:start+32]);pred=model(x,t);loss=((pred-y)**2).mean();optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
  model.eval();errs=[]
  with torch.no_grad():
   for start in range(0,len(dev),32):
    x,t,y=batch(devsets,dev[start:start+32]);errs.append(float(((model(x,t)-y)**2).mean()))
  score=float(np.mean(errs));history.append(dict(epoch=epoch,train=float(np.mean(losses)),dev=score))
  if score<best:best=score;best_state=copy.deepcopy(model.state_dict())
  if epoch%10==0:print(json.dumps(history[-1]),flush=True)
 model.load_state_dict(best_state);path=root/'TRANSLATOR.npz';model.export(path)
 npmodel=NumpyFullBrainTranslator(path);x,t,_=batch(devsets,dev[:4]);expected=model(x,t).detach().numpy();actual=npmodel(x.numpy(),t.numpy());err=float(abs(expected-actual).max());assert err<1e-5
 dump(root/'TRAINING.json',dict(importance='核心必需',history=history,best_dev_mse=best,numpy_parity_max=err,model=str(path),sha256=sha(path),train_samples=len(ids),dev_samples=len(dev),split='entire independent physical trajectories',input='all12000 brain rates plus own42 target state',label='actual teacher incremental correction; not torque',brain_specificity='not established by fitting'))

def evaluate():
 from ann_translator import NumpyFullBrainTranslator
 model=NumpyFullBrainTranslator(O/'learning/TRANSLATOR.npz');results=[]
 for ndt in [.125,.0625]:
  rt=Runtime(ndt)
  for mode in (['ann','brain_cut','hold','motor_cut'] if ndt==.125 else ['ann']):
   results.append(rt.run(f'eval_{mode}_dt{ndt}',mode,55,model))
 dump(O/'learning/EVALUATION.json',results)
if __name__=='__main__':
 {'collect':collect,'train':train,'evaluate':evaluate}[sys.argv[1]]()
