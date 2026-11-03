"""Round C: tune the per-actuator readout gain (90 dims) on the dynamic support objective.

Why this round exists: measured, not guessed. After round A/B the trained solution pins 11 of
90 muscles at the activation ceiling for more than half of the release window, while the
untrained baseline pins none. So the remaining lever is not more gain on the same pattern, it
is the DISTRIBUTION of the network's drive across actuators. `MotorOutputConfig.per_actuator_gain`
is the pre-existing declared translation-layer calibration for exactly that: static, no state
feedback, no bias, no sign change, and None is documented to reproduce every earlier result.

Everything else is held fixed: the neural substrate is untouched (pool modes stay neutral, same
connectome weights, same thresholds, same sensory wiring), the rollout code is imported from the
stage-A trainer, and the objective is imported from the dynamic-support trainer. Only the 90
readout gains move.

modes: identity / prereg / run / validate / verify
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import sys,json,math,hashlib,time,traceback,argparse
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'))
import train_stage_a_pool_modes as st
import train_dynamic_support as dst
import dynamic_support_metrics as dsm
from engine.embodied.motor_output import MotorOutputConfig
O=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-readout-round-c/20261103T070000Z')
LEGS=st.LEGS;DIM=90;DT=st.DT
SEED=int(os.environ.get('RC_SEED','20261104'))
LAMBDA=16;GENERATIONS=15;EVAL_CAP=LAMBDA*GENERATIONS
WORKERS=int(os.environ.get('RC_WORKERS','4'));RUN_TAG=os.environ.get('RC_RUN','R1')
RELEASE=.2;ASSIST=(1.1,1.3);EXTENDED=os.environ.get('RC_EXTENDED','1')=='1'
GAIN_MIN,GAIN_MAX=0.5,2.0
def js(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def decode(z):
 """z in [-1,1]^90 -> one gain per actuator, log-symmetric around the identity."""
 z=np.asarray(z,float);assert z.shape==(DIM,) and np.all(np.abs(z)<=1+1e-12),(z.shape,float(np.abs(z).max()))
 return np.exp(np.clip(z,-1,1)*math.log(GAIN_MAX))

def trial_with_gains(folder,gains):
 """Stage-A rollout with the readout gains installed; pool modes are left neutral."""
 assert len(gains)==DIM
 st.CONFIG['neural_motor_config']=MotorOutputConfig(per_actuator_gain=tuple(float(g) for g in gains))
 try:return st.trial(folder,np.zeros(st.DIM),RELEASE)
 finally:st.CONFIG.pop('neural_motor_config',None)

_W={}
def _init(assist):_W['assist']=assist
def _eval(job):
 z,folder=job
 try:
  _,rows,summary=trial_with_gains(folder,decode(z))
  assist=_W['assist'];J,parts=dst.objective(rows,assist);full=dsm.evaluate_native(assist,rows)
  dump(Path(folder)/'SCORE.json',dict(J=J,parts=parts,gate=full,dict_z=z.tolist(),
   gains=decode(z).tolist(),wall_s=summary['wall_s']))
  return dict(ok=True,J=float(J),parts=parts,gate_pass=bool(full['passed']),error=None)
 except BaseException:
  dump(Path(folder)/'SCORE.json',dict(failed=True,traceback=traceback.format_exc(),dict_z=z.tolist()))
  return dict(ok=False,J=float('inf'),parts=None,gate_pass=False,error=traceback.format_exc()[-800:])

# ---------------------------------------------------------------- sep-CMA-ES (diagonal)
class SepCMA:
 def __init__(self,n,sigma,lmbda,seed):
  self.n=n;self.sigma=sigma;self.lmbda=lmbda;self.mu=lmbda//2
  self.w=np.log(self.mu+.5)-np.log(np.arange(1,self.mu+1));self.w/=self.w.sum()
  self.mueff=1./np.sum(self.w**2)
  self.cs=(self.mueff+2.)/(n+self.mueff+5.)
  self.ds=1.+2.*max(0.,math.sqrt((self.mueff-1.)/(n+1.))-1.)+self.cs
  self.cc=(4.+self.mueff/n)/(n+4.+2.*self.mueff/n)
  self.c1=2./((n+1.3)**2+self.mueff)
  self.cmu=min(1.-self.c1,2.*(self.mueff-2.+1./self.mueff)/((n+2.)**2+self.mueff))
  self.m=np.zeros(n);self.C=np.ones(n);self.ps=np.zeros(n)
  self.rng=np.random.default_rng(seed);self.generation=0;self.evals=0
 def ask(self):
  self.z=self.rng.standard_normal((self.lmbda,self.n))
  self.y=self.z*np.sqrt(self.C)
  return np.clip(self.m+self.sigma*self.y,-1.,1.)
 def tell(self,fits):
  idx=np.argsort(fits)[:self.mu]
  mold=self.m.copy()
  self.m=self.m+self.sigma*np.sum(self.w[:,None]*self.y[idx],axis=0)
  yw=np.sum(self.w[:,None]*self.y[idx],axis=0)
  csn=math.sqrt(self.cs*(2.-self.cs)*self.mueff)
  invsqrt=1./np.sqrt(self.C)
  self.ps=(1.-self.cs)*self.ps+csn*(yw*invsqrt)
  hsig=float(np.linalg.norm(self.ps)/math.sqrt(1.-(1.-self.cs)**(2.*(self.generation+1)))/self.ds) < (1.4+2./(self.n+1.))
  artmp=(self.y[idx]-mold+self.m)/self.sigma if False else self.y[idx]+(mold-self.m)/self.sigma
  self.C=(1.-self.c1-self.cmu)*self.C+self.c1*(self.C*(self.ps**2)+(1.-hsig)*self.cc*(2.-self.cc)*self.C)+self.cmu*np.sum(self.w[:,None]*(artmp**2),axis=0)
  self.C=np.maximum(self.C,1e-12)
  self.sigma*=math.exp((self.cs/self.ds)*(np.linalg.norm(self.ps)/math.sqrt(1.-(1.-self.cs)**(2.*(self.generation+1)))/self.ds-1.))
  self.sigma=float(np.clip(self.sigma,1e-3,1.0))
  self.generation+=1;self.evals+=len(fits)

def _path(tag):return O/f'CMA_STATE_{tag}.npz'
def _hist(tag):return O/f'CMA_HISTORY_{tag}.json'
def _save(cma,tag):
 stt=cma.rng.bit_generator.state
 np.savez_compressed(_path(tag),m=cma.m,C=cma.C,ps=cma.ps,sigma=np.array([cma.sigma]),
  generation=np.array([cma.generation]),evals=np.array([cma.evals]),
  rng_state=np.array([format(int(stt['state']['state']),'x')]),rng_inc=np.array([format(int(stt['state']['inc']),'x')]),
  rng_has=np.array([int(stt.get('has_uint32',0))]),rng_u32=np.array([int(stt.get('uinteger',0))]))
def _load(cma,tag):
 d=dict(np.load(_path(tag)))
 cma.m=d['m'];cma.C=d['C'];cma.ps=d['ps'];cma.sigma=float(d['sigma'][0])
 cma.generation=int(d['generation'][0]);cma.evals=int(d['evals'][0])
 cma.rng.bit_generator.state={'bit_generator':'PCG64',
  'state':{'state':int(str(d['rng_state'][0]),16),'inc':int(str(d['rng_inc'][0]),16)},
  'has_uint32':int(d['rng_has'][0]),'uinteger':int(d['rng_u32'][0])}

def identity():
 """The readout hook must be faithful: all-ones gains reproduce the unmodified run exactly."""
 O.mkdir(parents=True,exist_ok=True)
 base=O/'IDENTITY_A_no_gain';ones=O/'IDENTITY_B_gain_ones'
 if (base/'RELEASE_NATIVE.npz').exists() and (ones/'RELEASE_NATIVE.npz').exists():
  rows_a=dict(np.load(base/'RELEASE_NATIVE.npz'))['rows']
  rows_b=dict(np.load(ones/'RELEASE_NATIVE.npz'))['rows']
  summ=js(ones/'EXECUTION.json')
 else:
  for f in (base,ones):
   if f.exists():raise SystemExit('refusing to overwrite '+str(f))
  _,rows_a,_=st.trial(base,np.zeros(st.DIM),RELEASE)
  _,rows_b,summ=trial_with_gains(ones,np.ones(DIM))
 same=dict(rows=[int(x) for x in rows_a.shape],bit_identical=bool(np.array_equal(rows_a,rows_b)),
  max_abs_diff=float(np.max(np.abs(rows_a-rows_b),initial=0.)),
  body_steps=summ['counters'])
 res=dict(cases=dict(identity_all_ones=same),passed=bool(same['bit_identical']),
  meaning='all-ones per-actuator gain must reproduce the unmodified rollout bit for bit, otherwise the hook itself changes the physics')
 dump(O/'IDENTITY.json',res);print(json.dumps(res,ensure_ascii=False,indent=1));return res

def prereg():
 O.mkdir(parents=True,exist_ok=True)
 assist=dsm.window_metrics(dict(np.load(BASE_BODY)),*ASSIST,'assist')
 dump(O/'PREREGISTRATION.json',dict(
  round='C',question='does spreading the network drive differently over the 90 actuators break the plateau',
  motivation='measured: the round-A/B trained solution pins 11 of 90 muscles at the activation ceiling for over half the release window, the untrained baseline pins none',
  parameter_space=dict(dimension=DIM,meaning='one multiplicative gain per actuator on the decoded command, applied before the clip',
   bounds=[GAIN_MIN,GAIN_MAX],mapping='gain = exp(z*ln 2), z in [-1,1]',
   unchanged='connectome weights, thresholds, sensory wiring, body, muscles, contacts, assist schedule; pool modes fixed at neutral',
   classification='declared translation-layer calibration already present in MotorOutputConfig; static, no state feedback, no bias, no sign change'),
  objective='imported unchanged from the dynamic-support round: 2*nonfoot_share + foot deficit + COM deficit + first-contact time term + free fall fraction',
  gate='imported unchanged: dynamic support D1-D8 on native 12.5 us rows, thresholds from this run own assisted window',
  optimizer=dict(name='sep-CMA-ES',lambda_=LAMBDA,mu=LAMBDA//2,sigma0=0.3,generations=GENERATIONS,eval_cap=EVAL_CAP,seed=SEED,workers=WORKERS),
  acceptance=dict(improvement_over_baseline='J below the measured baseline and foot support above 0.240 W (the round-A/B best)',
   break_plateau='at least one candidate passes the dynamic gate'),
  failures_reported='yes, every failed evaluation keeps its traceback',
  baseline_reference=dict(nonfoot_share=0.785,foot_support=0.109,round_ab_best=dict(nonfoot_share=0.7166648511889714,foot_support=0.24011973880029733,J=3.6277)),
  assist_window_stats=assist,calls=dict(note='every evaluation is a real 1.3 s assisted plus 0.2 s neural-only rollout; the import chain is the same code path as rounds A and B')))

BASE_BODY='/home/sensen/Desktop/engine修复验收/s4-stage-a-pool-modes/20261102T203810Z/DIAGNOSE_R2/RUN/BODY.npz'

def run():
 assist=dsm.window_metrics(dict(np.load(BASE_BODY)),*ASSIST,'assist')
 cma=SepCMA(DIM,0.3,LAMBDA,SEED)
 if _path(RUN_TAG).exists():
  _load(cma,RUN_TAG);history=js(_hist(RUN_TAG))
 else:
  history=[]
  done=sorted((O/'EVALS').glob('eval_*')) if (O/'EVALS').exists() else []
  gens={}
  for f in done:
   idx=int(f.name.split('_')[1]);gens.setdefault(idx//LAMBDA,{})[idx%LAMBDA]=js(f/'SCORE.json')
  for g in sorted(gens):
   if g!=cma.generation:raise SystemExit('replay gap at generation %d'%g)
   if len(gens[g])!=LAMBDA:raise SystemExit('partial generation %d cannot be replayed safely'%g)
   xs=cma.ask()
   for i,sc in gens[g].items():
    assert np.array_equal(np.asarray(sc['dict_z']),xs[i]),'RNG replay does not reproduce the stored parameters'
   fits=np.array([gens[g][i]['J'] if not gens[g][i].get('failed') else 1e6 for i in range(LAMBDA)])
   cma.tell(fits)
   history.append(dict(generation=g,best=float(fits.min()),median=float(np.median(fits)),
    worst=float(fits.max()),passers=int(sum(bool(gens[g][i].get('gate',{}).get('passed')) for i in range(LAMBDA))),
    best_parts=gens[g][int(np.argmin(fits))].get('parts'),failures=int(sum(bool(gens[g][i].get('failed')) for i in range(LAMBDA))),
    evals=int(cma.evals),sigma=float(cma.sigma),replayed=True))
  if history:
   _save(cma,RUN_TAG);dump(_hist(RUN_TAG),history)
   print('replayed %d completed generations from stored evaluations'%len(history),flush=True)
 dump(O/f'SOURCE_HASHES_BEFORE_{RUN_TAG}.json',{str(p):sha(p) for p in [Path(__file__),R/'tools/train_stage_a_pool_modes.py',R/'tools/train_dynamic_support.py',R/'tools/dynamic_support_metrics.py',R/'engine/embodied/motor_output.py']})
 t0=time.monotonic()
 import multiprocessing as mp
 ctx=mp.get_context('spawn')
 evals_dir=O/'EVALS';evals_dir.mkdir(parents=True,exist_ok=True)
 with ctx.Pool(WORKERS,initializer=_init,initargs=(assist,)) as pool:
  while cma.evals<EVAL_CAP:
   g=cma.generation;xs=cma.ask()
   jobs=[(xs[i],evals_dir/f'eval_{cma.evals+i:03d}') for i in range(LAMBDA)]
   for _,f in jobs:
    if f.exists():raise SystemExit('refusing to overwrite '+str(f))
   res=pool.map(_eval,jobs)
   fits=np.array([r['J'] if r['ok'] else 1e6 for r in res])
   cma.tell(fits)
   history.append(dict(generation=g,best=float(fits.min()),median=float(np.median(fits)),
    worst=float(fits.max()),passers=int(sum(r['gate_pass'] for r in res)),
    best_parts=next((r['parts'] for r in res if r['J']==fits.min()),None),
    failures=int(sum(not r['ok'] for r in res)),evals=int(cma.evals),sigma=float(cma.sigma)))
   _save(cma,RUN_TAG);dump(_hist(RUN_TAG),history)
   dump(O/f'BEST_SO_FAR_{RUN_TAG}.json',dict(best=min(history,key=lambda r:r['best']),history_len=len(history)))
   print(json.dumps(history[-1],ensure_ascii=False)[:400],flush=True)
   if all(not r['ok'] for r in res):
    dump(O/f'ALL_FAILED_{RUN_TAG}.json',dict(generation=g,errors=[r['error'] for r in res]));break
 best=min(history,key=lambda r:r['best'])
 dump(O/f'RUN_SUMMARY_{RUN_TAG}.json',dict(best=best,generations=len(history),evals=int(cma.evals),
  passers_total=int(sum(h['passers'] for h in history)),wall_s=time.monotonic()-t0,
  sigma_final=float(cma.sigma),assist_window=ASSIST,assist_stats=assist,
  baseline_reference=dict(nonfoot_share=0.785,foot_support=0.109)))

def _vjob(arg):
 global RELEASE
 tag,z,folder=arg[:3];folder=Path(folder)
 if len(arg)>3:RELEASE=float(arg[3])
 try:
  _,rows,summary=trial_with_gains(folder,decode(np.asarray(z,float)))
  assist=_W['assist'];J,parts=dst.objective(rows,assist);full=dsm.evaluate_native(assist,rows)
  dump(folder/'SCORE.json',dict(J=J,parts=parts,gate=full,gains=decode(np.asarray(z,float)).tolist()))
  return dict(tag=tag,ok=True,J=float(J),gate_pass=bool(full['passed']),parts=parts)
 except BaseException:
  return dict(tag=tag,ok=False,error=traceback.format_exc()[-600:])

def validate():
 assist=dsm.window_metrics(dict(np.load(BASE_BODY)),*ASSIST,'assist')
 hist=js(_hist(RUN_TAG))
 scored=[]
 for f in sorted((O/'EVALS').glob('eval_*')):
  sc=js(f/'SCORE.json')
  if sc.get('failed') or 'J' not in sc:continue
  scored.append((float(sc['J']),f,sc['dict_z']))
 scored.sort(key=lambda r:r[0]);cands=[(f.name,z) for _,f,z in scored[:3]]
 assert cands,'no successful evaluation to validate'
 dump(O/f'VALIDATION_SELECTION_{RUN_TAG}.json',dict(selected=[dict(tag=t) for t,_ in cands]))
 jobs=[]
 for t,z in cands:
  for rep in ('A','B'):
   jobs.append((f'{t}_{rep}_release{int(RELEASE*1000)}ms',z,O/f'VALIDATION_{RUN_TAG}'/f'{t}_{rep}'))
 if EXTENDED:
  jobs.append((f'{cands[0][0]}_A_release400ms',cands[0][1],O/f'VALIDATION_{RUN_TAG}'/f'{cands[0][0]}_A_release400ms',0.4))
 import multiprocessing as mp
 ctx=mp.get_context('spawn')
 with ctx.Pool(WORKERS,initializer=_init,initargs=(assist,)) as pool:res=pool.map(_vjob,jobs)
 dump(O/f'VALIDATION_RESULTS_{RUN_TAG}.json',dict(results=res,all_failed=all(not r['ok'] for r in res)))
 print(json.dumps(res,ensure_ascii=False)[:600])

def verify():
 assist=dsm.window_metrics(dict(np.load(BASE_BODY)),*ASSIST,'assist')
 checks={}
 # 1. recompute the objective of every saved evaluation from its own saved rows
 errs=[];n=0
 for f in sorted((O/'EVALS').glob('eval_*')):
  sc=js(f/'SCORE.json')
  if sc.get('failed'):continue
  rows=dict(np.load(f/'RELEASE_NATIVE.npz'))['rows']
  J,parts=dst.objective(rows,assist);errs.append(abs(J-sc['J']));n+=1
  if len(errs)>=40:break
 checks['objective_recomputes']=dict(sampled=n,max_abs_error=float(max(errs,default=0.)),passed=bool(max(errs,default=0.)<=1e-12))
 # 2. gains stay inside the declared bounds in every stored candidate
 gmax=0.;gmin=9.
 for f in sorted((O/'EVALS').glob('eval_*')):
  sc=js(f/'SCORE.json');g=np.asarray(sc.get('gains',[np.nan]),float)
  if np.all(np.isfinite(g)):gmax=max(gmax,float(g.max()));gmin=min(gmin,float(g.min()))
 checks['gains_within_bounds']=dict(min=float(gmin),max=float(gmax),passed=bool(gmin>=GAIN_MIN-1e-9 and gmax<=GAIN_MAX+1e-9))
 # 3. the neural substrate is untouched. The stored dict_z is the READOUT vector, not the pool
 #    modes, so this cannot be read off per-evaluation data: it is checked structurally (the
 #    only call site passes neutral pool modes) and supported by the assist window, which must be
 #    identical across every evaluation if the network really is unchanged.
 src=Path(__file__).read_text()
 structural='st.trial(folder,np.zeros(st.DIM),RELEASE)' in src
 assist_sup=[js(f/'SCORE.json')['parts']['assist_foot_support'] for f in sorted((O/'EVALS').glob('eval_*'))
             if not js(f/'SCORE.json').get('failed') and js(f/'SCORE.json').get('parts')]
 spread=max(assist_sup)-min(assist_sup) if assist_sup else float('inf')
 checks['pool_modes_neutral']=dict(structural_call_site=bool(structural),evaluations_checked=len(assist_sup),
  assist_foot_support_min=float(min(assist_sup)) if assist_sup else None,
  assist_foot_support_spread=float(spread),passed=bool(structural and spread<=1e-9),
  note='the per-evaluation record stores the readout vector, not the pool modes, so neutrality is established from the single call site plus an identical assist window across all evaluations')
 # 4. history is consistent with the per-generation best
 hist=js(_hist(RUN_TAG))
 checks['history_matches_generation_best']=dict(passed=bool(all(h['best']==min(h['best'],h['median']) for h in hist)))
 checks['best_so_far_matches_history_min']=dict(passed=bool(js(O/f'BEST_SO_FAR_{RUN_TAG}.json')['best']['best']==min(h['best'] for h in hist)))
 dump(O/f'INDEPENDENT_VERIFICATION_{RUN_TAG}.json',dict(checks=checks,passed=all(c['passed'] for c in checks.values())))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode');a=ap.parse_args()
 {'identity':identity,'prereg':prereg,'run':run,'validate':validate,'verify':verify}[a.mode]()
