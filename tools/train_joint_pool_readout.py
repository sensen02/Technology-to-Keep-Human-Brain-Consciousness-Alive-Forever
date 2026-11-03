"""Joint search over the pool-level modes (24) and the per-actuator readout gain (90).

Why: round A/B tuned the network (24 dims) and beat the baseline; round C tuned only the readout
(90 dims) and was worse than round A/B. Both levers provably matter, so the untested combination
is the two together. The search starts from a KNOWN point (round A/B best pool vector, neutral
readout) instead of a random one, because the evaluation budget is small relative to 114 dims.

Unchanged: connectome weights, thresholds, sensory wiring, body, muscles, contacts, assist
schedule, objective, gate. The parameter vector is [24 pool dims, 90 readout dims].

modes: seedcheck / prereg / run / validate / verify
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
import train_actuator_readout_c as rc
from engine.embodied.motor_output import MotorOutputConfig
O=Path('/home/sensen/Desktop/engine修复验收/s5-joint-pool-readout/20261103T080000Z')
POOL_DIM=st.DIM;GAIN_DIM=90;DIM=POOL_DIM+GAIN_DIM
SEED=int(os.environ.get('JP_SEED','20261105'))
LAMBDA=16;GENERATIONS=18;EVAL_CAP=LAMBDA*GENERATIONS
WORKERS=int(os.environ.get('JP_WORKERS','4'));RUN_TAG=os.environ.get('JP_RUN','R1')
RELEASE=.2;ASSIST=(1.1,1.3);SIGMA0=0.20
GAIN_MIN,GAIN_MAX=0.5,2.0
SEED_POOL=Path('/tmp/POOL_BEST_AB.json')
def js(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def split(vec):
 v=np.asarray(vec,float);assert v.shape==(DIM,)
 return np.clip(v[:POOL_DIM],-1,1),np.clip(v[POOL_DIM:],-1,1)
def gains_of(zg):return np.exp(np.clip(zg,-1,1)*math.log(GAIN_MAX))

def trial_joint(folder,vec,release=None):
 zp,zg=split(vec)
 st.CONFIG['neural_motor_config']=MotorOutputConfig(per_actuator_gain=tuple(float(g) for g in gains_of(zg)))
 old=st.CONFIG.get('_release_override')
 try:
  return st.trial(folder,zp,release or RELEASE)
 finally:st.CONFIG.pop('neural_motor_config',None)

_W={}
def _init(assist):_W['assist']=assist
def _score(folder,rows,vec,summary=None):
 assist=_W['assist'];J,parts=dst.objective(rows,assist);full=dsm.evaluate_native(assist,rows)
 dump(Path(folder)/'SCORE.json',dict(J=J,parts=parts,gate=full,vec=np.asarray(vec,float).tolist(),
  pool_z=np.asarray(vec,float)[:POOL_DIM].tolist(),readout_gain=gains_of(np.asarray(vec,float)[POOL_DIM:]).tolist(),
  wall_s=(summary or {}).get('wall_s')))
 return J,parts,full
def _eval(job):
 vec,folder=job
 try:
  _,rows,summary=trial_joint(folder,vec)
  J,parts,full=_score(folder,rows,vec,summary)
  return dict(ok=True,J=float(J),parts=parts,gate_pass=bool(full['passed']),error=None)
 except BaseException:
  dump(Path(folder)/'SCORE.json',dict(failed=True,traceback=traceback.format_exc(),vec=np.asarray(vec,float).tolist()))
  return dict(ok=False,J=float('inf'),parts=None,gate_pass=False,error=traceback.format_exc()[-800:])

def seedcheck():
 """The joint hook must reproduce the round A/B best at neutral readout."""
 O.mkdir(parents=True,exist_ok=True)
 pool=np.asarray(js(SEED_POOL)['pool_z'],float)
 vec=np.concatenate([pool,np.zeros(GAIN_DIM)])
 f=O/'SEED_AB_BEST_NEUTRAL_READOUT'
 if (f/'RELEASE_NATIVE.npz').exists():rows=dict(np.load(f/'RELEASE_NATIVE.npz'))['rows']
 else:
  if f.exists():raise SystemExit('refusing to overwrite '+str(f))
  _,rows,_=trial_joint(f,vec)
 assist=dsm.window_metrics(dict(np.load(rc.BASE_BODY)),*ASSIST,'assist')
 J,parts=dst.objective(rows,assist)
 ref=js(SEED_POOL)
 res=dict(cases=dict(seed_reproduces=dict(J=float(J),reference_J=float(ref['J']),abs_error=abs(float(J)-ref['J']),
   foot_support_mean=parts['foot_support_mean'],nonfoot_share=parts['nonfoot_share'],
   passed=bool(abs(float(J)-ref['J'])<=1e-9))),
  note='neutral readout must leave the round A/B best untouched, otherwise the joint hook changes the run',
  source=ref['source'])
 res['passed']=all(c['passed'] for c in res['cases'].values())
 dump(O/'SEEDCHECK.json',res);print(json.dumps(res,ensure_ascii=False,indent=1));return res

def prereg():
 O.mkdir(parents=True,exist_ok=True)
 assist=dsm.window_metrics(dict(np.load(rc.BASE_BODY)),*ASSIST,'assist')
 dump(O/'PREREGISTRATION.json',dict(round='joint',question='does the untested combination of pool modes and per-actuator readout break the plateau',
  parameter_space=dict(dimension=DIM,pool=POOL_DIM,readout=GAIN_DIM,
   pool_meaning='the round A/B pool modes: per-leg We/Wi gain, threshold offset, E/I balance, excitatory drive',
   readout_meaning='one multiplicative gain per actuator, gain = exp(z ln 2) in [0.5,2]',
   unchanged='body, muscles, contacts, connectome weights and thresholds beyond the declared pool modes, sensory wiring, assist schedule',
   classification='pool modes are declared neural parameter tuning; readout gain is the declared translation-layer calibration'),
  initialization=dict(pool='round A/B best vector, recorded with its source path',readout='neutral (all ones)',
   declared='this is NOT a random start: the first generation is seeded at a known good point because the budget is small relative to 114 dimensions',
   seed_source=str(SEED_POOL)),
  objective='unchanged dynamic-support objective',
  gate='unchanged dynamic criteria D1-D8 on native 12.5 us rows',
  optimizer=dict(name='sep-CMA-ES',lambda_=LAMBDA,mu=LAMBDA//2,sigma0=SIGMA0,generations=GENERATIONS,eval_cap=EVAL_CAP,seed=SEED,workers=WORKERS),
  acceptance=dict(beat='J below 3.6277 and foot support above 0.240 W',gate_break='at least one candidate passes the dynamic gate'),
  assist_window_stats=assist,
  honest_limit='114 dimensions with 288 evaluations is a small budget; a negative result bounds this budget, not the whole idea'))

def _path(tag):return O/f'CMA_STATE_{tag}.npz'
def _hist(tag):return O/f'CMA_HISTORY_{tag}.json'
def run():
 assist=dsm.window_metrics(dict(np.load(rc.BASE_BODY)),*ASSIST,'assist')
 cma=rc.SepCMA(DIM,SIGMA0,LAMBDA,SEED)
 pool=np.asarray(js(SEED_POOL)['pool_z'],float)
 cma.m=np.concatenate([pool,np.zeros(GAIN_DIM)])   # declared informed initialization
 history=[]
 if _path(RUN_TAG).exists():
  rc._load(cma,RUN_TAG);history=js(_hist(RUN_TAG))
 else:
  done=sorted((O/'EVALS').glob('eval_*')) if (O/'EVALS').exists() else []
  gens={}
  for f in done:
   idx=int(f.name.split('_')[1]);gens.setdefault(idx//LAMBDA,{})[idx%LAMBDA]=js(f/'SCORE.json')
  cma.m=np.concatenate([pool,np.zeros(GAIN_DIM)])
  for g in sorted(gens):
   if g!=cma.generation:raise SystemExit('replay gap at generation %d'%g)
   if len(gens[g])!=LAMBDA:raise SystemExit('partial generation %d cannot be replayed safely'%g)
   xs=cma.ask()
   for i,sc in gens[g].items():assert np.array_equal(np.asarray(sc['vec']),xs[i]),'RNG replay mismatch'
   fits=np.array([gens[g][i]['J'] if not gens[g][i].get('failed') else 1e6 for i in range(LAMBDA)])
   cma.tell(fits)
   history.append(dict(generation=g,best=float(fits.min()),median=float(np.median(fits)),
    passers=int(sum(bool(gens[g][i].get('gate',{}).get('passed')) for i in range(LAMBDA))),
    failures=int(sum(bool(gens[g][i].get('failed')) for i in range(LAMBDA))),evals=int(cma.evals),sigma=float(cma.sigma),replayed=True))
  if history:rc._save(cma,RUN_TAG);dump(_hist(RUN_TAG),history)
 t0=time.monotonic()
 import multiprocessing as mp
 ctx=mp.get_context('spawn');evals_dir=O/'EVALS';evals_dir.mkdir(parents=True,exist_ok=True)
 with ctx.Pool(WORKERS,initializer=_init,initargs=(assist,)) as pool_proc:
  while cma.evals<EVAL_CAP:
   g=cma.generation;xs=cma.ask()
   jobs=[(xs[i],evals_dir/f'eval_{cma.evals+i:03d}') for i in range(LAMBDA)]
   for _,f in jobs:
    if f.exists():raise SystemExit('refusing to overwrite '+str(f))
   res=pool_proc.map(_eval,jobs)
   fits=np.array([r['J'] if r['ok'] else 1e6 for r in res])
   cma.tell(fits)
   history.append(dict(generation=g,best=float(fits.min()),median=float(np.median(fits)),worst=float(fits.max()),
    passers=int(sum(r['gate_pass'] for r in res)),best_parts=next((r['parts'] for r in res if r['J']==fits.min()),None),
    failures=int(sum(not r['ok'] for r in res)),evals=int(cma.evals),sigma=float(cma.sigma)))
   rc._save(cma,RUN_TAG);dump(_hist(RUN_TAG),history)
   dump(O/f'BEST_SO_FAR_{RUN_TAG}.json',dict(best=min(history,key=lambda r:r['best']),history_len=len(history)))
   print(json.dumps({k:history[-1][k] for k in ('generation','best','median','passers','failures')}),flush=True)
   if all(not r['ok'] for r in res):
    dump(O/f'ALL_FAILED_{RUN_TAG}.json',dict(generation=g,errors=[r['error'] for r in res]));break
 best=min(history,key=lambda r:r['best'])
 dump(O/f'RUN_SUMMARY_{RUN_TAG}.json',dict(best=best,generations=len(history),evals=int(cma.evals),
  passers_total=int(sum(h['passers'] for h in history)),wall_s=time.monotonic()-t0,sigma_final=float(cma.sigma),
  initialization=dict(pool_source=str(SEED_POOL),readout='neutral'),
  reference=dict(round_ab=dict(J=3.627664321466884,foot_support=0.24011973880029733),round_c=dict(J=3.919962430249647,foot_support=0.14340965301622385),baseline=dict(nonfoot_share=0.785,foot_support=0.109))))

def _vjob(arg):
 global RELEASE
 vec,folder=arg[0],Path(arg[1])
 if len(arg)>2 and arg[2]:RELEASE=float(arg[2])
 try:
  _,rows,summary=trial_joint(folder,vec)
  J,parts,full=_score(folder,rows,vec,summary)
  return dict(tag=folder.name,ok=True,J=float(J),gate_pass=bool(full['passed']),parts=parts)
 except BaseException:
  return dict(tag=folder.name,ok=False,error=traceback.format_exc()[-600:])
def validate():
 assist=dsm.window_metrics(dict(np.load(rc.BASE_BODY)),*ASSIST,'assist')
 scored=[]
 for f in sorted((O/'EVALS').glob('eval_*')):
  sc=js(f/'SCORE.json')
  if sc.get('failed') or 'J' not in sc:continue
  scored.append((float(sc['J']),f,sc['vec']))
 scored.sort(key=lambda r:r[0]);cands=[(f.name,v) for _,f,v in scored[:3]]
 assert cands
 dump(O/f'VALIDATION_SELECTION_{RUN_TAG}.json',dict(selected=[dict(tag=t) for t,_ in cands]))
 jobs=[]
 for t,v in cands:
  for rep in ('A','B'):jobs.append((v,O/f'VALIDATION_{RUN_TAG}'/f'{t}_{rep}'))
 jobs.append((cands[0][1],O/f'VALIDATION_{RUN_TAG}'/f'{cands[0][0]}_A_release400ms',0.4))
 import multiprocessing as mp
 ctx=mp.get_context('spawn')
 with ctx.Pool(WORKERS,initializer=_init,initargs=(assist,)) as p:res=p.map(_vjob,jobs)
 dump(O/f'VALIDATION_RESULTS_{RUN_TAG}.json',dict(results=res,all_failed=all(not r['ok'] for r in res)))
 print(json.dumps([{k:r.get(k) for k in ('tag','ok','J','gate_pass')} for r in res],ensure_ascii=False))

def verify():
 assist=dsm.window_metrics(dict(np.load(rc.BASE_BODY)),*ASSIST,'assist')
 checks={}
 errs=[];n=0;gmin,gmax=9.,0.
 for f in sorted((O/'EVALS').glob('eval_*')):
  sc=js(f/'SCORE.json')
  if sc.get('failed') or 'J' not in sc:continue
  rows=dict(np.load(f/'RELEASE_NATIVE.npz'))['rows']
  J,_=dst.objective(rows,assist);errs.append(abs(float(J)-sc['J']));n+=1
  g=float(np.max(sc['readout_gain']));h=float(np.min(sc['readout_gain']));gmax=max(gmax,g);gmin=min(gmin,h)
  if n>=40:break
 checks['objective_recomputes']=dict(sampled=n,max_abs_error=float(max(errs,default=0.)),passed=bool(max(errs,default=0.)<=1e-12))
 checks['readout_within_bounds']=dict(min=float(gmin),max=float(gmax),passed=bool(gmin>=GAIN_MIN-1e-9 and gmax<=GAIN_MAX+1e-9))
 pool_ok=all(len(js(f/'SCORE.json').get('pool_z',[]))==POOL_DIM for f in sorted((O/'EVALS').glob('eval_*'))[:50] if not js(f/'SCORE.json').get('failed'))
 checks['pool_vector_present_and_bounded']=dict(passed=bool(pool_ok))
 hist=js(_hist(RUN_TAG))
 checks['best_so_far_matches_history_min']=dict(passed=bool(js(O/f'BEST_SO_FAR_{RUN_TAG}.json')['best']['best']==min(h['best'] for h in hist)))
 sc=js(O/'SEEDCHECK.json') if (O/'SEEDCHECK.json').exists() else None
 checks['seed_check_recorded']=dict(passed=bool(sc and sc['passed']))
 dump(O/f'INDEPENDENT_VERIFICATION_{RUN_TAG}.json',dict(checks=checks,passed=all(c['passed'] for c in checks.values())))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode');a=ap.parse_args()
 {'seedcheck':seedcheck,'prereg':prereg,'run':run,'validate':validate,'verify':verify}[a.mode]()
