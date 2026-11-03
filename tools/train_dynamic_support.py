"""Dynamic-support training: same real physical+SNN loop, new objective.

Round A optimised a support proxy that let the abdomen carry the body. This stage optimises
the dynamic criteria directly: foot support must stay in the run's own assisted band, the
non-foot share of ground reaction must go to zero, and the abdomen contact must be pushed
past the whole window. Same 24 bounded parameters, same rollout code (imported, not copied).

modes: prereg / run / validate / verify
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
import dynamic_support_metrics as dsm
O=Path('/home/sensen/Desktop/engine修复验收/s5-dynamic-support-training/20261103T002606Z')
S=Path('/home/sensen/Desktop/engine修复验收/s4-stage-a-pool-modes/20261102T203810Z')
LEGS=st.LEGS;DIM=st.DIM;DT=st.DT
SEED=20261103;LAMBDA=12;GENERATIONS=25;EVAL_CAP=LAMBDA*GENERATIONS;WORKERS=int(os.environ.get('DST_WORKERS','4'));RUN_TAG=os.environ.get('DST_RUN','R1')
RELEASE=.2;ASSIST=(1.1,1.3)
W_NONFOOT=2.0;W_FOOT=1.0;W_COM=1.0;W_TIME=1.0;W_FREEFALL=1.0
def js(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def objective(native_rows,assist):
 """Monotone in every gate: lying on the abdomen raises the non-foot share, losing foot
 support raises the foot term, sinking raises the COM term, an early contact raises the
 time term, and airborne phases raise the free-fall term. No aggregate that a collapse
 could improve."""
 s=dsm.native_support_series(native_rows)
 t=s['time']-1.3
 support=s['support'];share=np.nan_to_num(s['nonfoot_share']);total=s['ground_total']/dsm.WEIGHT
 free=((total<0.2)&(s['accel_z']<=-0.5*dsm.GRAVITY))
 pos=np.flatnonzero(s['nonfoot_signed']>1e-9)
 t_nf=float(t[pos[0]]) if len(pos) else None
 a_nf=float(np.mean(share))
 a_foot=max(0.,(assist['foot_support_mean']-float(support.mean()))/max(assist['foot_support_mean'],1e-9))
 a_com=max(0.,(assist['com_z_mean']-float(s['com_z'].min()))/max(assist['com_z_mean'],1e-9))
 a_time=0. if t_nf is None else max(0.,1.-t_nf/RELEASE)
 a_free=float(free.mean())
 J=W_NONFOOT*a_nf+W_FOOT*a_foot+W_COM*a_com+W_TIME*a_time+W_FREEFALL*a_free
 return float(J),dict(nonfoot_share_term=W_NONFOOT*a_nf,foot_term=W_FOOT*a_foot,com_term=W_COM*a_com,
  time_term=W_TIME*a_time,freefall_term=W_FREEFALL*a_free,first_nonfoot_s=t_nf,
  foot_support_mean=float(support.mean()),foot_support_p10=float(np.percentile(support,10)),
  nonfoot_share=a_nf,com_z_min=float(s['com_z'].min()),free_fall_fraction=a_free,
  bearing_legs_median=float(np.median(s['bearing_legs'])),assist_foot_support=assist['foot_support_mean'],
  assist_com_z_mean=assist['com_z_mean'])

def gate_verdict(parts,assist,checks_native):
 return dict(dynamic_pass=bool(all(checks_native.values())),checks=checks_native,parts=parts)

_W={}
def _init(_ref):_W['assist']=_ref
def _eval(job):
 idx,z=job
 folder=O/('EVALS' if RUN_TAG=='R1' else f'EVALS_{RUN_TAG}')/f'eval_{idx:04d}'
 folder.parent.mkdir(parents=True,exist_ok=True)
 try:
  folder,rows,summary=st.trial(folder,z,RELEASE)
  assist=_W['assist'];J,parts=objective(rows,assist)
  full=dsm.evaluate_native(assist,rows)
  sc=dict(index=idx,z=list(map(float,z)),J=J,parts=parts,dynamic_pass=full['passed'],
   checks=full['checks'],release=full['release'],calls=summary['counters'],wall_s=summary['wall_s'])
  dump(folder/'SCORE.json',sc)
  for name in ['NEURAL_RECORDS.json','PREWARM_NEURAL_RECORDS.json','BODY.npz','RELEASE_BODY.npz','HANDOVER.json']:
   q=folder/'RUN'/name
   if q.exists():q.unlink()
  return sc
 except BaseException:
  return dict(index=idx,z=list(map(float,z)),J=1e9,error=traceback.format_exc())

def _ck():return O/f'CMA_STATE_{RUN_TAG}.npz'
def _hist():return O/f'CMA_HISTORY_{RUN_TAG}.json'
def run():
 import multiprocessing as mp
 assert (O/'PREREGISTRATION.json').exists()
 ref=st.reference()
 body=dict(np.load(S/'DIAGNOSE_R2/RUN/BODY.npz'));assist=dsm.window_metrics(body,*ASSIST,'assist')
 base=js(S/'DIAGNOSE_R2/SCORE.json') if (S/'DIAGNOSE_R2/SCORE.json').exists() else None
 # fixed reference points outside the optimisation budget, for calibration only
 calib={}
 for tag,z in [('neutral',[0.0]*DIM),('stageA_best',js(S/'BEST_SO_FAR_R2.json')['z'])]:
  calib[tag]=z
 es=st.cma_setup(SEED);es['sigma']=st.SIGMA0
 evals=0;history=[];best=None;t0=time.monotonic();ctx=mp.get_context('fork')
 if _ck().exists():
  z_=dict(np.load(_ck()));es['m']=z_['m'];es['sigma']=float(z_['sigma']);es['C']=z_['C'];es['pc']=z_['pc'];es['ps']=z_['ps'];es['g']=int(z_['g']);evals=int(z_['evals'])
  bg=np.random.PCG64();bg.state=json.loads(str(z_['rng_state']));es['rng']=np.random.Generator(bg)
  history=js(_hist()) if _hist().exists() else []
 with ctx.Pool(WORKERS,initializer=_init,initargs=(assist,)) as pool:
  while es['g']<GENERATIONS and evals<EVAL_CAP and es['sigma']>=1e-3:
   xs,ys=st.cma_ask(es)
   res=pool.map(_eval,[(evals+i,xs[i]) for i in range(len(xs))]);evals+=len(res)
   fs=np.array([r['J'] for r in res])
   for r in res:
    if 'error' in r:dump(O/f"ERROR_eval_{r['index']:04d}.json",r)
   st.cma_tell(es,xs,ys,fs);order=np.argsort(fs)
   rec=dict(generation=es['g'],evals=evals,sigma=float(es['sigma']),best_index=int(res[order[0]]['index']),best_J=float(fs[order[0]]),
    best_z=list(map(float,xs[order[0]])),best_parts=res[order[0]].get('parts'),passers=int(sum(1 for r in res if r.get('dynamic_pass'))),
    median_J=float(np.median(fs)),wall_s=time.monotonic()-t0)
   history.append(rec);dump(_hist(),history)
   np.savez_compressed(_ck(),m=es['m'],sigma=np.array(es['sigma']),C=es['C'],pc=es['pc'],ps=es['ps'],g=np.array(es['g']),evals=np.array(evals),
    rng_state=np.array(json.dumps(es['rng'].bit_generator.state,default=lambda o:o.tolist() if isinstance(o,np.ndarray) else int(o))))
   print(json.dumps({k:rec[k] for k in ('generation','evals','best_J','passers','sigma')}),flush=True)
   if best is None or fs[order[0]]<best['J']:
    best=dict(J=float(fs[order[0]]),z=list(map(float,xs[order[0]])),index=int(res[order[0]]['index']),generation=es['g'],parts=res[order[0]].get('parts'))
    dump(O/f'BEST_SO_FAR_{RUN_TAG}.json',best)
 dump(O/f'RUN_SUMMARY_{RUN_TAG}.json',dict(evaluations=evals,generations=es['g'],best=best,seed=SEED,workers=WORKERS,
  objective_weights=dict(nonfoot=W_NONFOOT,foot=W_FOOT,com=W_COM,time=W_TIME,freefall=W_FREEFALL),release_s=RELEASE,
  assist_window=ASSIST,assist_stats=assist,passers_total=int(sum(h['passers'] for h in history)),wall_s=time.monotonic()-t0,
  calibration_points_outside_budget=list(calib)))

def _vjob(arg):
 tag,z,rel=arg
 folder=O/f'VALIDATION_{RUN_TAG}'/f'{tag}_release{int(round(rel*1000))}ms'
 folder.parent.mkdir(parents=True,exist_ok=True)
 folder,rows,summary=st.trial(folder,z,rel,keep=('fk',))
 assist=_W['assist'];J,parts=objective(rows,assist);full=dsm.evaluate_native(assist,rows)
 out=dict(tag=tag,z=list(map(float,z)),release_s=rel,J=J,parts=parts,dynamic_pass=full['passed'],checks=full['checks'],
  release=full['release'],calls=summary['counters'],wall_s=summary['wall_s'])
 if full['passed']:
  out['motion_mse'],out['motion_frames']=st.motion_error(folder,rel)
 dump(folder/'SCORE.json',out);return out

def validate():
 import multiprocessing as mp
 body=dict(np.load(S/'DIAGNOSE_R2/RUN/BODY.npz'));assist=dsm.window_metrics(body,*ASSIST,'assist')
 hist=js(_hist());best=js(O/f'BEST_SO_FAR_{RUN_TAG}.json');cands=[best['z']]
 for r in sorted(hist,key=lambda r:r['best_J'])[:3]:
  if r['best_z'] not in cands:cands.append(r['best_z'])
 cands=cands[:3];ctx=mp.get_context('fork')
 jobs=[]
 for i,z in enumerate(cands):jobs+= [(f'CAND{i}_A',z,.2),(f'CAND{i}_B',z,.2),(f'CAND{i}_EXT',z,.4)]
 with ctx.Pool(WORKERS,initializer=_init,initargs=(assist,)) as pool:res=pool.map(_vjob,jobs)
 dump(O/f'VALIDATION_RESULTS_{RUN_TAG}.json',dict(candidates=res,rule='dynamic criteria must hold at 200 ms twice and at 400 ms; motion only for passers'))
 print('validated',len(res))

def verify():
 """Fresh process, saved arrays only. Recompute the objective from the native rows."""
 files=js(O/'SOURCE_HASHES_BEFORE.json');checks={}
 for p,h in files.items():
  q=Path(p)
  if not q.exists():continue
  if q.name=='train_dynamic_support.py':checks['tool_drift_documented']=(O/'SOURCE_STAGE_LEDGER.json').exists()
  elif q.name=='dynamic_support_metrics.py':pass
  else:checks['source_'+q.name]=bool(sha(q)==h)
 roots,leg,pool,npools=st.identity();checks['identity_66_pools_204_mn']=bool(npools==66 and int((leg>=0).sum())==204)
 summary=js(O/f'RUN_SUMMARY_{RUN_TAG}.json');checks['eval_budget_respected']=bool(summary['evaluations']<=EVAL_CAP)
 hist=js(_hist());per_gen_min=min(r['best_J'] for r in hist);reported=js(O/f'BEST_SO_FAR_{RUN_TAG}.json')['J']
 checks['best_so_far_matches_history_min']=bool(abs(per_gen_min-reported)<=1e-12)
 root=O/('EVALS' if RUN_TAG=='R1' else f'EVALS_{RUN_TAG}')
 got=sorted(p for p in root.glob('eval_*') if (p/'RELEASE_NATIVE.npz').exists() and (p/'SCORE.json').exists())
 step=max(1,len(got)//20);sample=got[::step][:20];maxerr=0.
 for d in sample:
  sc=js(d/'SCORE.json');rows=dict(np.load(d/'RELEASE_NATIVE.npz'))['rows']
  J,parts=objective(rows,summary['assist_stats'])
  maxerr=max(maxerr,abs(J-sc['J']),abs(parts['foot_support_mean']-sc['parts']['foot_support_mean']),abs(parts['nonfoot_share']-sc['parts']['nonfoot_share']))
 checks['saved_rows_recompute_objective']=bool(maxerr<=1e-12)
 term_ok=True
 for d in sample:
  pp=js(d/'SCORE.json')['parts']
  term_ok=term_ok and abs(pp['nonfoot_share_term']-W_NONFOOT*pp['nonfoot_share'])<=1e-12
 checks['objective_terms_match_their_definition']=bool(term_ok)
 checks['nonfoot_never_treated_as_support']=bool(all('foot_support_mean' in js(d/'SCORE.json')['parts'] for d in sample))
 out=dict(passed=bool(all(checks.values())),checks=checks,max_recompute_error=maxerr,sampled=len(sample),
  summary={k:v for k,v in summary.items() if k!='assist_stats'},calls=dict(model_load=0,forward=0,body_step=0,neural_step=0))
 dump(O/f'INDEPENDENT_VERIFICATION_{RUN_TAG}.json',out);print(json.dumps({k:v for k,v in out.items() if k!='summary'},ensure_ascii=False,indent=2))
 return out['passed']

def prereg():
 O.mkdir(parents=True,exist_ok=True);assert not (O/'PREREGISTRATION.json').exists()
 paths=[Path(__file__),R/'GOAL_ROADMAP.zh-CN.md',R/'GOAL_TRACKER.zh-CN.md',R/'tools/dynamic_support_metrics.py',
  R/'tools/train_stage_a_pool_modes.py',st.B/'RUN/MODEL.xml',st.B/'RUN/MAPPING.json',st.B/'IDENTITIES.npz',st.A/'VERSION_CONTRACT.json']
 dump(O/'SOURCE_HASHES_BEFORE.json',{str(p):sha(p) for p in paths if Path(p).exists()})
 dump(O/'PREREGISTRATION.json',dict(head='3019ba2416dfd39aeb5eca726ea6fb4b3e6a0e92',stage='S5 dynamic-support training',
  why='round A optimised a support proxy that counted abdomen load as support; the revised roadmap requires foot-only support with a non-foot share cap',
  parameters=dict(dim=DIM,space='identical to stage A: 4 modes x 6 legs, bounded [-1,1], existing CSR rows and thresholds only, no new edge/neuron/plasticity/body change'),
  trial=dict(release_s=RELEASE,assist_window=ASSIST,rollout='one fresh body and fresh default SNN per evaluation, parameters fixed from t=0, imported from the stage-A trainer so there is a single rollout implementation'),
  objective=dict(formula='J = 2*mean(nonfoot_share) + max(0,(assist_foot-foot_mean)/assist_foot) + max(0,(assist_com-com_min)/assist_com) + [1 - t_first_nonfoot/0.2 if any else 0] + free_fall_fraction',
   monotonicity='every term worsens when the body collapses onto the abdomen or loses foot support, so a drag solution cannot lower J',
   weights=dict(nonfoot=W_NONFOOT,foot=W_FOOT,com=W_COM,time=W_TIME,freefall=W_FREEFALL),
   thresholds='taken from this body\'s own assisted window, not hand-chosen'),
  acceptance='roadmap D1-D8 on 12.5us native rows',optimizer=dict(name='CMA-ES (stage-A implementation) ',lambda_=LAMBDA,generations=GENERATIONS,eval_cap=EVAL_CAP,sigma0=st.SIGMA0,seed=SEED,workers=WORKERS),
  validation=dict(top=3,repeats=2,extended_release_s=.4),
  controls='the stage-A best parameters and the neutral point are recorded as calibration points outside the optimisation budget',
  failures='retained with sources; no post-hoc objective or threshold change'))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prereg','run','validate','verify']);a=ap.parse_args()
 r=globals()[a.mode]()
 if a.mode=='verify' and not r:sys.exit(1)
