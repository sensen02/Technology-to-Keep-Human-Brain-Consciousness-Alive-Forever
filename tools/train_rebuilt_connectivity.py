"""Retrain the 114 parameters on the rebuilt connectivity, where load feedback reaches the muscles.

Motivation, measured: with the historical edge filter the tactile afferent class had zero outgoing
synapses, so silencing or tripling it left the motor command bit-identical; the rebuild adds those
37,429 entries (0 modified) and puts all 204 motor cells one hop downstream. Running the OLD
parameters on the rebuilt network changes the result, which proves the new edges are read, but the
parameters were tuned while the class was absent, so the score got worse.

This run therefore searches the same 114 dimensions again, on the rebuilt network, starting from the
old optimum. Everything except the connectivity and the parameter vector is held fixed.

modes: prereg / run / report
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import sys,json,math,hashlib,time,traceback,shutil
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'))
import train_stage_a_pool_modes as st
import train_joint_pool_readout as jp
import train_actuator_readout_c as rc
import train_dynamic_support as dst
import dynamic_support_metrics as dsm
from engine.embodied.motor_output import MotorOutputConfig
O=Path('/home/sensen/Desktop/engine修复验收/s5-rebuilt-connectivity-training/20261104T090000Z')
REBUILD=Path('/home/sensen/Desktop/engine修复验收/s5-bristle-edge-rebuild/20261104T050000Z')
BEST=Path('/home/sensen/Desktop/engine修复验收/s5-joint-pool-readout/20261103T080000Z/EVALS/eval_709/SCORE.json')
SET=O/'CONNECTIVITY_SET'
DIM=jp.DIM;POOL=jp.POOL_DIM
SEED=int(os.environ.get('RB_SEED','20261106'))
LAMBDA=16;GENERATIONS=int(os.environ.get('RB_GENERATIONS','24'));EVAL_CAP=LAMBDA*GENERATIONS
WORKERS=int(os.environ.get('RB_WORKERS','4'));TAG=os.environ.get('RB_RUN','R1')
SIGMA0=float(os.environ.get('RB_SIGMA','0.25'));PATIENCE=int(os.environ.get('RB_PATIENCE','5'));RESTART=float(os.environ.get('RB_RESTART','0.25'))
RELEASE=.2
def js(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def make_set():
 if SET.exists():shutil.rmtree(SET)
 SET.mkdir(parents=True)
 for p in sorted(st.B.iterdir()):
  if p.name in ('ACTUAL_We.npz','ACTUAL_Wi.npz'):continue
  (SET/p.name).symlink_to(p)
 shutil.copyfile(REBUILD/'REBUILT_We.npz',SET/'ACTUAL_We.npz')
 shutil.copyfile(REBUILD/'REBUILT_Wi.npz',SET/'ACTUAL_Wi.npz')
 return dict(we=sha(SET/'ACTUAL_We.npz'),wi=sha(SET/'ACTUAL_Wi.npz'))

def trial_rebuilt(folder,vec):
 import engine.embodied.adapters as A
 import run_neural_muscle_closure as c
 orig=A.build_neural_tier
 def patched(config=None,*a,**kw):
  from engine.embodied.adapters import TierConfig as TC
  import dataclasses
  return orig(dataclasses.replace(config or TC(),sign_fill='bristle_only'),*a,**kw)
 A.build_neural_tier=patched;c.build_neural_tier=patched
 old_B=st.B;st.B=SET
 zp,zg=np.asarray(vec,float)[:POOL],np.asarray(vec,float)[POOL:]
 st.CONFIG['neural_motor_config']=MotorOutputConfig(per_actuator_gain=tuple(float(g) for g in jp.gains_of(zg)))
 try:return st.trial(folder,np.clip(zp,-1,1),RELEASE)
 finally:
  st.CONFIG.pop('neural_motor_config',None);st.B=old_B;A.build_neural_tier=orig;c.build_neural_tier=orig

_W={}
def _init(assist):_W['assist']=assist
def _eval(job):
 vec,folder=job
 try:
  _,rows,summary=trial_rebuilt(folder,vec)
  assist=_W['assist'];J,parts=dst.objective(rows,assist);full=dsm.evaluate_native(assist,rows)
  dump(Path(folder)/'SCORE.json',dict(J=J,parts=parts,gate=full,vec=np.asarray(vec,float).tolist(),wall_s=summary['wall_s']))
  return dict(ok=True,J=float(J),parts=parts,gate_pass=bool(full['passed']),error=None)
 except BaseException:
  dump(Path(folder)/'SCORE.json',dict(failed=True,traceback=traceback.format_exc(),vec=np.asarray(vec,float).tolist()))
  return dict(ok=False,J=float('inf'),parts=None,gate_pass=False,error=traceback.format_exc()[-800:])

def prereg():
 O.mkdir(parents=True,exist_ok=True)
 hashes=make_set()
 seed_vec=js(BEST)['vec']
 dump(O/'PREREGISTRATION.json',dict(
  question='with load feedback finally reaching the motor cells, does the plateau at J=3.39 break',
  single_connectivity_change="sign_fill='bristle_only' adds 37,429 bristle rows, modifies none, and puts all 204 motor cells one hop downstream",
  connectivity=hashes,connectivity_set=str(SET),
  parameter_space=dict(dimension=DIM,pool=POOL,readout=DIM-POOL,seed_vector='the disconnected-network optimum'),
  unchanged=['body','muscles','contacts','assist schedule','sensory routing and weights','translation scales','objective','gate'],
  optimizer=dict(name='sep-CMA-ES',lambda_=LAMBDA,sigma0=SIGMA0,generations=GENERATIONS,eval_cap=EVAL_CAP,
   patience=PATIENCE,restart_sigma=RESTART,seed=SEED,workers=WORKERS),
  reference=dict(disconnected_network_best=dict(J=3.3902926855730198,foot_support=0.31795537182301425,nonfoot_share=0.6417682564053153),
   rebuilt_network_old_parameters=dict(J=3.570744035862009,foot_support=0.2537,nonfoot_share=0.6990)),
  acceptance='a break of the plateau means J below 3.3903 with foot support above 0.318 W; the gate is unchanged'))

def _path(tag):return O/f'CMA_STATE_{tag}.npz'
def _hist(tag):return O/f'CMA_HISTORY_{tag}.json'
def run():
 make_set()
 assist=dsm.window_metrics(dict(np.load(jp.rc.BASE_BODY)),*jp.ASSIST,'assist')
 cma=rc.SepCMA(DIM,SIGMA0,LAMBDA,SEED)
 cma.m=np.asarray(js(BEST)['vec'],float).copy()
 history=[]
 if _path(TAG).exists():
  d=dict(np.load(_path(TAG)));cma.m=d['m'];cma.C=d['C'];cma.ps=d['ps'];cma.sigma=float(d['sigma'][0]);cma.generation=int(d['generation'][0]);cma.evals=int(d['evals'][0])
  cma.rng.bit_generator.state={'bit_generator':'PCG64','state':{'state':int(str(d['rng_state'][0]),16),'inc':int(str(d['rng_inc'][0]),16)},'has_uint32':int(d['rng_has'][0]),'uinteger':int(d['rng_u32'][0])}
  history=js(_hist(TAG))
 best_ever=min((h['best'] for h in history),default=float('inf'));best_vec=None
 for f in sorted((O/'EVALS').glob('eval_*')):
  sc=js(f/'SCORE.json')
  if sc.get('failed') or 'J' not in sc:continue
  if abs(float(sc['J'])-best_ever)<=1e-12:best_vec=np.asarray(sc['vec'],float).copy()
 if best_vec is None:best_vec=cma.m.copy();best_ever=float('inf') if not history else best_ever
 stale=0;restarts=[]
 t0=time.monotonic()
 import multiprocessing as mp
 ctx=mp.get_context('spawn');ed=O/'EVALS';ed.mkdir(parents=True,exist_ok=True)
 with ctx.Pool(WORKERS,initializer=_init,initargs=(assist,)) as pool:
  while cma.evals<EVAL_CAP:
   g=cma.generation;xs=cma.ask()
   jobs=[(xs[i],ed/f'eval_{cma.evals+i:03d}') for i in range(LAMBDA)]
   for _,f in jobs:
    if f.exists():raise SystemExit('refusing to overwrite '+str(f))
   res=pool.map(_eval,jobs)
   fits=np.array([r['J'] if r['ok'] else 1e6 for r in res])
   cma.tell(fits)
   if fits.min()<best_ever-1e-12:
    best_ever=float(fits.min());best_vec=np.asarray(xs[int(np.argmin(fits))],float).copy();stale=0
   else:stale+=1
   if stale>=PATIENCE:
    cma.m=best_vec.copy();cma.C=np.ones(DIM);cma.ps=np.zeros(DIM);cma.sigma=RESTART;stale=0
    restarts.append(dict(generation=g,evals=int(cma.evals),best_ever=best_ever))
    dump(O/f'RESTARTS_{TAG}.json',dict(patience=PATIENCE,restart_sigma=RESTART,events=restarts))
   history.append(dict(generation=g,best=float(fits.min()),median=float(np.median(fits)),
    passers=int(sum(r['gate_pass'] for r in res)),failures=int(sum(not r['ok'] for r in res)),
    best_parts=next((r['parts'] for r in res if r['J']==fits.min()),None),evals=int(cma.evals),sigma=float(cma.sigma)))
   np.savez_compressed(_path(TAG),m=cma.m,C=cma.C,ps=cma.ps,sigma=np.array([cma.sigma]),
    generation=np.array([cma.generation]),evals=np.array([cma.evals]),
    rng_state=np.array([format(int(cma.rng.bit_generator.state['state']['state']),'x')]),
    rng_inc=np.array([format(int(cma.rng.bit_generator.state['state']['inc']),'x')]),
    rng_has=np.array([int(cma.rng.bit_generator.state['has_uint32'])]),rng_u32=np.array([int(cma.rng.bit_generator.state['uinteger'])]))
   dump(_hist(TAG),history)
   dump(O/f'BEST_SO_FAR_{TAG}.json',dict(best=min(history,key=lambda r:r['best']),history_len=len(history)))
   print(json.dumps({k:history[-1][k] for k in ('generation','best','median','passers','failures')}),flush=True)
 best=min(history,key=lambda r:r['best'])
 dump(O/f'RUN_SUMMARY_{TAG}.json',dict(best=best,generations=len(history),evals=int(cma.evals),
  passers_total=int(sum(h['passers'] for h in history)),restarts=restarts,wall_s=time.monotonic()-t0,
  connectivity=make_set(),reference=dict(disconnected_best_J=3.3902926855730198,disconnected_best_foot=0.31795537182301425)))

def tactile_sweep():
 """On the rebuilt network, does the newly connected tactile drive help or hurt?"""
 make_set()
 assist=dsm.window_metrics(dict(np.load(jp.rc.BASE_BODY)),*jp.ASSIST,'assist')
 best=None;bestJ=9e9
 for f in sorted((O/'EVALS').glob('eval_*')):
  sc=js(f/'SCORE.json')
  if sc.get('failed') or 'J' not in sc:continue
  if float(sc['J'])<bestJ:bestJ=float(sc['J']);best=np.asarray(sc['vec'],float)
 recs=[]
 for w in (0.0,0.1,0.3,1.0):
  folder=O/'TACTILE_WEIGHT'/('w%03d'%round(w*100))
  if not (folder/'RELEASE_NATIVE.npz').exists():
   if folder.exists():raise SystemExit('refusing to overwrite '+str(folder))
   zp,zg=best[:POOL],best[POOL:]
   st.CONFIG['neural_sensory_channel_weights']={'bristle_touch':float(w)}
   st.CONFIG['neural_motor_config']=MotorOutputConfig(per_actuator_gain=tuple(float(g) for g in jp.gains_of(zg)))
   try:
    import engine.embodied.adapters as A, dataclasses
    import run_neural_muscle_closure as cl
    orig=A.build_neural_tier
    def patched(config=None,*a,**kw):
     from engine.embodied.adapters import TierConfig as TC
     return orig(dataclasses.replace(config or TC(),sign_fill='bristle_only'),*a,**kw)
    A.build_neural_tier=patched;cl.build_neural_tier=patched
    old_B=st.B;st.B=SET
    try:_,rows,_=st.trial(folder,np.clip(zp,-1,1),RELEASE)
    finally:st.B=old_B;A.build_neural_tier=orig;cl.build_neural_tier=orig
   finally:
    st.CONFIG.pop('neural_sensory_channel_weights',None);st.CONFIG.pop('neural_motor_config',None)
   np.savez_compressed(folder/'RELEASE_NATIVE.npz',rows=rows)
  else:rows=dict(np.load(folder/'RELEASE_NATIVE.npz'))['rows']
  J,parts=dst.objective(rows,assist);full=dsm.evaluate_native(assist,rows)
  rec=dict(bristle_weight=float(w),J=float(J),parts=parts,dynamic_pass=bool(full['passed']))
  dump(folder/'SCORE.json',rec);recs.append(rec)
  print('w=%.1f J=%.4f 足载%.4f 非足%.4f'%(w,J,parts['foot_support_mean'],parts['nonfoot_share']),flush=True)
 dump(O/'TACTILE_WEIGHT_SWEEP.json',dict(per_channel_weights=[dict(bristle_weight=r['bristle_weight'],J=r['J'],
  foot_support=r['parts']['foot_support_mean'],nonfoot_share=r['parts']['nonfoot_share'],dynamic_pass=r['dynamic_pass']) for r in recs],
  vector='best vector of this round',meaning='bristle weight 0 removes the rebedded tactile drive without touching the rebuilt edges; comparing it with 1.0 isolates what the added afferent class contributes'))
def report():
 recs=[]
 for f in sorted((O/'EVALS').glob('eval_*')):
  sc=js(f/'SCORE.json')
  if sc.get('failed') or 'J' not in sc:continue
  recs.append((float(sc['J']),f.name,sc))
 recs.sort(key=lambda r:r[0])
 rows=[dict(eval=n,J=round(J,4),foot_support=round(s['parts']['foot_support_mean'],4),
  nonfoot_share=round(s['parts']['nonfoot_share'],4),com_z_min=round(s['parts']['com_z_min'],4),
  first_nonfoot_ms=round(s['parts']['first_nonfoot_s']*1000,2) if s['parts']['first_nonfoot_s'] is not None else None,
  dynamic_pass=bool(s['gate']['passed'])) for J,n,s in recs[:5]]
 dump(O/'SUMMARY.json',dict(top5=rows,evaluations=len(recs),
  reference=dict(disconnected_best=dict(J=3.3902926855730198,foot_support=0.31795537182301425),
   rebuilt_old_parameters=dict(J=3.570744035862009,foot_support=0.2537)),
  best_J=rows[0]['J'] if rows else None,
  beats_disconnected=bool(rows and rows[0]['J']<3.3902926855730198)))
 print(json.dumps(js(O/'SUMMARY.json'),ensure_ascii=False,indent=1))
if __name__=='__main__':
 {'prereg':prereg,'run':run,'tactile':tactile_sweep,'report':report}[sys.argv[1]]()
