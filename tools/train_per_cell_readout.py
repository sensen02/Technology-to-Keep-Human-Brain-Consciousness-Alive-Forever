"""Can per-motor-neuron readout weights improve on the best static-gain solution?

The analysis of why training stalled found the trained object was a set of scalar rescalings of a
fixed muscle pattern: the 204 motor neurons to 90 muscle assignment was fixed and each muscle had one
gain, so the search could turn the volume knobs but not change the score.

This adds one weight per (actuator, motor neuron) assignment, so the command of a muscle becomes a
WEIGHTED mean of its pool and the relative contribution of each motor neuron becomes searchable. The
pool sizes are small (median two cells), which is exactly why the relative weights matter.

The seed repeats the best recorded 115 dimensional vector and leaves every new weight at one, so the
first evaluation reproduces the recorded 3.4892 exactly; any improvement after that is attributable to
the new readout shape and nothing else. The budget is matched to the parent run, six generations.

modes: prereg / run / report
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import sys,json,time,traceback,shutil,multiprocessing as mp
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'))
O=Path(os.environ.get('PCR_OUT','/home/sensen/Desktop/engine修复验收/s5-per-cell-readout/20261105T220000Z'))
REBUILD=Path('/home/sensen/Desktop/engine修复验收/s5-bristle-edge-rebuild/20261104T050000Z')
V2EV=Path('/home/sensen/Desktop/engine修复验收/s5-rebuilt-connectivity-training/20261104T090000Z/EVALS_V2')
SET=O/'CONNECTIVITY_SET'
BASE_DIM=115
LAM=int(os.environ.get('PCR_LAM','16'))
MU=LAM//2
GENS=int(os.environ.get('PCR_GENS','6'))
CHANNELS=('claw_position','hook_direction','hairplate_angle','campaniform_strain',
          'club_vibration','bristle_touch','multidendritic_noxious')
NCH=len(CHANNELS)          # bristle_touch already rides in the base block, the rest are new
NCH_NEW=NCH-1              # number of extra channel weights appended after the cell block
USE_SYNTH=os.environ.get('PCR_SYNTH','0')=='1'   # owner authorised synthetic readout for empty pools
SIGMA_OLD, SIGMA_NEW = 0.25, 0.10
def js(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def best_v2():
 best=None;bestJ=9e9
 for f in sorted(V2EV.glob('eval_*')):
  sc=js(f/'SCORE.json')
  if sc.get('failed') or 'J' not in sc:continue
  if float(sc['J'])<bestJ:bestJ=float(sc['J']);best=np.asarray(sc['vec'],float)
 return best,bestJ
def make_set():
 if SET.exists():shutil.rmtree(SET)
 SET.mkdir(parents=True)
 import train_stage_a_pool_modes as st
 for p in sorted(st.B.iterdir()):
  if p.name in ('ACTUAL_We.npz','ACTUAL_Wi.npz'):continue
  (SET/p.name).symlink_to(p)
 shutil.copyfile(REBUILD/'REBUILT_We.npz',SET/'ACTUAL_We.npz')
 shutil.copyfile(REBUILD/'REBUILT_Wi.npz',SET/'ACTUAL_Wi.npz')
_CELL_CACHE={}
def cell_dim():
 if 'n' in _CELL_CACHE:return _CELL_CACHE['n']
 lay=O/'LAYOUT.json'
 if lay.exists():
  try:_CELL_CACHE['n']=int(json.loads(lay.read_text())['cell_dim']);return _CELL_CACHE['n']
  except Exception:pass
 man=json.loads((SET/'RUN/MAPPING.json').read_text())['rows']
 _CELL_CACHE['n']=sum(len(r.get('local_indices') or []) for r in man)
 return _CELL_CACHE['n']

def synthetic_plan():
 import json as _j
 from pathlib import Path as _P
 man=_j.loads((SET/'RUN/MAPPING.json').read_text())['rows']
 idx=[i for i,r in enumerate(man) if not (r.get('local_indices') or [])]
 legs={}
 for i,r in enumerate(man):
  if i in idx:continue
  legs.setdefault(r.get('leg'),[]).append(i)
 plan=[]
 for i in idx:
  src=tuple(legs.get(man[i].get('leg'),()))
  plan.append(dict(actuator=man[i].get('actuator'),index=i,leg=man[i].get('leg'),sources=src))
 return plan

def cell_layout():
 """Bootstrap one build to read the exact pool sizes in manifest order."""
 import train_stage_a_pool_modes as st
 import engine.embodied.adapters as A, dataclasses
 orig=A.build_neural_tier
 def patched(config=None,*a,**kw):
  from engine.embodied.adapters import TierConfig as TC
  return orig(dataclasses.replace(config or TC(),sign_fill='bristle_only'),*a,**kw)
 A.build_neural_tier=patched
 old=st.B;st.B=SET
 try:
  import run_neural_muscle_closure as cl  # noqa: F401
  r=st.build_only() if hasattr(st,'build_only') else None
 finally:
  st.B=old;A.build_neural_tier=orig
 return r
def eval_one(args):
 idx,vec,folder=args
 import train_stage_a_pool_modes as st
 import train_joint_pool_readout as jp
 import train_dynamic_support as dst
 import dynamic_support_metrics as dsm
 import engine.embodied.adapters as A, dataclasses
 import run_neural_muscle_closure as cl
 from engine.embodied.motor_output import MotorOutputConfig
 orig=A.build_neural_tier
 def patched(config=None,*a,**kw):
  from engine.embodied.adapters import TierConfig as TC
  return orig(dataclasses.replace(config or TC(),sign_fill='bristle_only'),*a,**kw)
 A.build_neural_tier=patched;cl.build_neural_tier=patched
 old=st.B;st.B=SET
 v=np.asarray(vec,float)
 zp,zg=v[:jp.POOL_DIM],v[jp.POOL_DIM:jp.DIM]
 # z = 0 must mean weight 1, so the seed reproduces the recorded best exactly
 NSYN=len(synthetic_plan()) if USE_SYNTH else 0
 ncell_here=cell_dim()                       # from the built manifest, not from the vector length
 wflat=np.exp(np.clip(v[BASE_DIM:BASE_DIM+ncell_here],-1,1)*np.log(2.0))
 synth_block=v[BASE_DIM+ncell_here+NCH_NEW:]
 kw=dict(per_actuator_gain=tuple(float(g) for g in jp.gains_of(zg)),
  per_cell_weight=tuple(float(x) for x in wflat) if wflat.size else None)
 if USE_SYNTH and NSYN:
  plan=synthetic_plan()
  kw.update(synthetic_index=tuple(p['index'] for p in plan),
   synthetic_sources=tuple(tuple(p['sources']) for p in plan),
   synthetic_gain=tuple(float(np.exp(np.clip(synth_block[j],-1,1)*np.log(4.0))) for j in range(min(NSYN,synth_block.size))))
 st.CONFIG['neural_motor_config']=MotorOutputConfig(**kw)
 chw={c:1.0 for c in CHANNELS}
 chw['bristle_touch']=float(np.exp(np.clip(v[jp.DIM],-1,1)*np.log(4.0)))
 extra=v[BASE_DIM+ncell_here:BASE_DIM+ncell_here+NCH_NEW]
 for j,c in enumerate([c for c in CHANNELS if c!='bristle_touch']):
  if j<extra.size:chw[c]=float(np.exp(np.clip(extra[j],-1,1)*np.log(4.0)))
 st.CONFIG['neural_sensory_channel_weights']=chw
 try:
  _,rows,_=st.trial(folder,np.clip(zp,-1,1),.2)
  assist=dsm.window_metrics(dict(np.load(jp.rc.BASE_BODY)),*jp.ASSIST,'assist')
  J,parts=dst.objective(rows,assist);full=dsm.evaluate_native(assist,rows)
  np.savez_compressed(folder/'RELEASE_NATIVE.npz',rows=rows)
  dump(folder/'SCORE.json',dict(vec=list(map(float,vec)),J=float(J),parts=parts,dynamic_pass=bool(full['passed'])))
  return dict(idx=idx,J=float(J),parts=parts,folder=str(folder))
 except BaseException:
  dump(folder/'SCORE.json',dict(vec=list(map(float,vec)),failed=True,traceback=traceback.format_exc()[-800:]))
  return dict(idx=idx,J=float('inf'),failed=True,folder=str(folder))
 finally:
  st.CONFIG.pop('neural_motor_config',None);st.CONFIG.pop('neural_sensory_channel_weights',None)
  st.B=old;A.build_neural_tier=orig;cl.build_neural_tier=orig
def run():
 make_set()
 import train_stage_a_pool_modes as st
 import train_joint_pool_readout as jp
 vec0,J0=best_v2()
 seedfile=Path(os.environ.get('PCR_SEED',''))
 if seedfile and seedfile.exists():
  sd=json.loads(seedfile.read_text());vec0=np.asarray(sd['vec'],float);J0=float(sd['J'])
  print('从外部种子出发 J=%.4f 维度 %d'%(J0,vec0.size),flush=True)
 # pool sizes in manifest order, needed to size the new block
 import json as _j
 man=_j.loads((st.B/'RUN/MAPPING.json').read_text())
 sizes=[len(r.get('local_indices') or []) for r in man['rows']]
 ncell=sum(sizes)
 synth=synthetic_plan() if USE_SYNTH else []
 NSN=len(synth)
 D=BASE_DIM+ncell+NCH_NEW+NSN
 dump(O/'LAYOUT.json',dict(base_dim=BASE_DIM,cell_dim=ncell,total_dim=D,pool_sizes=sizes,
  seed='best recorded 115 dimensional vector, all new cell weights at one so the first evaluation reproduces it',
  sigma=dict(old=SIGMA_OLD,new=SIGMA_NEW),lambda_pop=LAM,mu=MU,generations=GENS,
  note='a weight w_i enters as w_i/sum(w) inside the pool, so the per-actuator gain keeps the overall scale and the new degrees of freedom are purely relative'))
 v0=np.asarray(vec0,float).ravel()
 if v0.size==D:x=v0.copy()
 elif v0.size<D:x=np.concatenate([v0,np.zeros(D-v0.size)])
 elif v0.size in (BASE_DIM,BASE_DIM+ncell):x=np.concatenate([v0,np.zeros(D-v0.size)])
 else:raise SystemExit(f'unexpected seed size {v0.size}, want {BASE_DIM} or {D}')
 print('seed size %d, cell weights %s'%(x.size,'inherited' if np.any(x[BASE_DIM:]) else 'all one'),flush=True)
 sig=np.concatenate([np.full(BASE_DIM,SIGMA_OLD),np.full(ncell,SIGMA_NEW),np.full(NCH_NEW,SIGMA_NEW),np.full(NSN,SIGMA_NEW)])
 w=np.ones(MU)/MU;hist=[]
 ctx=mp.get_context('spawn');pool=ctx.Pool(4)
 for g in range(GENS):
  if g==0:z=np.zeros((LAM,D))
  else:z=np.random.randn(LAM,D)
  X=x[None,:]+sig[None,:]*z
  X[0]=x if g>0 else X[0]   # the first member is always the current mean, never a fresh sample
  # the trial runner creates its own folder, so only a refusal check is done here
  jobs=[(k,X[k],O/f'EVALS/eval_{g*LAM+k:03d}') for k in range(LAM)]
  for _,_,f in jobs:
   if f.exists():raise SystemExit('refusing to overwrite '+str(f))
  res=pool.map(eval_one,jobs)
  J=np.array([r['J'] for r in res]);order=np.argsort(J)[:MU]
  x=x+np.sum(w[:,None]*(X[order]-x),axis=0)
  rec=dict(generation=g,best=float(J.min()),median=float(np.median(J)),mean=float(J.mean()),best_idx=int(np.argmin(J)),best_folder=res[int(np.argmin(J))]['folder'])
  hist.append(rec);dump(O/'CMA_HISTORY.json',hist)
  print('gen %d 最优 %.4f 中位 %.4f 均值 %.4f'%(g,J.min(),np.median(J),J.mean()),flush=True)
 pool.close();pool.join()
 best=min(hist,key=lambda r:r['best'])
 dump(O/'RUN_SUMMARY.json',dict(best_generation=best['generation'],best_J=best['best'],history=hist,
  seed_J=float(J0),evaluations=GENS*LAM,note='seed evaluation is the recorded best, so improvement means the new readout shape helped'))
 print('种子 %.4f → 最优 %.4f（%d 次评估）'%(J0,best['best'],GENS*LAM))
def report():
 d=js(O/'RUN_SUMMARY.json')
 seed=d['seed_J'];best=d['best_J']
 dump(O/'SUMMARY.json',dict(seed_J=seed,best_J=best,improved=bool(best<seed-1e-9),
  delta=float(seed-best),evaluations=d['evaluations'],history=d['history'],
  comparability='the parent 115 dimensional run reached 3.5279 after 96 evaluations and 3.4892 after 768; the seed here is that same 3.4892, so a lower best after 96 evaluations is the comparison that matters'))
 print('种子 %.4f → 96 次评估后最优 %.4f（%s，Δ%.4f）'%(seed,best,'改善' if best<seed else '无改善',seed-best))
if __name__=='__main__':
 {'prereg':lambda: (O.mkdir(parents=True,exist_ok=True),make_set(),dump(O/'PREREGISTRATION.json',dict(
  question='does giving each motor neuron its own readout weight let the search improve on the recorded best',
  design='one weight per actuator times motor neuron assignment, entering as a normalised weighted mean inside the pool',
  seed='recorded best 115 dimensional vector with all new weights at one',generations=GENS,lambda_pop=LAM,mu=MU,
  acceptance='reported as it comes out; a low improvement is informative too'))),
  'run':run,'report':report}[sys.argv[1]]()
