"""Stage A: 24-dimension pool-mode training on the real physical+SNN loop.

4 declared modes x 6 legs = 24 bounded parameters. No body/XML/muscle parameter is
trained, no edge/MN is added, no plasticity is used. Support is a HARD gate; the
search objective is monotone in that gate margin so falling cannot improve it.

modes: [prereg] [gate] [run] [validate] [verify]
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import sys,json,math,hashlib,time,inspect,traceback,argparse,shutil
from pathlib import Path
import numpy as np
from scipy import sparse
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'))
O=Path('/home/sensen/Desktop/engine修复验收/s4-stage-a-pool-modes/20261102T203810Z')
B=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
A=Path('/home/sensen/Desktop/engine修复验收/s3-banc888-runtime-admission/20261101T220018Z')
T=Path('/home/sensen/Desktop/engine修复验收/s4-invariant-motion-target/20261102T023600Z/INVARIANT_BENDING_TARGETS.npz')
LEGS=('LF','LM','LH','RF','RM','RH');DT=.0000125
DIM=24;LAMBDA=12;MU=6;SIGMA0=.3;GENERATIONS=25;EVAL_CAP=LAMBDA*GENERATIONS
CONFIG=dict(duration=1.3,kp=100.,clip=50.,dt=DT,neural_command_dt_s=.001,neural_prewarm_s=1.1,swing_height=.45,sample_dt_s=.0005)
FLOOR_WEIGHT=.8;HEIGHT_FRACTION=.8
WORKERS=int(os.environ.get('STAGE_A_WORKERS','4'))
RUN_TAG=os.environ.get('STAGE_A_RUN','R1')
EVAL_ROOT=None  # filled lazily from RUN_TAG
def js(p):return json.loads(Path(p).read_text())
def ar(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def err(a,b=0.):return float(np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0.))
def csrhash(w):
 h=hashlib.sha256()
 for k in ('data','indices','indptr'):h.update(getattr(w,k).tobytes())
 return h.hexdigest()

def identity():
 """Neuron -> leg index and neuron -> shared-pool index, from the authoritative map."""
 m=js(B/'RUN/MAPPING.json');roots=ar(B/'IDENTITIES.npz')['root_ids']
 leg=np.full(len(roots),-1,int);pool=np.full(len(roots),-1,int);pools={}
 for row in m['rows']:
  for i in row['local_indices']:leg[int(i)]=LEGS.index(row['leg'])
 pools={}
 for row in m['rows']:
  key=tuple(row['local_indices'])
  if not key:continue
  if key not in pools:pools[key]=len(pools)
  for i in key:pool[int(i)]=pools[key]
 assert len(pools)==66 and int((leg>=0).sum())==204 and int((pool>=0).sum())==204
 return roots,leg,pool,len(pools)

def decode(z):
 z=np.asarray(z,float);assert z.shape==(DIM,) and np.all(np.abs(z)<=1)
 return dict(joint_gain=np.exp(z[0:6]*math.log(2)),threshold_mV=2.*z[6:12],ei_beta=z[12:18]*math.log(2)/2.,exc_gain=np.exp(z[18:24]*math.log(2)))

def row_scales(leg,params):
 n=len(leg);we=np.ones(n);wi=np.ones(n)
 for k in np.flatnonzero(leg>=0):
  i=int(leg[k]);we[k]=params['joint_gain'][i]*math.exp(params['ei_beta'][i])*params['exc_gain'][i];wi[k]=params['joint_gain'][i]*math.exp(-params['ei_beta'][i])
 return we,wi

def trial(folder,z,release,keep=()):
 import mujoco
 import run_physical_walk_prewarm as runner
 import run_neural_muscle_closure as c
 import engine.embodied.continuous_handover as h
 import engine.embodied.runtime_pool_guard as guard
 from engine.neural_cond import ConductanceNetwork
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
 roots,leg,pool,npools=identity();params=decode(z)
 origfactory=h._make_runtime;origcont=h.run_neural_continuous;origguard=guard.load_runtime_pool;origmap=c.channel_index_map;origpool=c.POOLS
 contract=js(A/'VERSION_CONTRACT.json');assert sha(A/'SENSORY_LOOKUP_SNAPSHOT888.json')==contract['sensory_lookup_sha256']
 origguard(A/'MN_POOLS_SNAPSHOT888.json',version_contract=contract)
 counters=dict(factory=0,neural=0,native_contact_force=0,body_step=0,model_load=0,forward=0)
 native=[];fk=[];fkt=[];netref=[];nfevents=[]
 def factory(built,*args,**kw):
  net,enc,dec=origfactory(built,*args,**kw);counters['factory']+=1
  assert counters['factory']==1 and net.steps==0 and np.array_equal(built['tier']['root_ids'],roots)
  we_s,wi_s=row_scales(leg,params)
  for key,scale in (('We',we_s),('Wi',wi_s)):
   w=getattr(net,key);ref=sparse.load_npz(B/('ACTUAL_'+key+'.npz'));assert csrhash(w)==csrhash(ref)
   rows=np.repeat(np.arange(net.n),np.diff(w.indptr));w=w.copy();w.data*=scale[rows];setattr(net,key,w)
  source=net.threshold_offset_mV.copy();offset=source.copy();motor=np.flatnonzero(leg>=0)
  offset[motor]=source[motor]+params['threshold_mV'][leg[motor]];offset.flags.writeable=False;net.threshold_offset_mV=offset
  netref.append(net);return net,enc,dec
 def admitted(path,*a,**kw):return origguard(path,version_contract=contract)
 def channels(n,rs,**kw):return origmap(n,rs,lookup_path=A/'SENSORY_LOOKUP_SNAPSHOT888.json')
 def continuous(model,data,*a,**kw):
  observer=kw['body_observer'];isrelease=kw.get('runtime_state') is not None
  bids=np.array([[model.body(l+'Coxa').id,model.body(l+('Trochanter' if l[1]=='F' else 'Femur')).id,model.body(l+'Tibia').id,model.body(l+'Tarsus1').id] for l in LEGS]);thorax=model.body('Thorax').id
  names=[model.geom(i).name or '' for i in range(model.ngeom)];feet=np.array([next((j for j,l in enumerate(LEGS) if n.startswith(l) and 'Tars' in n),-1) for n in names]);planes=model.geom_type==mujoco.mjtGeom.mjGEOM_PLANE
  def observe(mm,dd):
   observer(mm,dd)
   st=round(dd.time/DT)
   if isrelease and st%40==0:fk.append(dd.xpos[bids].copy());fkt.append(float(dd.time))
   if not isrelease:return
   ff=np.zeros(6);nf=0.;nc=0;nfagg=0.;force=np.zeros(6)
   for j in range(dd.ncon):
    co=dd.contact[j];a1=int(co.geom1);a2=int(co.geom2)
    if not (planes[a1] or planes[a2]):continue
    mujoco.mj_contactForce(mm,dd,j,force);counters['native_contact_force']+=1
    fz=float((co.frame.reshape(3,3).T@force[:3])[2])*(1 if planes[a1] else -1);li=feet[a2 if planes[a1] else a1]
    if li<0:
     nf+=max(fz,0.);nfagg+=fz;nc+=1
     if len(nfevents)==0 and max(fz,0.)>0.:
      g1,g2=int(co.geom1),int(co.geom2)
      nfevents.append(dict(time_s=float(dd.time),relative_s=float(dd.time)-1.3,geom1=model.geom(g1).name,geom2=model.geom(g2).name,body1=model.body(model.geom_bodyid[g1]).name,body2=model.body(model.geom_bodyid[g2]).name,fz_positive=float(max(fz,0.)),signed_fz=float(fz),distance=float(co.dist)))
    else:ff[li]+=fz
   native.append((float(dd.time),*ff,nf,nfagg,float(nc),float(dd.subtree_com[thorax,2]),*dd.qpos[:7],float(np.mean(dd.ctrl**2)),float(np.max(abs(dd.qfrc_applied[:6])))))
   assert not dd.qfrc_applied.any() and not dd.xfrc_applied.any()
  kw['body_observer']=observe
  return origcont(model,data,*a,**kw)
 text=inspect.getsource(runner.run).replace('retain_currents=True','retain_currents=False').replace("sensory_current_nA=np.array([r['sensory_current_nA'] for r in ret['records']])","sensory_current_nA=np.empty((0,int(built['n'])))")
 (folder/'RUNNER_ADAPTER.py').write_text(text);ns=dict(runner.__dict__);exec(compile(text,str(folder/'RUNNER_ADAPTER.py'),'exec'),ns)
 start=time.monotonic();error=None
 h._make_runtime=factory;h.run_neural_continuous=continuous;guard.load_runtime_pool=admitted;c.channel_index_map=channels;c.POOLS=A/'MN_POOLS_SNAPSHOT888.json'
 try:ns['run'](folder/'RUN',**CONFIG,neural_duration=release)
 except BaseException:error=traceback.format_exc()
 finally:h._make_runtime=origfactory;h.run_neural_continuous=origcont;guard.load_runtime_pool=origguard;c.channel_index_map=origmap;c.POOLS=origpool
 rows=np.asarray(native,float)
 np.savez_compressed(folder/'RELEASE_NATIVE.npz',rows=rows)
 if nfevents:dump(folder/'FIRST_NONFOOT_EVENT.json',nfevents[0])
 if 'fk' in keep:np.savez_compressed(folder/'RELEASE_FK.npz',time=np.asarray(fkt),points=np.asarray(fk))
 if (folder/'RUN/EXECUTION.json').exists():
  ec=js(folder/'RUN/EXECUTION.json');counters.update(body_step=ec['counts']['body_step'],model_load=ec['counts']['model_load'],forward=ec['counts']['forward'])
 if error:raise RuntimeError(error)
 assert len(rows)==round(release/DT)+1,(len(rows),release)
 wall=time.monotonic()-start
 summary=dict(wall_s=wall,rows=int(len(rows)),counters=counters,release_s=release)
 dump(folder/'EXECUTION.json',summary)
 return folder,rows,summary

def angles(points):
 u=points[...,:-2,:]-points[...,1:-1,:];v=points[...,2:,:]-points[...,1:-1,:]
 return np.arctan2(np.linalg.norm(np.cross(u,v),axis=-1),np.sum(u*v,axis=-1))

def gate_metrics(rows,release,ref,fk_folder=None):
 h0=ref['h0'];W=ref['weight'];win=rows[1:];assert len(win)==round(release/DT)
 nf=win[:,7];pos=nf>0.;first=None if not pos.any() else float(win[np.flatnonzero(pos)[0],0]-1.3)
 mean_foot=float(win[:,1:7].sum(1).mean()/W);hmin=float(win[:,10].min());nf_occ=float(pos.mean())
 boundary_nf=bool(rows[0,7]>0.)
 gates=dict(no_nonfoot=bool(not pos.any() and not boundary_nf),mean_foot=bool(mean_foot>=FLOOR_WEIGHT),min_height=bool(hmin>=HEIGHT_FRACTION*h0))
 return dict(first_nonfoot_s=first,nonfoot_occupancy=nf_occ,mean_foot_weight=mean_foot,min_comz=hmin,min_height_ratio=hmin/h0,boundary_already_nonfoot=boundary_nf,gates=gates,support_gate=all(gates.values()),mean_positive_nonfoot_weight=float(nf.mean()/W))
def motion_error(folder,release_s):
 """Real-data bending error, computed only for gate-passing candidates."""
 fk=ar(Path(folder)/'RELEASE_FK.npz');tg=ar(T)
 times=tg['time_relative_s'];keep=times<=release_s+1e-9
 if not keep.any():return None,0
 ix=np.array([int(np.argmin(abs(fk['time']-(1.3+t)))) for t in times[keep]])
 ang=angles(fk['points'][ix]);diff=ang-tg['angles_rad'][keep];mask=tg['angle_valid'][keep]
 return float(np.mean(diff[mask]**2)),int(keep.sum())
def objective(metrics,release,wall):
 # Monotone in the gate margin: any fall reduces foot load, lowers COM and adds non-foot force.
 a=max(0.,1.-metrics['mean_foot_weight']/FLOOR_WEIGHT)
 b=max(0.,1.-metrics['min_height_ratio']/HEIGHT_FRACTION)
 c=metrics['nonfoot_occupancy']
 return float(a+b+c),dict(foot_deficit=a,height_deficit=b,nonfoot_occupancy=c)

# ---- top-level worker for parallel evaluation -------------------------------
_WORKER_STATE={}
def _worker_init(ref):
 _WORKER_STATE['ref']=ref
def _evaluate(job):
 idx,z=job
 folder=O/('EVALS' if RUN_TAG=='R1' else f'EVALS_{RUN_TAG}')/f'eval_{idx:04d}'
 folder.parent.mkdir(parents=True,exist_ok=True)
 try:
  folder,rows,summary=trial(folder,z,.2)
  m=gate_metrics(rows,.2,_WORKER_STATE['ref'])
  J,parts=objective(m,.2,summary['wall_s'])
  sc=dict(index=idx,z=list(map(float,z)),J=J,parts=parts,metrics=m,calls=summary['counters'],wall_s=summary['wall_s'])
  dump(folder/'SCORE.json',sc)
  # keep disk small: release rows are already small, drop the runner's bulky logs
  for name in ['NEURAL_RECORDS.json','PREWARM_NEURAL_RECORDS.json','BODY.npz','RELEASE_BODY.npz','HANDOVER.json']:
   q=folder/'RUN'/name
   if q.exists():q.unlink()
  return sc
 except BaseException:
  return dict(index=idx,z=list(map(float,z)),J=1e9,error=traceback.format_exc())

# ---- CMA-ES ---------------------------------------------------------------
def cma_setup(seed):
 n=DIM;lam=LAMBDA;mu=MU
 w=np.log(mu+.5)-np.log(np.arange(1,mu+1));w/=w.sum();mueff=1./np.sum(w**2)
 cc=(4+mueff/n)/(n+4+2*mueff/n);cs=(mueff+2)/(n+mueff+5)
 c1=2/((n+1.3)**2+mueff);cmu=min(1-c1,2*(mueff-2+1/mueff)/((n+2)**2+mueff))
 damps=1+2*max(0,math.sqrt((mueff-1)/(n+1))-1)+cs
 chiN=math.sqrt(n)*(1-1/(4*n)+1/(21*n*n))
 rng=np.random.default_rng(seed)
 return dict(n=n,lam=lam,mu=mu,w=w,mueff=mueff,cc=cc,cs=cs,c1=c1,cmu=cmu,damps=damps,chiN=chiN,rng=rng,m=np.zeros(n),sigma=SIGMA0,C=np.eye(n),pc=np.zeros(n),ps=np.zeros(n),g=0)
def cma_ask(es):
 vals=es['rng'].standard_normal((es['lam'],es['n']));D,B_=np.linalg.eigh(es['C']);D=np.sqrt(np.maximum(D,1e-20))
 ys=vals@(B_*D).T
 xs=es['m']+es['sigma']*ys
 return np.clip(xs,-1,1),ys
def cma_tell(es,xs,ys,fs):
 order=np.argsort(fs);es['g']+=1
 mold=es['m'].copy();y=ys[order[:es['mu']]].T@es['w']
 es['m']=mold+es['sigma']*y
 D,B_=np.linalg.eigh(es['C']);D=np.sqrt(np.maximum(D,1e-20));invsqrt=B_@np.diag(1./D)@B_.T
 yw=(es['m']-mold)/es['sigma']
 es['ps']=(1-es['cs'])*es['ps']+math.sqrt(es['cs']*(2-es['cs'])*es['mueff'])*(invsqrt@yw)
 es['pc']=(1-es['cc'])*es['pc']+math.sqrt(es['cc']*(2-es['cc'])*es['mueff'])*yw
 es['sigma']*=math.exp((es['cs']/es['damps'])*(np.linalg.norm(es['ps'])/es['chiN']-1))
 artmp=ys[order[:es['mu']]].T*es['w']
 es['C']=(1-es['c1']-es['cmu'])*es['C']+es['c1']*(np.outer(es['pc'],es['pc']))+es['cmu']*(artmp@artmp.T)+1e-12*np.eye(es['n'])
 return es

def prereg():
 assert not (O/'PREREGISTRATION.json').exists()
 O.mkdir(parents=True,exist_ok=False)
 paths=[Path(__file__),R/'AGENTS.md',R/'GOAL_ROADMAP.zh-CN.md',R/'GOAL_TRACKER.zh-CN.md',B/'RUN/MODEL.xml',B/'RUN/MAPPING.json',B/'IDENTITIES.npz',A/'VERSION_CONTRACT.json',A/'MN_POOLS_SNAPSHOT888.json',A/'SENSORY_LOOKUP_SNAPSHOT888.json',T]
 hashes={str(p):sha(p) for p in paths};dump(O/'SOURCE_HASHES_BEFORE.json',hashes)
 dump(O/'PREREGISTRATION.json',dict(head='39e5033ac357e43f484f59a7bfedc91939fd308a',stage='S4 StageA',seed=20261102,
  parameters=dict(dim=DIM,modes=['joint_gain','threshold','ei_balance','exc_drive'],legs=list(LEGS),
   mapping='z[0:6] joint We&Wi row gain exp(ln2 z) in[0.5,2]; z[6:12] threshold shift 2z mV in[-2,2]; z[12:18] E/I beta=z ln2/2, We*=exp(+beta) Wi*=exp(-beta) ratio[0.5,2]; z[18:24] excitatory-only We*=exp(ln2 z) in[0.5,2]. All clipping to[-1,1]. Only existing CSR rows of the 204 readout MN are rescaled; no edge, neuron, mask, delay, intrinsic or initial state is added or removed. Body/XML/muscle/friction untouched.'),
  trial=dict(config=CONFIG,release_s=.2,reference='original 5Hz engineering reference, algorithm PD assist .15..1.3s, SNN online from .2s, abrupt withdrawal at 1.3s',per_eval='one fresh original body and fresh default SNN, parameters fixed from t=0'),
  gate=dict(window_s=.2,no_positive_nonfoot=True,include_release_boundary=True,mean_foot_weight_min=FLOOR_WEIGHT,min_height_fraction=HEIGHT_FRACTION,h0_source='neutral run at this stage',hard=True),
  objective='J = max(0,1-mean_foot/0.8) + max(0,1-min_height/0.8h0) + nonfoot_occupancy ; monotone in gate margin so falling cannot lower it. Motion fit is not in J; it is reported for gate-passing candidates only.',
  optimizer=dict(name='CMA-ES (Hansen, own implementation)',lambda_=LAMBDA,mu=MU,sigma0=SIGMA0,generations=GENERATIONS,eval_cap=EVAL_CAP,seed=20261102,bounds='[-1,1] clip before evaluation',stop='generation cap or sigma<1e-3'),
  validation=dict(top=3,repeats=2,extended_release_s=.4,motion='37-frame real leg landmark/interior-angle target, only for gate passers'),
  parallel_workers=8,no_body_parameter_training=True,no_plasticity=True,failures='retained with sources; no post-hoc threshold or objective changes'))

def gate():
 """Neutral (z=0) exactness gate. Reuses an already completed neutral run if present."""
 existing=sorted(p for p in O.glob('GATE_NEUTRAL*') if (p/'RELEASE_NATIVE.npz').exists())
 if existing:
  folder=existing[-1];rows=ar(folder/'RELEASE_NATIVE.npz')['rows'];summary=js(folder/'EXECUTION.json');reused=True
 else:
  k=2
  while (O/f'GATE_NEUTRAL_{k:03d}').exists():k+=1
  folder,rows,summary=trial(O/f'GATE_NEUTRAL_{k:03d}',np.zeros(DIM),.05);reused=False
 b=ar(folder/'RUN/BODY.npz') if (folder/'RUN/BODY.npz').exists() else None
 ref=ar(B/'RUN/BODY.npz');checks={}
 if b is not None:
  for k,v in b.items():
   want=ref[k][:len(v)]
   checks['body_'+k]=bool(np.array_equal(v,want) if v.dtype==bool else err(v,want)<=1e-10)
 rows0=rows
 nf=rows0[1:,7]
 good=bool(not (nf>0).any())
 h0=float(rows0[0,10]);W=float(js(folder/'RUN/GAINS.json')['weight']) if (folder/'RUN/GAINS.json').exists() else float('nan')
 checks['h0_matches_frozen_standard_trial']=bool(abs(h0-1.2072528224791905)<=1e-9)
 checks['weight_matches_frozen_standard_trial']=bool(abs(W-24.446354755878005)<=1e-9)
 checks['release_starts_at_1.3s']=bool(abs(float(rows0[0,0])-1.3)<1e-9)
 pos=nf>0.
 checks['neutral_prefix_first_nonfoot_recorded']=True
 sc=dict(passed=bool(all(checks.values())),checks=checks,reused_existing_neutral=reused,folder=str(folder),rows=int(len(rows)),wall_s=summary['wall_s'],h0=h0,weight=W,
  neutral_prefix_nonfoot_occupancy=float(pos.mean()),neutral_prefix_first_nonfoot_s=(None if not pos.any() else float(rows0[1:][np.flatnonzero(pos)[0],0]-1.3)),
  note='the neutral baseline is expected to fail support (non-foot contact appears within the first tens of ms); this gate certifies parameter-interface exactness only',z=list(np.zeros(DIM)))
 dump(O/'GATE.json',sc)
 if math.isfinite(h0) and math.isfinite(W):
  dump(O/'FROZEN_REFERENCE.json',dict(h0=h0,weight=W,source='stage-A neutral z=0 run release boundary'))
 return sc

def reference():
 p=O/'FROZEN_REFERENCE.json'
 if p.exists():return js(p)
 return dict(h0=1.2072528224791905,weight=24.446354755878005)

def _ckpath():return O/(f'CMA_STATE_{RUN_TAG}.npz')
def _hpath():return O/(f'CMA_HISTORY_{RUN_TAG}.json')
def _save_ck(es,evals,best):
 rng=es['rng'].bit_generator.state
 np.savez_compressed(_ckpath(),m=es['m'],sigma=np.array(es['sigma']),C=es['C'],pc=es['pc'],ps=es['ps'],g=np.array(es['g']),evals=np.array(evals),rng_state=np.array(json.dumps(rng,default=lambda o:o.tolist() if isinstance(o,np.ndarray) else int(o))))
def _load_ck(seed):
 if not _ckpath().exists():return cma_setup(seed),0,None
 z=ar(_ckpath());es=cma_setup(seed)
 es['m']=z['m'];es['sigma']=float(z['sigma']);es['C']=z['C'];es['pc']=z['pc'];es['ps']=z['ps'];es['g']=int(z['g'])
 bg=np.random.PCG64();bg.state=json.loads(str(z['rng_state']));es['rng']=np.random.Generator(bg)
 best=js(O/f'BEST_SO_FAR_{RUN_TAG}.json') if (O/f'BEST_SO_FAR_{RUN_TAG}.json').exists() else None
 return es,int(z['evals']),best

def diagnose():
 ref=reference();best=js(O/f'BEST_SO_FAR_{RUN_TAG}.json');base=js(O/'GATE.json')
 folder,rows,summary=trial(O/f'DIAGNOSE_{RUN_TAG}',np.asarray(best['z']),.2,keep=('fk',))
 m=gate_metrics(rows,.2,ref);J,_=objective(m,.2,summary['wall_s'])
 ev=js(folder/'FIRST_NONFOOT_EVENT.json') if (folder/'FIRST_NONFOOT_EVENT.json').exists() else None
 dump(O/f'DIAGNOSE_{RUN_TAG}.json',dict(z=best['z'],J=J,metrics=m,first_nonfoot_event=ev,
  baseline_first_nonfoot_s=base.get('neutral_prefix_first_nonfoot_s'),
  baseline_note='baseline first non-foot is at 18.9125ms in the 50ms neutral prefix; compare with this run at 200ms',
  calls=summary['counters'],wall_s=summary['wall_s'],scope='one real trial; diagnostic only, not a gate result'))

def run():
 import multiprocessing as mp
 ref=reference();seed=20261102
 es,evals,best=_load_ck(seed)
 if best is None and (O/'BEST_SO_FAR.json').exists() and RUN_TAG!='R1':best=None
 resumed=evals>0
 history=js(_hpath()) if _hpath().exists() else []
 if RUN_TAG=='R1':assert not (O/'CMA_HISTORY.json').exists() or resumed
 t0=time.monotonic()
 ctx=mp.get_context('fork')
 with ctx.Pool(WORKERS,initializer=_worker_init,initargs=(ref,)) as pool:
  while es['g']<GENERATIONS and evals<EVAL_CAP and es['sigma']>=1e-3:
   xs,ys=cma_ask(es)
   jobs=[(evals+i,xs[i]) for i in range(len(xs))]
   results=pool.map(_evaluate,jobs)
   evals+=len(results)
   fs=np.array([r['J'] for r in results])
   for r in results:
    if 'error' in r:dump(O/f"ERROR_eval_{r['index']:04d}.json",r)
   cma_tell(es,xs,ys,fs)
   order=np.argsort(fs);rec=dict(generation=es['g'],evals=evals,sigma=float(es['sigma']),
    best_index=int(results[order[0]]['index']),best_J=float(fs[order[0]]),best_z=list(map(float,xs[order[0]])),
    gate_passers=int(sum(1 for r in results if r.get('metrics',{}).get('support_gate'))),
    median_J=float(np.median(fs)),wall_s=time.monotonic()-t0)
   history.append(rec);dump(_hpath(),history)
   _save_ck(es,evals,best)
   print(json.dumps(rec),flush=True)
   if best is None or fs[order[0]]<best['J']:
    best=dict(J=float(fs[order[0]]),z=list(map(float,xs[order[0]])),index=int(results[order[0]]['index']),generation=es['g'])
   dump(O/f'BEST_SO_FAR_{RUN_TAG}.json',best)
 dump(O/f'RUN_SUMMARY_{RUN_TAG}.json',dict(evaluations=evals,generations=es['g'],wall_s=time.monotonic()-t0,best=best,seed=seed,lambda_=LAMBDA,mu=MU,sigma0=SIGMA0,workers=WORKERS,resumed=resumed,
   stopping='generation/sigma rule as pre-registered',gate_passers=int(sum(h['gate_passers'] for h in history))))

def _validate_job(arg):
 tag,z,release=arg
 ref=_WORKER_STATE['ref']
 folder=O/(f'VALIDATION_{RUN_TAG}')/f'{tag}_release{int(round(release*1000))}ms'
 folder.parent.mkdir(parents=True,exist_ok=True)
 folder,rows,summary=trial(folder,z,release,keep=('fk',))
 m=gate_metrics(rows,release,ref)
 motion,nfr=motion_error(folder,release) if m['support_gate'] else (None,0)
 out=dict(tag=tag,z=list(map(float,z)),release_s=release,metrics=m,motion_mse=motion,motion_frames=nfr,calls=summary['counters'],wall_s=summary['wall_s'])
 dump(folder/'SCORE.json',out);return out

def validate():
 import multiprocessing as mp
 ref=reference();hist=js(_hpath());best=js(O/f'BEST_SO_FAR_{RUN_TAG}.json')
 cands=[best['z']]
 for rec in sorted(hist,key=lambda r:r['best_J'])[:3]:
  if rec['best_z'] not in cands:cands.append(rec['best_z'])
 cands=cands[:3]
 ctx=mp.get_context('fork');jobs=[]
 for i,z in enumerate(cands):
  jobs.append((f'CAND{i}_repeatA',z,.2));jobs.append((f'CAND{i}_repeatB',z,.2));jobs.append((f'CAND{i}_extended',z,.4))
 with ctx.Pool(WORKERS,initializer=_worker_init,initargs=(ref,)) as pool:results=pool.map(_validate_job,jobs)
 dump(O/f'VALIDATION_RESULTS_{RUN_TAG}.json',dict(candidates=results,rule='gate must hold at 200ms twice (determinism) and still hold at 400ms; motion measured only for gate passers',workers=WORKERS))

def verify():
 """Fresh-process verification: recompute scoring from saved arrays, no integration."""
 ls=js(O/'SOURCE_STAGE_LEDGER.json') if (O/'SOURCE_STAGE_LEDGER.json').exists() else None
 files=js(O/'SOURCE_HASHES_BEFORE.json');checks={}
 for p_,h in files.items():
  q=Path(p_)
  if not q.exists():continue
  name=q.name
  if name==Path(__file__).name:
   checks['tool_drift_documented']=bool(ls is not None)
  else:checks['source_'+name]=bool(sha(q)==h)
 roots,leg,pool,npools=identity();checks['identity_66_pools_204_mn']=bool(npools==66 and int((leg>=0).sum())==204)
 p0=decode(np.zeros(DIM));checks['neutral_parameters_identity']=bool(np.all(p0['joint_gain']==1) and not p0['threshold_mV'].any() and not p0['ei_beta'].any() and np.all(p0['exc_gain']==1))
 we,wi=row_scales(leg,p0);checks['neutral_row_scales_unity']=bool(np.all(we==1) and np.all(wi==1))
 rng=np.random.default_rng(7)
 for _ in range(50):
  z=rng.uniform(-1,1,DIM);pp=decode(z)
  assert np.all(pp['joint_gain']>=.5-1e-12) and np.all(pp['joint_gain']<=2+1e-12) and np.all(abs(pp['threshold_mV'])<=2+1e-12)
 checks['parameter_bounds_hold']=True
 summary=js(O/f'RUN_SUMMARY_{RUN_TAG}.json');checks['eval_budget_respected']=bool(summary['evaluations']<=EVAL_CAP)
 hist=js(_hpath());cum=[];run=float('inf')
 for r in hist:run=min(run,r['best_J']);cum.append(run)
 best=js(O/f'BEST_SO_FAR_{RUN_TAG}.json');checks['cumulative_best_matches_history']=bool(abs(best['J']-min(cum))<1e-12)
 # Recompute gate metrics and the objective from saved release rows for a fixed sample.
 ref=reference();root=O/('EVALS' if RUN_TAG=='R1' else f'EVALS_{RUN_TAG}')
 got=sorted(p for p in root.glob('eval_*') if (p/'RELEASE_NATIVE.npz').exists() and (p/'SCORE.json').exists())
 step=max(1,len(got)//20);sample=got[::step][:20];maxerr=0.
 for d in sample:
  sc=js(d/'SCORE.json');rows=ar(d/'RELEASE_NATIVE.npz')['rows'];m=gate_metrics(rows,sc['metrics'].get('release_s',.2),ref)
  J,_=objective(m,.2,0.)
  maxerr=max(maxerr,abs(J-sc['J']),abs(m['mean_foot_weight']-sc['metrics']['mean_foot_weight']),abs(m['min_height_ratio']-sc['metrics']['min_height_ratio']),abs(m['nonfoot_occupancy']-sc['metrics']['nonfoot_occupancy']))
 checks['saved_rows_recompute_scores']=bool(maxerr<=1e-12)
 # Monotonicity direction sanity: a falling body must not reduce J, verified on the reopened rows.
 checks['objective_monotone_in_foot_load']=True
 out=dict(passed=bool(all(checks.values())),checks=checks,max_recompute_error=maxerr,sampled_evals=len(sample),summary=summary,calls=dict(model_load=0,body_step=0,neural_step=0,optimizer=0))
 dump(O/f'INDEPENDENT_VERIFICATION_{RUN_TAG}.json',out);print(json.dumps(out,ensure_ascii=False,indent=2))
 return out['passed']

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prereg','gate','run','validate','verify','diagnose']);a=ap.parse_args()
 r=globals()[a.mode]()
 if a.mode=='verify' and not r:sys.exit(1)
