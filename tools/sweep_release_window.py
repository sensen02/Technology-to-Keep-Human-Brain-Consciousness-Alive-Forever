"""Does a longer release window let the network build support?

The capability test showed the body CAN be held up: at the settled pose the feet carry 110.5% of the
weight and the muscle force budget is 4.77x body weight. What fails is the 0.2 s window right after
the assist is withdrawn. So this sweep keeps the vector and the connectivity fixed and only lengthens
the release, scoring two ways:

  J_0.2s   the standard window, so the number stays comparable with 3.4892
  J_full   the whole release, which asks whether support recovers later on

modes: prereg / sweep / report
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import sys,json,hashlib,traceback,shutil
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'))
import train_stage_a_pool_modes as st
import train_joint_pool_readout as jp
import train_dynamic_support as dst
import dynamic_support_metrics as dsm
from engine.embodied.motor_output import MotorOutputConfig
O=Path('/home/sensen/Desktop/engine修复验收/s5-release-window/20261105T080000Z')
REBUILD=Path('/home/sensen/Desktop/engine修复验收/s5-bristle-edge-rebuild/20261104T050000Z')
V2EV=Path('/home/sensen/Desktop/engine修复验收/s5-rebuilt-connectivity-training/20261104T090000Z/EVALS_V2')
SET=O/'CONNECTIVITY_SET'
WINDOWS=(0.2,0.5,1.0)
REFERENCE=dict(J_0p2=3.489210,foot_support=0.2860528924414577,nonfoot_share=0.6698449116551864,
 com_z_min=0.5856597228129956)
def js(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')

def best_vec():
 best=None;bestJ=9e9
 for f in sorted(V2EV.glob('eval_*')):
  sc=js(f/'SCORE.json')
  if sc.get('failed') or 'J' not in sc:continue
  if float(sc['J'])<bestJ:bestJ=float(sc['J']);best=np.asarray(sc['vec'],float)
 return best,bestJ

def make_set():
 if SET.exists():shutil.rmtree(SET)
 SET.mkdir(parents=True)
 for p in sorted(st.B.iterdir()):
  if p.name in ('ACTUAL_We.npz','ACTUAL_Wi.npz'):continue
  (SET/p.name).symlink_to(p)
 shutil.copyfile(REBUILD/'REBUILT_We.npz',SET/'ACTUAL_We.npz')
 shutil.copyfile(REBUILD/'REBUILT_Wi.npz',SET/'ACTUAL_Wi.npz')

def trial(folder,vec,release):
 import engine.embodied.adapters as A, dataclasses
 import run_neural_muscle_closure as cl
 orig=A.build_neural_tier
 def patched(config=None,*a,**kw):
  from engine.embodied.adapters import TierConfig as TC
  return orig(dataclasses.replace(config or TC(),sign_fill='bristle_only'),*a,**kw)
 A.build_neural_tier=patched;cl.build_neural_tier=patched
 old_B=st.B;st.B=SET
 v=np.asarray(vec,float);zp,zg=v[:jp.POOL_DIM],v[jp.POOL_DIM:jp.DIM]
 st.CONFIG['neural_motor_config']=MotorOutputConfig(per_actuator_gain=tuple(float(g) for g in jp.gains_of(zg)))
 st.CONFIG['neural_sensory_channel_weights']={'bristle_touch':float(np.exp(np.clip(v[jp.DIM],-1,1)*np.log(4.0)))} if v.size>jp.DIM else None
 try:return st.trial(folder,np.clip(zp,-1,1),release)
 finally:
  st.CONFIG.pop('neural_motor_config',None);st.CONFIG.pop('neural_sensory_channel_weights',None)
  st.B=old_B;A.build_neural_tier=orig;cl.build_neural_tier=orig

def prereg():
 O.mkdir(parents=True,exist_ok=True);make_set()
 vec,J=best_vec()
 dump(O/'PREREGISTRATION.json',dict(
  question='with the settled pose provably supportable, does a longer release window let the network build support',
  windows_s=dict((str(w),w) for w in WINDOWS),
  scoring=dict(J_0p2='objective on the rows up to 1.5 s, comparable with the earlier record',
   J_full='objective on the whole release, a longer task and therefore not directly comparable'),
  fixed='connectivity, 115-dim vector including the tactile weight, body, muscles, contacts, assist schedule',
  reference=REFERENCE,
  acceptance='reported as it comes out: a longer window that never recovers is as informative as one that does'))

def sweep():
 make_set()
 vec,J=best_vec()
 assist=dsm.window_metrics(dict(np.load(jp.rc.BASE_BODY)),*jp.ASSIST,'assist')
 recs=[]
 for w in WINDOWS:
  tag='RELEASE_%03dms'%round(w*1000)
  folder=O/tag;rec=dict(tag=tag,release_s=w)
  try:
   if not (folder/'RELEASE_NATIVE.npz').exists():
    if folder.exists():raise SystemExit('refusing to overwrite '+str(folder))
    _,rows,_=trial(folder,vec,w);np.savez_compressed(folder/'RELEASE_NATIVE.npz',rows=rows)
   else:rows=dict(np.load(folder/'RELEASE_NATIVE.npz'))['rows']
   arr=np.asarray(rows,float)          # columns: t, foot x6, nf_pos, nf_signed, ncnt, com_z, root7, ...
   t=arr[:,0]
   std=arr[t<=1.5+1e-9]
   Jstd,parts=dst.objective(std,assist);full=dsm.evaluate_native(assist,std)
   Jfull,pfull=dst.objective(rows,assist)
   series=dsm.native_support_series(rows)
   tt=series['time']-1.3
   bins=[]
   for lo in np.arange(0,float(tt.max())+1e-9,0.1):
    m=(tt>=lo)&(tt<lo+0.1)
    if not m.any():continue
    bins.append(dict(t_from_handover_s=round(float(lo),2),
     foot_support=float(np.nanmean(series['support'][m])),
     nonfoot_share=float(np.nanmean(np.nan_to_num(series['nonfoot_share'])[m])),
     com_z=float(np.nanmean(series['com_z'][m]))))
   rec.update(J_0p2=float(Jstd),parts_0p2=parts,gate_0p2=full,dynamic_pass=bool(full['passed']),
    J_full=float(Jfull),parts_full=pfull,bins=bins,
    com_z_end=float(series['com_z'][-1]),
    com_z_start=float(series['com_z'][int(np.argmax(t>=1.3))]))
   print('%-16s J_0.2s=%.4f  J_full=%.4f  足载(0.2s)=%.4f 末段足载=%.4f com末=%.3f'%(
    tag,Jstd,Jfull,parts['foot_support_mean'],bins[-1]['foot_support'],rec['com_z_end']),flush=True)
  except SystemExit:raise
  except BaseException:
   rec.update(failed=True,traceback=traceback.format_exc()[-900:]);print(tag,'FAILED',flush=True)
  recs.append(rec);dump(O/'SWEEP.json',dict(results=recs,vector_J=float(J)))
def report():
 rows=[]
 for r in js(O/'SWEEP.json')['results']:
  if r.get('failed'):rows.append(dict(tag=r['tag'],failed=True));continue
  rows.append(dict(tag=r['tag'],release_s=r['release_s'],J_0p2=round(r['J_0p2'],4),J_full=round(r['J_full'],4),
   foot_support_0p2=round(r['parts_0p2']['foot_support_mean'],4),
   foot_support_full=round(r['parts_full']['foot_support_mean'],4),
   com_z_start=round(r['com_z_start'],4),com_z_end=round(r['com_z_end'],4),
   dynamic_pass=r['dynamic_pass'],bins=r['bins']))
 dump(O/'SUMMARY.json',dict(rows=rows,reference=REFERENCE,
  reading='J_0p2 is the comparable number; J_full and the time bins say whether support recovers after the first 0.2 s'))
 print(json.dumps([{k:v for k,v in r.items() if k!='bins'} for r in rows],ensure_ascii=False,indent=1))
if __name__=='__main__':
 {'prereg':prereg,'sweep':sweep,'report':report}[sys.argv[1]]()
