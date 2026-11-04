"""Does a much longer, gentler assistance fade let the network take over?

The earlier probe tested fades of 0, 50, 100 and 200 ms and every non-zero fade was worse. The
mechanism it measured was that during a short fade the body simply drops, so there was no time for
the network to respond. A much longer fade is the untested end of that axis: the assist torque ramps
to zero over 300, 600 or 900 ms inside the assisted phase, and it is exactly zero at the handover, so
the release window itself is still fully neural and the red line is untouched.

modes: prereg / sweep / report
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import sys,json,hashlib,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'))
import train_stage_a_pool_modes as st
import train_joint_pool_readout as jp
import train_dynamic_support as dst
import dynamic_support_metrics as dsm
from engine.embodied.motor_output import MotorOutputConfig
O=Path('/home/sensen/Desktop/engine修复验收/s5-long-assist-fade/20261105T100000Z')
REBUILD=Path('/home/sensen/Desktop/engine修复验收/s5-bristle-edge-rebuild/20261104T050000Z')
V2EV=Path('/home/sensen/Desktop/engine修复验收/s5-rebuilt-connectivity-training/20261104T090000Z/EVALS_V2')
SET=O/'CONNECTIVITY_SET'
FADES=(0.3,0.6,0.9)
SHORT={0.0:3.3903,0.05:4.0354,0.10:4.0400,0.20:4.0191}
REFERENCE=dict(no_fade_on_rebuilt_connectivity=3.4892,short_fades_on_the_old_connectivity=SHORT)
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
 import shutil
 if SET.exists():shutil.rmtree(SET)
 SET.mkdir(parents=True)
 for p in sorted(st.B.iterdir()):
  if p.name in ('ACTUAL_We.npz','ACTUAL_Wi.npz'):continue
  (SET/p.name).symlink_to(p)
 shutil.copyfile(REBUILD/'REBUILT_We.npz',SET/'ACTUAL_We.npz')
 shutil.copyfile(REBUILD/'REBUILT_Wi.npz',SET/'ACTUAL_Wi.npz')
def trial(folder,vec,fade):
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
 st.CONFIG['assistance_fade_s']=float(fade)
 try:return st.trial(folder,np.clip(zp,-1,1),.2)
 finally:
  st.CONFIG.pop('neural_motor_config',None);st.CONFIG.pop('neural_sensory_channel_weights',None)
  st.CONFIG.pop('assistance_fade_s',None);st.B=old_B;A.build_neural_tier=orig;cl.build_neural_tier=orig
def prereg():
 O.mkdir(parents=True,exist_ok=True);make_set();vec,J=best_vec()
 dump(O/'PREREGISTRATION.json',dict(question='does a much longer assistance fade let the network take over',
  knob='assistance_fade_s inside the assisted phase; the assist is exactly zero at the handover, so the release window stays fully neural',
  values_tested=list(FADES),vector_J=float(J),reference=REFERENCE,
  discipline='same connectivity, same vector, same everything else; support means foot-only',
  acceptance='a long fade helps if J falls below 3.4892 with foot support above 0.286 and the non-foot share below 0.670'))
def sweep():
 make_set();vec,J=best_vec()
 assist=dsm.window_metrics(dict(np.load(jp.rc.BASE_BODY)),*jp.ASSIST,'assist')
 recs=[]
 for fade in FADES:
  tag='FADE_%03dms'%round(fade*1000)
  folder=O/tag;rec=dict(tag=tag,fade_s=fade)
  try:
   if not (folder/'RELEASE_NATIVE.npz').exists():
    if folder.exists():raise SystemExit('refusing to overwrite '+str(folder))
    _,rows,_=trial(folder,vec,fade);np.savez_compressed(folder/'RELEASE_NATIVE.npz',rows=rows)
   else:rows=dict(np.load(folder/'RELEASE_NATIVE.npz'))['rows']
   Jj,parts=dst.objective(rows,assist);full=dsm.evaluate_native(assist,rows)
   rec.update(J=float(Jj),parts=parts,dynamic_pass=bool(full['passed']))
   dump(folder/'SCORE.json',rec)
   print('FADE %4.0f ms  J=%.4f 足载%.4f 非足%.4f'%(fade*1000,Jj,parts['foot_support_mean'],parts['nonfoot_share']),flush=True)
  except SystemExit:raise
  except BaseException:
   rec.update(failed=True,traceback=traceback.format_exc()[-800:]);print(tag,'FAILED',flush=True)
  recs.append(rec);dump(O/'SWEEP.json',dict(results=recs,vector_J=float(J)))
def report():
 rows=[]
 for r in js(O/'SWEEP.json')['results']:
  if r.get('failed'):rows.append(dict(tag=r['tag'],failed=True));continue
  p=r['parts']
  rows.append(dict(tag=r['tag'],fade_ms=round(r['fade_s']*1000),J=round(r['J'],4),
   foot_support=round(p['foot_support_mean'],4),nonfoot_share=round(p['nonfoot_share'],4),
   com_z_min=round(p['com_z_min'],4),dynamic_pass=r['dynamic_pass']))
 rows.sort(key=lambda r:r.get('fade_ms',9e9))
 dump(O/'SUMMARY.json',dict(rows=rows,no_fade_baseline=3.4892,reference=REFERENCE,
  reading='monotone or not, this is the untested far end of the fade axis; the short-fade numbers come from the earlier probe on the old connectivity and are context only'))
 print(json.dumps(rows,ensure_ascii=False,indent=1))
if __name__=='__main__':
 {'prereg':prereg,'sweep':sweep,'report':report}[sys.argv[1]]()
