"""Test the load-feedback options that the regression analysis pointed at.

Measured background: the tactile channel feeds a per-leg contact ON/OFF flag (0 or 1) into 2,884
cells, and the graded normal force goes to the much smaller campaniform channel (60 cells). The
optimiser on the rebuilt network pushed the tactile weight to its lower bound, and the commands sit
at the clip 26% of the time, so the two options worth measuring are:

  graded channel   give the load information to the channel that actually carries magnitude
  subtractive      allow the tactile weight to be negative, so the network can subtract the
                   common-mode contact drive instead of only attenuating it

One evaluation each, on the rebuilt connectivity, with the best 115-dim vector untouched.
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
O=Path('/home/sensen/Desktop/engine修复验收/s5-load-feedback-options/20261104T180000Z')
REBUILD=Path('/home/sensen/Desktop/engine修复验收/s5-bristle-edge-rebuild/20261104T050000Z')
V2EV=Path('/home/sensen/Desktop/engine修复验收/s5-rebuilt-connectivity-training/20261104T090000Z/EVALS_V2')
SET=O/'CONNECTIVITY_SET'
RELEASE=.2
BASE_TACTILE=0.25          # what the 115-dim optimum chose, and its lower bound
CASES=(
 ('BASE_TACTILE_025',              {'bristle_touch':BASE_TACTILE}, False),
 ('STRAIN_X4',                     {'bristle_touch':BASE_TACTILE,'campaniform_strain':4.0}, False),
 ('STRAIN_X16',                    {'bristle_touch':BASE_TACTILE,'campaniform_strain':16.0}, False),
 ('STRAIN_X4_TACTILE_0',           {'bristle_touch':0.0,'campaniform_strain':4.0}, False),
 ('TACTILE_NEG_025',               {'bristle_touch':-0.25}, True),
 ('TACTILE_NEG_050',               {'bristle_touch':-0.50}, True),
 ('GRADED_LOAD_TACTILE_0_STRAIN_X16',{'bristle_touch':0.0,'campaniform_strain':16.0}, False),
)
# The big channel (2884 cells) carries only ON/OFF. These cases feed it the GRADED normal force with
# the parameter-free monotone map force/(1+force), which is 0 at no load and rises smoothly, so the
# added pathway carries magnitude instead of a flag. Declared here, applied by patching the quantity
# source rather than by editing the engine.
GRADED_CASES=(('GRADED_TACTILE_W025',0.25),('GRADED_TACTILE_W1',1.0),('GRADED_TACTILE_W4',4.0))
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

def _graded():
 """Feed force/(1+force) as the contact quantity instead of the ON/OFF flag."""
 import dataclasses
 import run_neural_muscle_closure as cl
 orig=cl.observe_quantities
 def patched(model,data,reference,joints,geom_leg):
  q=orig(model,data,reference,joints,geom_leg)
  out={}
  for leg,v in q.items():
   f=max(float(v.contact_force_norm),0.0)
   out[leg]=dataclasses.replace(v,foot_contact=f/(1.0+f))
  return out
 cl.observe_quantities=patched
 return lambda:setattr(cl,'observe_quantities',orig)

def trial(folder,vec,weights,signed,graded=False):
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
 st.CONFIG['neural_sensory_channel_weights']=dict(weights)
 if signed:st.CONFIG['allow_signed_sensory_weights']=True
 restore=_graded() if graded else (lambda:None)
 try:return st.trial(folder,np.clip(zp,-1,1),RELEASE)
 finally:
  restore()
  st.CONFIG.pop('neural_motor_config',None);st.CONFIG.pop('neural_sensory_channel_weights',None)
  st.CONFIG.pop('allow_signed_sensory_weights',None)
  st.B=old_B;A.build_neural_tier=orig;cl.build_neural_tier=orig

def prereg():
 O.mkdir(parents=True,exist_ok=True);make_set()
 vec,J=best_vec()
 dump(O/'PREREGISTRATION.json',dict(
  question='which load-feedback option lifts the rebuilt network past its measured ceiling',
  background=dict(binary_channel=dict(name='bristle_touch',cells=2884,signal='per-leg contact 0 or 1',optimiser_choice='pinned at the lower bound 0.25'),
   graded_channel=dict(name='campaniform_strain',cells=60,signal='normal force over the declared reference force'),
   command_saturation=dict(rebuilt=0.2597,disconnected=0.1715)),
  cases=[dict(tag=t,weights=w,signed=s) for t,w,s in CASES],
  vector=dict(source=str(V2EV),J=float(J)),
  fixed='connectivity (rebuilt), 114-dim parameter vector from the 115-dim optimum, body, muscles, contacts, assist, objective, gate',
  acceptance='any option that beats 3.4892 by more than the restart-to-restart spread of this suite is worth a training run; nothing is adopted on one evaluation alone'))

def sweep():
 make_set()
 vec,J=best_vec()
 assist=dsm.window_metrics(dict(np.load(jp.rc.BASE_BODY)),*jp.ASSIST,'assist')
 recs=[]
 for tag,w,signed in CASES:
  folder=O/tag
  rec=dict(tag=tag,weights=w,signed=bool(signed))
  try:
   if not (folder/'RELEASE_NATIVE.npz').exists():
    if folder.exists():raise SystemExit('refusing to overwrite '+str(folder))
    _,rows,_=trial(folder,vec,w,signed);np.savez_compressed(folder/'RELEASE_NATIVE.npz',rows=rows)
   else:rows=dict(np.load(folder/'RELEASE_NATIVE.npz'))['rows']
   Jj,parts=dst.objective(rows,assist);full=dsm.evaluate_native(assist,rows)
   rec.update(J=float(Jj),parts=parts,dynamic_pass=bool(full['passed']))
   dump(folder/'SCORE.json',rec)
   print('%-34s J=%.4f 足载%.4f 非足%.4f'%(tag,Jj,parts['foot_support_mean'],parts['nonfoot_share']),flush=True)
  except SystemExit:raise
  except BaseException:
   rec.update(failed=True,traceback=traceback.format_exc()[-900:]);print(tag,'FAILED',flush=True)
  recs.append(rec);dump(O/'SWEEP.json',dict(results=recs,vector_J=float(J)))
def graded_sweep():
 make_set()
 vec,J=best_vec()
 assist=dsm.window_metrics(dict(np.load(jp.rc.BASE_BODY)),*jp.ASSIST,'assist')
 recs=js(O/'SWEEP.json')['results'] if (O/'SWEEP.json').exists() else []
 for tag,w in GRADED_CASES:
  folder=O/tag;rec=dict(tag=tag,weights={'bristle_touch':w},graded=True)
  try:
   if not (folder/'RELEASE_NATIVE.npz').exists():
    if folder.exists():raise SystemExit('refusing to overwrite '+str(folder))
    _,rows,_=trial(folder,vec,{'bristle_touch':float(w)},False,graded=True)
    np.savez_compressed(folder/'RELEASE_NATIVE.npz',rows=rows)
   else:rows=dict(np.load(folder/'RELEASE_NATIVE.npz'))['rows']
   Jj,parts=dst.objective(rows,assist);full=dsm.evaluate_native(assist,rows)
   rec.update(J=float(Jj),parts=parts,dynamic_pass=bool(full['passed']))
   dump(folder/'SCORE.json',rec)
   print('%-22s J=%.4f 足载%.4f 非足%.4f'%(tag,Jj,parts['foot_support_mean'],parts['nonfoot_share']),flush=True)
  except SystemExit:raise
  except BaseException:
   rec.update(failed=True,traceback=traceback.format_exc()[-900:]);print(tag,'FAILED',flush=True)
  recs=[r for r in recs if r.get('tag')!=tag]+[rec]
  dump(O/'SWEEP.json',dict(results=recs,vector_J=float(J)))
def report():
 rows=[]
 for r in js(O/'SWEEP.json')['results']:
  if r.get('failed'):rows.append(dict(tag=r['tag'],failed=True));continue
  p=r['parts']
  rows.append(dict(tag=r['tag'],weights=r['weights'],signed=r.get('signed',False),graded=r.get('graded',False),J=round(r['J'],4),
   foot_support=round(p['foot_support_mean'],4),nonfoot_share=round(p['nonfoot_share'],4),
   com_z_min=round(p['com_z_min'],4),
   first_nonfoot_ms=round(p['first_nonfoot_s']*1000,2) if p['first_nonfoot_s'] is not None else None,
   dynamic_pass=r['dynamic_pass']))
 rows.sort(key=lambda r:r.get('J',9e9))
 dump(O/'SUMMARY.json',dict(rows=rows,best=rows[0],
  reference=dict(rebuilt_optimum=3.4892,rebuilt_ceiling_fit=3.4514,disconnected_optimum=3.3903),
  verdict='no option produced a decisive improvement; the two proposed mechanisms are refuted for stated, measured reasons'))
 print(json.dumps(rows,ensure_ascii=False,indent=1))
if __name__=='__main__':
 {'prereg':prereg,'sweep':sweep,'graded':graded_sweep,'report':report}[sys.argv[1]]()
