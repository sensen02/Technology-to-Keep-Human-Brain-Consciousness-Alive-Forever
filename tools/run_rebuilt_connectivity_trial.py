"""Run the same closed loop on the rebuilt connectivity, where load feedback finally has a path.

What changes and what does not:

  changes   the tier's edges, only by the declared bristle fill (37,429 added entries, 0 modified),
            because `build_neural_tier` is called with `sign_fill='bristle_only'` instead of the
            historical default 'none'
  unchanged the 114-dim parameter vector, the body, the muscles, the contacts, the assist schedule,
            the sensory routing, the translation scales, the objective and the gate

The trial path is the same stage-A rollout the whole series used. Its internal hash assertion is
kept: the connectivity file that the trial compares against is the rebuilt one, so the assertion
now verifies that the runtime really built the rebuilt edges rather than being bypassed.

modes: prereg / run / report
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import sys,json,hashlib,time,traceback,shutil
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'))
import train_stage_a_pool_modes as st
import train_joint_pool_readout as jp
import train_dynamic_support as dst
import dynamic_support_metrics as dsm
O=Path('/home/sensen/Desktop/engine修复验收/s5-rebuilt-connectivity-loop/20261104T070000Z')
REBUILD=Path('/home/sensen/Desktop/engine修复验收/s5-bristle-edge-rebuild/20261104T050000Z')
BEST=Path('/home/sensen/Desktop/engine修复验收/s5-joint-pool-readout/20261103T080000Z/EVALS/eval_709/SCORE.json')
SET=O/'CONNECTIVITY_SET'
RELEASE=.2
def js(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')

def make_set():
 """A directory that mirrors the archived input set, with only the two edge files replaced."""
 if SET.exists():shutil.rmtree(SET)
 SET.mkdir(parents=True)
 for p in sorted(st.B.iterdir()):
  if p.name in ('ACTUAL_We.npz','ACTUAL_Wi.npz'):continue
  (SET/p.name).symlink_to(p)
 for name,src in (('ACTUAL_We.npz',REBUILD/'REBUILT_We.npz'),('ACTUAL_Wi.npz',REBUILD/'REBUILT_Wi.npz')):
  shutil.copyfile(src,SET/name)
 return dict(we=hashlib.sha256((SET/'ACTUAL_We.npz').read_bytes()).hexdigest(),
             wi=hashlib.sha256((SET/'ACTUAL_Wi.npz').read_bytes()).hexdigest(),
             archived_we=hashlib.sha256((st.B/'ACTUAL_We.npz').read_bytes()).hexdigest())

def trial_rebuilt(folder,vec):
 import engine.embodied.adapters as A
 orig=A.build_neural_tier
 def patched(config=None,*a,**kw):
  from engine.embodied.adapters import TierConfig as TC
  base=config or TC()
  import dataclasses
  forced=dataclasses.replace(base,sign_fill='bristle_only')
  return orig(forced,*a,**kw)
 A.build_neural_tier=patched
 import run_neural_muscle_closure as c
 c.build_neural_tier=patched
 old_B=st.B
 st.B=SET
 try:
  zp,zg=np.asarray(vec,float)[:jp.POOL_DIM],np.asarray(vec,float)[jp.POOL_DIM:]
  from engine.embodied.motor_output import MotorOutputConfig
  st.CONFIG['neural_motor_config']=MotorOutputConfig(per_actuator_gain=tuple(float(g) for g in jp.gains_of(zg)))
  try:return st.trial(folder,np.clip(zp,-1,1),RELEASE)
  finally:st.CONFIG.pop('neural_motor_config',None)
 finally:
  st.B=old_B;A.build_neural_tier=orig
  import run_neural_muscle_closure as c2
  from engine.embodied.adapters import build_neural_tier as original_builder
  c2.build_neural_tier=original_builder

def prereg():
 O.mkdir(parents=True,exist_ok=True)
 hashes=make_set()
 dump(O/'PREREGISTRATION.json',dict(
  question='does the network use load feedback once the tactile afferent class actually reaches the motor cells',
  single_change="build_neural_tier is called with sign_fill='bristle_only' instead of the historical default 'none'",
  provenance='the same published connectome, the dataset predicted transmitter for the bristle rows, the repository declared option; nothing written by hand',
  connectivity=hashes,
  connectivity_set=str(SET),
  unchanged=['114-dim parameter vector','body','muscles','contacts','assist schedule','sensory routing and weights','translation scales','objective','gate'],
  reference_run=dict(J=3.3902926855730198,foot_support=0.31795537182301425,nonfoot_share=0.6417682564053153,first_nonfoot_s=0.01838749999859468),
  expectation='this is an evaluation, not a training run: the parameters were tuned on the disconnected network, so a worse score is a legitimate outcome and will be reported as such',
  acceptance='the loop is informative either way: if foot support moves, load feedback is being used; if it is bit-identical to the old network, then the added edges are not being read at all'))

def run():
 hashes=make_set()
 folder=O/'LOOP_REBUILT_BEST_VECTOR'
 rec=dict(connectivity=hashes,folder=str(folder))
 try:
  _,rows,summary=trial_rebuilt(folder,np.asarray(js(BEST)['vec'],float))
  np.savez_compressed(folder/'RELEASE_NATIVE.npz',rows=rows)
  assist=dsm.window_metrics(dict(np.load(jp.rc.BASE_BODY)),*jp.ASSIST,'assist')
  J,parts=dst.objective(rows,assist);full=dsm.evaluate_native(assist,rows)
  rec.update(J=float(J),parts=parts,gate=full,dynamic_pass=bool(full['passed']),wall_s=summary['wall_s'])
 except BaseException:
  rec.update(failed=True,traceback=traceback.format_exc()[-2000:])
 dump(O/'RESULT.json',rec)
 print(json.dumps({k:v for k,v in rec.items() if k in ('J','dynamic_pass','failed')},ensure_ascii=False),flush=True)
 if not rec.get('failed'):
  p=rec['parts'];print('足载 %.4f 非足 %.4f 质心 %.4f 首触 %s'%(p['foot_support_mean'],p['nonfoot_share'],p['com_z_min'],p['first_nonfoot_s']))

def report():
 rec=js(O/'RESULT.json')
 ref=dict(J=3.3902926855730198,foot_support=0.31795537182301425,nonfoot_share=0.6417682564053153)
 if rec.get('failed'):
  dump(O/'SUMMARY.json',dict(failed=True,traceback=rec['traceback']));print('FAILED');return
 p=rec['parts']
 dump(O/'SUMMARY.json',dict(rebuilt=dict(J=round(rec['J'],4),foot_support=round(p['foot_support_mean'],4),
   nonfoot_share=round(p['nonfoot_share'],4),com_z_min=round(p['com_z_min'],4),
   first_nonfoot_ms=round(p['first_nonfoot_s']*1000,2) if p['first_nonfoot_s'] is not None else None,
   dynamic_pass=rec['dynamic_pass']),
  disconnected_reference=ref,
  deltas=dict(J=round(rec['J']-ref['J'],4),foot_support=round(p['foot_support_mean']-ref['foot_support'],4),
   nonfoot_share=round(p['nonfoot_share']-ref['nonfoot_share'],4)),
  same_vector=True,note='one evaluation with the parameters tuned on the disconnected network'))
 print(json.dumps(js(O/'SUMMARY.json'),ensure_ascii=False,indent=1))
if __name__=='__main__':
 {'prereg':prereg,'run':run,'report':report}[sys.argv[1]]()
