"""Sweep the declared per-channel sensory weights: which feedback does the network use, and does more help?

Measured before touching anything: of the seven declared channel types only five are modelled and
carry data (club_vibration and multidendritic_noxious are declared unmodelled and emit exactly
zero), and the tactile channel owns 450 of the ~559 sensory cells per leg, so it dominates the
sensory budget by cell count.

Each trial silences (weight 0) or amplifies one group, with the 114-dim vector and everything else
fixed. Weight 1.0 reproduces the earlier arithmetic bit for bit; silence is archived by channel
name as required. No controller is added: the network is still the only source of commands.

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
O=Path('/home/sensen/Desktop/engine修复验收/s5-sensory-weights/20261103T230000Z')
BEST=Path('/home/sensen/Desktop/engine修复验收/s5-joint-pool-readout/20261103T080000Z/EVALS/eval_709/SCORE.json')
RELEASE=.2
PROPRIO=('claw_position','hook_direction','hairplate_angle')
CONTACT=('bristle_touch',)
STRAIN=('campaniform_strain',)
CASES=(
 ('BASE_ALL_ONES',            {}),
 ('SILENCE_TACTILE',          {c:0.0 for c in CONTACT}),
 ('AMPLIFY_TACTILE_x3',       {c:3.0 for c in CONTACT}),
 ('SILENCE_PROPRIO',          {c:0.0 for c in PROPRIO}),
 ('AMPLIFY_PROPRIO_x3',       {c:3.0 for c in PROPRIO}),
 ('SILENCE_STRAIN',           {c:0.0 for c in STRAIN}),
 ('AMPLIFY_STRAIN_x3',        {c:3.0 for c in STRAIN}),
 ('SILENCE_ALL_MODELLED',     {c:0.0 for c in PROPRIO+CONTACT+STRAIN}),
 ('AMPLIFY_ALL_MODELLED_x3',  {c:3.0 for c in PROPRIO+CONTACT+STRAIN}),
)
def js(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')

def trial_sensory(folder,vec,weights):
 zp,zg=np.asarray(vec,float)[:jp.POOL_DIM],np.asarray(vec,float)[jp.POOL_DIM:]
 st.CONFIG['neural_motor_config']=MotorOutputConfig(per_actuator_gain=tuple(float(g) for g in jp.gains_of(zg)))
 st.CONFIG['neural_sensory_channel_weights']=dict(weights) if weights else None
 try:return st.trial(folder,np.clip(zp,-1,1),RELEASE)
 finally:
  st.CONFIG.pop('neural_motor_config',None);st.CONFIG.pop('neural_sensory_channel_weights',None)

def run_one(tag,weights):
 folder=O/tag
 if (folder/'RELEASE_NATIVE.npz').exists():rows=dict(np.load(folder/'RELEASE_NATIVE.npz'))['rows']
 else:
  if folder.exists():raise SystemExit('refusing to overwrite '+str(folder))
  _,rows,_=trial_sensory(folder,np.asarray(js(BEST)['vec'],float),weights)
  np.savez_compressed(folder/'RELEASE_NATIVE.npz',rows=rows)
 assist=dsm.window_metrics(dict(np.load(jp.rc.BASE_BODY)),*jp.ASSIST,'assist')
 J,parts=dst.objective(rows,assist);full=dsm.evaluate_native(assist,rows)
 rec=dict(tag=tag,weights=dict(weights) if weights else 'unchanged default (all ones)',J=float(J),parts=parts,
  gate=full,dynamic_pass=bool(full['passed']))
 dump(folder/'SCORE.json',rec);return rec

def prereg():
 O.mkdir(parents=True,exist_ok=True)
 dump(O/'PREREGISTRATION.json',dict(
  question='which sensory feedback does the network actually use, and does amplifying it help',
  measured_channels='five of seven channel types are modelled and carry data; club_vibration and multidendritic_noxious are declared unmodelled and emit exactly zero, so they are not swept',
  measured_cell_counts='per leg: bristle_touch 450, hairplate 29, campaniform 28, claw 27, hook 25 -- the tactile channel owns about 80% of the sensory cells',
  groups=dict(proprioceptive=list(PROPRIO),tactile=list(CONTACT),strain=list(STRAIN)),
  cases=[dict(tag=t,weights=w) for t,w in CASES],
  fixed='the 114-dim vector, the connectome, the body, the muscles, the contacts, the assist schedule, the translation scales, the objective and the gate',
  discipline='weight 1.0 is bit-identical to the legacy default; a weight of zero extinguishes a channel and is archived by name; weights stay nonnegative and no external stimulus is added',
  baseline=dict(J=3.3902926855730198,foot_support=0.31795537182301425,nonfoot_share=0.6417682564053153),
  acceptance='the sweep is informative either way: silence shows whether a channel is used, amplification shows whether headroom exists'))

def sweep():
 recs=[]
 for tag,w in CASES:
  try:recs.append(run_one(tag,w))
  except BaseException:recs.append(dict(tag=tag,failed=True,traceback=traceback.format_exc()[-700:]))
  dump(O/'SWEEP.json',dict(results=recs,complete=tag==CASES[-1][0]))
  print(tag,recs[-1].get('J'),flush=True)

def report():
 rows=[]
 for r in js(O/'SWEEP.json')['results']:
  if r.get('failed'):rows.append(dict(tag=r['tag'],failed=True));continue
  p=r['parts']
  rows.append(dict(tag=r['tag'],J=round(r['J'],4),foot_support=round(p['foot_support_mean'],4),
   nonfoot_share=round(p['nonfoot_share'],4),com_z_min=round(p['com_z_min'],4),
   bearing_legs=p['bearing_legs_median'],
   first_nonfoot_ms=round(p['first_nonfoot_s']*1000,2) if p['first_nonfoot_s'] is not None else None,
   dynamic_pass=r['dynamic_pass']))
 rows.sort(key=lambda r:r.get('J',9e9))
 dump(O/'SUMMARY.json',dict(rows=rows,best=rows[0]))
 print(json.dumps(rows,ensure_ascii=False,indent=1))
if __name__=='__main__':
 {'prereg':prereg,'sweep':sweep,'report':report}[sys.argv[1]]()
