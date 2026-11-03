"""Handover-timing probe: does a gradual withdrawal help the network catch the body?

The remaining failure is a timing one: at withdrawal only three legs are loaded and the body is
still rising, so the abdomen arrives 17-18 ms later. The assist can be faded out BEFORE the
handover instead of stopping abruptly (`assistance_fade_s`), which changes when and how the
network has to take over. That is a handover-protocol parameter, not a controller: the assisted
phase stays the same engineering reference, the neural runtime is unchanged, and after the fade
ends the assist is exactly zero.

The scoring objective and the gate are imported unchanged from the dynamic-support round.

modes: prereg / sweep / report
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import sys,json,math,hashlib,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'))
import train_stage_a_pool_modes as st
import train_dynamic_support as dst
import dynamic_support_metrics as dsm
import train_joint_pool_readout as jp
O=Path('/home/sensen/Desktop/engine修复验收/s5-handover-timing/20261103T170000Z')
BEST=Path('/home/sensen/Desktop/engine修复验收/s5-joint-pool-readout/20261103T080000Z/EVALS/eval_709/SCORE.json')
FADES=(0.0,0.05,0.10,0.20)
DURATIONS=(1.25,1.28,1.35,1.40)
RELEASE=.2
def js(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run_one(tag,fade):
 folder=O/tag
 if (folder/'RELEASE_NATIVE.npz').exists():
  rows=dict(np.load(folder/'RELEASE_NATIVE.npz'))['rows']
 else:
  if folder.exists():raise SystemExit('refusing to overwrite '+str(folder))
  vec=np.asarray(js(BEST)['vec'],float)
  st.CONFIG['assistance_fade_s']=float(fade)
  try:_,rows,_=jp.trial_joint(folder,vec,RELEASE)
  finally:st.CONFIG.pop('assistance_fade_s',None)
  np.savez_compressed(folder/'RELEASE_NATIVE.npz',rows=rows)
 assist=dsm.window_metrics(dict(np.load(jp.rc.BASE_BODY)),*jp.ASSIST,'assist')
 J,parts=dst.objective(rows,assist);full=dsm.evaluate_native(assist,rows)
 s=dsm.native_support_series(rows)
 t=s['time']-1.3
 i0=int(np.searchsorted(t,0.0))
 hand=dict(at_withdrawal=dict(per_leg_load=[float(s['foot'][i0,i]/dsm.WEIGHT) for i in range(6)],
   bearing=int(s['bearing_legs'][i0]),com_z=float(s['com_z'][i0]),
   com_vz=float(np.gradient(s['com_z'],s['time'])[i0]),support=float(s['support'][i0])))
 pos=np.flatnonzero(s['nonfoot_signed']>1e-9)
 rec=dict(tag=tag,fade_s=float(fade),J=float(J),parts=parts,gate=full,dynamic_pass=bool(full['passed']),
  handover=hand,first_nonfoot_s=(float(t[pos[0]]) if len(pos) else None))
 dump(folder/'SCORE.json',rec)
 return rec

def run_duration(tag,duration):
 folder=O/tag
 if (folder/'RELEASE_NATIVE.npz').exists():
  rows=dict(np.load(folder/'RELEASE_NATIVE.npz'))['rows']
 else:
  if folder.exists():raise SystemExit('refusing to overwrite '+str(folder))
  vec=np.asarray(js(BEST)['vec'],float)
  st.CONFIG['duration']=float(duration)
  try:_,rows,_=jp.trial_joint(folder,vec,RELEASE)
  finally:st.CONFIG['duration']=1.3
  np.savez_compressed(folder/'RELEASE_NATIVE.npz',rows=rows)
 # re-base so the handover sits at 1.3 in the analysis, which the objective and gate assume
 rebased=rows.copy();rebased[:,0]=rows[:,0]-(float(duration)-1.3)
 assist=dsm.window_metrics(dict(np.load(jp.rc.BASE_BODY)),*jp.ASSIST,'assist')
 J,parts=dst.objective(rebased,assist);full=dsm.evaluate_native(assist,rebased)
 s=dsm.native_support_series(rebased);t=s['time']-1.3;i0=int(np.searchsorted(t,0.0))
 hand=dict(at_withdrawal=dict(per_leg_load=[float(s['foot'][i0,i]/dsm.WEIGHT) for i in range(6)],
   bearing=int(s['bearing_legs'][i0]),com_z=float(s['com_z'][i0]),
   com_vz=float(np.gradient(s['com_z'],s['time'])[i0]),support=float(s['support'][i0])))
 pos=np.flatnonzero(s['nonfoot_signed']>1e-9)
 rec=dict(tag=tag,duration_s=float(duration),fade_s=0.0,rebase_s=float(duration)-1.3,J=float(J),parts=parts,
  gate=full,dynamic_pass=bool(full['passed']),handover=hand,
  first_nonfoot_s=(float(t[pos[0]]) if len(pos) else None))
 dump(folder/'SCORE.json',rec);return rec

def durations():
 recs=[]
 for d in DURATIONS:
  tag='HANDOVER_%03dms'%round(d*1000)
  try:recs.append(run_duration(tag,d))
  except BaseException:recs.append(dict(tag=tag,duration_s=d,failed=True,traceback=traceback.format_exc()[-800:]))
  dump(O/'DURATION_SWEEP.json',dict(results=recs,note='rows re-based so the handover sits at 1.3 s, which the objective and gate assume; assist is exactly zero in the whole evaluation window'))
  print(tag,recs[-1].get('J'),flush=True)
 dump(O/'DURATION_SWEEP.json',dict(results=recs,complete=True,note='rows re-based so the handover sits at 1.3 s'))

def prereg():
 O.mkdir(parents=True,exist_ok=True)
 assist=dsm.window_metrics(dict(np.load(jp.rc.BASE_BODY)),*jp.ASSIST,'assist')
 dump(O/'PREREGISTRATION.json',dict(
  question='does moving the withdrawal earlier and making it gradual let the network catch the body',
  knob='assistance_fade_s: inside the last part of the assisted phase the assist torque is scaled linearly to zero and is exactly zero at the handover',
  values_tested=list(FADES),
  fixed='the best 114-dim parameter vector from the joint search, the release window of 0.2 s, the objective and the gate, the body, muscles, contacts and the neural substrate',
  classification=dict(assist='engineering reference and closed-loop assist that MUST be withdrawn; fading it earlier is a handover-protocol change, not a controller replacing neurons',
   neural='unchanged: same connectome, same synapses, same sensory routing'),
  metric_discipline='support means foot-only; the reference band is taken from the no-fade assisted window of the archived run and this is declared, because with a fade the assisted window is no longer the same physical situation',
  measured_reference=dict(round_ab_joint_best_J=3.3902926855730198,foot_support=0.31795537182301425,nonfoot_share=0.6417682564053153,first_nonfoot_s=0.01838749999859468),
  assist_window_stats=assist,
  acceptance='a fade helps if foot support rises materially above 0.318 W with the non-foot share falling below 0.642, and at least one setting passes the gate'))

def sweep():
 recs=[]
 for fade in FADES:
  tag='FADE_%03dms'%round(fade*1000)
  try:recs.append(run_one(tag,fade))
  except BaseException:
   recs.append(dict(tag=tag,fade_s=fade,failed=True,traceback=traceback.format_exc()[-800:]))
  dump(O/'SWEEP.json',dict(results=recs))
  print(json.dumps({k:(v if not isinstance(v,dict) else None) for k,v in recs[-1].items()},ensure_ascii=False)[:200],flush=True)
 dump(O/'SWEEP.json',dict(results=recs,complete=True))
def report():
 recs=js(O/'SWEEP.json')['results']
 rows=[]
 for r in recs:
  if r.get('failed'):rows.append(dict(tag=r['tag'],failed=True));continue
  rows.append(dict(tag=r['tag'],fade_ms=round(r['fade_s']*1000),J=round(r['J'],4),
   foot_support=round(r['parts']['foot_support_mean'],4),nonfoot_share=round(r['parts']['nonfoot_share'],4),
   com_z_min=round(r['parts']['com_z_min'],4),bearing_legs_median=r['parts']['bearing_legs_median'],
   first_nonfoot_ms=None if r['first_nonfoot_s'] is None else round(r['first_nonfoot_s']*1000,2),
   at_withdrawal_bearing=r['handover']['at_withdrawal']['bearing'],
   at_withdrawal_support=round(r['handover']['at_withdrawal']['support'],4),
   com_vz_at_withdrawal=round(r['handover']['at_withdrawal']['com_vz'],2),
   dynamic_pass=r['dynamic_pass']))
 dump(O/'SUMMARY.json',dict(rows=rows))
 print(json.dumps(rows,ensure_ascii=False,indent=1))
if __name__=='__main__':
 {'prereg':prereg,'sweep':sweep,'durations':durations,'report':report}[sys.argv[1]]()
