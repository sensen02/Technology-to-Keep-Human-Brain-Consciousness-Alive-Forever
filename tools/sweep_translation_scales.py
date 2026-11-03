"""Sweep the translation-layer scales that set how many Newtons a spike rate buys.

The ablation measured the neural contribution at about 0.25 body weight when holding needs 1.0,
and the readout gains of the trained vector sit on both bounds, which says the optimiser wanted
more control authority than the declared scale allowed. Three declared scales decide that:

  rate_for_full_control_hz   ctrl = clip(gain * rate / this, 0, 1)   -- lower means more force
  filter_tau_s               spike-rate low-pass                     -- shorter keeps peak rates
  muscle activation timeconstants (dynprm) -- measured here as 0.1/0.4 ms, i.e. not limiting

Only these declared translation/model scales move; the connectome, the body, the muscles and the
114-dim vector stay fixed. Nothing here adds a controller: the drive still comes from the spiking
network, and the scale is a declared interface assumption that remains NOT_ESTABLISHED.

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
O=Path('/home/sensen/Desktop/engine修复验收/s5-translation-scales/20261103T210000Z')
BEST=Path('/home/sensen/Desktop/engine修复验收/s5-joint-pool-readout/20261103T080000Z/EVALS/eval_709/SCORE.json')
RELEASE=.2
RATES=(100.0,50.0,25.0,12.5)
TAUS=(0.005,0.001,0.002,0.020)
def js(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')

def trial_scaled(folder,vec,rate,tau):
 zp,zg=np.asarray(vec,float)[:jp.POOL_DIM],np.asarray(vec,float)[jp.POOL_DIM:]
 gains=jp.gains_of(zg)
 st.CONFIG['neural_motor_config']=MotorOutputConfig(filter_tau_s=float(tau),
   rate_for_full_control_hz=float(rate),max_control=1.0,per_actuator_gain=tuple(float(g) for g in gains))
 try:return st.trial(folder,np.clip(zp,-1,1),RELEASE)
 finally:st.CONFIG.pop('neural_motor_config',None)

def run_one(tag,rate,tau):
 folder=O/tag
 if (folder/'RELEASE_NATIVE.npz').exists():rows=dict(np.load(folder/'RELEASE_NATIVE.npz'))['rows']
 else:
  if folder.exists():raise SystemExit('refusing to overwrite '+str(folder))
  _,rows,_=trial_scaled(folder,np.asarray(js(BEST)['vec'],float),rate,tau)
  np.savez_compressed(folder/'RELEASE_NATIVE.npz',rows=rows)
 assist=dsm.window_metrics(dict(np.load(jp.rc.BASE_BODY)),*jp.ASSIST,'assist')
 J,parts=dst.objective(rows,assist);full=dsm.evaluate_native(assist,rows)
 body=dict(np.load(folder/'RUN/BODY.npz')) if (folder/'RUN/BODY.npz').exists() else None
 sat=None
 if body is not None:
  c=body['command'];sat=dict(fraction_at_upper_clip=float(np.mean(c>=0.9999)),
   fraction_at_lower_clip=float(np.mean(c<=1e-6)),mean_command=float(np.mean(c)),
   mean_active_muscles=float(np.mean(np.sum(c>0.01,axis=1))))
 rec=dict(tag=tag,rate_for_full_control_hz=float(rate),filter_tau_s=float(tau),J=float(J),parts=parts,
  gate=full,dynamic_pass=bool(full['passed']),command_statistics=sat)
 dump(folder/'SCORE.json',rec);return rec

def prereg():
 O.mkdir(parents=True,exist_ok=True)
 dump(O/'PREREGISTRATION.json',dict(
  question='is the neural drive short because the declared spike-rate-to-force scale is too conservative',
  evidence='the ablation measured a neural contribution of 0.249 body weight against 1.000 needed, and the trained readout gains sit on BOTH bounds, so the optimiser ran out of authority',
  knobs=dict(rate_for_full_control_hz=list(RATES),filter_tau_s=list(TAUS)),
  equivalent_meaning='lowering rate_for_full_control_hz multiplies every actuator command by a constant; 100->50 is a global x2, 100->25 is x4, 100->12.5 is x8',
  measured_muscle_timeconstants='activate 0.1 ms, deactivate 0.4 ms from dynprm of the compiled model: already far faster than the 5 ms command period, so activation memory is not the limiting scale',
  fixed='the 114-dim vector, the connectome, the body, the muscles, the contacts, the assist schedule, the objective and the gate',
  classification='declared translation-layer scale and its declared interface assumption; the spiking network still produces every command, no controller is added',
  metric='support is foot only; non-foot share is reported separately and never counted as support',
  baseline=dict(J=3.3902926855730198,foot_support=0.31795537182301425,nonfoot_share=0.6417682564053153),
  acceptance='a scale helps if foot support rises clearly above 0.318 W with the non-foot share falling below 0.642, and at least one setting passes the gate'))

def sweep():
 recs=[]
 for rate in RATES:
  tag='RATE_%05.1fHz_TAU_5ms'%rate
  try:recs.append(run_one(tag,rate,0.005))
  except BaseException:recs.append(dict(tag=tag,failed=True,traceback=traceback.format_exc()[-600:]))
  dump(O/'SWEEP_1_RATE.json',dict(results=recs,complete=rate==RATES[-1]))
  print(tag,recs[-1].get('J'),flush=True)

def taus():
 ok=[r for r in js(O/'SWEEP_1_RATE.json')['results'] if not r.get('failed')]
 best=min(ok,key=lambda r:r['J'])['rate_for_full_control_hz']
 recs=[]
 for tau in TAUS:
  tag='RATE_%05.1fHz_TAU_%04.1fms'%(best,tau*1000)
  try:recs.append(run_one(tag,best,tau))
  except BaseException:recs.append(dict(tag=tag,failed=True,traceback=traceback.format_exc()[-600:]))
  dump(O/'SWEEP_2_TAU.json',dict(rate_for_full_control_hz=best,results=recs,complete=tau==TAUS[-1]))
  print(tag,recs[-1].get('J'),flush=True)

def report():
 rows=[]
 for f in ('SWEEP_1_RATE.json','SWEEP_2_TAU.json'):
  p=O/f
  if not p.exists():continue
  for r in js(p)['results']:
   if r.get('failed'):rows.append(dict(tag=r['tag'],failed=True));continue
   s=r['command_statistics'] or {}
   rows.append(dict(tag=r['tag'],rate=r['rate_for_full_control_hz'],tau_ms=round(r['filter_tau_s']*1000,1),
    J=round(r['J'],4),foot_support=round(r['parts']['foot_support_mean'],4),
    nonfoot_share=round(r['parts']['nonfoot_share'],4),com_z_min=round(r['parts']['com_z_min'],4),
    bearing_legs=r['parts']['bearing_legs_median'],first_nonfoot_ms=round(r['parts']['first_nonfoot_s']*1000,2) if r['parts']['first_nonfoot_s'] is not None else None,
    cmd_at_clip=round(s.get('fraction_at_upper_clip',0),4),mean_cmd=round(s.get('mean_command',0),4),
    dynamic_pass=r['dynamic_pass']))
 dump(O/'SUMMARY.json',dict(rows=rows,best=min((r for r in rows if 'J' in r),key=lambda r:r['J'])))
 print(json.dumps(rows,ensure_ascii=False,indent=1))
if __name__=='__main__':
 {'prereg':prereg,'sweep':sweep,'taus':taus,'report':report}[sys.argv[1]]()
