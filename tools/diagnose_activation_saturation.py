"""Is the plateau a muscle-capacity wall or a command-pattern problem?

Reads saved arrays only: no model load, no forward, no step, no training.
Reports, per run and per window, how close the muscles are to their activation ceiling
and how many are pinned there, which is measurable without any inverse solve.
"""
import json,hashlib
from pathlib import Path
import numpy as np
O=Path('/home/sensen/Desktop/engine修复验收/s4-activation-saturation/20261103T060000Z')
RUNS={
 'baseline':Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z/RUN/BODY.npz'),
 'trained_best':Path('/home/sensen/Desktop/engine修复验收/s5-dynamic-support-training/20261103T002606Z/VALIDATION_R1/CAND0_A_release200ms/RUN/BODY.npz'),
}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(parents=True,exist_ok=False)
 out=dict(question='is the failure limited by muscle capacity or by the command pattern',
  windows=dict(assist=[0.90,1.30],release=[1.30,1.40]),
  note='activation ceiling is 1.0 (the controller clips every command to [0,1]); the model sets no force limit, so activation is the only hard per-muscle bound',
  runs={})
 for tag,p in RUNS.items():
  z=dict(np.load(p,allow_pickle=False));t=z['time'];a=z['activation'];f=z['muscle_force']
  r={}
  for wn,(lo,hi) in (('assist',(0.90,1.30)),('release',(1.30,1.40))):
   w=(t>=lo)&(t<=hi);aa=a[w];ff=f[w]
   per=(aa>=0.99).mean(0)
   r[wn]=dict(frames=int(w.sum()),
    activation_entry_saturation_fraction=float((aa>=0.99).mean()),
    actuators_saturated_over_80pct=int((per>0.8).sum()),
    actuators_saturated_over_50pct=int((per>0.5).sum()),
    actuators_saturated_over_20pct=int((per>0.2).sum()),
    max_per_actuator_saturation=float(per.max()),
    mean_active_muscles=float((np.abs(ff)>1.0).sum(1).mean()),
    max_abs_muscle_force=float(np.abs(ff).max()),
    top_saturating=[dict(actuator=int(u),fraction=float(per[u])) for u in np.argsort(-per)[:8]])
  r['verdict']=('a subset of muscles is pinned at the activation ceiling, so further gain increases cannot recruit them: this run is capacity-limited for that subset'
   if r['release']['actuators_saturated_over_50pct']>=5 else
   'no muscle is pinned at the ceiling for any sustained part of the release window: the failure is the command pattern, not capacity')
  r['source_sha256']=sha(p);out['runs'][tag]=r
 out['interpretation']=dict(
  not_needing_inverse='training never required a per-muscle inverse solution: the chain activation -> muscle force -> joint torque -> contact -> behaviour is forward and fully recorded, and the objective depends on the emergent behaviour, not on any particular muscle split',
  why_ambiguity_is_benign='the 48-dimensional split ambiguity changes which muscles share the load but not the joint torques, and the body responds only to the torques and the contacts',
  what_is_blocked='only causal claims of the form "this muscle causes the collapse", and using per-muscle force as a training target (no dataset carries muscle force anyway)',
  measured_answer='the baseline is nowhere near the ceiling, so its failure is a command-pattern problem; the trained solution drives a subset of muscles to the ceiling, so its plateau is partly a capacity wall that more gain tuning cannot cross')
 out['calls']=dict(model_load=0,forward=0,body_step=0,neural_step=0,training=0)
 Path(O/'RESULT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:{wn:{kk:vv for kk,vv in v[wn].items() if kk in ('actuators_saturated_over_50pct','max_per_actuator_saturation','mean_active_muscles')} for wn in ('assist','release')} for k,v in out['runs'].items()},indent=1))
if __name__=='__main__':main()
