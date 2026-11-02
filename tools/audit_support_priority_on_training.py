"""Transparent post-hoc metric audit on sealed training; never retrains/reselects it."""
from pathlib import Path
import sys,json,hashlib,math
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from engine.embodied.support_priority import evaluate_support,select_support_first
S=Path('/home/sensen/Desktop/engine修复验收/s4-bending-closedloop-training/20261102T042848Z')
O=Path('/home/sensen/Desktop/engine修复验收/s4-support-priority-metric/20261102T053639Z')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 O.mkdir(parents=True,exist_ok=True)
 assert not (O/'RESULT.json').exists(),'Do not overwrite accepted audit results'
 files=[Path(__file__),R/'engine/embodied/support_priority.py',R/'test_support_priority.py',S/'TRAINING_SCORES.json',S/'SELECTED_CANDIDATE.json',S/'FROZEN_REFERENCE.json'];hashes={str(f):sha(f) for f in files}
 dump(O/('FINAL_METHOD_PREREGISTRATION.json' if (O/'PREREGISTRATION.json').exists() else 'PREREGISTRATION.json'),dict(scope='Post-hoc diagnostic metric audit only, not retrospective experiment selection or claimed training success',source_sha256=hashes,horizon_s=.1,force_deadband_weight=1e-8,early_weight_tau_s=.02,policy='A candidate must complete the declared foot-only support horizon with no positive nonfoot load over fixed numericaldeadband,noexternalassistance,meanfootload>=.8W,minCOMheight>=.8h0. These are explicit engineering screening gates,not universalanimalgait. No pass: admitted_label=None,timeordereddiagnostic only.',sensitivity_deadbands=[0,1e-8,1e-6],no_new_body_neural_or_training=True))
 history=json.loads((S/'TRAINING_SCORES.json').read_text());selected=json.loads((S/'SELECTED_CANDIDATE.json').read_text());ref=json.loads((S/'FROZEN_REFERENCE.json').read_text());out=[];sensitivity=[]
 for record in history:
  p=S/record['label']/'BODYLOSS_NATIVE.npz';hashes[str(p)]=sha(p);rows=np.load(p)['rows'][1:];times=rows[:,0]-1.3
  results={}
  for eps in [0.,1e-8,1e-6]:
   stats=evaluate_support(times,rows[:,1:7],rows[:,7],rows[:,10],weight=ref['weight'],h0=ref['h0'],horizon_s=.1,force_epsilon_weight=eps,early_tau_s=.02,assistance=rows[:,19,None]);results[str(eps)]=stats
   # Independent scalar endpoint rule and weighted costs.
   positive=rows[:,7]/ref['weight']>eps;where=np.flatnonzero(positive);first=None if not len(where) else float(times[where[0]])
   assert stats['first_nonfoot_s']==first and stats['positive_nonfoot_endpoint_count']==int(positive.sum())
   dt=np.diff(np.r_[0.,times]);weighted=[];values=[]
   for i,t in enumerate(times):
    w=float(dt[i])*math.exp(-float(t)/.02);weighted.append(w);values.append(w*((math.fsum(map(float,rows[i,1:7]))/ref['weight']-1)**2+float(rows[i,7])/ref['weight']+((float(rows[i,10])-ref['h0'])/ref['h0'])**2))
   assert abs(math.fsum(values)/math.fsum(weighted)-stats['early_weighted_support_cost'])<1e-12
  out.append(dict(label=record['label'],motion_loss=record['terms']['motion'],old_weighted_total=record['loss'],support=results['1e-08']))
  sensitivity.append(dict(label=record['label'],deadbands=results))
 result=select_support_first(out);assert result['admitted_label'] is None and result['feasible_candidates']==0
 assert selected==min(history,key=lambda h:h['loss'])
 dump(O/'RECORD_METRICS.json',out);dump(O/'DEADBAND_SENSITIVITY.json',sensitivity)
 report=dict(passed=True,source_sha256=hashes,completed_training_records=len(history),new_metric_selection=result,old_experiment_selected=selected['label'],old_selection_and_frozen_outputs_unchanged=True,interpretation='All19previouslearningrecords fail this declared100mscleanfoot-support horizon; diagnostic leader is not an admitted model, not a new trained candidate. New rule separates event feasibility from motion/error compensation. This is post-hoc audit only and cannot retroactively produce preregistered success.',calls=dict(model_compile=0,body_step=0,network=0,neural_step=0,training=0,optimizer=0))
 assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'RESULT.json',report);print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
