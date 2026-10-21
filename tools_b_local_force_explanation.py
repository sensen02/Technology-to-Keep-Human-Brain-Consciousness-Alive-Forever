"""Offline per-DOF force decomposition; not acceleration or biological calibration."""
from pathlib import Path
import json,hashlib
import numpy as np
P=Path('/home/sensen/Desktop/engine修复验收/B-local-replay-20261014T213817')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def first(v,t):
 idx=np.flatnonzero(np.asarray(v)!=0);return None if not len(idx) else dict(sample=int(idx[0]),time_s=float(t[idx[0]]),value=float(v[idx[0]]))
def main():
 out=Path('/home/sensen/Desktop/engine修复验收')/'B-local-force-explanation-20261014-v1';out.mkdir(exist_ok=False)
 report=json.loads((P/'report.json').read_text());vd=report['joint']['dof'];qp=report['joint']['qpos'];assert (vd,qp)==(12,13)
 base=load(P/'baseline.npz');t=base['time_s'];result=dict(scope='OFFLINE_ENDPOINT_DECOMPOSITION_NOT_MASS_ACCELERATION_IDENTITY',dof=vd,qpos=qp,cases={},baseline_targets={});arrays={'time_s':t}
 for ai in (13,14):
  mom=base['moment'][:,ai,vd];f=base['actuator_force'][:,ai];tau=mom*f
  result['baseline_targets'][str(ai)]=dict(moment_min=float(mom.min()),moment_max=float(mom.max()),force_min=float(f.min()),force_max=float(f.max()),generalized_torque_min=float(tau.min()),generalized_torque_max=float(tau.max()),positive_torque_samples=int(np.sum(tau>0)),negative_torque_samples=int(np.sum(tau<0)))
 for name,ai in [('cut_flex',13),('cut_extensor',14)]:
  a=load(P/(name+'.npz'));assert np.array_equal(a['time_s'],t)
  bt=base['moment'][:,:,vd]*base['actuator_force'];ct=a['moment'][:,:,vd]*a['actuator_force'];delta=ct-bt
  target=delta[:,ai];other=delta.sum(axis=1)-target
  terms=dict(target_actuator=target,other_actuators=other,passive=a['qfrc_passive'][:,vd]-base['qfrc_passive'][:,vd],constraint=a['qfrc_constraint'][:,vd]-base['qfrc_constraint'][:,vd],negative_bias=-(a['qfrc_bias'][:,vd]-base['qfrc_bias'][:,vd]))
  net=sum(terms.values());direct=(a['qfrc_actuator'][:,vd]+a['qfrc_passive'][:,vd]+a['qfrc_constraint'][:,vd]-a['qfrc_bias'][:,vd])-(base['qfrc_actuator'][:,vd]+base['qfrc_passive'][:,vd]+base['qfrc_constraint'][:,vd]-base['qfrc_bias'][:,vd]);err=float(np.max(np.abs(net-direct)));assert np.allclose(net,direct,rtol=1e-10,atol=1e-10)
  # Algebraic split uses cross-state factors only; NOT a rerun at a common state.
  df=a['actuator_force'][:,ai]-base['actuator_force'][:,ai];dm=a['moment'][:,ai,vd]-base['moment'][:,ai,vd];force_term=base['moment'][:,ai,vd]*df;geometry_term=dm*a['actuator_force'][:,ai];assert np.allclose(force_term+geometry_term,target,atol=1e-10,rtol=1e-10)
  onset={k:first(v,t) for k,v in dict(ctrl=a['ctrl'][:,ai]-base['ctrl'][:,ai],activation=a['act'][:,ai]-base['act'][:,ai],force=df,target_torque=target,qpos=a['qpos'][:,qp]-base['qpos'][:,qp],qvel=a['qvel'][:,vd]-base['qvel'][:,vd]).items()}
  entry=dict(command_start_s=.020,first_exact_differences=onset,decomposition_max_error=err,term_max_abs={k:float(np.max(np.abs(v))) for k,v in terms.items()},target_product_split_max_error=float(np.max(np.abs(force_term+geometry_term-target))),interpretation='trajectory differences include geometry and all state feedback; no same-state intervention or acceleration reconstruction',first_torque_terms={k:float(v[200]) for k,v in terms.items()},first_torque_states=dict(qpos_exact=bool(np.array_equal(a['qpos'][200],base['qpos'][200])),qvel_exact=bool(np.array_equal(a['qvel'][200],base['qvel'][200])),other_activation_exact=bool(np.array_equal(np.delete(a['act'][200],ai),np.delete(base['act'][200],ai)))))
  result['cases'][name]=entry
  for k,v in terms.items():arrays[name+'_'+k]=v
  arrays[name+'_net']=net;arrays[name+'_target_force_factor']=force_term;arrays[name+'_target_geometry_factor']=geometry_term
 paths=[P/n for n in ('baseline.npz','cut_flex.npz','cut_extensor.npz','report.json')]+[Path(__file__)];result['hashes']={str(p):sha(p) for p in paths}
 np.savez_compressed(out/'terms.npz',**arrays);(out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
 for ax,name in zip(axes,('cut_flex','cut_extensor')):
  for k in ('target_actuator','other_actuators','passive','constraint','negative_bias','net'):ax.plot(t*1000,arrays[name+'_'+k],label=k)
  ax.set_ylabel('delta model torque');ax.set_title(name+' (cut minus baseline)');ax.legend(ncol=3,fontsize=8)
 axes[-1].set_xlabel('endpoint time (ms)');fig.suptitle('Saved LF tibia force decomposition; NOT acceleration attribution / NOT standing');fig.tight_layout();fig.savefig(out/'force-terms.png');plt.close(fig)
 print(out);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
