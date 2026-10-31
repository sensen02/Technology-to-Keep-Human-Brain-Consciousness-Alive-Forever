"""S128 frozen failure-boundary registration, no simulator imports."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
B=Path('/home/sensen/Desktop/engine修复验收');O=B/'s128-closed-loop-failure-boundary/20261024T083500Z'
P=B/'s121-constraint-row-recording/20261024T072500Z/init80ms';Q=B/'s127-joint-candidate-closed-loop/20261024T082500Z/init80ms';C=B/'s126-fixed-contraction/20261024T081500Z';F=B/'s119-training-constraints/20261024T070600Z/TRAINING_CONSTRAINTS.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n,v):(O/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def table(n,rs):
 with (O/n).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def onset(x,y,t):
 d=abs(y-x);ix=np.flatnonzero(d>0);jx=np.flatnonzero(d>1e-12+1e-10*np.maximum(abs(x),abs(y)))
 return dict(first_exact_ms=None if not ix.size else float(t[ix[0]]*1000),first_numerical_ms=None if not jx.size else float(t[jx[0]]*1000))
def main():
 files=[F,Path(__file__).resolve()]
 for p in (P.parent,Q.parent,C):files.extend(f for f in p.rglob('*') if f.is_file())
 hashes={str(p):sha(p) for p in files};report=json.loads((Q/'JOINT_CANDIDATE_INDEPENDENT_CHECK.json').read_text());assert report['passed'] and not report['S119_short_window_pass']
 for p in (P,Q):
  for name,h in json.loads((p/'RESULT.json').read_text())['hashes'].items():assert sha(name)==h
 a=dict(np.load(P/'TRAJECTORY.npz'));b=dict(np.load(Q/'TRAJECTORY.npz'));m=json.loads((P/'MOMENT_METADATA.json').read_text());t=a['time'];assert np.array_equal(t,b['time']);limits=[];e=[]
 for label,z in (('S121',a),('S127',b)):
  ex=[]
  for j in np.flatnonzero(m['jnt_limited']):
   q=z['qpos'][:,m['jnt_qposadr'][j]];lo,hi=m['jnt_range'][j];ex.append(np.maximum(np.maximum(lo-q,q-hi),0))
  assert len(ex)==42;e.append(np.array(ex).T);assert np.array_equal(e[-1].max(1),z['limit_excess'])
 for i in range(121):
  for k,j in enumerate(np.flatnonzero(m['jnt_limited'])):limits.append(dict(sample=i,time_ms=float(t[i]*1000),joint=m['joint_names'][j],reference=float(e[0][i,k]),actual=float(e[1][i,k]),delta=float(e[1][i,k]-e[0][i,k])))
 coupling=[]
 for i in range(121):
  r=dict(sample=i,time_ms=float(t[i]*1000),reference_excess=float(a['limit_excess'][i]),actual_excess=float(b['limit_excess'][i]),delta_excess=float(b['limit_excess'][i]-a['limit_excess'][i]))
  for label,z,p in (('reference',a,P),('actual',b,Q)):
   r[label+'_q']=float(z['qpos'][i,9]);r[label+'_v']=float(z['qvel'][i,8]);tau=z['actuator_force'][i]*z['actuator_moment'][i,:,8];assert abs(tau.sum()-z['qfrc_actuator'][i,8])<1e-10
   r[label+'_all90_torque']=json.dumps(tau.tolist());r[label+'_muscle_total']=float(tau.sum())
   with np.load(p/'CONSTRAINT_ROWS.npz') as rows:
    pref=f'sample{i:03d}_';J=rows[pref+'efc_J_dense'];force=rows[pref+'efc_force'];types=rows[pref+'efc_type'];v=J.T@force;assert abs(v-z['qfrc_constraint'][i]).max()<1e-8
    enum=json.loads((p/'CONSTRAINT_ROW_METADATA.json').read_text())['constraint_enum'];contact=[value for key,value in enum.items() if 'CONTACT' in key];lim=enum['mjCNSTR_LIMIT_JOINT'];r[label+'_contact']=float(J[np.isin(types,contact),8]@force[np.isin(types,contact)]);r[label+'_limit']=float(J[types==lim,8]@force[types==lim]);r[label+'_other']=float(J[~(np.isin(types,contact)|(types==lim)),8]@force[~(np.isin(types,contact)|(types==lim))])
   for key in ('qfrc_passive','qfrc_bias','Ma'):r[label+'_'+key]=float(z[key][i,8])
  coupling.append(r)
 peak=[dict(sample=int(z['limit_excess'].argmax()),time_ms=float(t[z['limit_excess'].argmax()]*1000),value=float(z['limit_excess'].max()),joint='joint_LFCoxa_roll') for z in (a,b)]
 result=dict(passed=True,decision='S127_REJECTED_UNCHANGED',cause='INSUFFICIENT_EVIDENCE_FOR_INDEPENDENT_CAUSE',new_weight_proposal=False,precision='exact numeric difference and abs1e-12 +rel1e-10; not a physical acceptance tolerance',onsets=dict(all42maximum=onset(a['limit_excess'],b['limit_excess'],t),roll_position=onset(a['qpos'][:,9],b['qpos'][:,9],t),roll_velocity=onset(a['qvel'][:,8],b['qvel'][:,8],t)),peaks=peak,peak_difference=peak[1]['value']-peak[0]['value'],s113_bound=json.loads(F.read_text())['safety_exact_bounds']['extrema']['limit_excess']['max'],model_loads=0,forward=0,steps=0,neural_ticks=0)
 O.mkdir(parents=True,exist_ok=False);dump('INPUT_HASHES.json',hashes);table('ALL42_PER_STATE_DIFFERENCES.csv',limits);table('ROLL_COUPLING.csv',coupling);dump('FAILURE_BOUNDARY.json',result)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(3,1,figsize=(10,10));axs[0].plot(t*1000,b['limit_excess']-a['limit_excess']);axs[0].set_ylabel('Delta all42 excess');axs[1].plot(t*1000,b['qpos'][:,9]-a['qpos'][:,9],label='delta roll q');axs[1].plot(t*1000,b['qvel'][:,8]-a['qvel'][:,8],label='delta roll v');axs[1].legend()
 for key in ('muscle_total','contact','limit'):axs[2].plot(t*1000,[r['actual_'+key]-r['reference_'+key] for r in coupling],label=key)
 axs[2].legend();axs[2].set_xlabel('physical ms');fig.suptitle('S128 frozen boundary: S127 rejected; algebra, NOT independent causal attribution');fig.tight_layout();fig.savefig(O/'FAILURE_BOUNDARY.png',dpi=130);plt.close(fig)
 assert all(sha(p)==h for p,h in hashes.items());dump('OUTPUT_HASHES.json',{str(p):sha(p) for p in O.iterdir() if p.is_file()});print(json.dumps(result))
if __name__=='__main__':main()
