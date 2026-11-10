"""Independent saved-event, handover, current, muscle and manifest verification."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
from scipy.signal import lfilter
from numpy.polynomial.legendre import leggauss
import mujoco as mj
D=Path(__file__).parent;sys.path.insert(0,str(D))
import seconds_warmup as run
O=run.O

def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 summary=json.loads((O/'SUMMARY.json').read_text());p=json.loads((O/'PREREGISTRATION.json').read_text())
 checks={};results=[];warm_end=load(O/'warmup3s/END.npz');dt=p['neural_dt_ms'];sub=round(1/dt);decay=np.exp(-dt/5)
 m=run.build();m.opt.gravity[:]=[0,0,-980.1];j=m.joint(run.JOINT).id;qa=m.jnt_qposadr[j];qref=m.key_qpos[0,qa]
 nodes,weights=leggauss(4)
 for r in summary['records']:
  tag=r['tag'];folder=O/tag;z=load(folder/'TRACE.npz');start=load(folder/'START.npz');end=load(folder/'END.npz');ms=r['duration_ms'];ns=ms*sub
  if tag in ['withdraw','sustained','withdraw_sensory_cut','withdraw_motor_cut']:
   checks[tag+'_handover']=all(np.array_equal(start[k],warm_end[k]) for k in start)
  steps=z['event_steps'];cells=z['event_cells'];checks[tag+'_events_valid']=bool(len(steps)==len(cells) and np.all((steps>=0)&(steps<ns)) and np.all((cells>=0)&(cells<12000)) and np.all(np.diff(steps)>=0))
  counts=np.zeros((ms,3),dtype=np.int64);counts[:,2]=np.bincount(steps//sub,minlength=ms)
  forcing=np.zeros((ns,2))
  for c,pool in enumerate([summary['motor_pools']['flex'],summary['motor_pools']['ext']]):
   mask=np.isin(cells,pool);st=steps[mask];counts[:,c]=np.bincount(st//sub,minlength=ms)
   forcing[:,c]=np.bincount(st,minlength=ns)*1000/dt/100/len(pool)
  checks[tag+'_counts']=np.array_equal(counts,z['counts']) and np.array_equal(counts[-500:,:2].sum(axis=0),r['motor_tail500'])
  checks[tag+'_neuron_counts']=np.array_equal(np.bincount(cells,minlength=12000),end['neural_spike_count']-start['neural_spike_count'])
  # Independent scipy recurrence followed by Gauss quadrature split at clip crossings.
  endpoints=np.column_stack([lfilter([1-decay],[1,-decay],forcing[:,c],zi=[decay*start['filtered'][c]])[0] for c in range(2)])
  starts=np.vstack([start['filtered'],endpoints[:-1]])
  crossing=((starts<1)&(endpoints>1))|((starts>1)&(endpoints<1))
  split=np.full_like(starts,dt)
  split[crossing]=-5*np.log((1-forcing[crossing])/(starts[crossing]-forcing[crossing]))
  integral=np.zeros_like(starts)
  for lo,hi in [(np.zeros_like(split),split),(split,np.full_like(split,dt))]:
   for node,weight in zip(nodes,weights):
    t=(hi+lo)/2+(hi-lo)/2*node
    value=forcing+(starts-forcing)*np.exp(-t/5)
    integral += (hi-lo)/2*weight*np.clip(value,0,1)
  targets=integral.reshape(ms,sub,2).sum(axis=1)
  expected=np.vstack([start['target'],targets[:-1]])
  if tag=='withdraw_motor_cut':expected[:]=0
  command_error=float(abs(expected-z['commands'][:,1:3]).max())
  checks[tag+'_commands']=command_error<1e-10
  checks[tag+'_filter_end']=bool(np.max(abs(endpoints[-1]-end['filtered']))<1e-10 and np.max(abs(targets[-1]-end['target']))<1e-10)
  checks[tag+'_premotor_withdrawal']=bool(np.all(z['commands'][:,3]==(.3 if tag in ['warmup3s','sustained'] else 0)))
  data=mj.MjData(m);mj.mj_setState(m,data,start['body_integration'],mj.mjtState.mjSTATE_INTEGRATION)
  prior_q=np.r_[data.qpos[qa],z['rows'][79::80,1][:-1]]
  expected_current=np.full(ms,.03) if tag=='withdraw_sensory_cut' else .03+(prior_q-qref)
  current_error=float(np.max(abs(z['claw_current']-expected_current[:,None])))
  checks[tag+'_live_position_current']=current_error<1e-12
  checks[tag+'_body_record']=bool(np.array_equal(z['bodies'][:,1+qa],z['rows'][79::80,1]))
  rows=z['rows'];a=start['activation'].copy();activation=[]
  for target in z['commands'][:,1:3]:
   tau=np.where(target>a,.01,.03)
   block=target+(a-target)*np.exp(-np.arange(1,81)[:,None]*run.DT/tau)
   activation.append(block);a=block[-1]
  activation=np.concatenate(activation);activation_error=float(np.max(abs(activation-rows[:,3:5])))
  checks[tag+'_activation']=activation_error<1e-10
  lam=rows[:,5:7]/summary['l0'];vel=rows[:,7:9]/(10*np.array(summary['l0']))
  fl=np.exp(-((lam-1)/.25)**2)
  fv=np.where(vel<0,np.maximum(0,1+vel)/(1+4*np.maximum(-vel,0)),1+.5*np.maximum(vel,0)/(np.maximum(vel,0)+.25))
  force=np.clip(np.array(summary['fmax'])*(activation*fl*fv+.05*(np.maximum(lam-1,0)/.1)**2+.02*np.maximum(vel,0)),0,summary['fmax'])
  force_error=float(np.max(abs(force-rows[:,9:11])))
  checks[tag+'_force']=force_error<1e-10
  checks[tag+'_safety']=bool(np.isfinite(rows).all() and np.all(rows[:,1]>=m.jnt_range[j,0]) and np.all(rows[:,1]<=m.jnt_range[j,1]) and r['warnings']==0)
  checks[tag+'_time']=bool(abs(float(end['neural_t_ms']-start['neural_t_ms'])-ms)<1e-9 and abs(rows[-1,0]-z['commands'][0,0]-ms/1000)<1e-8 and int(end['neural_steps']-start['neural_steps'])==ns and np.max(abs(np.diff(rows[:,0])-run.DT))<1e-12 and np.max(abs(z['bodies'][:,0]-rows[79::80,0]))<1e-12 and np.max(abs(z['commands'][:,0]+.001-z['bodies'][:,0]))<1e-10)
  mj.mj_setState(m,data,end['body_integration'],mj.mjtState.mjSTATE_INTEGRATION)
  checks[tag+'_end_state']=bool(np.array_equal(data.qpos,z['bodies'][-1,1:1+m.nq]) and np.array_equal(data.qvel,z['bodies'][-1,1+m.nq:]) and np.array_equal(end['activation'],rows[-1,3:5]))
  checks[tag+'_unique_events']=bool(len(np.unique(steps.astype(np.int64)*12000+cells))==len(steps))
  results.append(dict(tag=tag,motor_tail500=counts[-500:,:2].sum(axis=0).tolist(),counts100=counts.reshape(-1,100,3).sum(axis=1).tolist(),command_error=command_error,current_error=current_error,activation_error=activation_error,force_error=force_error,last_motor_ms=[float((steps[np.isin(cells,pool)].max()+1)*dt) if np.any(np.isin(cells,pool)) else None for pool in [summary['motor_pools']['flex'],summary['motor_pools']['ext']]],final_q=float(rows[-1,1])))
 old_body=load(run.ROOT/'premotor_claw_body/20261106T225055Z/v0.0/muscle_tendon_normal_p250_dt0.0078125.npz')
 warm=load(O/'warmup3s/TRACE.npz')
 checks['first250ms_body_exactly_reproduces_previous']=np.array_equal(warm['rows'][:20000],old_body['rows'][:,:11])
 checks['first250ms_command_exactly_reproduces_previous']=bool(np.max(abs(warm['commands'][:250,:3]-old_body['commands'][:,:3]))<1e-12)
 checks['input_hashes']=all(run.sha(Path(path))==h for path,h in p['hashes'].items())
 by={r['tag']:r for r in results};w=np.array(by['withdraw']['motor_tail500']);s=np.array(by['sustained']['motor_tail500']);c=np.array(by['cold_none']['motor_tail500'])
 persistence=bool(np.all(w>c)&np.all(s>0)&np.all(w>=.5*s))
 # Independently finish verification of preceding 100ms static probe, without new simulation.
 old=run.ROOT/'warmup/20261106T225055Z';short=[]
 for folder in sorted(old.glob('*_off*')):
  if not (folder/'SUMMARY.json').exists():continue
  record=json.loads((folder/'SUMMARY.json').read_text())['records'][0];z=load(folder/'trail.npz');sp=np.unpackbits(z['packed'],axis=1)[:,:12000];oldp=json.loads((folder/'PREREGISTRATION.json').read_text());cut=round(oldp['readout_window_ms'][0]/oldp['dt_ms'])
  actual=[int(sp[cut:,pool].sum()) for pool in [summary['motor_pools']['flex'],summary['motor_pools']['ext']]]
  checks[folder.name+'_short_counts']=actual==[record['flex_post'],record['ext_post']]
  short.append(dict(tag=folder.name,post100_motor=actual))
 verification=dict(importance='核心必需',checks=checks,passed=all(checks.values()),results=results,short100ms=short,local_activity_persistence=persistence,withdraw_tail500_both_zero=bool(np.all(w==0)),verdict='LOCAL_ACTIVITY_PERSISTED' if persistence else 'LOCAL_ACTIVITY_PERSISTENCE_CRITERION_NOT_MET',scope='local full-body-fixture test only; no proof about all possible warmups or full-body assisted gait')
 dump(O/'VERIFICATION.json',verification)
 print(json.dumps(verification,ensure_ascii=False),flush=True)
 assert verification['passed'],[k for k,v in checks.items() if not v]
if __name__=='__main__':main()
