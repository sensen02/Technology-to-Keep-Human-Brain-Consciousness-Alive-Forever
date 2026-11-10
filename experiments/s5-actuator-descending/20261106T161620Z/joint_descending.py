"""Zero-load full-body isolated neural feedback diagnostic, not gait handover."""
import sys,copy,json,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
import mujoco as mj
R=Path('/run/media/sensen/Data2/cell_wound_prototype');sys.path.insert(0,str(R))
from calibrate_v3 import build,JOINT,DT,CDT,P
from engine.neural_cond import ConductanceNetwork,ConductanceParams
from engine.embodied.sensory_channels import ChannelSensoryEncoder,ChannelConfig
from run_neural_muscle_closure import channel_index_map,observe_quantities
from run_integrated_repaired_closure import identities
NDT=float(sys.argv[1]) if len(sys.argv)>1 else .0005
assert NDT in [.0005,.0000625]
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z')
O=ROOT/('joint_descending_'+str(NDT))
selection=json.loads((ROOT/'descending_probe/PREREGISTRATION.json').read_text())
CAND=selection['candidate'];MATCHED=selection['matched']
A=Path('/home/sensen/Desktop/engine修复验收/s5-route-b-ablation/20261108T060000Z/ARMS/A')
H=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
LOOK=Path('/home/sensen/Desktop/engine修复验收/s3-banc888-runtime-admission/20261101T220018Z/SENSORY_LOOKUP_SNAPSHOT888.json')
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 O.mkdir(parents=True,exist_ok=True);assert not (O/'SUMMARY.json').exists()
 files=[ROOT/'descending_probe/PREREGISTRATION.json',Path(__file__),Path(__file__).with_name('calibrate_v3.py'),P/'MODEL_JOINT_TORQUE.xml',P/'POOL_TO_JOINT.json',H/'IDENTITIES.npz',H/'RUN/MAPPING.json',LOOK,A/'ACTUAL_We.npz',A/'ACTUAL_Wi.npz',R/'engine/neural_cond.py',R/'engine/embodied/sensory_channels.py',R/'run_neural_muscle_closure.py',R/'run_integrated_repaired_closure.py']
 hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 protocol=dict(importance='核心必需',stage='rest-state zero-gravity isolated joint diagnostic, NOT teacher handover or support',arms=['torque','incremental','muscle'],conditions=['normal','zero_output','sensory_cut','matched','candidate_cut'],descending_current_nA=.1,candidate=CAND,matched=MATCHED,initial_velocity_rad_s=[-.1,.1],duration_s=.1,neural_dt_s=NDT,command_dt_s=CDT,physics_dt_s=DT,
  initial_state='network rest, physical keyframe; one declared initial angular velocity perturbation; no posture rewrites during trial',
  fixtures='all contacts disabled, zero gravity, thorax weld and other 41 joints held',readout='u=clip((mean flex rate-mean ext rate)/100Hz,-1,1); no fitted offsets; declared tonic .1nA per selected descending cell from t=0, no gait waveform; selected as lower effective probe dose, not by physics',
  actuator_parameters='same as calibration_v3; neural u may exceed calibration pulse; failures retained',
  outcome_rule='diagnostic only: neural contribution requires BOTH perturbation directions lower integrated absolute angle deviation than same-actuator zero_output AND sensory_cut, with range/finite/warning checks; this is not full-body validation',
  hashes=hashes)
 dump(O/'PREREGISTRATION.json',protocol)
 m=build();j=m.joint(JOINT).id;qa=m.jnt_qposadr[j];va=m.jnt_dofadr[j];a=int(np.flatnonzero(m.actuator_trnid[:,0]==j)[0]);lim=m.actuator_ctrlrange[a,1];kp=lim/.2;damp=lim*.005
 qref=m.key_qpos[0].copy();joints,geom=identities(m)
 roots=np.load(H/'IDENTITIES.npz')['root_ids'];groups,_=channel_index_map(12000,roots,lookup_path=LOOK)
 pools=[p for p in json.loads((P/'POOL_TO_JOINT.json').read_text())['muscle_to_joint'] if p['joint']==JOINT]
 flex=next(p['local_indices'] for p in pools if p['sign']==1);ext=next(p['local_indices'] for p in pools if p['sign']==-1)
 we=sparse.load_npz(A/'ACTUAL_We.npz').tocsr();wi=sparse.load_npz(A/'ACTUAL_Wi.npz').tocsr()
 rh=sorted({i for p in json.loads((H/'RUN/MAPPING.json').read_text())['rows'] if p['leg']=='RH' for i in p['local_indices']})
 # Preserve the established RH multiplier, once, even though only LF is actuated.
 scale=np.ones(12000);scale[rh]=.375;we=sparse.diags(scale)@we
 runs=[]
 for arm in protocol['arms']:
  for cond in protocol['conditions']:
   for vel in protocol['initial_velocity_rad_s']:
    np.random.seed(0);net=ConductanceNetwork(12000,ConductanceParams(),dt_ms=NDT*1000)
    e=we.copy(); inh=wi.copy()
    if cond=='candidate_cut':
     mask=np.ones(12000);mask[CAND]=0;e=e@sparse.diags(mask);inh=inh@sparse.diags(mask)
    net.set_connectivity(e.tocsr(),inh.tocsr())
    enc=ChannelSensoryEncoder(groups,12000,ChannelConfig());d=mj.MjData(m);mj.mj_resetDataKeyframe(m,d,0);d.qvel[va]=vel;mj.mj_forward(m,d)
    q0=float(d.qpos[qa]);ref=q0;rates=np.zeros(12000);act=np.zeros(2);rows=[];spikes=[];commands=[]
    for k in range(100):
     source=rates.copy();u=float(np.clip((source[flex].mean()-source[ext].mean())/100,-1,1));u=0. if cond=='zero_output' else u
     ref=float(np.clip(ref+CDT*2*u,*m.jnt_range[j]))
     current=np.asarray(enc.encode(observe_quantities(m,d,qref,joints,geom))[0]) if cond!='sensory_cut' else np.zeros(12000)
     sensory_max=float(np.max(np.abs(current)))
     current[MATCHED if cond=='matched' else CAND]+=.1
     commands.append([d.time,u,source[flex].mean(),source[ext].mean(),sensory_max])
     for _ in range(round(CDT/NDT)):
      sp=net.step(current_nA=current);spikes.append(np.packbits(sp));decay=np.exp(-NDT/.005);rates=decay*rates+(1-decay)*sp/NDT
     for sub in range(80):
      if arm=='torque':control=lim*u
      elif arm=='incremental':control=kp*(ref-d.qpos[qa])-damp*d.qvel[va]
      else:
       target=np.array([max(u,0),max(-u,0)]);act=target+(act-target)*np.exp(-DT/.01)
       control=lim*(act[0]-act[1])+.1*kp*(q0-d.qpos[qa])-damp*d.qvel[va]
      d.ctrl[:]=0;d.ctrl[a]=np.clip(control,-lim,lim);mj.mj_step(m,d)
      rows.append([d.time,d.qpos[qa],d.qvel[va],d.ctrl[a],ref,*act,float(np.linalg.norm(d.qfrc_constraint))])
    arr=np.array(rows);tag=f'{arm}_{cond}_{"neg" if vel<0 else "pos"}'
    np.savez_compressed(O/(tag+'.npz'),rows=arr,commands=commands,spikes_packed=np.array(spikes),flex=flex,ext=ext)
    r=dict(tag=tag,arm=arm,condition=cond,velocity=vel,angle_iae=float(np.abs(arr[:,1]-q0).sum()*DT),terminal_error=float(abs(arr[-1,1]-q0)),max_error=float(abs(arr[:,1]-q0).max()),spikes=int(np.unpackbits(np.array(spikes),axis=1)[:,:12000].sum()),within_range=bool((arr[:,1]>=m.jnt_range[j,0]).all() and (arr[:,1]<=m.jnt_range[j,1]).all()),finite=bool(np.isfinite(arr).all()),warnings=int(d.warning.number.sum()))
    runs.append(r);print(json.dumps(r),flush=True)
 effects={}
 for arm in protocol['arms']:
  effects[arm]=all(next(r for r in runs if r['arm']==arm and r['velocity']==v and r['condition']=='normal')['angle_iae']<next(r for r in runs if r['arm']==arm and r['velocity']==v and r['condition']==c)['angle_iae'] for v in [-.1,.1] for c in ['zero_output','sensory_cut']) and all(r['within_range'] and r['finite'] and r['warnings']==0 for r in runs if r['arm']==arm)
 dump(O/'SUMMARY.json',dict(importance='核心必需',runs=runs,neural_contribution_screen=effects,physics_runs=len(runs),autonomous_support=False))
 assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 print('COMPLETE',effects,flush=True)
if __name__=='__main__':main()
