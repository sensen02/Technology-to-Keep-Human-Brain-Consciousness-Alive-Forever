"""Matched tendon-force budgets, separate antagonist neural drives, full body fixture."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
import mujoco as mj
R=Path('/run/media/sensen/Data2/cell_wound_prototype');sys.path.insert(0,str(R))
from tendon_muscle import build,force_law,NAMES,JOINT,DT,P
from engine.neural_cond import ConductanceNetwork,ConductanceParams
from engine.embodied.sensory_channels import ChannelSensoryEncoder,ChannelConfig
from run_neural_muscle_closure import channel_index_map,observe_quantities
from run_integrated_repaired_closure import identities
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');BIAS=float(sys.argv[1]);VEL=float(sys.argv[2]);GAIN=float(sys.argv[3]);assert BIAS in [.02,.05] and VEL in [-.1,0,.1] and GAIN in [.1,.3]
O=ROOT/'sensory_grid/20261106T185758Z'/('b'+str(BIAS)+'_g'+str(GAIN)+'_v'+str(VEL))
A=Path('/home/sensen/Desktop/engine修复验收/s5-route-b-ablation/20261108T060000Z/ARMS/A');H=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z');LOOK=Path('/home/sensen/Desktop/engine修复验收/s3-banc888-runtime-admission/20261101T220018Z/SENSORY_LOOKUP_SNAPSHOT888.json')
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 O.mkdir(parents=True,exist_ok=True);assert not (O/'SUMMARY.json').exists()
 cal=json.loads((ROOT/'tendon_muscle/PREREGISTRATION.json').read_text());audit=json.loads((ROOT/'descending/PATH_AUDIT.json').read_text());cand=audit['ranked_candidate_indices'];flex=audit['targets']['1'];ext=audit['targets']['-1'];l0=np.array(cal['l0']);fmax=np.array(cal['fmax'])
 files=[Path(__file__),Path(__file__).with_name('tendon_muscle.py'),ROOT/'tendon_muscle/PREREGISTRATION.json',ROOT/'descending/PATH_AUDIT.json',P/'MODEL_JOINT_TORQUE.xml',H/'IDENTITIES.npz',H/'RUN/MAPPING.json',LOOK,A/'ACTUAL_We.npz',A/'ACTUAL_Wi.npz',R/'engine/neural_cond.py',R/'engine/embodied/sensory_channels.py',R/'run_neural_muscle_closure.py',R/'run_integrated_repaired_closure.py'];hashes={str(p):sha(p) for p in files}
 protocol=dict(importance='核心必需',dynamic_sensory_gain=GAIN,sensory_bias_nA=BIAS,bias_cells='27 actual LF claw_position cells only, also retained in sensory_cut',initial_velocity_rad_s=VEL,arms=['muscle_tendon'],conditions=['normal','sensory_cut'],pulse_ms=[250],dt_ms=[.5,.0625],duration_ms=250,gravity=0,readout='independent clip(mean flex/100Hz,0,1), clip(mean ext/100Hz,0,1); no subtraction; prior completed neural rates',current_nA=.1,candidates=cand,flex=flex,ext=ext,l0=l0.tolist(),fmax=fmax.tolist(),matched_budget='same actual tendon paths and same two tensile force caps; direct force=fmax*neural activation target; muscle adds activation dynamics, force-length/velocity/passive terms',assistance='thorax and41 joints held, no contact or gravity; old42 torques zero; no body state compensation',criterion='same-bias normal must reduce disturbance trajectory IAE relative to its own zero-velocity run compared with sensory_cut relative to its own zero-velocity run, for BOTH +/-0.1rad/s and BOTH neural dt; all range/finite/warning checks must pass; bias is engineering hypothesis, not biological rate',hashes=hashes)
 dump(O/'PREREGISTRATION.json',protocol)
 m=build();j=m.joint(JOINT).id;qa=m.jnt_qposadr[j];va=m.jnt_dofadr[j];tids=[m.tendon(n).id for n in NAMES];qref=m.key_qpos[0].copy();joints,geom=identities(m)
 roots=np.load(H/'IDENTITIES.npz')['root_ids'];groups,_=channel_index_map(12000,roots,lookup_path=LOOK)
 we=sparse.load_npz(A/'ACTUAL_We.npz').tocsr();wi=sparse.load_npz(A/'ACTUAL_Wi.npz').tocsr();rh=sorted({i for row in json.loads((H/'RUN/MAPPING.json').read_text())['rows'] if row['leg']=='RH' for i in row['local_indices']});scale=np.ones(12000);scale[rh]=.375;we=(sparse.diags(scale)@we).tocsr();records=[]
 for dt in protocol['dt_ms']:
  for pulse in protocol['pulse_ms']:
   for arm in protocol['arms']:
    for cond in protocol['conditions']:
     np.random.seed(0);net=ConductanceNetwork(12000,ConductanceParams(),dt_ms=dt);net.set_connectivity(we.copy(),wi.copy());enc=ChannelSensoryEncoder(groups,12000,ChannelConfig());d=mj.MjData(m);mj.mj_resetDataKeyframe(m,d,0);d.qvel[va]=VEL;mj.mj_forward(m,d);q0=float(d.qpos[qa]);rates=np.zeros(12000);activation=np.zeros(2);rows=[];packed=[];commands=[]
     for k in range(250):
      target=np.clip([rates[flex].mean()/100,rates[ext].mean()/100],0,1)
      if cond=='zero_output':target[:]=0
      mj.mj_forward(m,d);current=GAIN*np.asarray(enc.encode(observe_quantities(m,d,qref,joints,geom))[0]) if cond!='sensory_cut' else np.zeros(12000);sense_max=float(abs(current).max())
      current[groups['lf']['claw_position']]+=BIAS
      if k<pulse:current[cand]+=.1
      commands.append([k*.001,*target,sense_max])
      for _ in range(round(1/dt)):
       sp=net.step(current_nA=current);packed.append(np.packbits(sp));decay=np.exp(-dt/5);rates=decay*rates+(1-decay)*sp/(dt/1000)
      for _ in range(80):
       tau=np.where(target>activation,.01,.03);activation=target+(activation-target)*np.exp(-DT/tau)
       mj.mj_forward(m,d);length=d.ten_length[tids].copy();velocity=d.ten_velocity[tids].copy()
       force=force_law(activation,length,velocity,l0,fmax)[0] if arm=='muscle_tendon' else fmax*target
       d.ctrl[:]=0;d.ctrl[-2:]=-force;mj.mj_step(m,d)
       rows.append([d.time,d.qpos[qa],d.qvel[va],*activation,*length,*velocity,*force,float(np.linalg.norm(d.qfrc_constraint))])
     arr=np.array(rows);tag=f'{arm}_{cond}_p{pulse}_dt{dt}';np.savez_compressed(O/(tag+'.npz'),rows=arr,commands=commands,spikes_packed=packed)
     r=dict(tag=tag,arm=arm,condition=cond,pulse_ms=pulse,dt_ms=dt,shift_at_off=float(arr[pulse*80-1,1]-q0),final_shift=float(arr[-1,1]-q0),max_abs_shift=float(abs(arr[:,1]-q0).max()),peak_abs_velocity=float(abs(arr[:,2]).max()),within_range=bool((arr[:,1]>=m.jnt_range[j,0]).all() and (arr[:,1]<=m.jnt_range[j,1]).all()),forces_bounded=bool((arr[:,9:11]>=0).all() and (arr[:,9:11]<=fmax).all()),finite=bool(np.isfinite(arr).all()),warnings=int(d.warning.number.sum()),spikes=int(np.unpackbits(np.array(packed),axis=1)[:,:12000].sum()))
     records.append(r);print(json.dumps(r),flush=True)
 assert hashes=={str(p):sha(p) for p in files}
 dump(O/'SUMMARY.json',dict(importance='核心必需',records=records,physical_runs=len(records),inputs_unchanged=True,autonomous_support=False))
if __name__=='__main__':main()
