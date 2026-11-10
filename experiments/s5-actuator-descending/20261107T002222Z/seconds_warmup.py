"""S5: continuous physical/neural 3s warmup followed by 1s withdrawal.
Full body local fixture diagnostic; NOT autonomous standing or walking.
"""
import sys,json,hashlib,copy,time
from pathlib import Path
import numpy as np
from scipy import sparse
import mujoco as mj
R=Path('/run/media/sensen/Data2/cell_wound_prototype')
OLD=R/'experiments/s5-actuator-descending/20261106T161620Z'
sys.path[:0]=[str(OLD),str(R)]
from tendon_muscle import build,force_law,NAMES,JOINT,DT,P
from interval_readout import step as readout_step
from engine.neural_cond import ConductanceNetwork,ConductanceParams
from engine.embodied.sensory_channels import ChannelSensoryEncoder,ChannelConfig
from run_neural_muscle_closure import channel_index_map,observe_quantities
from run_integrated_repaired_closure import identities
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z')
O=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261107T002222Z')
A=ROOT/'synaptic_scale/20261106T193236Z/0.25'
H=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
LOOK=Path('/home/sensen/Desktop/engine修复验收/s3-banc888-runtime-admission/20261101T220018Z/SENSORY_LOOKUP_SNAPSHOT888.json')
NDT=.0078125; WARM=3000; POST=1000; PRE=.3; BIAS=.03; GAIN=1.
FP=[8179,8164,1496,11176,8422,6341];EP=[11051,9574,6642,4476,4183,3003]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def state(m,s):
 d,net,enc,filtered,target,activation=s
 spec=mj.mjtState.mjSTATE_INTEGRATION
 body=np.empty(mj.mj_stateSize(m,spec));mj.mj_getState(m,d,body,spec)
 out={'body_integration':body,'filtered':filtered.copy(),'target':target.copy(),'activation':activation.copy()}
 for key in ['v','ge','gi','adaptation','refractory_until_ms','spike_count','_ring','_ring_i','t_ms','steps']:
  out['neural_'+key]=np.asarray(getattr(net,key)).copy()
 return out

def main():
 O.mkdir(parents=True,exist_ok=True);assert not (O/'PREREGISTRATION.json').exists()
 inputs=[Path(__file__),OLD/'premotor_body_claw_20261106T225055Z.py',OLD/'tendon_muscle.py',OLD/'interval_readout.py',ROOT/'tendon_muscle/PREREGISTRATION.json',ROOT/'descending/PATH_AUDIT.json',A/'EFFECTIVE_We.npz',A/'EFFECTIVE_Wi.npz',P/'MODEL_JOINT_TORQUE.xml',H/'IDENTITIES.npz',LOOK,R/'engine/neural_cond.py',R/'engine/embodied/sensory_channels.py',R/'run_neural_muscle_closure.py',R/'run_integrated_repaired_closure.py']
 hashes={str(p):sha(p) for p in inputs}
 protocol=dict(importance='核心必需',stage='S5 seconds warmup',warmup_ms=WARM,post_ms=POST,neural_dt_ms=NDT,physics_dt_s=DT,premotor_nA=PRE,sensory_bias_nA=BIAS,sensory_gain=GAIN,gravity_fraction=.1,branches=['withdraw','sustained','withdraw_sensory_cut','withdraw_motor_cut','cold_none'],scope='full body nq49 nv48 mesh71, thorax welded and 41 other joints held, no contact; local diagnostic not standing/walking',handover='same continuous neural, encoder, filter, target, muscle, full MjData state; exact deep copies for matched branches; no reset or qpos/qvel overwrite',warmup='live neural premotor injection plus real muscle/physics and live claw feedback for 3 seconds; no scripted motion',retained_inputs='claw bias .03 nA throughout; sensory_cut removes dynamic position current only; no descending drive',criterion='withdrawal counts in final 500ms must exceed cold-none and reach >=50% of sustained counts in EACH motor pool; passing only establishes local activity persistence, never standing/walking; report every 100ms bin and body drift',hashes=hashes,change_ledger=dict(old_file=str(inputs[1]),old_sha256=hashes[str(inputs[1])],new_file=str(inputs[0]),new_sha256=hashes[str(inputs[0])],reason='extend real continuous warmup to 3s and preserve every state through current withdrawal, add 1s matched controls'))
 dump(O/'PREREGISTRATION.json',protocol)
 m=build();m.opt.gravity[:]=[0,0,-980.1];j=m.joint(JOINT).id;qa=m.jnt_qposadr[j];va=m.jnt_dofadr[j];tids=[m.tendon(n).id for n in NAMES];qref=m.key_qpos[0].copy();joints,geom=identities(m)
 roots=np.load(H/'IDENTITIES.npz')['root_ids'];groups,_=channel_index_map(12000,roots,lookup_path=LOOK);claw=groups['lf']['claw_position']
 audit=json.loads((ROOT/'descending/PATH_AUDIT.json').read_text());flex=audit['targets']['1'];ext=audit['targets']['-1']
 cal=json.loads((ROOT/'tendon_muscle/PREREGISTRATION.json').read_text());l0=np.array(cal['l0']);fmax=np.array(cal['fmax'])
 we=sparse.load_npz(A/'EFFECTIVE_We.npz');wi=sparse.load_npz(A/'EFFECTIVE_Wi.npz')
 def fresh():
  np.random.seed(0);net=ConductanceNetwork(12000,ConductanceParams(),dt_ms=NDT);net.set_connectivity(we,wi)
  d=mj.MjData(m);mj.mj_resetDataKeyframe(m,d,0);mj.mj_forward(m,d)
  return [d,net,ChannelSensoryEncoder(groups,12000,ChannelConfig()),np.zeros(2),np.zeros(2),np.zeros(2)]
 records=[]
 def segment(s,tag,ms,pre_on,sensory=True,motor=True):
  folder=O/tag;folder.mkdir();initial=state(m,s);np.savez_compressed(folder/'START.npz',**initial)
  d,net,enc,filtered,averaged,activation=s
  # Event lists avoid storing gigabytes of empty spike bins; all cells and exact step indices retained.
  event_steps=[];event_cells=[];rows=[];commands=[];bodies=[];currents=[];counts=[];started=time.monotonic()
  for k in range(ms):
   target=averaged.copy() if motor else np.zeros(2)
   mj.mj_forward(m,d);raw=np.asarray(enc.encode(observe_quantities(m,d,qref,joints,geom))[0]);current=np.zeros(12000)
   if sensory:current[claw]=GAIN*raw[claw]
   current[claw]+=BIAS
   if pre_on:current[FP]+=PRE;current[EP]+=PRE
   currents.append(current[claw].copy());commands.append([d.time,*target,PRE if pre_on else 0.])
   area_sum=np.zeros(2);bin_counts=np.zeros(3,dtype=np.int64)
   for sub in range(round(1/NDT)):
    sp=net.step(current_nA=current);ix=np.flatnonzero(sp)
    if len(ix):event_steps.append(np.full(len(ix),k*round(1/NDT)+sub,dtype=np.int32));event_cells.append(ix.astype(np.int32))
    nf=int(sp[flex].sum());ne=int(sp[ext].sum());bin_counts += [nf,ne,len(ix)]
    forcing=np.array([nf/len(flex),ne/len(ext)])/(NDT/1000)/100
    filtered,area=readout_step(filtered,forcing,NDT);area_sum+=area
   averaged=area_sum;counts.append(bin_counts)
   for _ in range(80):
    tau=np.where(target>activation,.01,.03);activation=target+(activation-target)*np.exp(-DT/tau)
    mj.mj_forward(m,d);length=d.ten_length[tids].copy();velocity=d.ten_velocity[tids].copy();force=force_law(activation,length,velocity,l0,fmax)[0]
    d.ctrl[:]=0;d.ctrl[-2:]=-force;mj.mj_step(m,d)
    rows.append([d.time,d.qpos[qa],d.qvel[va],*activation,*length,*velocity,*force])
   bodies.append(np.r_[d.time,d.qpos,d.qvel])
   if (k+1)%500==0:print(json.dumps(dict(tag=tag,simulated_ms=k+1,wall_s=round(time.monotonic()-started,1),last500_motor=np.sum(counts[-500:],axis=0).tolist())),flush=True)
  s[3:]=[filtered,averaged,activation]
  arr=np.asarray(rows);counts=np.array(counts)
  np.savez_compressed(folder/'TRACE.npz',rows=arr,commands=commands,bodies=bodies,claw_current=currents,counts=counts,event_steps=np.concatenate(event_steps) if event_steps else np.empty(0,dtype=np.int32),event_cells=np.concatenate(event_cells) if event_cells else np.empty(0,dtype=np.int32))
  np.savez_compressed(folder/'END.npz',**state(m,s))
  r=dict(tag=tag,duration_ms=ms,motor_tail500=counts[-500:,:2].sum(axis=0).tolist(),counts100=counts.reshape(-1,100,3).sum(axis=1).tolist(),within_range=bool(((arr[:,1]>=m.jnt_range[j,0])&(arr[:,1]<=m.jnt_range[j,1])).all()),force_bounded=bool(((arr[:,9:11]>=0)&(arr[:,9:11]<=fmax)).all()),finite=bool(np.isfinite(arr).all() and np.isfinite(net.v).all()),warnings=int(d.warning.number.sum()),final_q=float(d.qpos[qa]),wall_s=time.monotonic()-started)
  records.append(r);dump(folder/'SUMMARY.json',r);print(json.dumps(r),flush=True)
  return s
 warm=segment(fresh(),'warmup3s',WARM,True)
 for tag,pre_on,sensory,motor in [('withdraw',False,True,True),('sustained',True,True,True),('withdraw_sensory_cut',False,False,True),('withdraw_motor_cut',False,True,False)]:
  branch=copy.deepcopy(warm)
  before=state(m,branch);source=state(m,warm)
  assert all(np.array_equal(before[k],source[k]) for k in source)
  segment(branch,tag,POST,pre_on,sensory,motor)
 segment(fresh(),'cold_none',POST,False)
 assert hashes=={str(p):sha(p) for p in inputs}
 dump(O/'SUMMARY.json',dict(importance='核心必需',records=records,total_integrated_seconds=(WARM+5*POST)/1000,continuous_handover_verified=True,physics=dict(nq=m.nq,nv=m.nv,nmesh=m.nmesh,neq=m.neq,gravity=m.opt.gravity.tolist(),joint_range=m.jnt_range[j].tolist()),motor_pools=dict(flex=flex,ext=ext),fmax=fmax.tolist(),l0=l0.tolist(),autonomous_standing_or_walking=False,absolute_SI_force='NOT_ESTABLISHED'))
if __name__=='__main__':main()
