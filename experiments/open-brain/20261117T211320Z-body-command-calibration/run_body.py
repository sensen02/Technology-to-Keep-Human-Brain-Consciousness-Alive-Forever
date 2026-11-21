"""Independent teacher or student physical rollout driven by identical postneck traces."""
from scope import *

@torch.inference_mode()
def run(family,variant,mode,out,checkpoint=WEIGHT):
 torch.set_num_threads(1);started=time.monotonic()
 command_dir=HERE/'COMMANDS'/f'{family}_{variant}';meta=json.loads((command_dir/'RESULT.json').read_text())
 with np.load(command_dir/'TRACE.npz',allow_pickle=False) as z:commands=z['command'].copy()
 assert sha(command_dir/'TRACE.npz')==meta['sha256']
 b=Body(height=0,adhesion=True,seed=meta['seed']);b.restore(meta['seed']);setup(b,'terminal')
 actor=read(checkpoint) if mode=='student' else None
 cpg=copy.deepcopy(b.motor.cpg_network);freq=cpg.intrinsic_freqs.copy();out.mkdir(parents=True,exist_ok=False)
 sources=freeze([checkpoint,command_dir/'TRACE.npz',command_dir/'RESULT.json'])
 write(out/'INPUTS.json',dict(source_sha256=sources,family=family,variant=variant,mode=mode,seed=meta['seed'],checkpoint=str(checkpoint),
   source='archived actual BANC fixed-dynamics DN arrival readout, identical for teacher/student'))
 mujoco.mj_saveModel(b.m,str(out/'BODY.mjb'),None);initial=b.d.xpos[b.thorax].copy()
 rows={k:[] for k in ('qpos','qvel','position','yaw','upright','nonfoot','control','pads','actuator_force','foot_forward','normal_load')}
 examples={k:[] for k in ('x','phase','y','lambda','pad_y','applied')}
 teacher_calls=0
 for tick,command in enumerate(commands):
  obs=observation(b)
  if mode=='teacher':action=b.motor.step(command,obs);teacher_calls+=1
  advance(cpg,freq,command);phase=phase_state(cpg)
  if mode=='teacher':assert np.max(abs(cpg.curr_phases-b.motor.cpg_network.curr_phases))<1e-10
  if tick%10==0:
   x=np.r_[feature(b.d.qpos.copy(),b.d.qvel.copy(),obs.stumbling_contact_forces,command,phase,b.weight),extra_live(b)].astype(np.float32)
   if mode=='teacher':control=np.clip(action.joint_angles,b.minimum,b.maximum);pads=action.adhesion_onoff.copy()
   else:control,pads,_=actor.act(x,phase)
   for k,v in dict(x=x,phase=phase,applied=control).items():examples[k].append(v.copy())
   if mode=='teacher':
    examples['y'].append(control.copy());examples['lambda'].append(b.motor.last_info['net_corrections'].copy());examples['pad_y'].append(pads.copy())
  b.apply(control,pads);load,force,nf,bf=b.contact();pos=b.d.xpos[b.thorax];rot=b.d.xmat[b.thorax].reshape(3,3)
  values=dict(qpos=b.d.qpos.copy(),qvel=b.d.qvel.copy(),position=pos.copy(),yaw=np.arctan2(rot[1,0],rot[0,0]),upright=rot[2,2],
    nonfoot=nf,control=control.copy(),pads=pads.copy(),actuator_force=b.d.actuator_force[b.actuators].copy(),
    foot_forward=(b.d.xpos[b.feet]-pos)@rot[:,0],normal_load=load.copy())
  for key,value in values.items():rows[key].append(value)
  if b.d.warning.number.any() or not np.isfinite(b.d.qpos).all():break
 arrays={k:np.array(v) for k,v in rows.items()};arrays.update({k:np.array(v) for k,v in examples.items()});arrays['initial_position']=initial
 np.savez_compressed(out/'TRACE.npz',**arrays)
 from metrics import measure
 result=measure(arrays,b.m,meta);result.update(mode=mode,family=family,variant=variant,teacher_calls=teacher_calls,
   weight_sha256=sha(checkpoint) if actor else None,steps=len(arrays['qpos']),trace_sha256=sha(out/'TRACE.npz'),warning=b.d.warning.number.tolist(),wall_seconds=time.monotonic()-started)
 verify_sources(sources);write(out/'RESULT.json',result)
 print(family,variant,mode,'v',round(result['forward_speed'],2),'yaw',round(result['yaw_degrees'],1),'stand',round(result['stand_rms'],3),flush=True)
 return result

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('family',choices=FAMILIES);p.add_argument('variant',type=int);p.add_argument('mode',choices=('teacher','student'));p.add_argument('out',type=Path);p.add_argument('--checkpoint',type=Path,default=WEIGHT);a=p.parse_args();run(a.family,a.variant,a.mode,a.out,a.checkpoint)
