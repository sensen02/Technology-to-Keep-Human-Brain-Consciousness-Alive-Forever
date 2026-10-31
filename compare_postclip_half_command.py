"""Saved closure comparison; restore last50ms for true per-foot world reactions."""
from pathlib import Path
import json,numpy as np,mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_postclip_half_command import O,BASE,c,sha,dump

def main():
 m=c.cp.load_model();m.opt.timestep=.0000125;mapping=json.loads((O/'MAPPING.json').read_text());names=[r['actuator'] for r in mapping['rows']];summary={};forces={};forwards=0;fig,axs=plt.subplots(4,2,figsize=(15,16));allivs={};peakrows={}
 for label,p in [('scale1',BASE/'normal'),('scale0p5',O)]:
  z=np.load(p/'BODY.npz');iv=np.load(p/'INTERVALS.npz');allivs[label]=iv;t=z['time'];last=np.flatnonzero(t>=.2);ids=np.flatnonzero(z['nonfoot_floor_count']);cmd=iv['commands'];rec=json.loads((p/'RECORDS.json').read_text());first=np.flatnonzero(np.any(z['command']>0,axis=1));actdiff=np.flatnonzero(np.any(abs(z['activation']-z['activation'][0])>1e-10,axis=1));world=[];load=[];errors=[]
  for idx in last:
   d=mujoco.MjData(m);d.qpos[:]=z['qpos'][idx];d.qvel[:]=z['qvel'][idx];d.act[:]=z['activation'][idx];d.ctrl[:]=z['command'][idx];d.time=t[idx];d.xfrc_applied[:]=z['external_force'][idx];mujoco.mj_forward(m,d);forwards+=1
   fs=np.zeros((7,3));counts=np.zeros(7,int)
   for k in range(d.ncon):
    co=d.contact[k];a,b=int(co.geom1),int(co.geom2);pa=m.geom_type[a]==mujoco.mjtGeom.mjGEOM_PLANE;pb=m.geom_type[b]==mujoco.mjtGeom.mjGEOM_PLANE
    if not(pa or pb):continue
    g=b if pa else a;gn=(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or '').lower();bn=(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) or '').lower();leg=next((i for i,l in enumerate(['lf','lm','lh','rf','rm','rh']) if (gn.startswith(l) or bn.startswith(l)) and 'tars' in gn+bn),6);f=np.zeros(6);mujoco.mj_contactForce(m,d,k,f);fs[leg]+=(co.frame.reshape(3,3).T@f[:3])*(1 if pa else -1);counts[leg]+=1
   errors.append(float(abs(fs.sum(0)-z['ground_force_world'][idx]).max()));world.append(fs);load.append(counts)
  world=np.array(world);load=np.array(load);forces[label]=dict(time=t[last],world=world,geometric_ground_contact_counts=load);np.savez_compressed(O/(label+'_LAST50_GROUND.npz'),**forces[label]);peakrows[label]=[dict(actuator=name,command_peak=float(z['command'][:,i].max()),activation_peak=float(z['activation'][:,i].max()),applied_at_upper_bound_samples=int((z['command'][:,i]>=1-1e-12).sum()),decoder_clipped_windows=sum(bool(r['decoder']['saturated'][i]) for r in rec['records'])) for i,name in enumerate(names)]
  summary[label]=dict(first_command_s=None if not len(first) else float(t[first[0]]),first_activation_change_s=None if not len(actdiff) else float(t[actdiff[0]]),first_nonfoot_s=None if not len(ids) else float(t[ids[0]]),final_thorax_height=float(z['thorax_position'][-1,2]),final_com_height=float(z['com_position'][-1,2]),final_com_vertical_velocity=float(z['com_velocity'][-1,2]),last50_nonfoot_samples=int((z['nonfoot_floor_count'][last]>0).sum()),last50_samples=len(last),last50_contact_legs_minmax=[int((load[:,:6]>0).sum(1).min()),int((load[:,:6]>0).sum(1).max())],last50_ground_force_mean_by_leg=world.mean(0),restored_total_ground_max_error=max(errors),warnings=int(z['warnings'].max()),nonfinite={k:int((~np.isfinite(z[k])).sum()) for k in ['qpos','qvel','activation','command','com_position','com_velocity']},total_neural_spikes=int(iv['spikes'].sum()),stable_support='NOT_JUDGED',max_command=float(z['command'].max()))
  vals=[(t,z['thorax_position'][:,2]),(t,z['com_position'][:,2]),(t,z['com_velocity'][:,2]),(t,2*np.arccos(np.clip(abs(z['qpos'][:,3]),0,1))),(np.arange(len(cmd))*.005,cmd.max(1)),(np.arange(1,len(cmd)+1)*.005,iv['spikes'].sum(1)),(t,z['nonfoot_floor_count']),(t[last],world[:,6,2])]
  for ax,(x,y) in zip(axs.flat,vals):ax.plot(x,y,label=label)
 old=allivs['scale1'];new=allivs['scale0p5'];dump(O/'NEURAL_CHANGE.json',dict(neuron_interval_count_differences=int(np.count_nonzero(old['spikes']!=new['spikes'])),per_neuron_total_old=old['spikes'].sum(0),per_neuron_total_new=new['spikes'].sum(0),note='saved emitted-spike interval counts; actual feedback closure, not replay'))
 dump(O/'PEAKS_90.json',peakrows);dump(O/'RESPONSE_SUMMARY.json',summary)
 for ax,title in zip(axs.flat,['Thorax height / model units','COM height / model units','COM vertical velocity / model units','Root rotation from identity / rad','Maximum applied command (post clip scale)','All network emitted spike counts per5ms','Nonfoot geometric ground contacts','Last50ms NONFOOT ground world vertical force']):ax.set_title(title);ax.legend();ax.grid(alpha=.2);ax.set_xlabel('physical seconds')
 fig.suptitle('Fresh half postclip command closure vs saved baseline / NOT stable-support certification');fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(O/'HALF_COMMAND_COMPARISON.png',dpi=130);plt.close(fig);Path('/home/sensen/Desktop/S2_HALF_COMMAND_COMPARISON.png').write_bytes((O/'HALF_COMMAND_COMPARISON.png').read_bytes());dump(O/'ANALYSIS_EXECUTION.json',dict(exit_code=0,model_load=1,mj_forward=forwards,mj_step=0,neural_step=0,reason='restore last50ms both saved arms to recover world contact frames, not baseline rerun'));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(summary,default=lambda a:a.tolist()))
if __name__=='__main__':main()
