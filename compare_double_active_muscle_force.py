"""Three configurations: restore ONLY new arm; reuse hash-verified old world forces."""
from pathlib import Path
import json,numpy as np,mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_double_active_muscle_force import O,TOTAL,BASE,c,sha,dump,scale_active,restore,components

def world_forces(m,d):
 fs=np.zeros((7,3));counts=np.zeros(7,int)
 for k in range(d.ncon):
  co=d.contact[k];a,b=int(co.geom1),int(co.geom2);pa=m.geom_type[a]==mujoco.mjtGeom.mjGEOM_PLANE;pb=m.geom_type[b]==mujoco.mjtGeom.mjGEOM_PLANE
  if not(pa or pb):continue
  g=b if pa else a;gn=(m.geom(g).name or '').lower();bn=(m.body(int(m.geom_bodyid[g])).name or '').lower();leg=next((i for i,l in enumerate(['lf','lm','lh','rf','rm','rh']) if (gn.startswith(l) or bn.startswith(l)) and 'tars' in gn+bn),6);f=np.zeros(6);mujoco.mj_contactForce(m,d,k,f);fs[leg]+=(co.frame.reshape(3,3).T@f[:3])*(1 if pa else -1);counts[leg]+=1
 return fs,counts

def main():
 manifest=json.loads((TOTAL/'MANIFEST_SHA256.json').read_text());references={}
 for name in ['force_x1_LAST50_GROUND.npz','force_x2_LAST50_GROUND.npz','COMPONENTS_90_AT_51_STATES.json','RESPONSE_SUMMARY.json']:
  assert sha(TOTAL/name)==manifest[name];references[str(TOTAL/name)]=manifest[name]
 for p in [BASE/'normal',TOTAL]:
  root=BASE if p==BASE/'normal' else p
  man=json.loads((root/'MANIFEST_SHA256.json').read_text())
  for name in ['BODY.npz','INTERVALS.npz','RECORDS.json']:
   key='normal/'+name if p==BASE/'normal' else name
   assert sha(p/name)==man[key];references[str(p/name)]=man[key]
 dump(O/'ARCHIVE_REFERENCES.json',references)
 m=c.cp.load_model();m.opt.timestep=.0000125;scale_active(m,2);z=np.load(O/'BODY.npz');last=np.flatnonzero(z['time']>=.2);world=[];counts=[];errors=[];rows=[];calls=0
 for idx in last:
  d=restore(m,z,idx);calls+=1;fs,co=world_forces(m,d);world.append(fs);counts.append(co);errors.append(float(abs(fs.sum(0)-z['ground_force_world'][idx]).max()))
 np.savez_compressed(O/'ACTIVE2_BIAS1_LAST50_GROUND.npz',time=z['time'][last],world=np.array(world),geometric_ground_contact_counts=np.array(counts))
 for idx in range(0,20001,400):
  d=restore(m,z,idx);calls+=1;a,b=components(m,d);np.testing.assert_allclose(a+b,d.actuator_force,atol=1e-10,rtol=1e-12);rows.append(dict(time=float(d.time),active=a,bias=b,total=d.actuator_force.copy(),command=d.ctrl.copy(),activation=d.act.copy()))
 dump(O/'COMPONENTS_90_AT_51_STATES.json',rows)
 oldcomponents=json.loads((TOTAL/'COMPONENTS_90_AT_51_STATES.json').read_text())['states'];mapping=json.loads((O/'MAPPING.json').read_text());motor=sorted({i for r in mapping['rows'] for i in r['local_indices']});summary={};peaks={};early={};fig,axs=plt.subplots(5,2,figsize=(16,20));baseline=np.load(BASE/'normal/BODY.npz');baseiv=np.load(BASE/'normal/INTERVALS.npz')
 configurations=[('historical',BASE/'normal',TOTAL/'force_x1_LAST50_GROUND.npz',oldcomponents['force_x1']),('active2_bias2',TOTAL,TOTAL/'force_x2_LAST50_GROUND.npz',oldcomponents['force_x2']),('active2_bias1',O,O/'ACTIVE2_BIAS1_LAST50_GROUND.npz',rows)]
 for label,p,ground,parts in configurations:
  zz=np.load(p/'BODY.npz');iv=np.load(p/'INTERVALS.npz');rec=json.loads((p/'RECORDS.json').read_text());gf=np.load(ground);t=zz['time'];ids=np.flatnonzero(zz['nonfoot_floor_count']);first=np.flatnonzero(np.any(zz['command']>0,axis=1));saturated=sum(sum(r['decoder']['saturated']) for r in rec['records'])
  summary[label]=dict(first_nonzero_command_saved_s=float(t[first[0]]) if len(first) else None,first_nonfoot_s=float(t[ids[0]]) if len(ids) else None,final_thorax=float(zz['thorax_position'][-1,2]),final_com=float(zz['com_position'][-1,2]),final_com_vz=float(zz['com_velocity'][-1,2]),last50_nonfoot_samples=int((zz['nonfoot_floor_count'][t>=.2]>0).sum()),last50_samples=len(gf['time']),last50_perleg_geometric_contact_samples=(gf['geometric_ground_contact_counts']>0).sum(0),last50_mean_world_xyz=gf['world'].mean(0),all_spikes=int(iv['spikes'].sum()),unique_motor_spikes=int(iv['spikes'][:,motor].sum()),neuron_interval_differences_vs_historical=int((iv['spikes']!=baseiv['spikes']).sum()),decoder_saturated_windows=int(saturated),activation_over1=int((zz['activation']>1).sum()),activation_below0=int((zz['activation']<0).sum()),warnings=int(zz['warnings'].max()),nonfinite={k:int((~np.isfinite(zz[k])).sum()) for k in ['qpos','qvel','activation','command','com_velocity']},stable_support='NOT_JUDGED')
  peaks[label]=[dict(actuator=r['actuator'],command_peak=float(zz['command'][:,i].max()),activation_peak=float(zz['activation'][:,i].max()),decoder_saturated_windows=sum(bool(v['decoder']['saturated'][i]) for v in rec['records'])) for i,r in enumerate(mapping['rows'])]
  early[label]={}
  for leg in ['lf','lm','lh','rf','rm','rh']:
   rr=[r for r in mapping['rows'] if r['actuator'].lower().startswith(leg)];cols=[r['actuator_index'] for r in rr];local={i for r in rr for i in r['local_indices']};events=[dict(time=k*.005+(st+1)*.0005,local=rec['probe_local_indices'][co]) for k,r in enumerate(rec['records'][:3]) for st,co in r['probe_emitted_spike_events_step_column'] if rec['probe_local_indices'][co] in local];early[label][leg]=dict(events=events,time=t[:1201],command=zz['command'][:1201,cols],activation=zz['activation'][:1201,cols])
  vals=[(t,zz['thorax_position'][:,2]),(t,zz['com_position'][:,2]),(t,zz['com_velocity'][:,2]),(t,2*np.arccos(np.clip(abs(zz['qpos'][:,3]),0,1))),([r['time'] for r in parts],[sum(r['active']) for r in parts]),([r['time'] for r in parts],[sum(r['bias']) for r in parts]),(np.arange(1,51)*.005,iv['spikes'][:,motor].sum(1)),(t,zz['command'].max(1)),(t,zz['nonfoot_floor_count']),(gf['time'],gf['world'][:,6,2])]
  for ax,(xx,yy) in zip(axs.flat,vals):ax.plot(xx,yy,label=label)
 for key in ['qpos','qvel','activation','command','external_force']:np.testing.assert_array_equal(z[key][0],baseline[key][0])
 np.testing.assert_array_equal(z['time'],baseline['time']);np.testing.assert_array_equal(np.load(O/'INTERVALS.npz')['sensory_current_nA'][0],baseiv['sensory_current_nA'][0]);assert max(errors)<1e-10
 for ax,title in zip(axs.flat,['Thorax height','COM height','COM vertical velocity','Root rotation /rad','Signed active-force sum /NOT support','Signed bias-force sum /NOT support','Unique motor node events /5ms','Maximum applied command','Nonfoot plane contacts','Last50ms NONFOOT world vertical reaction']):ax.set_title(title);ax.legend(fontsize=8);ax.grid(alpha=.2);ax.set_xlabel('physical seconds')
 fig.suptitle('Artificial active capacity x2, bias historical / three real saved closures / NOT physiological validation');fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(O/'THREE_CONFIG_COMPARISON.png',dpi=125);plt.close(fig);Path('/home/sensen/Desktop/S2_ACTIVE_FORCE_DOUBLE.png').write_bytes((O/'THREE_CONFIG_COMPARISON.png').read_bytes());dump(O/'RESPONSE_SUMMARY.json',summary);dump(O/'PEAKS_90.json',peaks);dump(O/'EARLY_LEG_RECRUITMENT.json',early);dump(O/'ANALYSIS_EXECUTION.json',dict(exit_code=0,model_load=1,mj_forward=calls,mj_step=0,neural_step=0,restored_new_ground_states=len(last),new_component_states=51,restored_ground_force_max_error=max(errors),old_arms_restored=0,initial_body_and_encoding_exact=True));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(summary,default=lambda a:a.tolist()))
if __name__=='__main__':main()
