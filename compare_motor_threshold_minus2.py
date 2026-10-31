"""Saved-only fixed threshold intervention comparison; no forward or physics."""
from pathlib import Path
import json,sys,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_motor_threshold_minus2 import O,BASE,TARGET,sha,dump

def main():
 target=json.loads(TARGET.read_text());nodes=target['unique_local_nodes'];mapping=json.loads((O/'MAPPING.json').read_text());rows=mapping['rows'];motor=sorted({i for r in rows for i in r['local_indices']});other=sorted(set(motor)-set(nodes));tick=np.load(O/'ALL_NODE_TICK_EVENTS.npz')['spikes'];sides={};pools=[]
 for label,path in [('historical',BASE/'normal'),('minus2mV',O)]:
  z=np.load(path/'BODY.npz');iv=np.load(path/'INTERVALS.npz');counts=iv['spikes'];rec=json.loads((path/'RECORDS.json').read_text());t=z['time'];non=np.flatnonzero(z['nonfoot_floor_count']);last=t>=.2-1e-12;sides[label]=dict(first_nonfoot_s=float(t[non[0]]) if len(non) else None,final_thorax=z['thorax_position'][-1],final_root_qpos=z['qpos'][-1,:7],final_COM=z['com_position'][-1],final_COM_velocity=z['com_velocity'][-1],total_spikes=int(counts.sum()),target_spikes=int(counts[:,nodes].sum()),other_motor_spikes=int(counts[:,other].sum()),all_motor_spikes=int(counts[:,motor].sum()),decoder_saturations=int(np.count_nonzero(iv['commands']>=1-1e-12)),early15ms_perleg={leg:dict(events=int(counts[:3,sorted({i for r in rows if r.get('leg','').lower()==leg for i in r['local_indices']})].sum()),max_command=float(z['command'][:1201,[r['actuator_index'] for r in rows if r.get('leg','').lower()==leg]].max()),max_activation=float(z['activation'][:1201,[r['actuator_index'] for r in rows if r.get('leg','').lower()==leg]].max())) for leg in ['lf','lm','lh','rf','rm','rh']},last50ms=dict(samples=int(last.sum()),nonfoot_samples=int(np.count_nonzero(z['nonfoot_floor_count'][last])),saved_foot_contact_count=np.count_nonzero(z['foot_contact'][last],axis=0),saved_foot_mean_normal_force=z['foot_normal_force'][last].mean(0),normal_load_caveat='historical saved detector may include nonplane contacts; explicit plane rows classified separately below'),finite_all=all(np.isfinite(z[k]).all() for k in z.files),activation_min=float(z['activation'].min()),activation_max=float(z['activation'].max()),warning_max=z['warnings'].max(axis=0))
  for tr in target['source_11_muscles']:
   r=next(r for r in rows if r['actuator']==tr['actuator']);ai=r['actuator_index'];pools.append(dict(arm=label,actuator=r['actuator'],local_indices=r['local_indices'],pool_events_per5ms=counts[:,r['local_indices']].sum(1),command=z['command'][:,ai],activation=z['activation'][:,ai]))
  # Explicit plane-vs-tarsus saved contact geometry loads; no world-frame restore needed for local normal.
  contacts=json.loads((path/'CONTACTS.json').read_text());footmeta=json.loads(Path('/home/sensen/Desktop/engine修复验收/s2-early-support-region/20261027T200000Z/PROJECTION_AND_CLASSIFICATION.json').read_text());geomleg={int(k):v['leg'] for k,v in footmeta['foot_geoms'].items()};plane=footmeta.get('plane_geom_ids',[0]);loads=[];contactcounts=[]
  for i in np.flatnonzero(last):
   load={l:0. for l in ['lf','lm','lh','rf','rm','rh','NONFOOT']};ct={l:0 for l in load}
   for co in contacts[i]:
    a,b=int(co[1]),int(co[2]);g=b if a in plane else a if b in plane else None
    if g is None:continue
    l=geomleg.get(g,'NONFOOT');ct[l]+=1;load[l]+=max(float(co[7]),0.)
   loads.append(load);contactcounts.append(ct)
  dump(O/(label+'_LAST50_CONTACT_NORMAL.json'),dict(times=t[last],plane_geom_ids=plane,normal_load=loads,contact_count=contactcounts,not_world_vertical=True))
 targetrows=[]
 historical_rec=json.loads((BASE/'normal/RECORDS.json').read_text());historical_counts=np.load(BASE/'normal/INTERVALS.npz')['spikes'];historical_probe=historical_rec['probe_local_indices']
 for j in nodes:
  hit=np.flatnonzero(tick[:,j]);targetrows.append(dict(local_node=j,new_events=int(tick[:,j].sum()),first_new_spike_ms=float((hit[0]+1)*.5) if len(hit) else None,historical_events=int(historical_counts[:,j].sum()),historical_first_spike_ms=next((float(k*5+(p[0]+1)*.5) for k,w in enumerate(historical_rec['records']) for p in w['probe_raw_spike_events_step_column'] if historical_probe[p[1]]==j),None),historical_first_time_caveat='historical saved probe exact tick, not assumed zero over250ms'))
 dump(O/'TARGET_EVENTS.json',targetrows);dump(O/'MUSCLE_OUTPUTS_11.json',pools);dump(O/'COMPARISON.json',sides)
 fig,axs=plt.subplots(3,2,figsize=(17,13));axs[0,0].plot(np.arange(1,501)*.5,np.cumsum(tick[:,nodes].sum(1)),label='target29 minus2mV');axs[0,0].step(np.arange(1,51)*5,np.cumsum(np.load(BASE/'normal/INTERVALS.npz')['spikes'][:,nodes].sum(1)),where='post',color='k',ls='--',label='historical targets (5ms counts)');axs[0,0].legend();axs[0,0].set_title('Deduplicated target spikes cumulative')
 for label,path in [('historical',BASE/'normal'),('minus2mV',O)]:
  z=np.load(path/'BODY.npz');t=z['time']*1000;ais=[next(r['actuator_index'] for r in rows if r['actuator']==tr['actuator']) for tr in target['source_11_muscles']];axs[0,1].plot(t,z['command'][:,ais].max(1),label=label);axs[1,0].plot(t,z['activation'][:,ais].max(1),label=label);axs[1,1].plot(t,z['thorax_position'][:,2],label=label+' thorax');axs[1,1].plot(t,z['com_position'][:,2],ls='--',label=label+' COM');axs[2,0].plot(t,z['com_velocity'][:,2],label=label);axs[2,1].plot(t,z['nonfoot_floor_count'],label=label)
 for ax,title in zip(axs.flat,['Target29 recruitment','11 muscles max command','11 muscles max activation','Body heights','Full COM vz','NONFOOT plane contacts']):ax.set_title(title);ax.set_xlabel('physical/neural ms');ax.legend();ax.grid(alpha=.2)
 fig.suptitle('Artificial29-node base threshold -2mV / one new dynamic closed loop / not physiological correction');fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(O/'THRESHOLD_COMPARISON.png',dpi=130);plt.close(fig);Path('/home/sensen/Desktop/S3_THRESHOLD_MINUS2.png').write_bytes((O/'THRESHOLD_COMPARISON.png').read_bytes());dump(O/'ANALYSIS_EXECUTION.json',dict(command=sys.executable+' '+str(Path(__file__)),exit_code=0,model_load=0,forward=0,body_step=0,neural_step=0));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(sides,default=lambda x:x.tolist()));print('targets newly firing',sum(r['new_events']>0 for r in targetrows))
if __name__=='__main__':main()
