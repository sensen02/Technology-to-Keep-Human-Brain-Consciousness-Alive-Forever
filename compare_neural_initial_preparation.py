from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_neural_initial_preparation import O,BASE,sha,dump

def main():
 mapping=json.loads((O/'MAPPING.json').read_text());prep=json.loads((O/'PREPARATION.json').read_text());transfer=json.loads((O/'RELEASE_TRANSFER.json').read_text());summary={};early={};fig,axs=plt.subplots(3,2,figsize=(15,12));initial={}
 for label,p in [('baseline',BASE/'normal'),('prepared20ms',O)]:
  z=np.load(p/'BODY.npz');iv=np.load(p/'INTERVALS.npz');t=z['time'];rec=json.loads((p/'RECORDS.json').read_text());ids=np.flatnonzero(z['nonfoot_floor_count']);last=t>=.2;cmd=np.flatnonzero(np.any(z['command']>0,axis=1));act=np.flatnonzero(np.any(z['activation']>1e-4+1e-10,axis=1));initial[label]={k:z[k][0].copy() for k in ['qpos','qvel','activation','command','external_force']}
  summary[label]=dict(first_command_saved_s=float(t[cmd[0]]) if len(cmd) else None,first_activation_above_floor_s=float(t[act[0]]) if len(act) else None,first_nonfoot_s=float(t[ids[0]]) if len(ids) else None,final_thorax=float(z['thorax_position'][-1,2]),final_com=float(z['com_position'][-1,2]),final_com_vz=float(z['com_velocity'][-1,2]),last50_nonfoot_samples=int((z['nonfoot_floor_count'][last]>0).sum()),last50_samples=int(last.sum()),last50_true_ground_foot_minmax=[int(z['foot_contact'][last].sum(1).min()),int(z['foot_contact'][last].sum(1).max())],max_activation=float(z['activation'].max()),activation_above1=int((z['activation']>1).sum()),warnings=int(z['warnings'].max()),nonfinite={k:int((~np.isfinite(z[k])).sum()) for k in ['qpos','qvel','activation','command']},stable_support='NOT_JUDGED')
  early[label]={}
  for leg in ['lf','lm','lh','rf','rm','rh']:
   rows=[r for r in mapping['rows'] if r['actuator'].lower().startswith(leg)];indices=[r['actuator_index'] for r in rows];locals=sorted({i for r in rows for i in r['local_indices']});events=[]
   for k,r in enumerate(rec['records'][:3]):
    events.extend([dict(time_s=k*.005+(st+1)*.0005,local=rec['probe_local_indices'][col]) for st,col in r['probe_emitted_spike_events_step_column'] if rec['probe_local_indices'][col] in locals])
   early[label][leg]=dict(events_first15ms=events,command_time=z['time'][:1201],commands=z['command'][:1201,indices],activation=z['activation'][:1201,indices],actuators=[r['actuator'] for r in rows])
  for ax,y in zip(axs.flat,[z['thorax_position'][:,2],z['com_position'][:,2],z['com_velocity'][:,2],2*np.arccos(np.clip(abs(z['qpos'][:,3]),0,1)),z['command'].max(1),z['nonfoot_floor_count']]):ax.plot(t,y,label=label)
 assert all(np.array_equal(initial['baseline'][k],initial['prepared20ms'][k]) for k in initial['baseline']);np.testing.assert_array_equal(transfer['body_qpos'],initial['baseline']['qpos']);np.testing.assert_array_equal(transfer['body_activation'],initial['baseline']['activation']);np.testing.assert_array_equal(transfer['body_velocity'],initial['baseline']['qvel']);iv=np.load(O/'INTERVALS.npz');np.testing.assert_array_equal(iv['commands'][0],transfer['pending'])
 for ax,title in zip(axs.flat,['Thorax height','COM height','COM vertical velocity','Root rotation /rad','Maximum applied command / release at0','Nonfoot true ground geometric contacts']):ax.set_title(title);ax.legend();ax.grid(alpha=.2);ax.set_xlabel('physical seconds')
 fig.suptitle('Fixed-sensory neural preparation20ms (not physical experience) then250ms closure');fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(O/'PREPARATION_COMPARISON.png',dpi=130);plt.close(fig);Path('/home/sensen/Desktop/S2_NEURAL_PREPARATION.png').write_bytes((O/'PREPARATION_COMPARISON.png').read_bytes());dump(O/'RESPONSE_SUMMARY.json',summary);dump(O/'EARLY_LEG_RECRUITMENT.json',early);dump(O/'TRANSFER_TESTS.json',dict(initial_body_exact=True,release_pending_applied_exact=True,prep_intervals=len(prep['events']),prep_spike_events=sum(len(r['spikes']) for r in prep['events']),release_command_positive_count=int(np.count_nonzero(transfer['pending'])),exit_code=0));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(summary))
if __name__=='__main__':main()
