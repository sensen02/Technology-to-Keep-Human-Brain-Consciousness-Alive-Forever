from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_split_write_diagnostic import O,HISTORY
from run_one_second_motor_pair import dump,sha

def main():
 assert all(json.loads((O/'DEFAULT_REGRESSION.json').read_text()).values());summary={};fig,axs=plt.subplots(3,2,figsize=(14,12));legs=['lf','lm','lh','rf','rm','rh'];manifest=json.loads((O/'MAPPING.json').read_text());rows=manifest['rows']
 for tag,path,dt in [('old5ms',HISTORY/'normal',.005),('new1ms',O/'write1ms',.001)]:
  s=np.load(path/'BODY.npz');iv=np.load(path/'INTERVALS.npz');t=s['time'];cmd=iv['commands'];times=np.arange(len(cmd))*dt;checks={};events={}
  if tag=='old5ms':
   rec=json.loads((path/'RECORDS.json').read_text());probe=rec['probe_local_indices']
   for k,r in enumerate(rec['records']):
    for step,col in r['probe_emitted_spike_events_step_column']:events.setdefault(probe[col],[]).append(k*.005+(step+1)*.0005)
   sensory=np.arange(50)*.005
  else:
   for tick,idx in zip(iv['event_ticks'],iv['event_local_indices']):events.setdefault(int(idx),[]).append(float(tick*.0005))
   sensory=iv['sensory_sample_start_s'];np.testing.assert_allclose(sensory,np.arange(50)*.005,atol=1e-14)
   np.testing.assert_allclose(iv['motor_start_s'],times,atol=1e-14);np.testing.assert_array_equal(s['command'][1:],np.repeat(cmd,80,axis=0));assert np.all(iv['motor_source_end_tick'][1:]<=np.rint(times[1:]/.0005))
  for leg in legs:
   ai=[j for j,r in enumerate(rows) if r['actuator'][:2].lower()==leg];pool=set(idx for j in ai for idx in rows[j]['local_indices']);sp=sorted(v for idx in pool for v in events.get(idx,[]));ids=np.flatnonzero((cmd[:,ai]>0).any(1));actids=np.flatnonzero((s['activation'][:,ai]>.0001+1e-12).any(1));e=None if not sp else sp[0];c=None if not len(ids) else float(times[ids[0]]);a=None if not len(actids) else float(t[actids[0]])
   checks[leg]=dict(first_motor_event_s=e,first_command_s=c,first_activation_s=a,event_to_command_s=None if e is None or c is None else c-e,command_to_activation_s=None if a is None or c is None else a-c)
   axs[0,0].scatter([legs.index(leg)],[None if e is None or c is None else (c-e)*1000],label=tag if leg=='lf' else None)
  non=s['nonfoot_floor_count'];nz=np.flatnonzero(non);last=t>=.2
  summary[tag]=dict(legs=checks,first_nonfoot_s=None if not len(nz) else float(t[nz[0]]),max_activation=float(s['activation'].max()),activation_gt1=int((s['activation']>1).sum()),final_height=float(s['thorax_position'][-1,2]),last50ms_nonfoot_samples=int((non[last]>0).sum()),last50ms_samples=int(last.sum()),last50ms_foot_count_minmax=[int(s['foot_contact'][last].sum(1).min()),int(s['foot_contact'][last].sum(1).max())],stability='NOT_JUDGED')
  axs[0,1].plot(t,s['thorax_position'][:,2],label=tag);axs[1,0].plot(t,2*np.arccos(np.clip(abs(s['qpos'][:,3]),0,1)),label=tag);axs[1,1].plot(t,s['activation'].max(1),label=tag);axs[2,0].plot(t,s['foot_contact'].sum(1),label=tag);axs[2,1].plot(t,non,label=tag)
 axs[0,0].set_xticks(range(6),legs)
 for ax,title in zip(axs.flat,['First motor event to command wait / ms','Thorax height / model units','Root rotation from identity / rad','Max activation','True floor-contact legs','Nonfoot floor contacts']):ax.set_title(title);ax.grid(alpha=.2);ax.legend()
 fig.suptitle('Single cadence change: sensory5ms/neural0.5ms fixed; not standing certification');fig.tight_layout();fig.savefig(O/'WRITE_COMPARISON.png',dpi=140);plt.close(fig);Path('/home/sensen/Desktop/S2_WRITE_CADENCE.png').write_bytes((O/'WRITE_COMPARISON.png').read_bytes());dump(O/'COMPARISON.json',summary);(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(summary))
if __name__=='__main__':main()
