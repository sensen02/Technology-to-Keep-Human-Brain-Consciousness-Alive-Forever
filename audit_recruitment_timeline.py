"""Saved-only pre-abdomen-contact recruitment timeline; no simulator calls."""
from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_one_second_motor_pair import sha,dump
S=Path('/home/sensen/Desktop/engine修复验收/s2-release-250ms/20261027T161000Z')
M=Path('/home/sensen/Desktop/engine修复验收/s2-one-second-motor-pair/20261027T153000Z/MAPPING.json')
O=Path('/home/sensen/Desktop/engine修复验收/s2-recruitment-timeline/20261027T162000Z')
def first(mask,t):
 ids=np.flatnonzero(mask);return None if not len(ids) else float(t[ids[0]])
def main():
 O.mkdir(parents=True,exist_ok=True);manifest=json.loads((S/'MANIFEST_SHA256.json').read_text());assert all(sha(S/k)==v for k,v in manifest.items());mapping=json.loads(M.read_text());rows=mapping['rows'];legs=['lf','lm','lh','rf','rm','rh'];limit=.0387375
 dump(O/'INPUTS.json',dict(input_manifest=str(S/'MANIFEST_SHA256.json'),sha256=sha(S/'MANIFEST_SHA256.json'),mapping=str(M),mapping_sha256=sha(M),cutoff_s=limit,rule='all source windows starting before cutoff retained; last window partly beyond cutoff clearly marked; act>floor+1e-12 is recruitment above floor, not physiological threshold',mapping_level=mapping['provenance'],missing=['state-dependent actuator moment-arm Jacobian/direction not saved; antagonism not identifiable from names','active gain force and passive bias separately not saved']))
 results={};by_leg={};fig,axs=plt.subplots(7,1,figsize=(15,15),sharex=True)
 for arm in ('normal','motor_cut'):
  z=np.load(S/arm/'BODY.npz');i=np.load(S/arm/'INTERVALS.npz');doc=json.loads((S/arm/'RECORDS.json').read_text());records=doc['records'][:8];t=z['time'];valid=t<=limit+1e-12;ts=t[valid];probe=doc['probe_local_indices'];lookup={idx:j for j,idx in enumerate(probe)};allrows=[]
  # Reconstruct only motor-pool substep spikes actually present in probe recording.
  rates=np.zeros(90);decay=np.exp(-.0005/.005);maxerr=0
  for k,r in enumerate(records):
   events=r['probe_emitted_spike_events_step_column'];sp=np.zeros((10,len(probe)),int)
   for step,col in events:sp[step,col]=1
   counts=[]
   for a,row in enumerate(rows):counts.append(int(sp[:,[lookup[idx] for idx in row['local_indices']]].sum()) if row['local_indices'] else 0)
   np.testing.assert_array_equal(counts,r['decoder']['pool_spike_counts'])
   if r['decoder']['connected']:
    for sample in sp:
     instant=np.array([sample[[lookup[x] for x in row['local_indices']]].mean()/.0005 if row['local_indices'] else 0 for row in rows]);rates=decay*rates+(1-decay)*instant
   else:rates[:]=0
   maxerr=max(maxerr,float(np.max(abs(rates-r['decoder']['rates_hz']))));np.testing.assert_allclose(np.clip(rates/100,0,1),i['decoded_next_controls'][k],atol=1e-14)
  for a,row in enumerate(rows):
   idx=row['local_indices'];events=[]
   for k,r in enumerate(records):
    for step,col in r['probe_emitted_spike_events_step_column']:
     when=k*.005+(step+1)*.0005
     if probe[col] in idx and when<=limit:events.append(when)
   applied=i['commands'][:8,a];decoded=i['decoded_next_controls'][:8,a];act=z['activation'][valid,a];force=z['muscle_force'][valid,a];leg=row.get('side','')+row.get('leg','');name=row['actuator'];leg=name[:2].lower() if name[:2].lower() in legs else leg
   cmdtime=first(applied>0,np.arange(8)*.005);evtime=min(events) if events else None;acttime=first(act>.0001+1e-12,ts)
   rec=dict(actuator=name,leg=leg,mapping_status=row['status'],pool_present_cells=len(idx),first_motor_event_s=evtime,first_decoded_valid_s=first(decoded>0,np.arange(1,9)*.005),first_written_command_s=cmdtime,first_activation_above_floor_s=acttime,first_nonzero_total_force_s=first(abs(force)>1e-12,ts),motor_event_to_written_delay_s=None if cmdtime is None or evtime is None else cmdtime-evtime,written_to_activation_delay_s=None if cmdtime is None or acttime is None else acttime-cmdtime,windows=[dict(start_s=k*.005,end_s=(k+1)*.005,straddles_cutoff=(k+1)*.005>limit,pool_spikes=r['decoder']['pool_spike_counts'][a],rate_hz=r['decoder']['rates_hz'][a],decoded_next=float(decoded[k]),applied=float(applied[k]),effective_ctrl=float(np.clip(applied[k],.0001,1)),activation_at_window_start=float(z['activation'][k*400,a]),total_force_at_window_start=float(z['muscle_force'][k*400,a]),saturated=r['decoder']['saturated'][a]) for k,r in enumerate(records)],total_force_is_active_plus_bias=True)
   allrows.append(rec)
  results[arm]=dict(actuators=allrows,decoder_rate_max_error=maxerr)
  by_leg[arm]={}
  for j,leg in enumerate(legs):
   group=[r for r in allrows if r['leg']==leg];ai=[r['actuator_index'] for r in rows if r['actuator'][:2].lower()==leg]
   def earliest(field):return min([r[field] for r in group if r[field] is not None],default=None)
   by_leg[arm][leg]={f:earliest(f) for f in ['first_motor_event_s','first_decoded_valid_s','first_written_command_s','first_activation_above_floor_s','first_nonzero_total_force_s']}
   by_leg[arm][leg]['commanded_muscle_count_before_cutoff']=sum(r['first_written_command_s'] is not None and r['first_written_command_s']<limit for r in group)
   axs[j].step(np.arange(8)*5,(i['commands'][:8,ai]>0).sum(1),where='post',label=arm+' commanded bundles')
   axs[j].plot(ts*1000,z['activation'][valid][:,ai].max(1),label=arm+' max activation',alpha=.6);axs[j].set_ylabel(leg);axs[j].legend(fontsize=7,ncol=2);axs[j].grid(alpha=.2)
  axs[6].plot(ts*1000,z['thorax_position'][valid,2],label=arm+' thorax')
 axs[6].legend();axs[6].set_xlabel('Physical ms; commands interval start, activation post-step; partial last interval');
 for ax in axs:ax.axvline(limit*1000,c='red',ls='--',label='normal first abdomen contact');ax.axvline(29.25,c='purple',ls='--',label='cut first abdomen contact');ax.axvline(15,c='grey',ls=':');ax.set_xlim(0,limit*1000)
 fig.suptitle('Pre-normal-abdomen-contact recruitment / counts and activation have different meanings / no support labels');fig.tight_layout();fig.savefig(O/'RECRUITMENT.png',dpi=140);plt.close(fig);Path('/home/sensen/Desktop/S2_RECRUITMENT_TIMELINE.png').write_bytes((O/'RECRUITMENT.png').read_bytes())
 sensory={}
 for arm in ('normal','motor_cut'):
  recs=json.loads((S/arm/'RECORDS.json').read_text())['records'][:8]
  sensory[arm]=[dict(start_s=r['time_start_s'],end_s=r['time_end_s'],channel_quantities=r['channel_record'],actual_network_input_by_channel=r['actual_network_input_by_channel'],motor_spikes=r['motor_spikes'],total_network_spikes=r['total_network_spikes']) for r in recs]
  for leg in legs:
   by_leg[arm][leg]['first_nonzero_sensory_input_s']=next((r['time_start_s'] for r in recs if any(ch['max_nA']>0 or ch['min_nA']<0 for ch in r['actual_network_input_by_channel'][leg].values())),None)
 dump(O/'SENSORY_WINDOWS.json',sensory)
 dump(O/'ACTUATOR_LEDGER.json',results);dump(O/'LEG_TIMES.json',by_leg);dump(O/'EXECUTION.json',dict(command=sys.executable+' '+str(Path(__file__)),exit_code=0,physics_step=0,forward=0,model_load=0,neural_step=0,probe_motor_event_times='actual saved500us bins, not inferred from interval counts',decoder_reconstruction_pass=True));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(by_leg));print('exit0 saved-only timeline')
if __name__=='__main__':main()
