"""S163 independent saved-array/source audit; imports no project or simulator modules."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
ROOT=Path('/home/sensen/Desktop/engine修复验收/s163-sensory-readout-audit/20261024T224712953805523Z')
PROJECT=Path('/run/media/sensen/Data2/cell_wound_prototype')
A=Path('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z')
B=Path('/home/sensen/Desktop/engine修复验收/s162-fast-sensory/20261024T224135142077Z')
F=B/'BOUND_INPUT_FREEZE'
manifest=json.loads((F/'0003_MANIFEST.json').read_text())
groups=json.loads((B/'init80ms/ENCODER_PARAMETERS.json').read_text())['groups']
pools=sorted({tuple(sorted(row['local_indices'])) for row in manifest['rows'] if row['local_indices']})
readout=sorted({i for p in pools for i in p})
roots={int(i):str(r) for row in manifest['rows'] for i,r in zip(row['local_indices'],row['tier_root_ids'])}
data={}
inputs=[]
for label,base in [('S154',A),('S162',B)]:
 data[label]={}
 for key,file in [('final','NEURAL_FINAL.npz'),('pre','INITIALIZATION.npz'),('traj','TRAJECTORY.npz'),('cold','NEURAL_COLD.npz')]:
  path=base/'init80ms'/file; inputs.append(path)
  with np.load(path,allow_pickle=False) as z:data[label][key]={k:z[k].copy() for k in z.files}
 for file in ['NEURAL_COLD_META.json','NEURAL_FINAL_META.json','NEURAL_RELEASE_META.json']:
  path=base/'init80ms'/file;inputs.append(path);data[label][file]=json.loads(path.read_text())
a,b=data['S154'],data['S162'];i=int(np.argmax(abs(a['final']['v']-b['final']['v'])))
rows=[]
for label,d in data.items():
 for n in readout:
  final=d['final'];until=float(final['refractory_until_ms'][n]);v=float(final['v'][n]);now=82.5
  rows.append(dict(arm=label,local_index=n,root_id=roots[n],pools=';'.join(str(j) for j,p in enumerate(pools) if n in p),v_final_mV=v,threshold_mV=-50.,algebraic_gap_mV=-50-v,refractory_until_ms=until,eligible_next_start=(now+1e-10>=until),remaining_refractory_ms=max(0.,until-now),adaptation_final_nA=float(final['adaptation'][n]),ge_final_uS=float(final['ge'][n]),gi_final_uS=float(final['gi'][n]),count_PRE=int(d['pre']['spike_count'][n]),count_release=int(d['traj']['spikes'][:,n].sum()),count_final=int(final['spike_count'][n]),last_release_spike=bool(d['traj']['spikes'][-1,n])))
csvpath=ROOT/'SOURCE_READOUT_FINAL_STATE.csv'
with csvpath.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={}
for label,d in data.items():
 t=d['traj'];pre=d['pre'];final=d['final'];rr=[r for r in rows if r['arm']==label];drive=(t['features']*t['weights']).sum(2);mapped=t['actuator_pool']>=0
 ctrl=np.zeros_like(t['decoded']);ctrl[:,mapped]=np.clip(drive,0,1)[:,t['actuator_pool'][mapped]]
 # Independent finite history algebra from SAVED spikes, no neuron state updates.
 history=np.zeros((80,len(pools)));feats=[]
 for sp in np.concatenate([pre['spikes'],t['spikes']]):
  act=np.array([sp[list(p)].mean()/.0005/100 for p in pools]);history=np.concatenate([act[None,:],history])[:80]
  feats.append(np.stack([history[lo:hi].mean(0) for lo,hi in [(0,5),(5,10),(10,20),(20,40),(40,80)]],axis=1))
 feats=np.array(feats)
 assert np.array_equal(final['spike_count'],pre['spike_count']+t['spikes'].sum(0))
 assert np.max(abs(ctrl-t['decoded']))==0
 summary[label]=dict(metadata_final=d['NEURAL_FINAL_META.json'],metadata_cold=d['NEURAL_COLD_META.json'],metadata_release=d['NEURAL_RELEASE_META.json'],release_tick_count=len(t['spikes']),PRE_total_spikes=int(pre['spikes'].sum()),release_total_spikes=t['spikes'].sum(1).tolist(),readout_union_n=len(readout),readout_PRE_spikes=int(pre['spikes'][:,readout].sum()),readout_release_spikes=t['spikes'][:,readout].sum(1).tolist(),readout_final_v_range_mV=[min(r['v_final_mV'] for r in rr),max(r['v_final_mV'] for r in rr)],readout_final_gap_range_mV=[min(r['algebraic_gap_mV'] for r in rr),max(r['algebraic_gap_mV'] for r in rr)],readout_eligible_next_start=sum(r['eligible_next_start'] for r in rr),readout_refractory_next_start=sum(not r['eligible_next_start'] for r in rr),readout_adaptation_range_nA=[min(r['adaptation_final_nA'] for r in rr),max(r['adaptation_final_nA'] for r in rr)],FIR_saved_reconstruction_maxabs=float(np.max(abs(feats-np.concatenate([pre['features'],t['features']])))),decode_reconstruction_maxabs=float(np.max(abs(ctrl-t['decoded']))),pool_drive_min=float(drive.min()),pool_drive_max=float(drive.max()),pool_drive_above_1=int((drive>1).sum()),decoded_range=[float(t['decoded'].min()),float(t['decoded'].max())],applied_range=[float(t['applied'].min()),float(t['applied'].max())],mapped_actuators=int(mapped.sum()),weights_shape=list(t['weights'].shape),weights_range=[float(t['weights'].min()),float(t['weights'].max())])
maxnode=dict(local_index=i,root_id='720575941430177161',sensory_membership=[dict(leg=l,channel=ch) for l,cs in groups.items() for ch,ids in cs.items() if i in ids],readout_membership=i in readout,delta_S162_minus_S154_mV=float(b['final']['v'][i]-a['final']['v'][i]),arms={})
for label,d in data.items():
 maxnode['arms'][label]={k:d['final'][k][i].item() for k in ['v','ge','gi','adaptation','refractory_until_ms','spike_count']}
 maxnode['arms'][label].update(PREcount=int(d['pre']['spikes'][:,i].sum()),release_spikes=d['traj']['spikes'][:,i].tolist(),release_current_nA=d['traj']['current'][:,i].tolist(),final_gap_mV=float(-50-d['final']['v'][i]))
# saved-array inventory, only direct files in the two actual arm directories
inventory={label:{p.name:list(np.load(p,allow_pickle=False).files) for p in base.joinpath('init80ms').glob('*.npz') if p.name in ['INITIALIZATION.npz','NEURAL_COLD.npz','NEURAL_FINAL.npz','TRAJECTORY.npz','STOP_FULL_STATE.npz']} for label,base in [('S154',A),('S162',B)]}
source_ranges=[(F/'0049_neural_cond.py',[(1,9),(17,30),(63,78),(99,155)]),(B/'SOURCE_FREEZE/PRIVATE_RUNNER_SOURCE.py',[(185,188),(225,237),(294,309),(338,373),(376,399)]),(F/'0087_sensory_channels.py',[(255,297),(320,352)]),(F/'0104_motor_translation.py',[(12,29),(35,59)])]
sources=[]
for path,ranges in source_ranges:
 lines=path.read_text().splitlines();inputs.append(path)
 sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),quotes=[dict(start=lo,end=hi,text='\n'.join(f'{j}: {lines[j-1]}' for j in range(lo,min(hi,len(lines))+1))) for lo,hi in ranges]))
inputs.extend([F/'0003_MANIFEST.json',B/'init80ms/ENCODER_PARAMETERS.json'])
report=dict(scope='S163/P1/P2 independent source and saved-numpy audit, no model/physics/neuron execution',new_engine_calls=dict(load=0,forward=0,step=0,neural_ticks=0),max_final_v_node=maxnode,arms=summary,saved_inventory=inventory,source_evidence=sources,input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in dict.fromkeys(inputs)},limitations=['No per-tick voltage/ge/gi/adaptation/refractory trajectory is saved in actual TRAJECTORY; do not recover or reconstruct hidden neural dynamics.','Final gap is an algebraic terminal post-reset voltage gap, not the missing pre-reset candidate crossing margin.','Fixed threshold -50mV, dynamically gated by refractory eligibility; adaptation changes current, not the numeric threshold.','No new run, no parameter change, no physics conclusion or calibrated biological units.'])
text=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
(PROJECT/'s163_neural_source_audit.json').write_text(text)
(ROOT/'s163_neural_source_audit.json').write_text(text)
print(json.dumps(dict(maxnode=maxnode,summary=summary,csv=str(csvpath)),ensure_ascii=False,indent=2))
