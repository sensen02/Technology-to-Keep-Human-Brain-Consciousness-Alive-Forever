"""S164 saved neural-pair boundary analysis; no neural execution."""
from pathlib import Path
import argparse,json,csv
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',required=True,type=Path);p=pa.parse_args().output
 # Schema supplied by producer before execution; analyze only actual recorded arrays.
 data={}
 for arm in ['A','B']:
  with np.load(p/f'{arm}_STATES.npz') as z:data[arm]={k:z[k].copy() for k in z.files}
  data[arm].update(candidate=data[arm]['candidate_v_BE'],eligible=data[arm]['active'],unclipped=data[arm]['unclipped_pool'],end_ms=data[arm]['post_t_ms'])
 a,b=data['A'],data['B'];ids=np.array(sorted({int(r['local_index']) for r in csv.DictReader(Path('/home/sensen/Desktop/engine修复验收/s163-sensory-readout-audit/20261024T224712953805523Z/GRAPH_READOUT_NEURONS.csv').open())}));rows=list(csv.DictReader(Path('/home/sensen/Desktop/engine修复验收/s163-sensory-readout-audit/20261024T224712953805523Z/GRAPH_ALL_NEURONS.csv').open()));changed=np.any(a['input']!=b['input'],axis=0)
 def first(diff):
  at=np.argwhere(diff)
  if not len(at):return None
  tick=int(at[0,0]);ii=np.flatnonzero(diff[tick]);return dict(tick=tick,relative_end_ms=float(a['end_ms'][tick]-80),neural_end_ms=float(a['end_ms'][tick]),indices=ii.tolist())
 spike=a['spikes']!=b['spikes'];event=first(spike);read=first(spike[:,ids]);cmd=first(a['applied']!=b['applied']);uncl=first(a['unclipped']!=b['unclipped']);cand=first(a['candidate']!=b['candidate']);evrows=[]
 for k,i in np.argwhere(spike):evrows.append(dict(tick=int(k),relative_end_ms=float(a['end_ms'][k]-80),local_index=int(i),root_id=rows[i]['root_id'],change='ADDED_B' if b['spikes'][k,i] else 'OMITTED_B',readout=bool(i in set(ids)),direct_changed_input=bool(changed[i]),candidate_A=float(a['candidate'][k,i]),candidate_B=float(b['candidate'][k,i]),eligible_A=bool(a['eligible'][k,i]),eligible_B=bool(b['eligible'][k,i])))
 with (p/'DIFFERENTIAL_SPIKE_EVENTS.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(evrows[0]) if evrows else ['tick','local_index']);w.writeheader();w.writerows(evrows)
 result=dict(first_candidate_difference=cand,first_spike_difference=event,first_readout_spike_difference=read,first_unclipped_difference=uncl,first_applied_difference=cmd,n_differential_spike_events=len(evrows),first_readout_note='indices in first_readout are positions within saved sorted readout list; see readout_local_indices',readout_local_indices=ids.tolist(),changed_input_count=int(changed.sum()),experiment='Held real sample neural only, NOT body closed loop')
 fig,axs=plt.subplots(3,2,figsize=(14,12));t=a['end_ms']-80
 for ax,key,title in [(axs[0,0],'candidate','Candidate voltage max difference (mV)'),(axs[2,0],'unclipped','Unclipped FIR pool drive max difference'),(axs[2,1],'applied','Gain .5 command max difference')]:ax.plot(t,np.max(abs(a[key]-b[key]),axis=1));ax.set_title(title)
 axs[0,1].plot(t,spike.sum(1));axs[0,1].set_title('Differing spikes all neurons');axs[1,0].plot(t,spike[:,ids].sum(1));axs[1,0].set_title('Differing readout spikes');axs[1,1].plot(t,np.count_nonzero(a['arrivals']!=b['arrivals'],axis=1));axs[1,1].set_title('Differing presynaptic arrival events')
 for ax in axs.flat:ax.axvline(.5,color='gray',ls='--');ax.set_xlabel('Neural time since PRE160 (ms)');ax.grid()
 fig.suptitle('S164 fixed real sensory sample neural response, NOT body experiment');fig.tight_layout();fig.savefig(p/'NEURAL_RESPONSE_TIMELINE.png',dpi=150);(p/'PARENT_RESPONSE_SUMMARY.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
