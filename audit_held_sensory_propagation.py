"""S164 saved-event propagation attribution candidates, not invented causal paths."""
from pathlib import Path
import argparse,csv,json
import numpy as np
from scipy.sparse import csr_matrix
def load(p):
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',required=True,type=Path);p=pa.parse_args().output;a=load(p/'A_STATES.npz');b=load(p/'B_STATES.npz');con=load(Path('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z/init80ms/CONNECTIVITY.npz'));W={k:csr_matrix((con[k+'_data'],con[k+'_indices'],con[k+'_indptr']),shape=tuple(con[k+'_shape'])) for k in ['We','Wi']};rows=list(csv.DictReader(Path('/home/sensen/Desktop/engine修复验收/s163-sensory-readout-audit/20261024T224712953805523Z/GRAPH_ALL_NEURONS.csv').open()));reads=np.array([int(r['local_index']) for r in rows if r['pool_ids']]);d=a['spikes']!=b['spikes'];at=np.argwhere(d[:,reads]);k=int(at[0,0]);target=int(reads[at[0,1]]);arr=[]
 # Actual immediate incoming differential events to first differing readout, before its spike.
 for tick in range(k+1):
  changed=np.flatnonzero(a['arrivals'][tick]!=b['arrivals'][tick])
  for typ,mat in W.items():
   for source in changed:
    weight=float(mat[target,source])
    if weight:arr.append(dict(arrival_tick=tick,arrival_neural_start_ms=float(a['pre_t_ms'][tick]),emission_tick=tick-5,source_index=int(source),source_root_id=rows[source]['root_id'],target_index=target,kind=typ,weight_uS=weight,event_A=bool(a['arrivals'][tick,source]),event_B=bool(b['arrivals'][tick,source])))
 actual=[]
 for tick in range(41):
  if tick>=5:assert np.array_equal(a['arrivals'][tick],a['spikes'][tick-5]) and np.array_equal(b['arrivals'][tick],b['spikes'][tick-5])
 firstarr=np.argwhere(a['arrivals']!=b['arrivals']);firstarrtick=int(firstarr[0,0]);result=dict(first_differential_arrival_tick=firstarrtick,first_differential_arrival_start_neural_ms=float(a['pre_t_ms'][firstarrtick]),first_readout_index=target,first_readout_root_id=rows[target]['root_id'],first_readout_tick=k,first_readout_end_neural_ms=float(a['post_t_ms'][k]),candidate_A=float(a['candidate_v_BE'][k,target]),candidate_B=float(b['candidate_v_BE'][k,target]),eligible_A=bool(a['active'][k,target]),eligible_B=bool(b['active'][k,target]),spike_A=bool(a['spikes'][k,target]),spike_B=bool(b['spikes'][k,target]),actual_incoming_differential_events_to_first_readout=arr,interpretation='Actual nonzero incoming edges carrying differential arrival events, consistent temporal association not a channel-isolated causal attribution; accumulated conductance and history matter')
 (p/'PARENT_PROPAGATION_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
