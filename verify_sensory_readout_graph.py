"""S163 independent vectorized CSR reachability/index/readout saved-data audit."""
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np
B=Path('/home/sensen/Desktop/engine修复验收/s162-fast-sensory/20261024T224135142077Z')
A=Path('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z')
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);p=pa.parse_args().output
 rows=list(csv.DictReader((p/'GRAPH_ALL_NEURONS.csv').open()));n=len(rows);assert n==17099 and [int(r['local_index']) for r in rows]==list(range(n))
 with np.load(A/'init80ms/TRAJECTORY.npz') as a,np.load(B/'init80ms/TRAJECTORY.npz') as b:changed=np.any(a['current']!=b['current'],axis=0);assert sum(changed)==552;assert np.array_equal(a['spikes'],b['spikes']);assert all((r['input_delta']=='True')==changed[i] for i,r in enumerate(rows));counts=b['spikes'].sum(0)
 with np.load(B/'init80ms/CONNECTIVITY.npz') as a:
  pre=[];post=[]
  for k in ['We','Wi']:
   mask=a[k+'_data']!=0;pre.extend(a[k+'_indices'][mask].tolist());post.extend(np.repeat(np.arange(n),np.diff(a[k+'_indptr']))[mask].tolist())
 pre=np.array(pre);post=np.array(post);dist=np.full(n,-1);dist[changed]=0;layer=0
 while True:
  target=np.unique(post[dist[pre]==layer]);target=target[dist[target]<0]
  if not len(target):break
  layer+=1;dist[target]=layer
 assert np.array_equal(dist,np.array([int(r['min_hops_from_any_changed']) for r in rows]))
 read=list(csv.DictReader((p/'GRAPH_READOUT_NEURONS.csv').open()));ids=[int(r['local_index']) for r in read];assert len(set(ids))==len(ids)
 with np.load(B/'init80ms/NEURAL_FINAL.npz') as a:
  for r in read:
   i=int(r['local_index']);assert float(r['s162_final_v_mV'])==a['v'][i] and int(r['s162_release_spike_count'])==counts[i]
 sources=json.loads((p/'GRAPH_SOURCE_HASHES.json').read_text());assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==v['sha256'] for f,v in sources.items())
 result=dict(passed=True,independent_method='layered vectorized edge traversal, not producer deque BFS',n=n,readout_cells=len(ids),changed_inputs=int(changed.sum()),index_unit_mapping_checked=True,source_hashes_checked=True,new_neural_ticks=0,checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(p/'PARENT_GRAPH_INDEPENDENT_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
