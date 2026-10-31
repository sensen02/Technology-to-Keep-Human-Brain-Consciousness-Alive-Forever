"""S163 offline graph/identity audit. Imports only stdlib and numpy; no neural execution."""
from pathlib import Path
import argparse, csv, json, hashlib
from collections import deque
import numpy as np
REF=Path('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z')
NEW=Path('/home/sensen/Desktop/engine修复验收/s162-fast-sensory/20261024T224135142077Z')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def writecsv(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def identity(a):
 # Literal numpy transcription of frozen adapters/neck_cut_data selector, command union, endpoints.
 roots=a['root_ids'];pre=a['pre'];post=a['post'];n=len(roots)
 mech=np.isin(a['ann_class'],['bristle_neuron','chordotonal_organ_neuron','campaniform_sensillum_neuron','hair_plate_neuron','multidendritic_neuron','taste_bristle_tactile_neuron'])
 head=np.array([any(k in s for k in ('antennal_nerve','eye_nerve','maxillary-labial_nerve','occipital_nerve','frontal_nerve')) for s in a['ann_nerve']])
 keep=mech & np.array(['leg_nerve' in s for s in a['ann_nerve']]) & ~head
 for _ in range(2):
  hit=keep[pre];inflow=np.bincount(post[hit],weights=a['syn'][hit].astype(float),minlength=n)
  candidates=np.flatnonzero((inflow>=10)&~keep);order=candidates[np.lexsort((candidates,-inflow[candidates]))]
  keep[order[:12000-int(keep.sum())]]=True
 base=np.flatnonzero(keep);byid={int(r):i for i,r in enumerate(roots)}
 reached=np.array([byid[r] for r in [720575941433155799,720575941566493282]])
 pure=np.isin(a['ann_nt_verified'],['acetylcholine','gaba','glutamate'])
 for _ in range(2): reached=np.union1d(reached,post[np.isin(pre,reached)&pure[pre]])
 gi=np.union1d(base,reached)
 for r in [720575941391252886,720575941511883985,720575941497037218,720575941559306355]:
  if byid[r] not in set(gi):gi=np.append(gi,byid[r])
 return gi

def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',required=True,type=Path);args=pa.parse_args();out=args.output;out.mkdir(exist_ok=True,parents=True)
 records=json.loads((NEW/'BOUND_INPUT_FREEZE.json').read_text())['records'];sources={}
 def frozen(suffix):
  r=next(r for r in records if r['source'].endswith(suffix));p=Path(r['frozen']);assert sha(p)==r['sha256'];sources[str(p)]=dict(original=r['source'],sha256=sha(p));return p
 bp=frozen('/data/flywire/banc_connectome.npz');mp=frozen('/s4-endpoints/20261023T011559Z/MANIFEST.json')
 for suffix in ['/engine/neural_cond.py','/engine/neck_cut_data.py','/engine/embodied/adapters.py','/engine/embodied/command_tier.py','/engine/embodied/motor_endpoints.py','/engine/embodied/neural_command.py']:frozen(suffix)
 a=load(bp);gi=identity(a);roots=a['root_ids'][gi];n=len(gi);assert n==17099
 manifest=json.loads(mp.read_text());rows=manifest['rows'];assert len(rows)==90
 t=[];final=[]
 for arm in [REF,NEW]:
  t.append(load(arm/'init80ms/TRAJECTORY.npz'));final.append(load(arm/'init80ms/NEURAL_FINAL.npz'))
  for name in ['TRAJECTORY.npz','CONNECTIVITY.npz','NEURAL_FINAL.npz','NEURAL_FINAL_META.json']:
   p=arm/'init80ms'/name;sources[str(p)]=dict(sha256=sha(p))
 meta=json.loads((NEW/'init80ms/NEURAL_FINAL_META.json').read_text());cfg=meta['neural_configs']['p'];tm=meta['neural_scalars']['t_ms']
 assert np.array_equal(t[0]['actuator_pool'],t[1]['actuator_pool']);ap=t[1]['actuator_pool'];pools={};members=[[] for _ in range(n)];muscles=[[] for _ in range(n)]
 for r in rows:
  j=r['actuator_index'];p=int(ap[j]);idx=tuple(r['local_indices']);assert all(0<=i<n for i in idx)
  if p<0:
   assert not idx
   continue
  assert list(map(int,roots[list(idx)]))==r['tier_root_ids']
  if p in pools:assert pools[p]==idx
  pools[p]=idx
  for i in idx:
   if p not in members[i]:members[i].append(p)
   muscles[i].append(j)
 assert sorted(pools)==list(range(66))
 graph=load(NEW/'init80ms/CONNECTIVITY.npz');other=load(REF/'init80ms/CONNECTIVITY.npz');assert all(np.array_equal(v,other[k]) for k,v in graph.items())
 outgoing=[[] for _ in range(n)];edgecounts={}
 for typ in ['We','Wi']:
  data,indices,ptr=map(graph.get,[typ+'_data',typ+'_indices',typ+'_indptr']);assert tuple(graph[typ+'_shape'])==(n,n)
  assert len(ptr)==n+1 and ptr[0]==0 and ptr[-1]==len(data)==len(indices) and np.all(np.diff(ptr)>=0)
  assert np.all((indices>=0)&(indices<n)) and np.all(np.isfinite(data)) and np.all(data>=0)
  post=np.repeat(np.arange(n),np.diff(ptr));nz=data!=0
  for pre,target in zip(indices[nz],post[nz]):outgoing[int(pre)].append(int(target))
  edgecounts[typ]=int(nz.sum())
 ticks=min(len(t[0]['current']),len(t[1]['current']));delta=t[1]['current'][:ticks]-t[0]['current'][:ticks];maximum=abs(delta).max(0);changed=maximum!=0
 dist=np.full(n,-1,dtype=int);q=deque(np.flatnonzero(changed).tolist());dist[changed]=0
 while q:
  i=q.popleft()
  for j in outgoing[i]:
   if dist[j]<0:dist[j]=dist[i]+1;q.append(j)
 def cell(i):
  g=gi[i];d=dict(local_index=i,global_dataset_index=int(g),root_id=str(roots[i]),input_delta=bool(changed[i]),maxabs_input_delta_nA=float(maximum[i]),first_changed_tick=int(np.flatnonzero(delta[:,i]!=0)[0]) if changed[i] else -1,pool_ids=';'.join(map(str,members[i])),muscle_indices=';'.join(map(str,muscles[i])),min_hops_from_any_changed=int(dist[i]),source_class=str(a['ann_class'][g]),source_super_class=str(a['ann_super_class'][g]),source_nerve=str(a['ann_nerve'][g]),source_primary_type=str(a['ann_primary_type'][g]),source_nt_verified=str(a['ann_nt_verified'][g]))
  for k in range(2):
   label=['s154','s162'][k];f=final[k];d[label+'_release_spike_count']=int(t[k]['spikes'][:ticks,i].sum());d[label+'_total_actual_spike_count']=int(f['spike_count'][i]);d[label+'_final_v_mV']=float(f['v'][i]);d[label+'_threshold_mV']=float(cfg['threshold_mV']);d[label+'_final_adaptation_nA']=float(f['adaptation'][i]);d[label+'_refractory_until_ms']=float(f['refractory_until_ms'][i]);d[label+'_final_ge_uS']=float(f['ge'][i]);d[label+'_final_gi_uS']=float(f['gi'][i]);d[label+'_final_active_next_interval']=bool(tm+1e-10>=f['refractory_until_ms'][i])
  return d
 allcells=[cell(i) for i in range(n)];writecsv(out/'GRAPH_ALL_NEURONS.csv',allcells)
 readout=sorted(set(i for ids in pools.values() for i in ids));writecsv(out/'GRAPH_READOUT_NEURONS.csv',[allcells[i] for i in readout]);poolrows=[]
 for p,ids in sorted(pools.items()):
  reach=[int(dist[i]) for i in ids if dist[i]>=0];js=np.flatnonzero(ap==p).tolist()
  poolrows.append(dict(pool_id=p,local_indices=';'.join(map(str,ids)),root_ids=';'.join(str(roots[i]) for i in ids),n_cells=len(ids),n_changed_cells=int(changed[list(ids)].sum()),n_reachable_cells=len(reach),min_hops=min(reach) if reach else -1,muscle_indices=';'.join(map(str,js)),actuators=';'.join(rows[j]['actuator'] for j in js)))
 writecsv(out/'GRAPH_66_POOL_MEMBERSHIP.csv',poolrows)
 muscle_rows=[]
 for r in rows:
  j=r['actuator_index'];p=int(ap[j]);pr=poolrows[p] if p>=0 else dict(local_indices='',root_ids='',n_changed_cells=0,n_reachable_cells=0,min_hops=-1);muscle_rows.append(dict(actuator_index=j,actuator=r['actuator'],leg=r['leg'],muscle_target=r['muscle_target'],pool_id=p,mapping_status='MAPPED' if p>=0 else 'UNMAPPED_NO_TIER_CELLS',declared_pool_root_ids=';'.join(map(str,r['pool_root_ids'])),missing_tier_root_ids=';'.join(map(str,r.get('missing_tier_roots',sorted(set(r['pool_root_ids'])-set(r.get('tier_root_ids',[])))))),local_indices=pr['local_indices'],root_ids=pr['root_ids'],n_changed_cells=pr['n_changed_cells'],n_reachable_cells=pr['n_reachable_cells'],min_hops=pr['min_hops']))
 writecsv(out/'GRAPH_90_MUSCLE_MAPPING.csv',muscle_rows)
 hist=lambda values:{str(int(k)):int(np.sum(np.asarray(values)==k)) for k in np.unique(values)}
 result=dict(n_neurons=n,changed_current_neurons=int(changed.sum()),n_unique_readout_cells=len(readout),n_readout_pools=66,n_muscles=90,n_mapped_muscles=74,n_unmapped_muscles=16,development_failures=['Offline assertion exit1: actual actuator_pool=-1 for16 empty-index manifest rows; corrected to explicit unmapped rather than claiming90mapped.','Offline assertion exit1: declared pool_root_ids row69 includes out-of-tier root; corrected crosscheck to tier_root_ids and retained missing_tier_roots in output. No model or neural calls in either failure.','Offline CSV enhancement KeyError exit1: missing_tier_roots omitted for some empty rows; corrected to declared-minus-tier fallback. The shell then rendered earlier valid summary; final successful regeneration follows.'],nonzero_entries=edgecounts,reachable_neurons=int((dist>=0).sum()),readout_cell_hops_histogram=hist(dist[readout]),pool_min_hops_histogram=hist([r['min_hops'] for r in poolrows]),muscle_min_hops_histogram=hist([r['min_hops'] for r in muscle_rows]),readout_changed_current_count=int(changed[readout].sum()),readout_release_spike_counts=[int(tk['spikes'][:,readout].sum()) for tk in t],all_release_spike_mismatches=int(np.count_nonzero(t[0]['spikes']!=t[1]['spikes'])),index945=allcells[945],identity_crosschecks=dict(actual_n_17099=True,all_manifest_local_root_pairs=True,all66_pools_consistent=True,both_graph_arrays_equal=True),orientation='CSR row=postsynaptic target, column=presynaptic source; ge += We @ arrivals and gi += Wi @ arrivals',delay=dict(dt_ms=meta['neural_scalars']['dt'],delay_steps=meta['neural_scalars']['delay_steps'],configured_emission_to_arrival_ms=cfg['delay_ms'],ring_length=5),limitations=['Graph unions actual nonzero excitatory and inhibitory entries; reachability does not imply excitation, spikes or causal effect.','Hop count is structural; no propagation time inferred from current injection. An emitted spike arrives after 2ms at interval start; target integrates and thresholds at interval end, not instantaneously.','First current delta appears at release tick1; only5 release ticks observed. No readout cell has direct current delta; all readout graph paths require synaptic edges.','Final state is post-reset/post-decay; no per-tick pre-threshold voltage reconstructed here. Threshold is fixed; adaptation is subtractive current, not a dynamic threshold.','Root identity reconstructed using literal numpy selector transcription from frozen sources, not by executing tier/neural/model builders; all manifest readout pairs crosschecked.'])
 (out/'GRAPH_SUMMARY.json').write_text(json.dumps(result,indent=2));(out/'GRAPH_SOURCE_HASHES.json').write_text(json.dumps(sources,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
