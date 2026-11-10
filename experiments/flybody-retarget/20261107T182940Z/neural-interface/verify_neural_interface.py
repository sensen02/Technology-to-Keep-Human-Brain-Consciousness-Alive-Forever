"""Independent read-only artifact verification. No imports from generator or engine."""
from pathlib import Path
import hashlib,json,gzip,csv,resource
import numpy as np
ROOT=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 m=json.loads((ROOT/'NEURAL_INTERFACE_MANIFEST.json').read_text());checks={}
 for key,v in m['sources'].items():assert sha(v['path'])==v['sha256'],key
 checks['all_source_hashes']=True
 for key,v in m['artifacts'].items():assert sha(v['path'])==v['sha256'],key
 checks['all_declared_artifact_hashes']=True
 with np.load(m['sources']['cache']['path'],allow_pickle=False) as z:
  rid=z['root_ids'];pre=z['pre'];post=z['post'];syn=z['syn'];sc=z['ann_super_class'];cl=z['ann_class'];nt=z['ann_nt_verified'];nerve=z['ann_nerve'];fun=z['ann_function'];body=z['ann_body_part']
 n=len(rid);assert len(set(map(int,rid)))==n and np.all(rid[1:]>rid[:-1]);assert pre.min()>=0 and post.min()>=0 and max(pre.max(),post.max())<n
 checks['cache_id_and_edge_bounds']=True
 with np.load(ROOT/'GROUP_CACHE_INDICES.npz',allow_pickle=False) as z:groups={k:z[k] for k in z.files}
 assert set(groups)==set(m['groups'])
 for k,ix in groups.items():
  assert len(ix)==m['groups'][k]['cache_count'];assert len(ix)==len(np.unique(ix));assert np.all((ix>=0)&(ix<n));assert np.all(ix[1:]>ix[:-1])
  rule=m['groups'][k]['rule']
  if 'ann_super_class_equals' in rule:expected=np.flatnonzero(sc==rule['ann_super_class_equals']);assert np.array_equal(ix,expected)
  if 'ann_super_class_in' in rule:expected=np.flatnonzero(np.isin(sc,rule['ann_super_class_in']));assert np.array_equal(ix,expected)
  if 'ann_class_equals' in rule:assert np.array_equal(ix,np.flatnonzero(cl==rule['ann_class_equals']))
  if 'ann_function_exact_token' in rule:assert np.array_equal(ix,np.flatnonzero([rule['ann_function_exact_token'] in str(x).split(',') for x in fun]))
 checks['group_membership_bounds_and_source_rules']=True
 asc=set(map(int,groups['ascending_boundary']));desc=set(map(int,groups['descending_boundary']));assert not asc&desc;assert asc|desc==set(map(int,groups['neck_crossing_union']))
 assert len(asc)==2359 and len(desc)==1327
 checks['ascending_descending_disjoint_exhaustive']=True
 with np.load(ROOT/'DIRECTED_ROUTE_EDGE_ROWS.npz',allow_pickle=False) as z:
  for name,v in m['directed_routes'].items():
   src=np.zeros(n,bool);dst=np.zeros(n,bool);src[groups[v['source_group']]]=True;dst[groups[v['target_group']]]=True
   expected=np.flatnonzero(src[pre]&dst[post]);assert np.array_equal(z[name],expected);assert len(expected)==v['edge_rows'];assert int(syn[expected].sum(dtype=np.float64))==v['synapse_sum']
 checks['all_directed_route_rows_exact']=True
 with np.load(ROOT/'LOCAL_MODULE_CANDIDATE.npz',allow_pickle=False) as z:c={k:z[k] for k in z.files}
 ix=c['cache_indices'];rr=c['source_edge_rows'];assert len(ix)<=20000;assert np.array_equal(c['root_ids'],rid[ix]);assert np.array_equal(groups['local_module_selected'],ix)
 global_to_local=np.full(n,-1,np.int64);global_to_local[ix]=np.arange(len(ix));expected=np.flatnonzero((global_to_local[pre]>=0)&(global_to_local[post]>=0));assert np.array_equal(rr,expected)
 assert np.array_equal(c['pre_local'],global_to_local[pre[rr]]);assert np.array_equal(c['post_local'],global_to_local[post[rr]]);assert np.array_equal(c['synapse_counts'],syn[rr]);assert c['pre_local'].min()>=0 and c['post_local'].min()>=0
 assert c['pre_local'].max()<len(ix) and c['post_local'].max()<len(ix)
 known=np.isin(nt[pre[rr]],['acetylcholine','gaba','glutamate']);assert np.array_equal(c['pure_verified_label_mask'],known);assert np.array_equal(c['strict_source_edge_rows'],rr[known]);assert np.array_equal(c['source_verified_nt'],nt[pre[rr]])
 checks['local_candidate_is_exact_induced_cache_subgraph']=True
 checks['known_label_mask_exact_no_unknown_sign_fill']=True
 # Reproduce ranking independently, including complete mandatory port preservation.
 mandatory=np.zeros(n,bool)
 for name in ['neck_crossing_union','leg_sensory_nerve_union','leg_motor_class']:mandatory[groups[name]]=True
 assert np.all(global_to_local[np.flatnonzero(mandatory)]>=0)
 score=np.zeros(n,float)
 np.add.at(score,post[mandatory[pre]],syn[mandatory[pre]])
 np.add.at(score,pre[mandatory[post]],syn[mandatory[post]])
 candidates=np.flatnonzero((sc=='ventral_nerve_cord_intrinsic')&~mandatory);rank=sorted(map(int,candidates),key=lambda i:(-score[i],i));chosen=sorted(list(np.flatnonzero(mandatory))+rank[:20000-int(mandatory.sum())]);assert np.array_equal(ix,chosen)
 checks['candidate_selection_exact']=True
 in_all=np.bincount(post[rr],minlength=n);out_all=np.bincount(pre[rr],minlength=n);in_strict=np.bincount(post[rr[known]],minlength=n);out_strict=np.bincount(pre[rr[known]],minlength=n)
 coverage={}
 for k in ['ascending_boundary','descending_boundary','leg_sensory_nerve_union','leg_motor_class','vnc_intrinsic']:
  g=groups[k];g=g[global_to_local[g]>=0]
  coverage[k]={'selected_cells':len(g),'any_incoming_cells':int((in_all[g]>0).sum()),'any_outgoing_cells':int((out_all[g]>0).sum()),'strict_label_incoming_cells':int((in_strict[g]>0).sum()),'strict_label_outgoing_cells':int((out_strict[g]>0).sum()),'strict_no_incoming_cells':int((in_strict[g]==0).sum())}
 with gzip.open(ROOT/'NODE_IDENTITIES.csv.gz','rt') as f:
  count=0
  for row in csv.DictReader(f):
   i=int(row['cache_index_0based']);assert i==count;assert row['root_id_string']==str(rid[i]);assert row['ann_super_class']==sc[i];assert row['ann_class']==cl[i];assert row['ann_nt_verified']==nt[i];count+=1
  assert count==n
 checks['all_exported_node_ids_and_labels_exact']=True
 for name,column in [('SENSORY_CHANNEL_CANDIDATES.csv','root_888'),('MOTOR_TARGET_CANDIDATES.csv','root_888'),('LOCAL_MOTOR_READOUT_PORTS.csv','root_id_string')]:
  with (ROOT/name).open() as f:
   for row in csv.DictReader(f):
    i=int(row['cache_index_0based']);assert i==-1 or row[column]==str(rid[i]);assert i>=-1 and i<n
 checks['sensory_motor_csv_mapping_indices_exact']=True
 for name,summary in m['coverage'].items():
  total=unique=matched=0
  with gzip.open(summary['id_mapping_file']['path'],'rt') as f:
   for row in csv.DictReader(f):
    unique+=1;total+=int(row['source_row_count']);i=int(row['cache_index_0based'])
    if i>=0:assert str(rid[i])==row['source_id_string'];matched+=1
  assert total==summary['source_rows_with_id'] and unique==summary['unique_ids'] and matched==summary['unique_in_cache']
 checks['mapping_row_multiplicities_and_exact_coverage']=True
 review=json.loads((ROOT/'SENSORY_CHANNEL_REVIEW.json').read_text())
 with (ROOT/'SENSORY_CHANNEL_CANDIDATES.csv').open() as f:bristle=[r for r in csv.DictReader(f) if r['channel']=='bristle_touch']
 assert len(bristle)==review['bristle_touch_rows']
 classcounts={}
 for r in bristle:
  k=r['cache_class'] if int(r['cache_index_0based'])>=0 else 'missing_cache_id';classcounts[k]=classcounts.get(k,0)+1
 assert classcounts==review['matched_cache_class_counts']
 amb={r['root_888']:r for r in bristle if r['cache_class'] not in ('bristle_neuron','')}
 assert set(amb)=={r['root_id_string'] for r in review['ambiguous_records']}
 for r in review['ambiguous_records']:
  s=amb[r['root_id_string']];assert int(s['cache_index_0based'])==r['cache_index_0based'];assert s['cache_class']==r['cache_class'];assert s['cell_sub_class']==r['source_subclass']
 checks['sensory_substring_ambiguities_exact']=True
 result={'status':'PASS','manifest_sha256':sha(ROOT/'NEURAL_INTERFACE_MANIFEST.json'),'verifier_path':str(Path(__file__).resolve()),'verifier_sha256':sha(__file__),'checks':checks,'local_module_structural_port_coverage':coverage,'limitation':'Static source, membership, bounds and sparse topology validation only. Not dynamics, physiology, functional path sufficiency or behavior. Strict label edge mask is an engineering sign-eligibility rule.','calls':{'simulation':0,'neural_steps':0,'training':0,'lesions':0},'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 (ROOT/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
