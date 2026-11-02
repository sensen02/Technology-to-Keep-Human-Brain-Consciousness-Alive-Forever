"""Separate raw-row reconstruction and path audit; no dynamics or graph admission."""
from pathlib import Path
import csv,json,math,hashlib,collections
import numpy as np
from scipy import sparse
R=Path(__file__).resolve().parents[1]
O=Path('/home/sensen/Desktop/engine修复验收/s3-sourced-nt-candidate-graph/20261102T004326Z')
B=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
P=Path('/home/sensen/Desktop/engine修复验收/s3-banc888-nt-prediction-provenance/20261101T230101Z')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def main():
 assert not (O/'INDEPENDENT_VERIFICATION.json').exists()
 identities=np.load(O/'IDENTITIES_AND_SELECTED.npz');roots=identities['root_ids'];glob=identities['global_index'];selected=set(map(int,identities['selected_local']));d=np.load(R/'data/flywire/banc_connectome.npz');n=len(roots)
 assert len(selected)==963 and np.array_equal(d['root_ids'][glob],roots)
 rawpre=d['pre'];rawpost=d['post'];rawsynapses=d['syn']
 receipt={int(r['tier_index']):r for r in csv.DictReader((P/'DRIVEN_RECIPIENTS.csv').open())}
 provenance=json.loads((O/'SELECTED_SOURCE_PROVENANCE.json').read_text());assert len(provenance)==963
 for p in provenance:
  i=p['local_index'];r=receipt[i]
  assert p['root_888']==int(roots[i]) and r['cache_verified']==r['local_verified']=='' and r['modeled_zero_output']=='1' and r['motor_readout']=='0'
  assert p['label']==r['official_predicted'] and p['score']==float(r['official_score']) and p['score']>=.9 and p['count']==int(r['official_synapse_count'])>=20
 builds={'EXC':collections.defaultdict(list),'INH':collections.defaultdict(list)};raw_seen=set();syn=0.
 for row in csv.DictReader((O/'ADDED_RAW_ROWS.csv').open()):
  k=int(row['raw_row']);i=int(row['pre_local']);j=int(row['post_local']);value=float(row['conductance_increment_uS']);raw_syn=float(rawsynapses[k])
  assert k not in raw_seen;raw_seen.add(k);assert i in selected and int(rawpre[k])==int(glob[i]) and int(rawpost[k])==int(glob[j])
  assert int(row['pre_root888'])==int(roots[i]) and int(row['post_root888'])==int(roots[j]) and float(row['synapses'])==raw_syn
  label=receipt[i]['official_predicted'];sign='EXC' if label=='acetylcholine' else 'INH';assert row['predicted_NT']==label and row['receptor_assumption']==sign
  assert value==min(raw_syn,8)*.0005*.2;builds[sign][(j,i)].append(value);syn+=raw_syn
 assert len(raw_seen)==17264 and syn==98807
 base={};candidate={}
 for sign,suffix in [('EXC','We'),('INH','Wi')]:
  a=sparse.load_npz(O/f'ADDED_{suffix}.npz').tocoo();dictionary={(int(j),int(i)):float(v) for j,i,v in zip(a.row,a.col,a.data)}
  expected={key:math.fsum(values) for key,values in builds[sign].items()};assert set(dictionary)==set(expected)
  assert all(abs(dictionary[k]-v)<1e-15 for k,v in expected.items())
  baseline=sparse.load_npz(B/f'ACTUAL_{suffix}.npz').tocsc();c=sparse.load_npz(O/f'CANDIDATE_{suffix}.npz').tocsc()
  assert all(baseline.indptr[i]==baseline.indptr[i+1] for i in selected)
  for i in range(n):
   got={int(j):float(v) for j,v in zip(c.indices[c.indptr[i]:c.indptr[i+1]],c.data[c.indptr[i]:c.indptr[i+1]])}
   want={int(j):float(v) for j,v in zip(baseline.indices[baseline.indptr[i]:baseline.indptr[i+1]],baseline.data[baseline.indptr[i]:baseline.indptr[i+1]])}
   if i in selected:want={j:v for (j,source),v in expected.items() if source==i}
   assert set(got)==set(want) and all(abs(got[j]-v)<1e-15 for j,v in want.items())
  base[sign]=baseline;candidate[sign]=c
 graph=(candidate['EXC']+candidate['INH']).astype(bool).tocsc();paths=json.loads((O/'BOOLEAN_REACHABILITY.json').read_text())
 mapping=json.loads((B/'RUN/MAPPING.json').read_text());motors={int(i) for row in mapping['rows'] for i in row['local_indices']}
 for target,path in paths['candidate']['paths'].items():
  assert path[0] in selected and path[-1]==int(target) and int(target) in motors
  for i,j in zip(path[:-1],path[1:]):assert bool(graph[j,i])
 # Reverse multi-source traversal determines which individual sources can reach ANY MN.
 reverse=graph.tocsr();queue=collections.deque(motors);can_reach=set(motors)
 while queue:
  j=queue.popleft()
  for i in reverse.indices[reverse.indptr[j]:reverse.indptr[j+1]]:
   if int(i) not in can_reach:can_reach.add(int(i));queue.append(int(i))
 pernode={str(int(roots[i])):dict(local_index=i,can_reach_any_readout_MN=i in can_reach) for i in sorted(selected)}
 (O/'SOURCE_TO_MN_REACHABILITY.json').write_text(json.dumps(pernode,ensure_ascii=False,indent=2)+'\n')
 result=json.loads((O/'RESULT.json').read_text())
 for path,h in result['source_sha256'].items():assert sha(Path(path))==h
 report=dict(passed=True,raw_rows_verified=len(raw_seen),raw_synapses=syn,selected_nodes=len(selected),existing_weights_preserved=True,known_labels_overridden=0,path_witnesses_verified=len(paths['candidate']['paths']),sources_can_reach_any_MN=len(selected&can_reach),sources_cannot_reach_any_MN=len(selected-can_reach),all_source_hashes_unchanged=True,runtime_allowed=False,calls=dict(model=0,network=0,neural_step=0,body_step=0,training=0),limits='Boolean topology only; not dynamics, physiological receptor proof or embodied behavior.')
 (O/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
