"""Exactly two root provenance scan; no tier/network/model construction."""
from pathlib import Path
import csv,gzip,json,sys,hashlib
import numpy as np
import pyarrow.feather as feather
P=Path(__file__).parent
O=Path('/home/sensen/Desktop/engine修复验收/s2-two-root-cache-source/20261027T171000Z')
ROOTS=[720575941434056663,720575941500775137]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 O.mkdir(parents=True,exist_ok=False);paths=[P/'build_banc_cns.py',P/'flywire_download.py',P/'data/flywire/banc_neurons.csv.gz',P/'data/flywire/banc_connections_princeton.csv.gz',P/'data/flywire/banc_connectome.npz',P/'data/banc888/banc_888_meta.feather',Path(__file__)];hashes={str(p):sha(p) for p in paths};out={str(r):dict(root=r,node_csv=[],raw_edge_rows=[],feather_matches=[]) for r in ROOTS}
 with gzip.open(paths[2],'rt') as f:
  for line,row in enumerate(csv.DictReader(f),2):
   if int(row['Root ID']) in ROOTS:out[row['Root ID']]['node_csv'].append(dict(logical_line=line,record=row))
 with gzip.open(paths[3],'rt') as f:
  reader=csv.reader(f);header=next(reader);total=0
  for line,row in enumerate(reader,2):
   total+=1
   for r in ROOTS:
    if int(row[0])==r or int(row[1])==r:out[str(r)]['raw_edge_rows'].append(dict(logical_line=line,record=dict(zip(header,row))))
 table=feather.read_table(paths[5]);cols=[c for c in table.column_names if c in ['root_id','banc_888_id','flow','super_class','cell_type','peripheral_target_type','body_part_effector','side'] or c.startswith('root_')];records=table.select(cols).to_pylist()
 for pos,row in enumerate(records):
  for r in ROOTS:
   matched=[k for k,v in row.items() if (k=='banc_888_id' or k.startswith('root_')) and str(v)==str(r)]
   if matched:out[str(r)]['feather_matches'].append(dict(row_position=pos,matched_columns=matched,record={k:str(v) if k=='banc_888_id' or k.startswith('root_') else v for k,v in row.items()}))
 with np.load(paths[4],allow_pickle=False) as z:
  cache=set(map(int,z['root_ids']));cache_raw_rows=int(z['n_rows'][0])
 for v in out.values():v.update(in_cache=v['root'] in cache,conclusion='NODE_HAS_NO_ROWS_IN_LOCAL_RAW_CONNECTION_PRODUCT' if v['node_csv'] and not v['raw_edge_rows'] else 'LOCAL_SOURCE_GAP_OR_VERSION_ID_MISMATCH',cache_rule='build_banc_cns.py:68 root_ids=union(unique(raw_pre),unique(raw_post)); annotations not edge endpoints skipped109–110',version='Codex dataset=banc download URL unversioned; exact bytes pinned by SHA256. Feather filename888 and explicit version columns retained; no assertion that root_id is v888')
 result=dict(roots=out,raw_edge_header=header,raw_edge_rows_scanned=total,cache_recorded_raw_rows=cache_raw_rows,raw_row_count_matches_cache=total==cache_raw_rows,hashes=hashes,command=sys.executable+' '+str(Path(__file__)),exit_code=0,calls=dict(model=0,tier=0,forward=0,neural_step=0,body_step=0),online_query=False)
 (O/'RESULT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());assert all(sha(Path(k))==v for k,v in hashes.items());print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
