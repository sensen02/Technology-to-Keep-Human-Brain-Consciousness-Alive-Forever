from pathlib import Path
import json,csv,gzip,hashlib,numpy as np
P=Path(__file__).parent
O=Path('/home/sensen/Desktop/engine修复验收/s2-two-root-cache-source/20261027T171000Z')
def main():
 result=json.loads((O/'RESULT.json').read_text());links={r:v['feather_matches'][0]['record']['banc_888_id'] for r,v in result['roots'].items()};out={r:dict(requested_root=r,same_annotation_row_banc888_id=b,node_records=[],edge_rows=[],warning='version evidence only; no ID replacement or new mapping') for r,b in links.items()};reverse={b:r for r,b in links.items()}
 with gzip.open(P/'data/flywire/banc_neurons.csv.gz','rt') as f:
  for line,row in enumerate(csv.DictReader(f),2):
   if row['Root ID'] in reverse:out[reverse[row['Root ID']]]['node_records'].append(dict(logical_line=line,record=row))
 with gzip.open(P/'data/flywire/banc_connections_princeton.csv.gz','rt') as f:
  rd=csv.DictReader(f)
  for line,row in enumerate(rd,2):
   for k in ('pre_root_id','post_root_id'):
    if row[k] in reverse:out[reverse[row[k]]]['edge_rows'].append(dict(logical_line=line,record=row));break
 with np.load(P/'data/flywire/banc_connectome.npz',allow_pickle=False) as z:
  idx={str(int(r)):i for i,r in enumerate(z['root_ids'])}
  for r,b in links.items():out[r]['linked_version_id_cache_global_index']=idx.get(b)
 (O/'VERSION_RECORDS.json').write_text(json.dumps(out,indent=2));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps({r:dict(link=x['same_annotation_row_banc888_id'],node_rows=len(x['node_records']),edge_rows=len(x['edge_rows']),cache_index=x['linked_version_id_cache_global_index']) for r,x in out.items()}))
if __name__=='__main__':main()
