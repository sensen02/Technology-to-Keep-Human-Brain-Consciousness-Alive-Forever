"""Candidate-only derivative of rebuild11; explicit identity/product required."""
from pathlib import Path
import json,hashlib,argparse,csv,gzip
import numpy as np
from run_neural_rebuild_11_mn_pools import ALIASES,parse_actuator

def identity(v):
 if isinstance(v,(int,np.integer)):return int(v)
 if isinstance(v,str) and v.isdecimal():return int(v)
 raise ValueError('missing or noninteger identity: '+repr(v))
def project(records,*,identity_column,connection_product,node_ids):
 if not identity_column or not connection_product:raise ValueError('explicit identity column and target connection product required')
 out=[];errors=[];targets={};sources={}
 for pos,r in records:
  if identity_column not in r:errors.append(dict(row=pos,error='missing_identity_column'));continue
  try:old=identity(r['root_id']);target=identity(r[identity_column])
  except ValueError as e:errors.append(dict(row=pos,error=str(e)));continue
  if r.get('banc_888_id') is not None and r.get('root_888') is not None and identity(r['banc_888_id'])!=identity(r['root_888']):errors.append(dict(row=pos,error='conflicting_explicit_version_columns'));continue
  if target in targets:errors.append(dict(row=pos,error='duplicate_target',target=target,other_row=targets[target]))
  if old in sources:errors.append(dict(row=pos,error='duplicate_or_multimapping_source',source=old,other_row=sources[old]))
  targets[target]=pos;sources[old]=pos
  out.append(dict(row=pos,old_root_id=old,target_id=target,target_node_present=target in node_ids,annotation=r))
  if target not in node_ids:errors.append(dict(row=pos,error='target_absent_node_table',target=target))
 return out,errors

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--identity-column',required=True,choices=['banc_888_id','root_888']);ap.add_argument('--connection-product',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);args=ap.parse_args()
 import pyarrow.feather as feather
 P=Path(__file__).parent;meta=P/'data/banc888/banc_888_meta.feather';oldpath=P/'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json';cachepath=P/'data/flywire/banc_connectome.npz';nodepath=P/'data/flywire/banc_neurons.csv.gz';old=json.loads(oldpath.read_text());O=args.output;O.mkdir(parents=True,exist_ok=False)
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();sources=[meta,oldpath,cachepath,nodepath,args.connection_product,Path(__file__),P/'run_neural_rebuild_11_mn_pools.py'];hashes={str(p):sha(p) for p in sources}
 with gzip.open(nodepath,'rt') as f:nodes={int(r['Root ID']) for r in csv.DictReader(f)}
 table=feather.read_table(meta);cols=['root_id','banc_888_id','root_888','flow','peripheral_target_type','body_part_effector','side'];raw=table.select(cols).to_pylist();wanted={(r['muscle_target'],parse_actuator(r['actuator'])['body_part'],r['side']) for r in old['measured']['resolved']};records=[(i,r) for i,r in enumerate(raw) if r['flow']=='efferent' and (r['peripheral_target_type'],r['body_part_effector'],r['side']) in wanted]
 projected,errors=project(records,identity_column=args.identity_column,connection_product=str(args.connection_product),node_ids=nodes)
 with np.load(cachepath,allow_pickle=False) as z:cacheids=set(map(int,z['root_ids']))
 # Pin final historical indices, do not rebuild or extend tier.
 frozen=json.loads(Path('/home/sensen/Desktop/engine修复验收/s2-runtime-root-identity/20261027T165000Z/MEMBER_DIFFERENCES.json').read_text());root_local={root:idx for r in frozen for root,idx in zip(r['roundtrip_roots'],r['network_local_indices'])}
 # Full historical tier index directory is obtained via exact saved metadata if available; source build is offline only.
 from engine.embodied.adapters import build_neural_tier,TierConfig
 built=build_neural_tier(TierConfig(background_mean_nA=0,background_sd_nA=0));tierlocal={int(r):i for i,r in enumerate(built['tier']['root_ids'])}
 endpoints=set();hits={p['target_id']:dict(pre_rows=0,post_rows=0) for p in projected}
 with gzip.open(args.connection_product,'rt') as f:
  for r in csv.DictReader(f):
   pre,post=int(r['pre_root_id']),int(r['post_root_id']);endpoints.update((pre,post))
   if pre in hits:hits[pre]['pre_rows']+=1
   if post in hits:hits[post]['post_rows']+=1
 candidates=[];diffs=[]
 for r in old['measured']['resolved']:
  parsed=parse_actuator(r['actuator']);key=(r['muscle_target'],parsed['body_part'],r['side']);match=[p for p in projected if (p['annotation']['peripheral_target_type'],p['annotation']['body_part_effector'],p['annotation']['side'])==key];oldroots=sorted(set(r['pool_root_ids']));newroots=sorted(p['target_id'] for p in match);old_reconstructed=sorted(set(p['old_root_id'] for p in match));assert old_reconstructed==oldroots
  candidate=dict(r,pool_root_ids=newroots,identity_evidence=[dict(row_position=p['row'],old_root_id=p['old_root_id'],target_id=p['target_id'],leg=parsed['leg'],side=r['side'],target_label=r['muscle_target'],source_sha256=hashes[str(meta)],target_node_present=p['target_node_present'],target_endpoint_present=p['target_id'] in endpoints,target_cache_present=p['target_id'] in cacheids,tier_local_index=tierlocal.get(p['target_id']),edge_counts=hits[p['target_id']]) for p in match]);candidates.append(candidate);diffs.append(dict(actuator=r['actuator'],old_members=oldroots,candidate_members=newroots,removed=sorted(set(oldroots)-set(newroots)),added=sorted(set(newroots)-set(oldroots)),old_path_exact=True))
 contract=dict(identity_column=args.identity_column,connection_product=str(args.connection_product),source_hashes=hashes,explicit_annotation_version_association=True,structural_compatibility=all(p['target_id'] in endpoints and p['target_node_present'] for p in projected),published_release_confirmed=False,runtime_allowed=False,block_reason='CANDIDATE_ONLY: exact Codex release unpinned; explicit projection not approved runtime mapping',errors=errors,unresolved_preserved=16,default_path_modified=False)
 for name,v in [('CONTRACT.json',contract),('CANDIDATE_POOLS.json',dict(status='CANDIDATE_NOT_RUNTIME',measured=dict(resolved=candidates,unresolved=old['measured']['unresolved']),contract=contract)),('MEMBER_DIFFS_74.json',diffs),('ASSOCIATION_ERRORS.json',errors)]: (O/name).write_text(json.dumps(v,indent=2,ensure_ascii=False))
 assert all(sha(Path(k))==v for k,v in hashes.items());(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(dict(changed_actuators=sum(bool(d['added'] or d['removed']) for d in diffs),association_errors=len(errors),structural_compatibility=contract['structural_compatibility'],runtime_allowed=False)))
if __name__=='__main__':main()
