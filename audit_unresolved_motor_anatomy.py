"""Saved-data exact annotation audit for16 unresolved output slots. No engine import/build."""
from pathlib import Path
import json,hashlib,csv,gzip,time
from collections import Counter
import pandas as pd,numpy as np
P=Path('/run/media/sensen/Data2/cell_wound_prototype')
O=Path('/home/sensen/Desktop/engine修复验收/s5-unresolved-motor-anatomy/20261028T085000Z')
T=Path('/home/sensen/Desktop/engine修复验收/s5-neural-tier-coverage/20261028T070000Z/TIER_IDENTITIES.npz')
M=Path('/home/sensen/Desktop/engine修复验收/s5-compatible-stance-handover/20261028T080000Z/MAPPING.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n')
def main():
 assert not O.exists();O.mkdir(parents=True);start=time.time();files=[Path(__file__),M,T,P/'data/banc888/banc_888_meta.feather',P/'data/flywire/banc_neurons.csv.gz',P/'data/flywire/banc_connectome.npz',P/'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json',P/'run_neural_rebuild_11_mn_pools.py',P/'engine/embodied/motor_output.py',P/'engine/embodied/runtime_pool_guard.py'];hashes={str(p):sha(p) for p in files}
 dump(O/'PREREGISTRATION.json',dict(scope='Read-only exact muscle-target/leg/side/efferent annotations and lexical CSV aliases; no same-name cross-leg substitutions, no invented targets or runtime change.',sources=hashes,counts=dict(model_load=0,forward=0,body_step=0,neural_step=0,tier_build=0),initial_preflight_failure='venv_body lacks pyarrow; failed feather read before analysis,0engine/model/physics/neural. Existing project venv has pyarrow and is used; no package installs.'))
 cols=['root_id','banc_888_id','flow','super_class','cell_type','peripheral_target_type','body_part_effector','side','nerve'];raw=pd.read_feather(P/'data/banc888/banc_888_meta.feather',columns=cols);texts={k:raw[k].astype(str).to_numpy() for k in cols};tier=set(map(int,np.load(T)['root_ids']));full=set(map(int,np.load(P/'data/flywire/banc_connectome.npz')['root_ids']));manifest=json.loads(M.read_text());missing=[r for r in manifest['rows'] if not r['local_indices']];assert len(missing)==16
 with gzip.open(P/'data/flywire/banc_neurons.csv.gz','rt') as f:csvrows=list(csv.DictReader(f))
 target='pleural_remotor_and_abductor_muscle';same=texts['peripheral_target_type']==target;remotor=[]
 for i in np.flatnonzero(same):
  row={k:str(texts[k][i]) for k in cols};rid=int(raw['root_id'].iloc[i]);row.update(raw_feather_row=int(i),root_id=str(rid),full_dataset_present=rid in full,tier12000_present=rid in tier);remotor.append(row)
 exact=[];expected={'F':'front_leg','M':'middle_leg','H':'hind_leg'}
 for r in missing:
  tg=r['muscle_target'];body=expected[r['leg'][1]];side=r['side'];s=texts['peripheral_target_type']==tg;hit=s&(texts['body_part_effector']==body)&(texts['side']==side)&(texts['flow']=='efferent');ids=[str(int(raw['root_id'].iloc[i])) for i in np.flatnonzero(hit)];exact.append(dict(actuator_index=r['actuator_index'],actuator=r['actuator'],target=tg,body_part_effector=body,side=side,reason=r['source_annotation'].get('reason'),same_target_any_route=int(s.sum()),exact_annotation_root_ids=ids,cause='ORIGINAL_NO_EXACT_CANDIDATES' if not ids else 'REAL_CANDIDATES_REQUIRE_REVIEW',tier_filter_cause=False if not ids else None))
 rm=[r for r in csvrows if 'pleural_remotor' in (r.get('Primary Cell Type','')+' '+r.get('Alternative Cell Type(s)',''))];csvout=[]
 for r in rm:
  rawmatch=raw[raw.root_id.astype(str)==r['Root ID']];csvout.append(dict(root_id=r['Root ID'],csv_primary_type=r.get('Primary Cell Type'),csv_alternative_types=r.get('Alternative Cell Type(s)'),csv_nerve=r.get('Nerve'),raw_feather_matches=[{k:str(row[k]) for k in cols} for _,row in rawmatch.iterrows()]))
 rmrow=next(r for r in manifest['rows'] if r['actuator_index']==47);assert rmrow['actuator']=='RMC_pleural_remotor_and_abductor' and not rmrow['pool_root_ids'];exactrm=[r for r in remotor if r['body_part_effector']=='middle_leg' and r['side']=='right' and r['flow']=='efferent'];assert not exactrm
 result=dict(unresolved47_manifest=rmrow,raw_feather_rows=len(raw),all16_unresolved=exact,exact47_raw_candidates=exactrm,remotor_all_target_rows=remotor,remotor_route_counts=dict(Counter(r['body_part_effector']+'/'+r['side'] for r in remotor)),csv_remotor_alias_records=csvout,all16_exact_empty=all(not r['exact_annotation_root_ids'] for r in exact),classification47='ORIGINAL_NO_EXACT_CANDIDATES_NOT_TIER_EXCLUSION_OR_ALIAS_BUG',repair_authorized=False,scope='No exact RM target row in current raw data. LM root720575941553503460 cannot be relabeled or copied toRM. Annotation-level target absence is not biological absence. No genuine new mapping established.')
 dump(O/'DATA_MATCH.json',result);dump(O/'EXECUTION.json',dict(exit_code=0,runtime_s=time.time()-start,counts=dict(model_load=0,forward=0,body_step=0,neural_step=0,tier_build=0),all_source_hashes_unchanged=all(sha(p)==h for p,h in hashes.items())));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(dict(remotor_rows=len(remotor),route_counts=result['remotor_route_counts'],csv_alias_records=len(csvout),all16_exact_empty=result['all16_exact_empty'],rm47_cause=result['classification47']),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
