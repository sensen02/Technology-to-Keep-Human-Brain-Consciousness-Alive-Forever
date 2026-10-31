"""S3/P3 fixed first unresolved muscle source audit, no simulation or web."""
from pathlib import Path
import json,hashlib,csv,gzip,sys
import numpy as np
P=Path(__file__).parent;O=Path('/home/sensen/Desktop/engine修复验收/s3-first-mapping-gap/20261027T223000Z');M=Path('/home/sensen/Desktop/engine修复验收/s4-command1ms/20261027T222000Z/MAPPING.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 import pyarrow.feather as feather
 pre=json.loads((O/'PREREGISTRATION.json').read_text());assert sha(M)==pre['sha256'];rows=json.loads(M.read_text())['rows'];g=[r for r in rows if not r['local_indices']];assert len(g)==16 and g[0]==pre['selected_row'];assert g[0]['actuator']=='LFC_pleural_promotor'
 files=[M,P/'data/banc888/banc_888_meta.feather',P/'data/flywire/banc_neurons.csv.gz',P/'data/flywire/banc_connectome.npz',P/'data/banc_dataverse/supplemental_data_2.txt',P/'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json',P/'tools_resolve_muscle_anatomy.py',P/'tools_build_design_runtime.py',P/'run_neural_rebuild_11_mn_pools.py',Path(__file__)];h={str(p):sha(p) for p in files};table=feather.read_table(files[1]);columns=table.column_names;raw=table.to_pylist();ann=['root_id','banc_888_id','root_626','root_850','root_888','root_890','flow','super_class','cell_type','peripheral_target_type','body_part_effector','side','nerve'];records=[];exact=[]
 for i,r in enumerate(raw):
  if r.get('peripheral_target_type')=='pleural_promotor_muscle':exact.append(i)
  if r.get('side')=='left' and r.get('body_part_effector')=='front_leg' and ('pleural_promotor' in str(r.get('cell_type')) or r.get('peripheral_target_type')=='pleural_promotor_muscle'):
   records.append(dict(feather_row0=i,record={k:str(r.get(k)) if r.get(k) is not None else None for k in ann}))
 assert not exact;assert len(records)==4
 ids={v for rec in records for k,v in rec['record'].items() if k in ann[:6] and v not in (None,'0')};csvmatches=[]
 with gzip.open(files[2],'rt') as f:
  for line,r in enumerate(csv.DictReader(f),2):
   if r.get('Root ID') in ids:csvmatches.append(dict(path=str(files[2]),line=line,record=r))
 supplementary=[]
 with files[4].open() as f:
  for line,r in enumerate(csv.DictReader(f),2):
   if str(r.get('root_id')) in ids:supplementary.append(dict(path=str(files[4]),line=line,record=r))
 with np.load(files[3],allow_pickle=False) as cache:cacheids=set(map(lambda r:str(int(r)),cache['root_ids']))
 for r in records:r['root_version_cache_presence']={k:(v in cacheids) for k,v in r['record'].items() if k in ann[:6]};r['interpretation']='compound naming association only; target is tergopleural_promotor_muscle, not separate pleural target'
 refs=[]
 for p,terms in [(files[6],['REJECT_SHARED_MILLER28_30_POOL','pleural_promotor']), (files[7],['pleural_promotor']), (files[8],['pleural_promotor'])]:
  for i,l in enumerate(p.read_text().splitlines(),1):
   if any(t in l for t in terms):refs.append(dict(path=str(p),line=i,text=l.strip(),sha256=h[str(p)]))
 result=dict(selected=pre['selected_row'],decision='UNRESOLVED_NO_CANDIDATE_PATCH',runtime_admission=False,exact_peripheral_target_rows_all_dataset=len(exact),feather_rows_scanned=len(raw),query=dict(exact='peripheral_target_type == pleural_promotor_muscle',compound='side==left AND body_part_effector==front_leg AND cell_type contains pleural_promotor'),compound_annotation_records=records,csv_version_id_matches=csvmatches,supplement_version_id_matches=supplementary,root_version_warning='same annotation row version columns and cache presence are not lineage/equivalence or independent muscle innervation proof; no root substitution',missing_records=['authoritative distinct LFC pleural-promotor motor innervation annotation with left/front identity, not shared Miller28_30 alias','immutable annotation/export version manifest: source revision materialization UTC checksum and field definitions','root_id version lineage/split-merge record linking source annotated motor neuron to runtime root if IDs differ','documented mapping of distinct model muscle LFC_pleural_promotor to source muscle identifier; compound name alone insufficient'],limits=['existing name rejection is project policy, not new biological evidence','other15 gaps not researched','no historical/default mapping changed','74resolved does not mean biologically certified','candidate patch not produced without adequate direct evidence'])
 dump(O/'RESULT.json',result);dump(O/'SOURCE_LINES.json',refs);dump(O/'INPUT_CODE_HASHES.json',h);tests={'fixed_first_gap_index3':g[0]['actuator_index']==3,'no_exact_target':not exact,'four_same_side_leg_compound_records':len(records)==4,'no_runtime_admission':not result['runtime_admission'],'input_hashes_unchanged':all(sha(Path(p))==v for p,v in h.items())};assert all(tests.values());dump(O/'TESTS.json',dict(exit_code=0,checks=tests));dump(O/'EXECUTION.json',dict(command=sys.executable+' '+str(Path(__file__)),exit_code=0,model_load=0,forward=0,body_step=0,neural_step=0,web=0,failures=['pre-research attempted historical root MAPPING.json path absent; corrected to latest frozen historical-pool manifest before selection'],status='missing direct evidence is audit outcome, not execution failure'));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print('slot3 LFC_pleural_promotor: exact0 compound4; no candidate patch, no runtime changes')
if __name__=='__main__':main()
