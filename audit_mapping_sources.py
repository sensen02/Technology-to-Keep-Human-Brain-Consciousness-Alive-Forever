"""Read original BANC rows and actual loaded manifest; no neural/body simulation."""
from pathlib import Path
import json,sys,csv,hashlib,subprocess
import pandas as pd
import numpy as np,mujoco
from validate_motor_mapping import validate_mapping
from run_neural_rebuild_11_mn_pools import ALIASES,parse_actuator,FEATHER
import engine.canonical_physics as cp
P=Path(__file__).parent
S=Path('/home/sensen/Desktop/engine修复验收/s2-split-write/20261027T163000Z')
O=Path('/home/sensen/Desktop/engine修复验收/s2-mapping-validation/20261027T164000Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False))
def main():
 O.mkdir(parents=True,exist_ok=True);source=P/'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json';mapping=json.loads((S/'MAPPING.json').read_text());original=json.loads(source.read_text());raw=json.loads((O/'ANNOTATION_COLUMNS.json').read_text());d=pd.DataFrame({k:[v[str(i)] for i in range(len(v))] for k,v in raw.items()});m=cp.load_model();names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,j) for j in range(m.nu)];assert names==[r['actuator'] for r in mapping['rows']]
 files=[Path(__file__),P/'validate_motor_mapping.py',P/'test_validate_motor_mapping.py',P/'run_neural_rebuild_11_mn_pools.py',P/'engine/embodied/motor_output.py',P/'run_split_write_diagnostic.py',S/'MAPPING.json',S/'write1ms/INTERVALS.npz',source,FEATHER,Path(cp.CANONICAL_XML)]
 dump(O/'INPUTS.json',dict(sha256={str(p):sha(p) for p in files},metadata_columns=['root_id','flow','peripheral_target_type','body_part_effector','side','cell_type'],metadata_rows=len(d),model_loads=1,forward=0,step=0,neural_step=0))
 ann={}
 for pos,r in d.iterrows():
  if str(r['flow'])=='efferent':ann.setdefault((str(r['peripheral_target_type']),str(r['body_part_effector']),str(r['side'])),[]).append((int(pos),int(r['root_id'])))
 ev=np.load(S/'write1ms/INTERVALS.npz');mask=ev['event_ticks']*.0005<=.0385625;idx,count=np.unique(ev['event_local_indices'][mask],return_counts=True);counts=dict(zip(map(int,idx),map(int,count)));pools={};records=[];mismatch=[]
 for a,r in enumerate(mapping['rows']):
  parsed=parse_actuator(r['actuator']);assert parsed['ok'];target=ALIASES[parsed['muscle']];key=(target,parsed['body_part'],parsed['side']);found=ann.get(key,[]);roots=sorted(set(root for pos,root in found));stored=sorted(set(r['pool_root_ids']));ok=roots==stored
  if not ok:mismatch.append(r['actuator'])
  pool='|'.join(key) if roots else None
  if pool:pools[pool]=roots
  # root IDs and local indices are paired by present tier_root_ids in the actual manifest.
  assert len(r['tier_root_ids'])==len(r['local_indices']);assert set(r['tier_root_ids'])<=set(roots)
  other=[dict(target=k[0],body_part=k[1],side=k[2],row_positions=[x[0] for x in vals],root_ids=sorted(set(x[1] for x in vals))) for k,vals in ann.items() if k[0]==target]
  records.append(dict(actuator=r['actuator'],actuator_index=a,alias=parsed['muscle'],expected=target,expected_body_part=parsed['body_part'],expected_side=parsed['side'],pool=pool,source_level='declared_engineering_alias' if roots else 'missing_basis',annotation_identity_match=ok,pool_root_ids=roots,tier_roots=r['tier_root_ids'],missing_tier_roots=r.get('missing_tier_roots',[]),runtime_status=r['status'],annotation_source=str(FEATHER),annotation_row_positions=[p for p,root in found],alias_source=str(P/'run_neural_rebuild_11_mn_pools.py')+':54–76',source_record=str(source)+('#/measured/resolved/' if roots else '#/measured/unresolved/')+str(next(j for j,x in enumerate(original['measured']['resolved' if roots else 'unresolved']) if x['actuator']==r['actuator'])),saved_event_count=sum(counts.get(i,0) for i in r['local_indices']) if pool else None,event_window_s=[0,.0385625],failure_reason=None if roots else r['source_annotation']['reason'],available_same_target_elsewhere=other if not roots else [],anatomical_confirmation='NOT_ESTABLISHED'))
 result=validate_mapping(names,pools,records);result['source_reconciliation_mismatches']=mismatch;assert not mismatch;assert result['engineering_valid'];assert len(result['unresolved'])==16
 dump(O/'MAPPING_VALIDATION.json',result);dump(O/'GAPS_16.json',result['unresolved']);dump(O/'SHARED_POOLS.json',result['shared_pools'])
 with (O/'TABLE_90.csv').open('w',newline='') as f:
  fields=['actuator','expected','expected_body_part','expected_side','source_level','annotation_identity_match','runtime_status','mapping_state','activity_state','saved_event_count','annotation_row_positions','source_record','failure_reason'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(result['rows'])
 md='# 16项未解决来源缺口\n\n|肌束|期望target / leg / side|实际失败|\n|---|---|---|\n'+''.join('|'+r['actuator']+'|'+r['expected']+' / '+r['expected_body_part']+' / '+r['expected_side']+'|'+r['failure_reason']+'|\n' for r in result['unresolved']);(O/'GAPS_16.md').write_text(md);Path('/home/sensen/Desktop/S2_MAPPING_GAPS.md').write_text(md)
 test=subprocess.run([sys.executable,'-m','unittest','-v','test_validate_motor_mapping'],capture_output=True,text=True);(O/'TEST.log').write_text(test.stdout+test.stderr);assert test.returncode==0
 assert all(sha(Path(k))==v for k,v in json.loads((O/'INPUTS.json').read_text())['sha256'].items())
 dump(O/'EXECUTION.json',dict(command=sys.executable+' '+str(Path(__file__)),exit_code=0,model_load=1,forward=0,step=0,neural_step=0,engineering_valid=result['engineering_valid'],biological_mapping_complete=False,unresolved=16,mapped_zero_events=sum(r['activity_state']=='mapped_zero_events' for r in result['rows']),shared_pools=len(result['shared_pools']),mapping_modified=False))
 for p in (Path(__file__),P/'validate_motor_mapping.py',P/'test_validate_motor_mapping.py'):(O/p.name).write_bytes(p.read_bytes())
 print((O/'EXECUTION.json').read_text())
if __name__=='__main__':main()
