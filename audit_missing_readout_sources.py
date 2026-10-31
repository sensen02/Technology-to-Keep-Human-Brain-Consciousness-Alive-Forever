"""S168 bounded saved-only provenance audit. No engine/project imports or simulation.
Run with venv/bin/python (pandas/pyarrow). Only new timestamp outputs are written.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse, ast, csv, gzip, hashlib, json, shutil, sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
P=Path('/run/media/sensen/Data2/cell_wound_prototype')
B=Path('/home/sensen/Desktop/engine修复验收')
S162=B/'s162-fast-sensory/20261024T224135142077Z'
S154=B/'s154-activation-rate/20261024T215751775489Z'
S163=B/'s163-sensory-readout-audit/20261024T224712953805523Z'
S150=B/'s150-leg-command-unloading/20261024T213608033385Z'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def load(p):
 with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def table(p,rows):
 assert rows
 with p.open('x',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def safe_function(path,name,namespace):
 tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
 # Execute ONLY one already-reviewed function definition; top-level imports/main never execute.
 assert not node.decorator_list
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
 return namespace[name]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);args=ap.parse_args()
 out=args.output or B/'s168-missing-readout-sources'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
 out.mkdir(parents=True,exist_ok=False)
 records=json.loads((S162/'BOUND_INPUT_FREEZE.json').read_text())['records']
 def bound(suffix):
  r=next(r for r in records if r['source'].endswith(suffix));p=Path(r['frozen']);assert sha(p)==r['sha256'];return p
 bp=bound('/data/flywire/banc_connectome.npz');mp=bound('/s4-endpoints/20261023T011559Z/MANIFEST.json');xp=bound('/MODEL_CALIB_H_VERIFIED.xml')
 sources=[Path(__file__),P/'analyze_sensory_graph.py',P/'analyze_leg_command_unloading.py',P/'run_neural_rebuild_11_mn_pools.py',P/'run_leg_motor_coverage_audit.py',P/'tools_resolve_muscle_anatomy.py',P/'tools_build_design_runtime.py',P/'data/banc888/banc_888_meta.feather',P/'data/flywire/banc_neurons.csv.gz',P/'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json',bp,mp,xp,S162/'BOUND_INPUT_FREEZE.json',S154/'INPUT_HASHES.json',S162/'SOURCE_FREEZE/PRIVATE_RUNNER_SOURCE.py',S162/'init80ms/CONNECTIVITY.npz',S154/'init80ms/CONNECTIVITY.npz',S154/'init80ms/MOMENT_METADATA.json',S163/'GRAPH_90_MUSCLE_MAPPING.csv',S163/'GRAPH_ALL_NEURONS.csv',S150/'S149_ACTUATOR_MAP.json',Path('/run/media/sensen/Data2/engine修复验收/s4-endpoints/20261023T011559Z/run.py'),Path('/run/media/sensen/Data2/engine修复验收/s2-physical-prewarm/20261023T010000Z/CONFIG.json')]
 for suffix in ['/motor_output.py','/motor_coverage.py','/motor_remap.py','/neural_command.py','/motor_endpoints.py','/command_tier.py','/adapters.py','/neck_cut_data.py','/motor_translation.py']:sources.append(bound(suffix))
 hashes={str(p):sha(p) for p in sources}
 dump(out/'INPUT_HASHES.json',hashes)
 dump(out/'PREREGISTRATION.json',dict(task='S168/P3-S3 bounded missing-readout source audit',scope='only16 rows unmapped in frozen S163; no optimization/model/engine/neural/physics',calls=dict(model_load=0,forward=0,physics_step=0,neural_step=0),identity='root IDs canonical integer strings; dataset global index distinct tier-local index; safe AST identity function only',candidate_rule='raw feather all same peripheral target across legs/sides, plus explicit compound-promotor type for pleural-promotor rows, plus exact/alternative CSV name matches; every route mismatch retained and disqualified',qualification='motor annotation, exact target+leg+side and efferent match, tier presence and actual nonzero incoming independently; annotation is not verified innervation',incoming='frozen CSR row=target,col=source; count data!=0 and distinct source indices, not diff(indptr) blindly; absent tier incoming is NOT_APPLICABLE rather than zero',causes=['ORIGINAL_NO_EXACT_CANDIDATES','DATASET_ID_ABSENT','TIER_EXCLUDED','IDENTITY_MATCH_FAILURE','POOL_OMISSION'],repair='recommend only; no changes to mapping/checkpoint/history',input_hashes=str(out/'INPUT_HASHES.json'),freeze='copy all consumed inputs before analysis; current raw annotation frozen NOW, not pretended to be S154 archival source'))
 freeze=out/'SOURCE_FREEZE';freeze.mkdir();frozen={}
 for i,p in enumerate(sources):
  dst=freeze/f'{i:03d}_{p.name}';shutil.copyfile(p,dst);assert sha(dst)==hashes[str(p)];frozen[str(p)]=str(dst)
 dump(out/'SOURCE_FREEZE.json',dict(files=[dict(source=str(p),frozen=frozen[str(p)],sha256=hashes[str(p)]) for p in sources]))
 def F(p):return Path(frozen[str(p)])
 manifest=json.loads(F(mp).read_text());graphrows=list(csv.DictReader(F(S163/'GRAPH_90_MUSCLE_MAPPING.csv').open()));missing=[r for r in manifest['rows'] if not r['local_indices']]
 assert len(missing)==16 and [r['actuator_index'] for r in missing]==[int(r['actuator_index']) for r in graphrows if int(r['pool_id'])<0]
 raw=pd.read_feather(F(P/'data/banc888/banc_888_meta.feather'));cols=['root_id','banc_888_id','flow','super_class','cell_type','peripheral_target_type','body_part_effector','side','nerve']
 assert all(k in raw for k in cols)
 dump(out/'EARLY_ROOT_SCHEMA.json',dict(raw_rows=len(raw),raw_columns=raw.columns.tolist(),raw_root_dtype=str(raw.root_id.dtype),raw_banc888_dtype=str(raw.banc_888_id.dtype),root_ids_equal_banc888=bool((raw.root_id.astype(str)==raw.banc_888_id.astype(str)).all()),missing_indices=[r['actuator_index'] for r in missing],empty_declared_pools=all(not r['pool_root_ids'] for r in missing),schema_refs=dict(manifest=frozen[str(mp)],raw_feather=frozen[str(P/'data/banc888/banc_888_meta.feather')]),known_provenance='Raw muscle-target fields are NOT present in banc_connectome.npz; independent raw feather/CSV required. Freeze is this audit, not historical raw-source freeze.'))
 a=load(F(bp));gi=safe_function(F(P/'analyze_sensory_graph.py'),'identity',{'np':np})(a);assert len(gi)==17099
 byroot={str(int(r)):i for i,r in enumerate(a['root_ids'])};local={str(int(a['root_ids'][g])):i for i,g in enumerate(gi)}
 assert len(byroot)==len(a['root_ids'])
 historical=list(csv.DictReader(F(S163/'GRAPH_ALL_NEURONS.csv').open()));assert [r['root_id'] for r in historical]==[str(int(r)) for r in a['root_ids'][gi]]
 for r in manifest['rows']:assert [str(int(a['root_ids'][gi[i]])) for i in r['local_indices']]==list(map(str,r['tier_root_ids']))
 c=load(F(S162/'init80ms/CONNECTIVITY.npz'));c154=load(F(S154/'init80ms/CONNECTIVITY.npz'));assert all(np.array_equal(v,c154[k]) for k,v in c.items())
 incoming={};totals={}
 for typ in ['We','Wi']:
  data,idx,ptr=[c[typ+'_'+s] for s in ['data','indices','indptr']]
  assert tuple(c[typ+'_shape'])==(17099,17099) and len(ptr)==17100 and ptr[-1]==len(data)==len(idx)
  assert np.all(np.diff(ptr)>=0) and np.isfinite(data).all()
  incoming[typ]=[(int(np.count_nonzero(data[ptr[i]:ptr[i+1]])),set(map(int,idx[ptr[i]:ptr[i+1]][data[ptr[i]:ptr[i+1]]!=0]))) for i in range(17099)]
  totals[typ]=int(np.count_nonzero(data))
 # Raw dataset edge pairs are separate from the filtered/signed frozen CSR.
 fullnz=(a['syn']!=0)&np.isfinite(a['syn']);full_in=np.bincount(a['post'][fullnz],minlength=len(a['root_ids']))
 with gzip.open(F(P/'data/flywire/banc_neurons.csv.gz'),'rt') as f:csvrows=list(csv.DictReader(f))
 csvby={r['Root ID']:(i+2,r) for i,r in enumerate(csvrows)}
 mm=json.loads(F(S154/'init80ms/MOMENT_METADATA.json').read_text());xml=ET.parse(F(xp)).getroot()
 maps=safe_function(F(P/'analyze_leg_command_unloading.py'),'mapping',{})(xml,mm,['lf','rf','lm','rm','lh','rh']);oldmaps=json.loads(F(S150/'S149_ACTUATOR_MAP.json').read_text());assert maps==oldmaps
 assert [r['name'] for r in maps]==[r['actuator'] for r in manifest['rows']]
 # Historical source ancestry, including unresolved preservation, not a new resolver.
 pools=json.loads(F(P/'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json').read_text())['measured'];unresolved={r['actuator']:r for r in pools['unresolved']}
 config=json.loads(F(Path('/run/media/sensen/Data2/engine修复验收/s2-physical-prewarm/20261023T010000Z/CONFIG.json')).read_text())
 assert all(r['source_annotation']==unresolved[r['actuator']] for r in missing)
 assert all(r['source_annotation']==config['manifest']['rows'][r['actuator_index']]['source_annotation'] for r in missing)
 # S154 and S162 frozen bound inputs refer to equal actual dataset+manifest+XML identities.
 rec154=json.loads(F(S154/'INPUT_HASHES.json').read_text())
 for p in [bp,mp,xp]:
  rec=next(r for r in records if r['frozen']==str(p));assert rec154[rec['source']]==rec['sha256']
 expected_body={'F':'front_leg','M':'middle_leg','H':'hind_leg'};detail=[];candidates=[];rawdetails=[]
 rawstr={k:raw[k].astype(str).to_numpy() for k in cols};rawroot={str(int(r)):i for i,r in enumerate(raw.root_id)}
 # Updated root_id can merge unrelated historical rows; never assume global uniqueness.
 relevant=raw.peripheral_target_type.astype(str).isin({r['muscle_target'] for r in missing})|raw.cell_type.astype(str).str.contains('tergopleural_promotor_pleural_promotor',regex=False)
 assert not raw.loc[relevant,'root_id'].duplicated().any()
 dump(out/'RAW_ROOT_SCHEMA_CAVEAT.json',dict(raw_rows=len(raw),unique_current_root_ids=len(rawroot),duplicate_current_root_rows=int(raw.root_id.duplicated().sum()),muscle_candidate_root_duplicates=0,policy='banc_888_id and current root_id distinct; retain both. Unrelated duplicate current IDs do not license arbitrary muscle joins.'))
 for r in missing:
  j=r['actuator_index'];name=r['actuator'];target=r['muscle_target'];body=expected_body[r['leg'][1]];side=r['side'];ma=maps[j]
  same=rawstr['peripheral_target_type']==target;exact=same&(rawstr['body_part_effector']==body)&(rawstr['side']==side)&(rawstr['flow']=='efferent')
  compound=np.array(['tergopleural_promotor_pleural_promotor' in str(s) for s in rawstr['cell_type']]) if target=='pleural_promotor_muscle' else np.zeros(len(raw),bool)
  indices=set(map(int,np.flatnonzero(same|compound)))
  key=target.removesuffix('_muscle').replace('pleural_remotor_and_abductor','pleural_remotor_abductor')
  csvmatches=[]
  for rid,(line,cr) in csvby.items():
   if key==cr['Primary Cell Type'] or key in cr['Alternative Cell Type(s)'].split(','):
    csvmatches.append(rid)
    if rid in rawroot:indices.add(rawroot[rid])
  own=[]
  for ri in sorted(indices,key=lambda i:int(rawroot_id(raw, i))):
   rid=str(int(raw.root_id.iloc[ri]));g=byroot.get(rid);li=local.get(rid);crline,cr=csvby.get(rid,(None,{}))
   targetmatch=bool(same[ri]);routematch=rawstr['body_part_effector'][ri]==body and rawstr['side'][ri]==side
   motor=rawstr['super_class'][ri]=='motor';eff=rawstr['flow'][ri]=='efferent';fullmotor=g is not None and str(a['ann_super_class'][g])=='motor'
   we,wi=(incoming['We'][li],incoming['Wi'][li]) if li is not None else ((None,set()),(None,set()))
   union=we[1]|wi[1]
   relation='EXACT_TARGET' if targetmatch else 'COMPOUND_PROMOTOR_AMBIGUITY' if compound[ri] else 'CSV_ALTERNATIVE_NAME_CONFLICT'
   qualified=targetmatch and routematch and motor and eff
   row=dict(actuator_index=j,actuator=name,root_id=rid,raw_feather_row=ri,raw_banc_888_id=str(raw.banc_888_id.iloc[ri]),candidate_relation=relation,raw_target=str(rawstr['peripheral_target_type'][ri]),raw_effector=str(rawstr['body_part_effector'][ri]),raw_side=str(rawstr['side'][ri]),raw_flow=str(rawstr['flow'][ri]),raw_super_class=str(rawstr['super_class'][ri]),raw_cell_type=str(rawstr['cell_type'][ri]),raw_nerve=str(rawstr['nerve'][ri]),same_leg_side=bool(routematch),exact_target=targetmatch,independent_motor_annotation=bool(motor),annotation_exact_pool_qualified=bool(qualified),verified_individual_muscle_innervation='NOT_ESTABLISHED',full_dataset_present=g is not None,full_dataset_global_index=g if g is not None else '',full_dataset_motor_annotation=bool(fullmotor),full_dataset_primary_type=str(a['ann_primary_type'][g]) if g is not None else '',tier_present=li is not None,tier_local_index=li if li is not None else '',actual_We_nonzero_incoming=we[0] if li is not None else 'NOT_APPLICABLE_OUTSIDE_TIER',actual_Wi_nonzero_incoming=wi[0] if li is not None else 'NOT_APPLICABLE_OUTSIDE_TIER',actual_unique_incoming_sources=len(union) if li is not None else 'NOT_APPLICABLE_OUTSIDE_TIER',actual_incoming_nonzero=bool(union) if li is not None else 'NOT_APPLICABLE_OUTSIDE_TIER',full_raw_edge_nonzero_incoming_pairs=int(full_in[g]) if g is not None else 'NOT_IN_DATASET',csv_logical_line=crline or '',csv_primary_type=cr.get('Primary Cell Type',''),csv_alternative_types=cr.get('Alternative Cell Type(s)',''),csv_nerve=cr.get('Nerve',''),source_raw=f'{frozen[str(P/"data/banc888/banc_888_meta.feather")]}:row[{ri}]',source_csv=f'{frozen[str(P/"data/flywire/banc_neurons.csv.gz")]}:logical_line[{crline}]',source_connectivity=f'{frozen[str(S162/"init80ms/CONNECTIVITY.npz")]}:We/Wi CSR row[{li}]',disqualifier='none_at_annotation_level_not_verified_innervation' if qualified else 'wrong_leg_or_side' if not routematch else relation)
   candidates.append(row);own.append(row)
   rawdetails.append(dict(actuator_index=j,root_id=rid,raw_fields={k:str(rawstr[k][ri]) for k in cols},csv_fields=cr,csv_logical_line=crline))
  exactids=[str(int(raw.root_id.iloc[i])) for i in np.flatnonzero(exact)]
  # Distinguish original no candidates from dataset absence, selector exclusion and omission.
  if not exactids:cause='ORIGINAL_NO_EXACT_CANDIDATES'
  elif any(x not in byroot for x in exactids):cause='DATASET_ID_ABSENT'
  elif any(x not in local for x in exactids):cause='TIER_EXCLUDED'
  elif not r['pool_root_ids']:cause='POOL_OMISSION'
  else:cause='IDENTITY_MATCH_FAILURE'
  record=dict(actuator_index=j,actuator=name,leg=r['leg'],side=side,muscle_target=target,manifest_status=r['status'],declared_pool_root_ids=json.dumps(r['pool_root_ids']),original_reason=r['source_annotation'].get('reason'),actual_exact_raw_candidates=len(exactids),actual_exact_raw_root_ids=';'.join(exactids),same_target_any_leg_side_count=int(same.sum()),candidate_rows=len(own),same_leg_side_ambiguous_root_ids=';'.join(x['root_id'] for x in own if x['same_leg_side'] and not x['exact_target']),same_leg_side_ambiguous_tier_nonzero_incoming_roots=';'.join(x['root_id'] for x in own if x['same_leg_side'] and not x['exact_target'] and x['tier_present'] and x['actual_incoming_nonzero']),cause=cause,annotation_cause_detail='NO_TARGET_VOCABULARY' if not same.any() else 'TARGET_EXISTS_OTHER_LEG_OR_SIDE',identity_match_failure=False,pool_omission=False,tier_filter_cause=False,original_declared_pool_empty=True,model_tendon=ma['tendon'],model_wrap_type=ma['wrap_type'],crossed_joints=';'.join(ma['referenced_crossed_joints']),dofs=';'.join(map(str,ma['dofs'])),joint_derived_leg=ma['leg'],functional_role=ma['functional_role'],minimal_verified_repair='NONE_ESTABLISHED',recommendation='ONLY_ANNOTATION_HYPOTHESIS_REVIEW_NOT_IMPLEMENT' if j in (3,18) else 'RETAIN_UNMAPPED_NEED_NEW_SOURCED_ANATOMY',source_manifest=frozen[str(mp)],source_model=frozen[str(xp)],source_joint_metadata=frozen[str(S154/'init80ms/MOMENT_METADATA.json')],source_original_resolver=frozen[str(P/'run_neural_rebuild_11_mn_pools.py')]+':119-165',source_manifest_builder=str(bound('/motor_output.py'))+':29-56')
  detail.append(record)
 table(out/'MISSING_16_DETAILED.csv',detail);table(out/'CANDIDATE_ROWS.csv',candidates);dump(out/'CANDIDATE_RAW_CROSSCHECK.json',rawdetails);dump(out/'MODEL_TENDON_MAPPING.json',maps)
 # Preserve full textual source with original line numbers, including actual model XML.
 text=[]
 for p in sources:
  if p.suffix in ('.py','.xml'):
   text += ['\n### '+str(p)+'\nFrozen: '+frozen[str(p)]+'\nSHA256: '+hashes[str(p)]]+[f'{i}: {s}' for i,s in enumerate(F(p).read_text().splitlines(),1)]
 (out/'SOURCE_LINE_REFERENCES.txt').write_text('\n'.join(text),encoding='utf-8')
 summary=dict(status='COMPLETE_SAVED_ONLY_AUDIT',task='S168/P3-S3',missing_rows=len(detail),candidate_rows=len(candidates),unique_candidate_roots=len({r['root_id'] for r in candidates}),causes=dict(Counter(r['cause'] for r in detail)),candidate_scope='all exact target anywhere, compound annotation and CSV lexical conflicts, never verified innervation',dataset_neurons=len(a['root_ids']),tier_neurons=len(gi),csr_actual_nonzero_entries=totals,csr_same_S154_S162=True,model_mapping_crosscheck_S150=True,all_missing_declared_pool_empty=True,verified_minimal_repair_slots=[],hypothesis_review_only_slots=[3,18],calls=dict(engine_import=0,model_load=0,forward=0,physics_step=0,neural_step=0,optimization=0),uncertainties=['raw feather and CSV frozen at S168; source annotations match historical unresolved records but their historical full raw-file hash not established here','motor annotation and incoming connectivity do not verify innervation of each model muscle','same-target other-leg or contralateral cells are not valid replacement sources','compound primary/alternative names cannot establish independent Miller28/30 recruitment or justify three-way sharing','full raw edge pairs are not signed filtered CSR; outside-tier actual incoming is not measured and is not zero','No physiological sign, recruitment, SI force units, or behavioral sufficiency established'])
 dump(out/'SUMMARY.json',summary)
 report=['# S168 / P3-S3 缺失读出来源审计（仅保存证据）','## 结论','16个槽位均在原始精确(target, effector, side, efferent)连接前没有候选，且历史declared pool为空；不是16个已知运动池被17099 tier裁掉。独立原始数据重连与历史source_annotation逐字一致。未发现可确证的identity match failure或pool omission；不据此断言动物没有这些肌肉运动神经元。','所有model/load/forward/physics/neural/optimization调用为0，未导入engine，没有改活动mapping/checkpoint/历史。本产物为来源审计不是物理或神经实验。','## 两个前腿候选不能漏报，也不能升级为已证修复','slot3 LFC_pleural_promotor和slot18 RFC_pleural_promotor可在compound cell_type（tergopleural_promotor_pleural_promotor / Miller28_30）各找到4个同侧前腿注释候选。真实peripheral_target_type仍是tergopleural_promotor_muscle，CSV primary是tergopleural_promotor，compound为alternative。motor身份、tier存在与实际incoming见候选CSV独立列；即使均通过，也不证明各细胞支配独立pleural模型肌肉。','历史run_leg_motor_coverage_audit.py:53-87把此列作两槽共享池提案，其meaning字段将注释解释为BOTH过强；tools_resolve_muscle_anatomy.py:32-51明确REJECT_SHARED_MILLER28_30_POOL，tools_build_design_runtime.py:146-156也保留pleural为空。两份历史工程政策不是新的解剖事实。最多建议仅对3/18做独立有来源解剖核验并预登记共享池假设，不推荐现在修改。没有evidence-backed verified minimal repair slots。','其他14槽位不跨腿/跨侧补源。36的目标仅右中腿及其他腿有标签；47的精确目标在左中腿等有，CSV右中腿femur_reductor的alternative虽提pleural_remotor_abductor，属命名冲突而非已证身份修复。肌腱路径derived joint leg独立于命名匹配，功能role保留UNKNOWN。','## 输入/来源/计数',json.dumps(summary,ensure_ascii=False,indent=2),'原始feather含peripheral_target_type/body_part_effector/side；153962节点缓存不含这些字段，不能用cache motor身份推导肌肉。每个raw candidate保留row index、banc_888_id、root_id、CSV逻辑行、全图global/tierlocal与实际CSR非零incoming（exc/inh及unique源）。原始full非零入边数与实际strict filtered tier CSR分列，未混用。','SOURCE_FREEZE.json与INPUT_HASHES在解析/分析前写入；所有使用输入完整复制并校验SHA，当前原始注释仅标S168冻结，不伪称S154当时原始全表hash已证。S154/S162实际bound dataset/manifest/XML hash相等；S4 run.py:37-38从旧CONFIG source_annotation重建，motor_output.py:29-40明确原unresolved保留空池，run_neural_rebuild_11_mn_pools.py:119-165原resolver只用efferent target+leg+side，未motor独立筛选，故本审计另查motor而非假定。','## 文件与行引用','所有输入原路径/冻结路径/SHA见SOURCE_FREEZE.json，源码及实际XML完整逐行引用见SOURCE_LINE_REFERENCES.txt。MISSING_16_DETAILED.csv逐16槽；CANDIDATE_ROWS.csv逐候选；CANDIDATE_RAW_CROSSCHECK.json保留原始注释；MODEL_TENDON_MAPPING.json由实际XML site祖先路径/关节引用与保存MOMENT_METADATA复算，并与S150全部90行完全相同。','未建立：细胞到独立肌肉终末的生物验证、Miller28/30拆分或共享招募、神经功能可达/输出、站立/行走、物理单位。NOT_ESTABLISHED。']
 (out/'CONCLUSION.zh-CN.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
 after={str(p):sha(p) for p in sources};assert hashes==after;dump(out/'INPUT_HASHES_AFTER.json',after)
 dump(out/'OUTPUT_HASHES.json',{str(p):sha(p) for p in out.iterdir() if p.is_file()})
 print(str(out));print(json.dumps(summary,ensure_ascii=False,indent=2))
def rawroot_id(raw,i):return str(int(raw.root_id.iloc[i]))
if __name__=='__main__':main()
