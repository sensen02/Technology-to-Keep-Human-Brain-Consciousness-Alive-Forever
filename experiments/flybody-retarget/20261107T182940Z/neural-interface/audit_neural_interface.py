"""Read-only anatomy audit. Writes only beside this script; no model or dynamics imports."""
from pathlib import Path
import csv, gzip, hashlib, json, platform, resource
from collections import Counter
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
OUT=Path(__file__).resolve().parent
DESKTOP=Path('/home/sensen/Desktop/engine修复验收/flybody-retarget/20261107T182940Z/neural-interface')
ASC=('ascending','sensory_ascending','ascending_visceral_circulatory')
DESC=('descending','sensory_descending')
BRAIN=('central_brain_intrinsic','optic_lobe_intrinsic','visual_projection','visual_centrifugal')
LEG={'lf':'left_prothoracic_leg_nerve','lm':'left_mesothoracic_leg_nerve','lh':'left_metathoracic_leg_nerve','rf':'right_prothoracic_leg_nerve','rm':'right_mesothoracic_leg_nerve','rh':'right_metathoracic_leg_nerve'}
CHANNEL={'claw_position':'claw_chordotonal_organ_neuron','hook_direction':'hook_chordotonal_organ_neuron','hairplate_angle':'hair_plate_neuron','campaniform_strain':'campaniform_sensillum_neuron','club_vibration':'club_chordotonal_organ_neuron','bristle_touch':'bristle_neuron','multidendritic_noxious':'multidendritic_neuron'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def info(p):
 p=Path(p);return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def dump(name,d):
 (OUT/name).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def writecsv(name,rows,fields):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def counts(v):return {str(k):int(n) for k,n in sorted(Counter(v).items())}
def tokens(v,terms):return np.array([bool(set(str(x).split(','))&set(terms)) for x in v])

def main():
 sources={k:ROOT/v for k,v in {
 'cache':'data/flywire/banc_connectome.npz','raw_neurons':'data/flywire/banc_neurons.csv.gz','raw_connections':'data/flywire/banc_connections_princeton.csv.gz',
 'builder':'build_banc_cns.py','neck_data':'engine/neck_cut_data.py','adapters':'engine/embodied/adapters.py','access_map':'engine/embodied/access_map.py',
 'feather':'data/banc888/banc_888_meta.feather','association':'data/banc888/b888_to_root.npy','supplement':'data/banc_dataverse/supplemental_data_2.txt',
 'cell_info':'data/banc_dataverse/cell_info.parquet','codex':'data/banc_dataverse/codex_annotations.parquet','somas':'data/banc/somas_v1.parquet',
 'sensory_lookup':'outputs/neural_rebuild/sensory_channel_lookup.json','sensory_exporter':'tools_export_sensory_channel_lookup.py',
 'motor_pools':'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json','motor_crosswalk':'tools_p3_motor_version_crosswalk.py',
 'candidate_exporter':'tools/export_banc888_interface_candidates.py'}.items()}
 source_info={k:info(p) for k,p in sources.items()};source_info['audit_script']=info(__file__)
 a=dict(np.load(sources['cache'],allow_pickle=False));rid=a['root_ids'];n=len(rid);pre=a['pre'];post=a['post'];syn=a['syn'];E=len(pre)
 assert n==153962 and E==3037361
 assert np.all(rid[1:]>rid[:-1]) and pre.min()>=0 and post.min()>=0 and max(pre.max(),post.max())<n
 keys=pre*n+post;assert np.all(keys[1:]>keys[:-1]);assert np.all(np.isfinite(syn)) and np.all(syn>0) and np.all(syn==np.floor(syn))
 idx={str(v):i for i,v in enumerate(rid)};sc=a['ann_super_class'];cl=a['ann_class'];nerve=a['ann_nerve'];body=a['ann_body_part'];fun=a['ann_function'];nt=a['ann_nt_verified'];typ=a['ann_primary_type']
 known=np.isin(nt,['acetylcholine','gaba','glutamate'])
 masks={};definitions={}
 def group(name,mask,rule,source='cache'):
  assert np.asarray(mask).shape==(n,);masks[name]=np.asarray(mask,dtype=bool);definitions[name]={'source':source,'rule':rule}
 for label in np.unique(sc):group('super_class__'+(label or 'EMPTY'),sc==label,{'ann_super_class_equals':str(label)})
 group('ascending_boundary',np.isin(sc,ASC),{'ann_super_class_in':ASC})
 group('descending_boundary',np.isin(sc,DESC),{'ann_super_class_in':DESC})
 group('neck_crossing_union',np.isin(sc,ASC+DESC),{'ann_super_class_in':ASC+DESC})
 group('brain_intrinsic_and_visual',np.isin(sc,BRAIN),{'ann_super_class_in':BRAIN})
 group('vnc_intrinsic',sc=='ventral_nerve_cord_intrinsic',{'ann_super_class_equals':'ventral_nerve_cord_intrinsic'})
 group('class_only_ascending_disagreement',(cl=='ascending_neuron')&(sc!='ascending'),{'class':'ascending_neuron','super_class_not':'ascending'})
 group('class_only_descending_disagreement',(cl=='descending_neuron')&(sc!='descending'),{'class':'descending_neuron','super_class_not':'descending'})
 sensory=np.isin(sc,['sensory','sensory_ascending','sensory_descending'])
 legall=np.zeros(n,bool);motorall=np.zeros(n,bool)
 for leg,term in LEG.items():
  legnerve=tokens(nerve,[term]);legall|=sensory&legnerve;motorall|=(sc=='motor')&legnerve
  group('leg_sensory_'+leg,sensory&legnerve,{'super_class_in':['sensory','sensory_ascending','sensory_descending'],'ann_nerve_contains_token':term})
  group('leg_motor_nerve_'+leg,(sc=='motor')&legnerve,{'super_class':'motor','ann_nerve_contains_token':term})
 group('leg_sensory_nerve_union',legall,{'union':['leg_sensory_'+x for x in LEG]})
 group('leg_motor_nerve_union',motorall,{'union':['leg_motor_nerve_'+x for x in LEG]})
 group('leg_motor_class',cl=='leg_motor_neuron',{'ann_class_equals':'leg_motor_neuron'})
 for label in ['proboscis_motor_neuron','pharynx_motor_neuron','crop_motor_neuron','taste_bristle_gustatory_neuron','pars_intercerebralis_neurosecretory_cell','pars_lateralis_neurosecretory_cell','subesophageal_zone_neurosecretory_cell','ventral_nerve_cord_neurosecretory_cell']:
  group('class__'+label,cl==label,{'ann_class_equals':label})
 for name,terms in {'mouth_sensory':['labellum','proboscis','pharynx','haustellum','labral_sense_organ_or_fishtrap_bristles','labral_or_ventral_cibarial_sense_organ','dorsal_cibarial_sense_organ'], 'digestive_body_labels':['anterior_digestive_tract','digestive_tract','crop'], 'hemolymph_body_labels':['hemolymph','hemolymph_sensory']}.items():
  m=tokens(body,terms)&sensory if name=='mouth_sensory' else tokens(body,terms)
  group(name,m,{'body_part_any_token':terms,'sensory_super_class_required':name=='mouth_sensory'})
 for token in ['feeding','feeding_suppression','interoceptive','hemolymph_sugar','sugar','water','pharyngeal_pumping','diuresis','oxygenation']:
  group('function_token__'+token,tokens(fun,[token]),{'ann_function_exact_token':token})
 # Read source neurons as strings to preserve 18-digit IDs and all raw labels.
 raw=pd.read_csv(sources['raw_neurons'],dtype=str,keep_default_na=False)
 assert not raw['Root ID'].duplicated().any();ri=raw.set_index('Root ID');aligned=ri.reindex([str(x) for x in rid]);assert aligned.index.is_unique
 fields={'Flow':'ann_flow','Super Class':'ann_super_class','Class':'ann_class','Nerve':'ann_nerve','Body Part':'ann_body_part','Function':'ann_function','Primary Cell Type':'ann_primary_type','Verified NT type':'ann_nt_verified'}
 raw_equal={k:bool(np.array_equal(aligned[k].fillna('').to_numpy(),a[v])) for k,v in fields.items()};assert all(raw_equal.values())
 coverage={}
 def coverage_record(name,ids):
  vals=[str(x) for x in ids if str(x) not in ('','0','nan','None','NA')];unique=set(vals);matched=unique&set(idx);missing=unique-set(idx);cache_missing=set(idx)-unique
  rec={'source_rows_with_id':len(vals),'unique_ids':len(unique),'duplicate_id_rows':len(vals)-len(unique),'unique_in_cache':len(matched),'unique_not_in_cache':len(missing),'cache_ids_without_source':len(cache_missing)}
  fn='mapping_'+name+'.csv.gz'
  with gzip.open(OUT/fn,'wt',newline='') as f:
   multiplicity=Counter(vals)
   w=csv.writer(f);w.writerow(['source_id_string','cache_index_0based','source_row_count']);w.writerows((r,idx.get(r,-1),multiplicity[r]) for r in sorted(unique,key=int))
  rec['id_mapping_file']=info(OUT/fn);coverage[name]=rec;return rec
 coverage_record('raw_neurons',raw['Root ID'])
 cols=['banc_888_id','root_id','root_626','root_850','root_888','root_890','side','flow','super_class','cell_class','cell_sub_class','cell_type','body_part_sensory','body_part_effector','peripheral_target_type','cell_function','cell_function_detailed','region','nerve','tract','neurotransmitter_verified','neurotransmitter_predicted','neurotransmitter_score','neuropeptide_verified']
 feather=pd.read_feather(sources['feather'],columns=cols).fillna('')
 for k in cols[:6]:coverage_record('feather_'+k,feather[k])
 assert feather.root_888.is_unique and set(map(str,rid))<=set(feather.root_888)
 version_row_disagreements=feather.loc[feather.root_888!=feather.banc_888_id,['banc_888_id','root_888','root_id','cell_type','cell_class','super_class']].to_dict('records')
 for row in version_row_disagreements:
  row['root888_in_cache']=row['root_888'] in idx;row['banc888_in_cache']=row['banc_888_id'] in idx
 fi=feather.set_index('root_888').reindex([str(x) for x in rid]);assert not fi.index.duplicated().any()
 comparisons={}
 for old,new in [('ann_super_class','super_class'),('ann_class','cell_class'),('ann_nerve','nerve'),('ann_primary_type','cell_type'),('ann_nt_verified','neurotransmitter_verified')]:
  mismatch=a[old]!=fi[new].to_numpy();comparisons[old+'_vs_'+new]={'different_count':int(mismatch.sum())}
  group('feather_disagreement__'+old,mismatch,{'cache_column':old,'root_888_joined_feather_column':new},'cache+feather')
 # Export exact sensory candidate rows, retaining legacy duplicate associations.
 sensory_rows=[];sensory_old={};sensory_new={}
 for rowno,row in feather.iterrows():
  sub=row.cell_sub_class;leg=None
  for bp,code in [('front_leg','f'),('middle_leg','m'),('hind_leg','h')]:
   if sub.startswith(bp) and row.side in ('left','right'):leg=row.side[0]+code;break
  channel=next((k for k,v in CHANNEL.items() if v in sub),None)
  if leg is None or channel is None:continue
  old=str(row.root_id);new=str(row.root_888);sensory_old[old]=[leg,channel];sensory_new[new]=[leg,channel]
  sensory_rows.append({'feather_row_0based':rowno,'root_id_legacy':old,'root_888':new,'cache_index_0based':idx.get(new,-1),'legacy_cache_index_0based':idx.get(old,-1),'leg':leg,'channel':channel,'cell_sub_class':sub,'cache_class':cl[idx[new]] if new in idx else '', 'cache_super_class':sc[idx[new]] if new in idx else ''})
 lookup=json.loads(sources['sensory_lookup'].read_text());assert sensory_old==lookup['leg_channel_by_root']
 writecsv('SENSORY_CHANNEL_CANDIDATES.csv',sensory_rows,list(sensory_rows[0]))
 channel_summary={}
 for leg in LEG:
  for channel in CHANNEL:
   rows=[r for r in sensory_rows if r['leg']==leg and r['channel']==channel];ids={r['root_888'] for r in rows};m=np.array([str(r) in ids for r in rid])
   name='sensory_channel__'+leg+'__'+channel
   group(name,m,{'feather_cell_sub_class_contains':CHANNEL[channel],'body_prefix_and_side_to_leg':leg,'identity_column':'root_888'},'feather')
   channel_summary[leg+'/'+channel]={'source_rows':len(rows),'unique_ids':len(ids),'in_cache':int(m.sum()),'legacy_in_cache':sum(r['legacy_cache_index_0based']>=0 for r in rows)}
 motor=json.loads(sources['motor_pools'].read_text());motor_rows=[];motor_ids_old=set();motor_ids_new=set()
 for pool in motor['measured']['resolved']:
  bp={'F':'front_leg','M':'middle_leg','H':'hind_leg'}[pool['leg'][1]]
  sub=feather[(feather.flow=='efferent')&(feather.peripheral_target_type==pool['muscle_target'])&(feather.body_part_effector==bp)&(feather.side==pool['side'])]
  assert set(map(str,pool['pool_root_ids']))==set(sub.root_id)
  for j,row in sub.iterrows():
   motor_ids_old.add(row.root_id);motor_ids_new.add(row.root_888)
   motor_rows.append({'actuator_legacy_90':pool['actuator'],'leg':pool['leg'],'muscle_target':pool['muscle_target'],'feather_row_0based':j,'root_id_legacy':row.root_id,'root_888':row.root_888,'cache_index_0based':idx.get(row.root_888,-1),'legacy_cache_index_0based':idx.get(row.root_id,-1),'cell_type':row.cell_type})
 writecsv('MOTOR_TARGET_CANDIDATES.csv',motor_rows,list(motor_rows[0]));dump('MOTOR_UNRESOLVED_SOURCE_RECORDS.json',motor['measured']['unresolved'])
 # All cache nodes retain both annotation vocabularies and exact row identities.
 identities=pd.DataFrame({'cache_index_0based':np.arange(n),'root_id_string':[str(x) for x in rid]})
 for k in ['ann_flow','ann_super_class','ann_class','ann_nerve','ann_body_part','ann_function','ann_primary_type','ann_nt_verified']:identities[k]=a[k]
 identities['raw_neurons_row_0based']=raw.set_index('Root ID').assign(rowno=np.arange(len(raw))).reindex([str(r) for r in rid])['rowno'].fillna(-1).astype(int).to_numpy()
 identities['raw_sub_class']=aligned['Sub Class'].fillna('').to_numpy();identities['raw_verified_neuropeptide']=aligned['Verified Neuropeptide'].fillna('').to_numpy()
 for k in ['root_id','banc_888_id','side','super_class','cell_class','nerve','neurotransmitter_verified','cell_sub_class','cell_type','body_part_sensory','body_part_effector','peripheral_target_type','cell_function','cell_function_detailed','region','tract','neurotransmitter_predicted','neurotransmitter_score','neuropeptide_verified']:identities['feather_'+k]=fi[k].to_numpy()
 identities.to_csv(OUT/'NODE_IDENTITIES.csv.gz',index=False)
 # Concrete local alternates: ID coverage and label inventories, no float-ID coercion.
 supplement=pd.read_csv(sources['supplement'],dtype=str,keep_default_na=False)
 coverage_record('supplement',supplement.root_id)
 supplement_identity_equal=set(supplement.root_id)==set(feather.root_888)
 del supplement
 alternate={}
 for name in ['cell_info','codex','somas']:
  pf=pq.ParquetFile(sources[name]);ids=[];labcounts=Counter();validrows=0
  columns=['pt_root_id','valid','tag','tag2'] if name=='cell_info' else ['pt_root_id','valid','valid_ref','classification_system','cell_type'] if name=='codex' else ['pt_root_id','valid']
  for batch in pf.iter_batches(batch_size=100000,columns=columns):
   dd=batch.to_pydict();ids.extend(dd['pt_root_id']);validrows+=sum(bool(v) for v in dd['valid'])
   if name=='cell_info':labcounts.update((str(x),str(y)) for x,y in zip(dd['tag2'],dd['tag']))
   elif name=='codex':labcounts.update((str(x),str(y)) for x,y in zip(dd['classification_system'],dd['cell_type']))
  coverage_record(name,ids)
  interesting=[{'field':k[0],'label':k[1],'rows':int(v)} for k,v in sorted(labcounts.items()) if any(t in (k[0]+' '+k[1]).lower() for t in ['descending','ascending','feeding','probosc','pharyn','crop','hemolymph','interocept','hunger','satiety','insulin'])]
  alternate[name]={'rows':pf.metadata.num_rows,'valid_rows':validrows,'interesting_label_counts':interesting,'note':'Coverage is exact ID overlap; repeated annotation rows do not imply duplicate neurons. No geometry or polarity inferred.'}
 # Raw pair streaming reaggregation: O(E+chunk), never dense N by N.
 accum=np.zeros(E,np.int64);rawrows=0;rawmin=None;rawnt=Counter();neuropils=Counter()
 for ch in pd.read_csv(sources['raw_connections'],chunksize=150000,dtype={'pre_root_id':np.int64,'post_root_id':np.int64,'neuropil':str,'syn_count':np.int64,'nt_type':str},keep_default_na=False):
  rawrows+=len(ch);rr=ch.pre_root_id.to_numpy();tt=ch.post_root_id.to_numpy();pp=np.searchsorted(rid,rr);qq=np.searchsorted(rid,tt)
  assert np.all(pp<n) and np.all(qq<n) and np.array_equal(rid[pp],rr) and np.array_equal(rid[qq],tt)
  kk=pp*n+qq;ee=np.searchsorted(keys,kk);assert np.all(ee<E) and np.array_equal(keys[ee],kk)
  ss=ch.syn_count.to_numpy();assert np.all(ss>0);np.add.at(accum,ee,ss)
  rawmin=int(ss.min()) if rawmin is None else min(rawmin,int(ss.min()));rawnt.update(ch.nt_type);neuropils.update(ch.neuropil)
 assert np.array_equal(accum,syn.astype(np.int64));assert rawrows==int(a['n_rows'][0]);del accum
 # Complete mutually exclusive source/target super-class directional matrix.
 labels,code=np.unique(sc,return_inverse=True);nc=len(labels);paircode=code[pre]*nc+code[post]
 countmat=np.bincount(paircode,minlength=nc*nc);synmat=np.bincount(paircode,weights=syn.astype(np.float64),minlength=nc*nc)
 directed=[]
 for i,src in enumerate(labels):
  for j,dst in enumerate(labels):
   directed.append({'source_super_class':str(src),'target_super_class':str(dst),'cache_edge_rows':int(countmat[i*nc+j]),'synapse_count_sum':int(synmat[i*nc+j])})
 writecsv('DIRECTED_SUPERCLASS_EDGE_COUNTS.csv',directed,list(directed[0]))
 assert sum(x['cache_edge_rows'] for x in directed)==E and sum(x['synapse_count_sum'] for x in directed)==int(syn.sum(dtype=np.float64))
 # Exact row selectors for the principal anatomical interface routes.
 routes={};route_info={}
 route_pairs=[('brain_intrinsic_and_visual','descending_boundary'),('descending_boundary','vnc_intrinsic'),('vnc_intrinsic','ascending_boundary'),('ascending_boundary','brain_intrinsic_and_visual'),('leg_sensory_nerve_union','vnc_intrinsic'),('vnc_intrinsic','leg_motor_class'),('brain_intrinsic_and_visual','vnc_intrinsic'),('vnc_intrinsic','brain_intrinsic_and_visual'),('mouth_sensory','brain_intrinsic_and_visual'),('brain_intrinsic_and_visual','class__proboscis_motor_neuron'),('brain_intrinsic_and_visual','class__pharynx_motor_neuron')]
 for src,dst in route_pairs:
  name=src+'__TO__'+dst;mask=masks[src][pre]&masks[dst][post];rows=np.flatnonzero(mask);routes[name]=rows
  route_info[name]={'source_group':src,'target_group':dst,'edge_rows':len(rows),'synapse_sum':int(syn[rows].sum(dtype=np.float64)),'strict_label_edge_rows':int(known[pre[rows]].sum()),'selector_array_key':name}
 np.savez_compressed(OUT/'DIRECTED_ROUTE_EDGE_ROWS.npz',**routes)
 # Candidate: reserve real sensory/MN ports and every annotated neck boundary cell.
 # Only add actual VNC intrinsic cells. Ranking is structural, never a functional claim.
 seed=masks['neck_crossing_union']|legall|masks['leg_motor_class']
 reserved=np.flatnonzero(seed);cap=20000;assert len(reserved)<=cap
 score=np.bincount(post[seed[pre]],weights=syn[seed[pre]],minlength=n)+np.bincount(pre[seed[post]],weights=syn[seed[post]],minlength=n)
 candidates=np.flatnonzero(masks['vnc_intrinsic']&~seed);order=np.lexsort((candidates,-score[candidates]));selected=np.sort(np.r_[reserved,candidates[order[:cap-len(reserved)]]]);local=np.full(n,-1,np.int32);local[selected]=np.arange(len(selected),dtype=np.int32)
 assert len(selected)<=cap and np.all(local[selected]>=0);em=(local[pre]>=0)&(local[post]>=0);erows=np.flatnonzero(em);strict=known[pre[erows]]
 assert np.all((local[pre[erows]]>=0)&(local[post[erows]]>=0))
 np.savez_compressed(OUT/'LOCAL_MODULE_CANDIDATE.npz',root_ids=rid[selected],cache_indices=selected,source_edge_rows=erows,pre_local=local[pre[erows]],post_local=local[post[erows]],synapse_counts=syn[erows],pure_verified_label_mask=strict,source_verified_nt=nt[pre[erows]],strict_source_edge_rows=erows[strict])
 group('local_module_selected',np.isin(np.arange(n),selected),{'reserved_groups':['neck_crossing_union','leg_sensory_nerve_union','leg_motor_class'],'fill':'VNC intrinsic ranked by sum of incident synapses to reserved ports, descending; tie cache index ascending','cap':cap})
 np.savez_compressed(OUT/'GROUP_CACHE_INDICES.npz',**{k:np.flatnonzero(v) for k,v in masks.items()})
 groups={}
 for k,m in masks.items():
  groups[k]={**definitions[k],'cache_count':int(m.sum()),'index_array_key':k,'index_file':'GROUP_CACHE_INDICES.npz','ids_resolve_as':'cache.root_ids[index]; decimal strings also in NODE_IDENTITIES.csv.gz','pure_verified_nt_cells':int((m&known).sum()),'nt_verified_counts':counts(nt[m]),'selected_in_local_module':int(m[selected].sum())}
 disagreements=identities[masks['class_only_ascending_disagreement']|masks['class_only_descending_disagreement']].to_dict('records')
 # Readout endpoints uniquely labelled to one leg only; never infer an MN-muscle map from the 42 joints.
 motor_port_rows=[]
 for i in np.flatnonzero(masks['leg_motor_class']):
  legs=[leg for leg,term in LEG.items() if term in set(nerve[i].split(','))]
  motor_port_rows.append({'root_id_string':str(rid[i]),'cache_index_0based':int(i),'local_index_0based':int(local[i]),'nerve_matched_legs':','.join(legs),'readout_eligible_single_leg':len(legs)==1,'feather_side':str(fi.iloc[i]['side']),'muscle_target':str(fi.iloc[i]['peripheral_target_type']),'cell_type':str(typ[i])})
 writecsv('LOCAL_MOTOR_READOUT_PORTS.csv',motor_port_rows,list(motor_port_rows[0]))
 motor_ports=Counter(r['nerve_matched_legs'] for r in motor_port_rows if r['readout_eligible_single_leg'])
 # Hash all inputs again: content was not modified during audit.
 assert all(sha(sources[k])==v['sha256'] for k,v in source_info.items() if k!='audit_script')
 artifacts={p.name:info(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['NEURAL_INTERFACE_MANIFEST.json','VERIFICATION.json','REPORT.zh-CN.md','verify_neural_interface.py']}
 result={
 'schema':'neural-interface-candidate/1','status':'CANDIDATE_SOURCE_AUDITED_NOT_TRAINED_NOT_RUNTIME_ADMITTED','task_relation':'S3/S4/S5 anatomy interface for approved locomotion/feeding/choice platform',
 'scope':{'simulations':0,'neural_steps':0,'training_updates':0,'lesion_experiments':0,'core_or_tracker_edits':0,'external_research':0,'output_root':str(OUT),'desktop_mirror_root':str(DESKTOP)},
 'requested_absolute_cache_path':{'path':'/data/flywire/banc_connectome.npz','exists':Path('/data/flywire/banc_connectome.npz').exists(),'actual_source':str(sources['cache'])},
 'sources':source_info,'artifacts':artifacts,
 'cache':{'nodes':n,'aggregated_directed_pairs':E,'raw_rows':rawrows,'raw_row_synapse_min':rawmin,'pair_synapse_min':float(syn.min()),'pair_synapse_max':float(syn.max()),'total_synapses':int(syn.sum(dtype=np.float64)),'nt_pair':counts(a['nt_pair']),'nt_neuron':counts(a['nt_neuron']),'verified_nt':counts(nt),'raw_nt_type':dict(rawnt),'raw_neuropil_row_counts':dict(neuropils),'threshold_statement':'Observed raw neuropil-row minimum1 and aggregated-pair minimum3; builder does not impose a threshold. Export filtering history remains unknown; absent pairs are not proven biologically absent.','raw_pairs_reaggregation_exact':True,'raw_annotation_alignment_exact':raw_equal,'identity_indexing':'root_ids sorted int64; pre/post zero-based into root_ids; external JSON/CSV IDs are decimal strings, not float'},
 'coverage':coverage,'feather_banc888_vs_root888_row_disagreements':version_row_disagreements,'feather_annotation_comparison':comparisons,'supplement_id_set_equals_full_feather':supplement_identity_equal,'alternate_tables':alternate,
 'groups':groups,'crossing':{'ascending':int(masks['ascending_boundary'].sum()),'descending':int(masks['descending_boundary'].sum()),'union':int(masks['neck_crossing_union'].sum()),'class_ascending':int((cl=='ascending_neuron').sum()),'class_descending':int((cl=='descending_neuron').sum()),'disagreement_records':disagreements,'meaning':'Super-class definition preserved; class contradictions exported, not reconciled. No independent single-neuron polarity verification; never relabel every crossing neuron descending.'},
 'sensory_channels':{'source_rows':len(sensory_rows),'legacy_unique':len(sensory_old),'root888_unique':len(sensory_new),'legacy_in_cache':len(set(sensory_old)&set(idx)),'root888_in_cache':len(set(sensory_new)&set(idx)),'by_leg_channel':channel_summary,'candidate_file':'SENSORY_CHANNEL_CANDIDATES.csv','limitations':['Source labels identify populations, not per-cell tuning or measured kinetics.','Channel substring rules inherited and explicit; bristle substring may overlap gustatory/tactile subtypes, inspect exported subclass before injection.','Nociceptive and load signals require actual physical sensing/encoding; no afferents or nerves are invented.']},
 'motor_crosswalk':{'resolved_legacy_actuator_pools':len(motor['measured']['resolved']),'unresolved_legacy_actuator_pools':len(motor['measured']['unresolved']),'legacy_unique_roots':len(motor_ids_old),'root888_unique_roots':len(motor_ids_new),'legacy_in_cache':len(motor_ids_old&set(idx)),'root888_in_cache':len(motor_ids_new&set(idx)),'candidate_file':'MOTOR_TARGET_CANDIDATES.csv','all_targets_are_42_joint_mapping':False,'limitations':['Same-row version association is not split/merge validation.','Legacy muscle-target pools are for90 actuators, not calibrated mappings onto current42 increments.','Muscle aliases, many-to-many pools, missing16, force/activation calibration and neuron identity qualification remain explicit.']},
 'directed_routes':route_info,'edge_location_limit':'Directed pairs are real cached pre->post aggregates across neuropils. Endpoint super-class is not a synapse location, neck tract geometry, conduction delay, efficacy or physiological sign.',
 'local_module':{'file':'LOCAL_MODULE_CANDIDATE.npz','cap':cap,'nodes':len(selected),'reserved_nodes':len(reserved),'vnc_intrinsic_nodes':int(masks['vnc_intrinsic'][selected].sum()),'edges':len(erows),'pure_verified_label_edges':int(strict.sum()),'excluded_from_strict_edges':int((~strict).sum()),'source_synapse_sum':int(syn[erows].sum(dtype=np.float64)),'all_edges_are_exact_cache_rows':True,'selection_is_functional_validation':False,'readout_eligible_motors_by_leg':dict(motor_ports),'readout_max_weights_for7_outputs_each_leg':7*sum(motor_ports.values()),'readout_biases':42,'recommendation':'Implement identity-preserving sparse recurrent local neuron module over saved source edges; reserve descending/ascending ports and actual leg sensory/MN identities. A separate brain module may send through descending cells and receive ascending cells; no direct raw-body-to-brain or brain-to-joint shortcut. Candidate is locomotor-only, not complete brain/feeding/choice.',
 'bounded_parameter_proposal':{'status':'ENGINEERING_PROPOSAL_NOT_FITTED','neuron_update':'bounded point-neuron state; preserve identities and sparse source topology; dynamics and units declared','strict_label_mode':'Only pure acetylcholine/gaba/glutamate source labels eligible initially; excitation/inhibition mapping remains receptor model assumption, glutamate need not imply universally inhibitory biology. Unknown/mixed/modulatory labels do not silently gain signs.','trainable_gain_count_max':8,'gain_bounds':[0.5,2.0],'gain_partition':'directed port/region groups, predeclare exact grouping before fit','neuron_time_constant_group_count_max':8,'time_constant_bounds_ms':[5,100],'threshold_group_count_max':8,'threshold_normalized_bounds':[-1,1],'external_drive':'only declared physically observed leg channels injected into listed sensory identities, plus declared commands on listed descending ports; no arbitrary all-neuron bias','readout':'per-leg bounded linear/tanh readout from unique leg motor ports to7 corresponding incremental joints; engineering decoder, not muscle anatomy','readout_weight_bounds':[-1,1],'readout_bias_bounds':[-1,1],'readout_output_bounds':[-1,1],'final_actuation':'existing42-joint angle/velocity/torque safeguards; coordinate registration from parent retarget audit required','no_new_edges':True,'unknown_edge_policy':'retained in anatomical artifact but inactive in strict dynamics; separate declared sensitivity model needed for any learned unknown sign','training_executed':False},
 'qualification_needed':['Recompute signed incoming/outgoing port coverage before training; disconnected signed ports cannot be declared restored.','Verify sparse scatter update, exact ID projection, finite state and timestep stability without full-body simulation first.','Then bounded offline teacher imitation plus real physics validation authorized by parent, with teacher/engineered readout labelled.','Add source-backed mouth/feeding/internal-state interfaces separately; labels alone do not implement ingestion, energy balance, reward or behavior choice.','No lesion/cut/electrode experiments until full platform implemented and preoperative model frozen.']},
 'missing_or_unknown':[
 {'item':'physical neck axon geometry and cell-specific crossing polarity','status':'UNKNOWN','existing_alternative':'cache crossing class labels; cell_info tag/tag2; feather tract/region; soma parquet only points, not axons'},
 {'item':'per-cell mechanosensory tuning and receptor kinetics','status':'UNKNOWN','existing_alternative':'raw Sub Class and feather cell_sub_class/side/body labels, not measured gains'},
 {'item':'calibrated MN-to-current42joint-or-muscle map','status':'NOT_ESTABLISHED','existing_alternative':'feather peripheral_target_type plus body_part_effector+side; legacy90actuator source pools with16unresolved; engineering motor-only readout'},
 {'item':'feeding/homeostatic circuit dynamics and calibrated abdomen physiology','status':'NOT_ESTABLISHED','existing_alternative':'raw/cache function, mouth motor, gut/hemolymph body labels; feather neuropeptide_verified and cell_function_detailed; no inferred hunger/satiety afferents'},
 {'item':'verified transmitter for every cell or receptor sign for every edge','status':'INCOMPLETE','existing_alternative':'ann_nt_verified; feather predicted NT with score, separately tagged predictions, no automatic fill'},
 {'item':'export threshold source policy/materialization provenance','status':'PARTLY_UNKNOWN','existing_alternative':'raw row min3 and exact root_888 full coverage establish compatibility, not an independent authenticated release history'},
 {'item':'spatial edge locations','status':'NOT_IN_CACHE','existing_alternative':'raw neuropil rows retain coarse locations but aggregated cache loses pair-by-neuropil partition; no micron axon coordinates'},
 {'item':'current default embodied tier is whole brain','status':'FALSE','existing_alternative':'adapters TierConfig12000 leg-seeded, head false and visual false; candidate here separately versioned, no legacy tier replacement'}],
 'geometry':{'source':'access_map','native_voxel_um':[0.004,0.004,0.045],'soma_table_contains':'nucleus centroid point only; not axon position','attribution':'BANC (2026), Distributed control circuits across a brain-and-cord connectome, Harvard Dataverse doi:10.7910/DVN/7WTH1N, somas_v1.parquet, CC BY4.0','no_geometry_exported':True},
 'invariants':{'unique_sorted_cache_ids':True,'edge_bounds':True,'unique_lexicographic_pairs':True,'positive_integer_synapse_counts':True,'raw_reaggregation_exact':True,'source_hashes_unchanged':True,'local_indices_no_negative_wraparound':True,'no_fabricated_edges':True,'crossing_ascending_descending_disjoint':bool(not np.any(masks['ascending_boundary']&masks['descending_boundary']))},
 'runtime':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'dense_n_by_n_arrays':0}}
 dump('NEURAL_INTERFACE_MANIFEST.json',result)
 print(json.dumps({'cache':{k:v for k,v in result['cache'].items() if k in ['nodes','aggregated_directed_pairs','raw_rows','pair_synapse_min','total_synapses']},'crossing':{k:v for k,v in result['crossing'].items() if k!='disagreement_records'},'sensory':{k:v for k,v in result['sensory_channels'].items() if k not in ['by_leg_channel','limitations']},'motor':result['motor_crosswalk'],'local':{k:v for k,v in result['local_module'].items() if k not in ['bounded_parameter_proposal','qualification_needed','recommendation']},'runtime':result['runtime']},indent=2))
if __name__=='__main__':main()
