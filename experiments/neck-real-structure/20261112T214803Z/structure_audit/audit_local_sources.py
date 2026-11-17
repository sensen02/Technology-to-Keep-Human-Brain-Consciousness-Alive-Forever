"""Read-only stdlib CSV/metadata audit; no numerical libraries, models, simulation or network."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,gzip,json,hashlib,zipfile,ast,struct,os,sys
P=Path('/run/media/sensen/Data2/cell_wound_prototype')
O=P/'experiments/neck-real-structure/20261112T214803Z/structure_audit'
CROSS=('ascending','sensory_ascending','ascending_visceral_circulatory','descending','sensory_descending')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
def fresh(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
def npy_schema(z,name):
 with z.open(name) as f:
  if f.read(6)!=b'\x93NUMPY':raise ValueError('notnpy')
  version=tuple(f.read(2));n=struct.unpack('<H' if version==(1,0) else '<I',f.read(2 if version==(1,0) else 4))[0]
  h=ast.literal_eval(f.read(n).decode('latin1').strip())
  return dict(version=list(version),**h)
def main():
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 if (O/'SOURCE_AUDIT.json').exists():raise FileExistsError('no audit overwrite')
 sources={};paths=['data/flywire/banc_neurons.csv.gz','data/flywire/banc_connections_princeton.csv.gz','data/flywire/banc_connectome.npz','data/flywire/fafb_neurons.csv.gz','data/flywire/fafb_classification.csv.gz','data/banc/somas_v1.parquet','data/banc888/banc_888_meta.feather','data/banc888/b888_to_root.npy','data/banc_dataverse/cell_info.parquet','data/banc_dataverse/codex_annotations.parquet','data/banc_dataverse/supplemental_data_2.txt','engine/neck_cut_data.py','build_banc_cns.py','flywire_download.py','outputs/brain_isolation/neck_cut_report.json','outputs/model_branches/banc_release_check/banc_release_check.json','experiments/flybody-retarget/20261107T182940Z/neural-interface/NEURAL_INTERFACE_MANIFEST.json','experiments/flybody-retarget/20261107T182940Z/neural-interface/audit_neural_interface.py']
 for name in paths:
  p=P/name;sources[str(p)]=dict(bytes=p.stat().st_size,sha256=sha(p))
 counts={key:Counter() for key in ('Flow','Super Class','Class','Nerve','Verified NT type')};rows=[];n=0;ids=set();duplicate=[];disagreement=[]
 raw=P/'data/flywire/banc_neurons.csv.gz'
 with gzip.open(raw,'rt',newline='') as f:
  reader=csv.DictReader(f);header=reader.fieldnames
  for line,row in enumerate(reader,2):
   n+=1;rid=row['Root ID']
   if rid in ids:duplicate.append(rid)
   ids.add(rid)
   for k,c in counts.items():c[row.get(k,'')]+=1
   sc=row.get('Super Class','');cl=row.get('Class','')
   if sc in CROSS or cl in ('ascending_neuron','descending_neuron'):
    record=dict(row);record['source_logical_line']=line;record['selection_evidence']='RAW_SUPER_CLASS_CROSSING_ANNOTATION' if sc in CROSS else 'CLASS_ONLY_DISAGREEMENT_NOT_ADMITTED_AS_CROSSING';record['anatomical_neck_crossing_verified']=False;rows.append(record)
    if (cl=='ascending_neuron' and sc!='ascending') or (cl=='descending_neuron' and sc!='descending'):disagreement.append(dict(root_id=rid,super_class=sc,cell_class=cl,source_logical_line=line))
 catalog=O/'RAW_CROSSING_ID_CATALOG.csv'
 with catalog.open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=header+['source_logical_line','selection_evidence','anatomical_neck_crossing_verified']);writer.writeheader();writer.writerows(rows)
 classes={sc:[r['Root ID'] for r in rows if r.get('Super Class')==sc] for sc in CROSS}
 fresh(O/'ANNOTATION_CROSSING_IDS.json',dict(source=str(raw),source_sha256=sources[str(raw)]['sha256'],IDs_are_decimal_strings=True,selection='Exact raw Super Class fields; annotations not independently verified morphology',classes=classes,class_only_disagreements=disagreement,neck_runtime_admitted=False))
 edge=P/'data/flywire/banc_connections_princeton.csv.gz'
 with gzip.open(edge,'rt',newline='') as f:
  reader=csv.DictReader(f);edgeheader=reader.fieldnames;edgepreview=[next(reader) for _ in range(5)]
 sup=P/'data/banc_dataverse/supplemental_data_2.txt'
 with sup.open(newline='') as f:
  reader=csv.DictReader(f);supheader=reader.fieldnames;suppreview=[next(reader) for _ in range(3)]
 with zipfile.ZipFile(P/'data/flywire/banc_connectome.npz') as z:cache_schema={name:npy_schema(z,name) for name in z.namelist() if name.endswith('.npy')}
 historical=read(P/'experiments/flybody-retarget/20261107T182940Z/neural-interface/NEURAL_INTERFACE_MANIFEST.json');release=read(P/'outputs/model_branches/banc_release_check/banc_release_check.json')
 skeleton=P/'data/flywire/fafb_skeletons_swc.zip'
 result=dict(status='LOCAL_SOURCE_INVENTORY_REAL_ANNOTATIONS_AVAILABLE_NECK_GEOMETRY_NOT_ESTABLISHED',scope=dict(no_network=True,model_forward_calls=0,Core_constructions=0,training_updates=0,numerical_imports=0,full_neuronCSV_scanned=True,edgeCSV_rows_read=5,supplementCSV_rows_read=3,parquet_feather_loaded=False,skeleton_zip_opened=False),sources=sources,actual_neuron_csv=dict(path=str(raw),header=header,rows=n,unique_root_ids=len(ids),duplicate_root_ids=duplicate,counts={k:dict(c) for k,c in counts.items()},crossing_exact_superclass_counts={k:len(v) for k,v in classes.items()},crossing_total=sum(map(len,classes.values())),class_only_disagreements=disagreement,catalog=str(catalog),catalog_sha256=sha(catalog),ids_file=str(O/'ANNOTATION_CROSSING_IDS.json'),ids_sha256=sha(O/'ANNOTATION_CROSSING_IDS.json')),actual_edge_header_and_preview=dict(path=str(edge),header=edgeheader,preview=edgepreview,full_edge_counts_recomputed=False,physical_synapse_xyz_present_in_header=any('position' in h.lower() or h.lower() in ('x','y','z') for h in edgeheader),warning='Pair-by-neuropil synapse aggregates. DN-DN/AN-AN endpoint classes do not locate synapses in neck.'),actual_supplement_limited=dict(path=str(sup),header=supheader,preview=suppreview,full_counts_recomputed=False,contains_cross_dataset_match_fields=[h for h in supheader if h.endswith('_match')],warning='Match labels/IDs are associations, not new independent split/merge validation or same-specimen provenance.'),actual_npz_header_only=dict(path=str(P/'data/flywire/banc_connectome.npz'),arrays=cache_schema,payload_loaded=False),historical_not_rerun=dict(interface_manifest_path=str(P/'experiments/flybody-retarget/20261107T182940Z/neural-interface/NEURAL_INTERFACE_MANIFEST.json'),crossing=historical.get('crossing'),cache=historical.get('cache'),edge_location_limit=historical.get('edge_location_limit'),geometry=historical.get('geometry'),missing_or_unknown=historical.get('missing_or_unknown'),release_source_claims=release['sources']),skeleton_archive=dict(path=str(skeleton),bytes=skeleton.stat().st_size,sha256='NOT_COMPUTED_13GB_OUT_OF_LIGHT_METADATA_SCOPE',content_inspected=False,known_scope_from_prior_local_code='FAFB brain-only truncated at neck; cannot establish cross-neck BANC anatomy'),source_provenance=dict(dataset='BANC adult female brain+nerve cord (local builder/archived report attribution)',release='v888 / Harvard Dataverse release3 per archived source records; not independently authenticated online this audit',doi='10.7910/DVN/7WTH1N',sex='female per local source attribution',stage='adult per local source attribution',coordinate='Historical soma point table native voxel .004,.004,.045um; no neck arbor coordinates inspected',same_sample='BANC brain+VNC reported same specimen; FAFB/MANC are different datasets/specimens and cannot silently splice IDs'),qualification=dict(annotation_ID_catalog=True,raw_directional_pair_data_available=True,complete_neck_axon_morphology_verified=False,neck_plane_or_boundary_coordinates=False,neck_local_synapses_verified=False,individual_polarity_independently_verified=False,percell_conduction_delay_or_electrode_mapping=False,complete_transmitter_or_receptor_sign=False,real_neck_structure_runtime_admitted=False,training_permitted_by_this_audit=False),next_minimal='Obtain/inspect BANC same-version neurite/skeleton geometry and explicit neck boundary/electrode spatial mapping. Optional parent-approved parquet/footer and feather schema metadata read can refine annotation/version provenance, not fill morphology. Keep neck ports unadmitted/failclosed until structural scope explicitly qualified.')
 fresh(O/'SOURCE_AUDIT.json',result)
 assert all(sha(Path(p))==v['sha256'] for p,v in sources.items())
 fresh(O/'VERIFICATION.json',dict(all_read_sources_unchanged=True,source_count=len(sources),actual_CSV_rows=n,annotation_crossing_count=result['actual_neuron_csv']['crossing_total'],catalog_rows=len(rows),audit_script_sha256=sha(Path(__file__)),result_sha256=sha(O/'SOURCE_AUDIT.json'),no_heavy_imports=not any(k in sys.modules for k in ('numpy','torch','mujoco','pandas','pyarrow')),no_training_or_simulation=True))
 print(json.dumps(dict(status=result['status'],neurons=n,crossing=result['actual_neuron_csv']['crossing_exact_superclass_counts'],class_disagreements=len(disagreement),catalog_rows=len(rows),no_heavy_imports=True),ensure_ascii=False))
if __name__=='__main__':main()
