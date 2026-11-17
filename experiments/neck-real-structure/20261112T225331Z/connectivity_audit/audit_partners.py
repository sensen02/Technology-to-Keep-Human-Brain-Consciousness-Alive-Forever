"""Stdlib exact-source incident edge audit; anatomical connectivity is not function."""
from pathlib import Path
import csv,gzip,json,hashlib,collections,time,os,sys
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3]
PRIMARY=['720575941352057904','720575941351919408']
DNP09=['720575941433155799','720575941566493282']
FIELDS=['Root ID','Super Class','Class','Sub Class','Primary Cell Type','Alternative Cell Type(s)','Body Part','Function','Flow','Nerve','Soma side','Verified NT type','Predicted NT type','Predicted NT confidence','Top in/out region']
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def coarse(row):
 if not row:return 'annotation_missing'
 sc=row['Super Class'];cl=row['Class']
 if cl=='leg_motor_neuron':return 'leg_motor'
 if sc=='motor':return 'other_motor'
 if sc=='ventral_nerve_cord_intrinsic':return 'VNC_intrinsic'
 if sc in ('ascending','sensory_ascending','ascending_visceral_circulatory'):return 'ascending'
 if sc in ('descending','sensory_descending'):return 'descending'
 if sc in ('central_brain_intrinsic','optic_lobe_intrinsic','visual_projection','visual_centrifugal'):return 'brain_or_visual'
 if sc=='sensory':return 'sensory'
 return sc or 'unclassified'
def main():
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});start=time.monotonic()
 ns=PROJECT/'data/flywire/banc_neurons.csv.gz';es=PROJECT/'data/flywire/banc_connections_princeton.csv.gz';prev=PROJECT/'experiments/neck-real-structure/20261112T214803Z'
 sourcepaths=[ns,es,HERE/'PREREG.json',Path(__file__),PROJECT/'engine/embodied/descending_command.py',PROJECT/'engine/embodied/sensory_channels.py',PROJECT/'experiments/banc-descending-learning/20261107T203049Z/descending/REPORT.md']+[prev/'geometry_downloads'/(rid+'_GEOMETRY_METADATA.json') for rid in PRIMARY]
 before={str(p):sha(p) for p in sourcepaths};neurons={}
 with gzip.open(ns,'rt',newline='') as f:
  for line,row in enumerate(csv.DictReader(f),2):
   rid=row['Root ID'];assert rid.isdecimal() and rid not in neurons
   neurons[rid]={k:row[k] for k in FIELDS};neurons[rid]['source_neuron_logical_line']=line
 assert len(neurons)==158262
 roots=PRIMARY+DNP09;writers={};handles=[];stats={rid:{d:{'rows':0,'syn_count':0,'partner_ids':set(),'partner_categories':collections.Counter(),'partner_classes':collections.Counter(),'neuropils':collections.Counter(),'direct_legmotor_ids':set(),'direct_leg_sensory_ids':set()} for d in ('incoming','outgoing')} for rid in roots};partnerdetails={rid:{'incoming':{},'outgoing':{}} for rid in roots}
 columns=['source_edge_logical_line','audited_root_id','edge_direction','partner_root_id','pre_root_id','post_root_id','neuropil','syn_count','nt_type']+['partner_'+k for k in FIELDS if k!='Root ID']+['partner_source_neuron_logical_line','partner_coarse_class','annotation_version_status']
 try:
  for rid in roots:
   path=HERE/(rid+'_INCIDENT.csv');h=path.open('x',newline='');handles.append(h);writer=csv.DictWriter(h,fieldnames=columns);writer.writeheader();writers[rid]=writer
  total=0;totalsyn=0
  with gzip.open(es,'rt',newline='') as f:
   for line,row in enumerate(csv.DictReader(f),2):
    total+=1;n=int(row['syn_count']);totalsyn+=n;pre=row['pre_root_id'];post=row['post_root_id']
    for rid,direction,pid in ((pre,'outgoing',post),(post,'incoming',pre)):
     if rid not in stats:continue
     partner=neurons.get(pid,{});cat=coarse(partner);s=stats[rid][direction];s['rows']+=1;s['syn_count']+=n;s['partner_ids'].add(pid);s['partner_categories'][cat]+=n;s['partner_classes'][partner.get('Class','MISSING')]+=n;s['neuropils'][row['neuropil']]+=n
     if cat=='leg_motor':s['direct_legmotor_ids'].add(pid)
     if ('leg_nerve' in partner.get('Nerve','')) and any(t in partner.get('Class','') for t in ('bristle','chordotonal','campaniform','hair_plate','multidendritic')):s['direct_leg_sensory_ids'].add(pid)
     if pid not in partnerdetails[rid][direction]:partnerdetails[rid][direction][pid]=dict(annotation=partner,syn_count=0,neuropils=collections.Counter())
     pd=partnerdetails[rid][direction][pid];pd['syn_count']+=n;pd['neuropils'][row['neuropil']]+=n
     out=dict(source_edge_logical_line=line,audited_root_id=rid,edge_direction=direction,partner_root_id=pid,**row,partner_source_neuron_logical_line=partner.get('source_neuron_logical_line'),partner_coarse_class=cat,annotation_version_status='RAW_CODEX_CSV_NOT_ASSUMED_V888')
     out.update({'partner_'+k:partner.get(k,'') for k in FIELDS if k!='Root ID'});writers[rid].writerow(out)
 finally:
  for h in handles:h.close()
 assert total==3990039 and totalsyn==23556214
 summarized={}
 for rid in roots:
  directions={}
  for d,s in stats[rid].items():
   directions[d]={k:(sorted(v) if isinstance(v,set) else dict(v) if isinstance(v,collections.Counter) else v) for k,v in s.items()}
   directions[d]['partner_count']=len(s['partner_ids']);directions[d]['top_partners_by_aggregate_syn_count']=sorted(partnerdetails[rid][d].values(),key=lambda v:-v['syn_count'])[:20]
  exact888=None
  if rid in PRIMARY:
   exact888=json.loads((prev/'geometry_downloads'/(rid+'_GEOMETRY_METADATA.json')).read_text())['version_identity_evidence']['source_row']
  summarized[rid]=dict(raw_annotation=neurons[rid],actual_v888_identity_previously_verified=exact888,raw_to_v888_not_assumed_for_partners=True,directions=directions,incident_csv=str(HERE/(rid+'_INCIDENT.csv')),incident_csv_sha256=sha(HERE/(rid+'_INCIDENT.csv')))
 for p,h in before.items():assert sha(p)==h
 report=dict(status='SOURCE_INCIDENT_CONNECTIVITY_AUDITED_NOT_WALKING_INTERFACE_VALIDATION',source_hashes=before,source_unchanged=True,raw_neuron_rows=len(neurons),edge_rows_scanned=total,edge_syn_count_sum=totalsyn,audited_roots=summarized,existing_evidence=dict(DNp09_roots=DNP09,source_code=str(PROJECT/'engine/embodied/descending_command.py'),probe_report=str(PROJECT/'experiments/banc-descending-learning/20261107T203049Z/descending/REPORT.md'),probe_scope='Historical fixed engineering current stimulated DNp09 but did not organize useful coordinated walking drive; graph reachability is not functional proof',sensory_source=str(PROJECT/'engine/embodied/sensory_channels.py'),sensory_scope='Subtype labels support modality groups; joint sign/gain/per-cell tuning are engineering assumptions; direct AN joint encoding not established'),no_shortest_path_computation=True,no_function_from_graph_claim=True,AN_joint_feedback_encoding='NOT_ESTABLISHED',DN_joint_decoder='NOT_ESTABLISHED',primary_samples_walking_ports_qualified=False,training_updates=0,Core_constructions=0,neural_forward_calls=0,seconds=time.monotonic()-start,heavy_imports=[x for x in ('numpy','torch','pandas','pyarrow','mujoco') if x in sys.modules])
 with (HERE/'PARTNER_AUDIT.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps(dict(status=report['status'],rows=total,seconds=report['seconds'],roots={rid:{'type':neurons[rid]['Primary Cell Type'],'in':stats[rid]['incoming']['rows'],'out':stats[rid]['outgoing']['rows'],'out_legmotor_partners':len(stats[rid]['outgoing']['direct_legmotor_ids']),'in_leg_sensory_partners':len(stats[rid]['incoming']['direct_leg_sensory_ids'])} for rid in roots},heavy_imports=report['heavy_imports']),indent=2))
if __name__=='__main__':main()
