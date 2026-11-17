"""Parent-owned full source streaming counts; no adjacency/model/physiology inference."""
from pathlib import Path
import csv,gzip,json,hashlib,collections,time,os
ROOT=Path(__file__).resolve().parent
CROSS={'ascending','sensory_ascending','ascending_visceral_circulatory','descending','sensory_descending'}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});start=time.monotonic();pr=json.loads((ROOT/'EDGE_AUDIT_PREREG.json').read_text());edge=Path(pr['source']);neurons=Path(pr['neuron_source']);catalog=Path(pr['catalog']);hashes={str(p):sha(p) for p in (edge,neurons,catalog,ROOT/'EDGE_AUDIT_PREREG.json',Path(__file__))};assert hashes[str(edge)]==pr['source_sha256']
 annotation={};allids=set();classes=collections.Counter();candidate=set()
 with gzip.open(neurons,'rt',newline='') as f:
  for row in csv.DictReader(f):
   rid=row['Root ID'];assert rid.isdecimal() and rid not in allids;allids.add(rid)
   if row['Super Class'] in CROSS:annotation[rid]=row['Super Class'];classes[row['Super Class']]+=1
 with catalog.open(newline='') as f:
  reader=csv.DictReader(f);fields=reader.fieldnames;idkey=next(k for k in fields if k.lower() in ('root_id','root id'))
  for row in reader:candidate.add(row[idkey])
 assert len(allids)==158262 and len(annotation)==3699 and len(candidate)==3704 and set(annotation)<=candidate
 pre_seen=set();post_seen=set();incident=collections.defaultdict(collections.Counter);counts=collections.Counter();regions=collections.defaultdict(collections.Counter);nt=collections.Counter();pairs=set()
 # Track unique pairs only for crossing-related rows; memory bound scales with their actual incident partners.
 with gzip.open(edge,'rt',newline='') as f:
  reader=csv.DictReader(f);assert reader.fieldnames==['pre_root_id','post_root_id','neuropil','syn_count','nt_type']
  for row in reader:
   pre,post=row['pre_root_id'],row['post_root_id'];n=int(row['syn_count']);assert n>0 and pre in allids and post in allids
   counts['rows']+=1;counts['synapses']+=n;nt[row['nt_type']]+=1
   a=pre in candidate;b=post in candidate
   if a:pre_seen.add(pre);incident[pre]['outgoing_rows']+=1;incident[pre]['outgoing_synapses']+=n
   if b:post_seen.add(post);incident[post]['incoming_rows']+=1;incident[post]['incoming_synapses']+=n
   if a or b:
    counts['candidate_incident_rows']+=1;counts['candidate_incident_synapses']+=n
    key=('candidate' if a else 'other')+'->'+('candidate' if b else 'other');regions[key][row['neuropil']]+=n
    pairs.add((pre,post))
 connected=pre_seen|post_seen;annotated_connected=connected&set(annotation)
 result=dict(status='FULL_AGGREGATE_EDGE_SOURCE_AUDITED_NOT_NECK_GEOMETRY',source_hashes=hashes,neuron_rows=len(allids),annotated_crossing=len(annotation),annotation_class_counts=dict(classes),candidate_union_count=len(candidate),edge_counts=dict(counts),candidate_incident_unique_pairs=len(pairs),nt_type_row_counts=dict(nt),candidate_connected_count=len(connected),annotated_crossing_connected_count=len(annotated_connected),annotated_crossing_missing_edges=sorted(set(annotation)-connected),class_only_connected_count=len(connected-set(annotation)),candidate_per_id_incident_counts={k:dict(incident[k]) for k in sorted(candidate)},candidate_direction_neuropil_synapses={k:dict(v) for k,v in regions.items()},geometry_xyz_present=False,anatomical_neck_admitted=False,training_updates=0,Core_constructions=0,neural_forward_calls=0,seconds=time.monotonic()-start,limitations='No spatial neck plane or local synapse location; endpoint neuropil counts are not conduction or sign evidence. Annotation mismatch retained; no discarded IDs.')
 for p,h in hashes.items():assert sha(p)==h
 with (ROOT/'FULL_EDGE_AUDIT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:result[k] for k in ('status','neuron_rows','annotated_crossing','candidate_union_count','edge_counts','candidate_connected_count','annotated_crossing_connected_count','class_only_connected_count','seconds','anatomical_neck_admitted')}))
if __name__=='__main__':main()
