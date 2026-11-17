"""Summarize retained exact incident rows; does not infer encoding or function."""
from pathlib import Path
import csv,json,collections,hashlib
HERE=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 report=json.loads((HERE/'PARTNER_AUDIT.json').read_text());sources={str(HERE/'PARTNER_AUDIT.json'):sha(HERE/'PARTNER_AUDIT.json')};summary={}
 for rid in report['audited_roots']:
  p=HERE/(rid+'_INCIDENT.csv');sources[str(p)]=sha(p);groups={}
  with p.open(newline='') as f:
   for x in csv.DictReader(f):
    sensory=rid=='720575941351919408' and x['edge_direction']=='incoming' and 'leg_nerve' in x['partner_Nerve'] and any(t in x['partner_Class'] for t in ('bristle','chordotonal','campaniform','hair_plate','multidendritic'))
    motor=x['edge_direction']=='outgoing' and x['partner_coarse_class'] in ('leg_motor','other_motor')
    if not (sensory or motor):continue
    key=x['partner_root_id'];g=groups.setdefault(key,dict(root_id=key,edge_direction=x['edge_direction'],primary_type=x['partner_Primary Cell Type'],cell_class=x['partner_Class'],sub_class=x['partner_Sub Class'],nerve=x['partner_Nerve'],body_part=x['partner_Body Part'],function_annotation=x['partner_Function'],verified_NT=x['partner_Verified NT type'],aggregate_syn_count=0,neuropils=set(),source_logical_rows=[],annotation_version='RAW_CSV_NOT_ASSUMED_V888'))
    g['aggregate_syn_count']+=int(x['syn_count']);g['neuropils'].add(x['neuropil']);g['source_logical_rows'].append(int(x['source_edge_logical_line']))
  for g in groups.values():g['neuropils']=sorted(g['neuropils'])
  summary[rid]=list(groups.values())
 an=summary['720575941351919408'];subtypes=collections.Counter(x['sub_class'] for x in an)
 result=dict(status='DIRECT_PARTNER_MODALITY_INVENTORY_ONLY',direct_partners=summary,AN08B018_leg_sensory_subtype_counts=dict(subtypes),AN08B018_leg_sensory_partner_count=len(an),AN08B018_leg_sensory_syn_count=sum(x['aggregate_syn_count'] for x in an),encoding_qualification=dict(AN_joint_angle='NOT_ESTABLISHED',AN_load='NOT_ESTABLISHED',AN_vibration='ANNOTATION_ASSOCIATION_ONLY_NOT_CALIBRATED',DNpe054_walking='NOT_ESTABLISHED',DNp09_42actuator_decoder='NOT_ESTABLISHED'),source_hashes=sources,training_updates=0,neural_forward_calls=0,physical_steps=0)
 with (HERE/'DIRECT_INTERFACE_EVIDENCE.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({'AN_leg_sensory_partners':len(an),'subtypes':dict(subtypes),'syn_count':result['AN08B018_leg_sensory_syn_count'],'new_training':0}))
if __name__=='__main__':main()
