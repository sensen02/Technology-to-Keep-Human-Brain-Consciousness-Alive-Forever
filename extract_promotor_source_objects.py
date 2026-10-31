"""S3/P3 local source-object/path extraction only; no model or BANC imports."""
from pathlib import Path
import xml.etree.ElementTree as ET,json,hashlib,sys
P=Path(__file__).parent;O=Path('/home/sensen/Desktop/engine修复验收/s3-promotor-source-objects/20261027T224000Z')
NAMES=['LFC_pleural_promotor','LFC_tergopleural_promotor_a','LFC_tergopleural_promotor_b']
UP=P/'venv_body/lib/python3.12/site-packages/flygym/assets/model/musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml'
XMLS=[UP,P/'outputs/muscles_six_legs/fruitfly_six_leg_muscles.xml',P/'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml']
OSIMS=[P/'outputs/upstream_muscle_round11'/n for n in ['best_combined.osim','best_combined_full.osim']]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def ref(p,tag,name):
 token='<'+tag+' name="'+name+'"';return dict(path=str(p),line=next(i for i,l in enumerate(p.read_text().splitlines(),1) if token in l),key=tag+'[@name="'+name+'"]',sha256=sha(p))
def main():
 O.mkdir(parents=True,exist_ok=False);scripts=[P/n for n in ['tools_mirror_muscles.py','tools_stand_and_ground.py','tools_repair_moment_arms.py','tools_repair_muscle_lengthrange.py']];files=OSIMS+XMLS+scripts+[Path(__file__)];h={str(p):sha(p) for p in files};dump(O/'PREREGISTRATION.json',dict(scope=NAMES,only_two_families_three_actual_objects=True,no_other_muscles=True,no_banc_web_physics=True,source_hashes=h))
 osrows=[];mjrows=[]
 for path in OSIMS:
  root=ET.parse(path).getroot()
  for name in NAMES:
   muscles=[e for e in root.iter() if e.get('name')==name];assert len(muscles)==1;mu=muscles[0];pts=[]
   for q in mu.findall('./GeometryPath/PathPointSet/objects/*'):
    pts.append(dict(order=len(pts),type=q.tag,name=q.get('name'),source=ref(path,q.tag,q.get('name')),parent_frame=q.findtext('socket_parent_frame'),location=q.findtext('location')))
   assert pts;osrows.append(dict(name=name,object_type=mu.tag,source=ref(path,mu.tag,name),external_anatomical_id=None,id_note='object name and raw Miller pathpoint labels are source identifiers; external atlas equivalence not certified',pathpoints=pts,properties={e.tag:e.text for e in mu if e.tag!='GeometryPath'}))
 for path in XMLS:
  root=ET.parse(path).getroot();parents={c:p for p in root.iter() for c in p};sites={e.get('name'):e for e in root.iter('site')};acts={e.get('name'):e for e in root.find('actuator')};tendons={e.get('name'):e for e in root.find('tendon')}
  for name in NAMES:
   a=acts[name];ten=tendons[a.get('tendon')];pts=[]
   for q in ten:
    site=sites[q.get('site')];parent=parents[site];assert parent.tag=='body';pts.append(dict(order=len(pts),reference_type=q.tag,site=site.get('name'),site_source=ref(path,'site',site.get('name')),host_body=parent.get('name'),local_position=site.get('pos')))
   mjrows.append(dict(name=name,source_path=str(path),actuator=dict(attributes=dict(a.attrib),source=ref(path,a.tag,name)),tendon=dict(attributes=dict(ten.attrib),source=ref(path,ten.tag,ten.get('name'))),pathpoints=pts))
 def pathsignature(row,osim=False):return [(p['parent_frame'],p['location']) if osim else (p['host_body'],p['local_position']) for p in row['pathpoints']]
 comparisons=[]
 for name in NAMES:
  a,b=[r for r in osrows if r['name']==name];assert a['pathpoints'] and pathsignature(a,True)==pathsignature(b,True)
  chain=[r for r in mjrows if r['name']==name];comparisons.append(dict(name=name,osim_two_files_same_path=True,installed_to_intermediate_same_path=pathsignature(chain[0])==pathsignature(chain[1]),intermediate_to_frozen_same_path=pathsignature(chain[1])==pathsignature(chain[2]),installed_to_frozen_actuator_same_attributes=chain[0]['actuator']['attributes']==chain[2]['actuator']['attributes']))
 distinct=[]
 for n in NAMES[1:]:
  a=next(r for r in osrows if r['name']==NAMES[0]);b=next(r for r in osrows if r['name']==n);distinct.append(dict(pair=[NAMES[0],n],separate_source_objects=True,ordered_paths_differ=pathsignature(a,True)!=pathsignature(b,True)))
 refs=[]
 for p,needles in [(scripts[0],['SRC =','tree = ET.parse(SRC)','if target == "LF"','new_name =','dict(src_act.attrib)','new_act.set']), (scripts[1],['SRC_XML =','OUT_XML =']), (scripts[2],['s.set("pos"']), (scripts[3],['lengthrange'])]:
  for i,l in enumerate(p.read_text().splitlines(),1):
   if any(s in l for s in needles):refs.append(dict(path=str(p),line=i,text=l.strip(),sha256=sha(p)))
 dump(O/'OPENSIM_OBJECTS.json',osrows);dump(O/'MJCF_TENDON_ACTUATOR_PATHS.json',mjrows);dump(O/'GENERATION_CHAIN_LINES.json',refs);dump(O/'RESULT.json',dict(objects=NAMES,distinct_object_comparisons=distinct,chain=comparisons,lf_clone_split_status='all three already exist as separate OpenSim objects and installed actuators; mirror explicitly skips LF and copies attrs for other legs; no evidence of LF project split/clone',missing=['exact OpenSim-to-installed conversion commit/config/invocation','authoritative anatomical atlas/source correspondence for raw Miller28a/28b/30 labels','independent evidence whether distinct model objects represent distinct physical muscle bundles and their neural targets'],limits='distinct source objects and different paths do not establish separate motor identity or innervation; not BANC evidence; no candidate patch',status='SOURCE_OBJECT_DISTINCTION_SUPPORTED_ANATOMICAL_IDENTITY_NOT_ESTABLISHED'))
 assert all(r['ordered_paths_differ'] for r in distinct);assert all(sha(Path(p))==v for p,v in h.items());dump(O/'VERIFICATION.json',dict(exit_code=0,source_object_count_per_osim=3,osim_files=2,mjcf_stages=3,all_named_objects_unique=True,source_hashes_unchanged=True,no_runtime_mapping_change=True,model_load=0,forward=0,body_step=0,neural_step=0,banc_scans=0,web=0,failures=[]));dump(O/'INPUT_CODE_HASHES.json',h);(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print('two families,3 actual objects, different paths; source extraction verified; no simulation')
if __name__=='__main__':main()
