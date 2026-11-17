"""Matched flat control: only stdlib/AST/JSON/XML; no payload or scientific execution."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROUGH=HERE.parent/'rough_terrain_long_window_38p4'
CASE='plane_long38p4_matched_control'
KIND='SOURCE_NECK_LONG_FLAT_MATCHED_CONTROL_V1'

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def read(path): return json.loads(Path(path).read_text())
def save(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def put(name,value):
    with (HERE/name).open('x') as f: f.write(value)
def main():
    if (HERE/'READY.json').exists():
        for path,digest in read(HERE/'FROZEN.json')['source_hashes'].items():
            if sha(path)!=digest: raise ValueError('Freeze drift '+path)
        if sha(HERE/'FROZEN.json')!=read(HERE/'READY.json')['freeze_sha256']: raise ValueError('Freeze identity')
        print(HERE/'READY.json');return
    checks={}
    def check(name,value):
        checks[name]=bool(value)
        if not value: raise ValueError(name)
    oldfreeze=read(ROUGH/'FROZEN.json')
    hashes=dict(oldfreeze['source_hashes'])
    for p,d in hashes.items(): check('source:'+p,sha(p)==d)
    for p in ROUGH.rglob('*'):
        if p.is_file(): hashes[str(p)]=sha(p)
    ledgerpath=ROUGH/'PREPARED/MODEL_LEDGER.json'
    oldledger=read(ledgerpath)
    check('source_case_unambiguous',list(oldledger['cases'])==['mild_blocks_seed17_start_repair_v1_long38p4'])
    old=next(iter(oldledger['cases'].values()))
    xml=Path(old['xml']);original=ET.fromstring(xml.read_bytes());flat=copy.deepcopy(original)
    world=flat.find('worldbody')
    deleted=[]
    expected=['terrain_block_%02d'%i for i in range(6)]
    for node in list(world):
        if node.tag=='geom' and node.get('name') in expected:
            check('static_box:'+node.get('name'),node.get('type')=='box')
            deleted.append(dict(index=list(original.find('worldbody')).index(next(x for x in original.find('worldbody') if x.get('name')==node.get('name'))),xml=ET.tostring(node,encoding='unicode'),attributes=dict(node.attrib)))
            world.remove(node)
    check('exact_six_removed',sorted(x['attributes']['name'] for x in deleted)==expected)
    restored=copy.deepcopy(flat)
    for row in sorted(deleted,key=lambda x:x['index']): restored.find('worldbody').insert(row['index'],ET.fromstring(row['xml']))
    check('reverse_XML_exact_tree',ET.tostring(restored)==ET.tostring(original))
    check('same_assets_body_actuators_defaults',all(ET.tostring(flat.find(tag))==ET.tostring(original.find(tag)) for tag in ('asset','actuator','default','option','compiler','keyframe') if original.find(tag) is not None))
    oldgeoms=original.findall('.//worldbody//geom');newgeoms=flat.findall('.//worldbody//geom')
    check('geoms_96_to90',len(oldgeoms)==96 and len(newgeoms)==90)
    check('only_terrain_plane',[n.get('name') for n in newgeoms if (n.get('name') or '').startswith('terrain_')]==['terrain_plane'])
    prepared=HERE/'PREPARED';prepared.mkdir()
    newxml=prepared/(CASE+'.xml')
    with newxml.open('xb') as f: f.write(ET.tostring(flat,encoding='utf-8',xml_declaration=True))
    entry=dict(xml=str(newxml),xml_sha256=sha(newxml),terrain=[copy.deepcopy(x) for x in old['terrain'] if x['name']=='terrain_plane'],observation_version=old['observation_version'],legacy_plane_only_values_retained=True,scene_version=CASE,source_case=next(iter(oldledger['cases'])),source_xml=str(xml),source_xml_sha256=sha(xml),change_reason='Remove exactly six static boxes for matched long flat control; all other XML elements/attributes identical')
    save(prepared/'MODEL_LEDGER.json',dict(kind=KIND+'_MODEL_LEDGER',cases={CASE:entry},source_ledger=str(ledgerpath),source_ledger_sha256=sha(ledgerpath),scientific_change='six static boxes removed',physical_steps=0))
    save(HERE/'XML_DIFF_PROOF.json',dict(source_xml=str(xml),source_xml_sha256=sha(xml),new_xml=str(newxml),new_xml_sha256=sha(newxml),deleted_worldbody_geoms=deleted,inverse_XML_tree_exact=True,old_ngeom_source=96,new_ngeom_source=90,all_common_elements_attributes_exact=True,compiled_common_arrays='PENDING_PARENT_ONE_CORE_ZERO_STEPS',optical_input='changes naturally with actual box removal; visual vs mechanical effects not separated'))
    source=(ROUGH/'runtime.py').read_text()
    oldcase='mild_blocks_seed17_start_repair_v1_long38p4';oldkind='SOURCE_NECK_ROUGH_TERRAIN_LONG38P4_V1'
    changed=source.replace("CASE = '"+oldcase+"'","CASE = '"+CASE+"'").replace("KIND = '"+oldkind+"'","KIND = '"+KIND+"'")
    parsed=ast.parse(changed);node=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=='scene_metadata')
    metadata='''def scene_metadata():
    proof = read(HERE/'XML_DIFF_PROOF.json')
    return dict(kind=KIND,scene_version=CASE,case=CASE,scene_class='matched_flat_control',
        new_scene_xml=proof['new_xml'],new_scene_xml_sha256=proof['new_xml_sha256'],
        source_scene_xml=proof['source_xml'],source_scene_xml_sha256=proof['source_xml_sha256'],
        only_scientific_change='remove exactly six static terrain boxes; entire scene contrast, not visual/mechanical isolation',
        schedule=dict(SCHEDULE),rough_coverage_status='NOT_APPLICABLE_FLAT_CONTROL',
        matched_rough_directory=str(HERE.parent/'rough_terrain_long_window_38p4'),
        compiled_common_physics_guard='PARENT_ZERO_STEP_GUARD.json',
        full_physiological_neck_or_biological_validity=False)
'''
    lines=changed.splitlines(keepends=True);oldmeta=''.join(lines[node.lineno-1:node.end_lineno]);lines[node.lineno-1:node.end_lineno]=[metadata];changed=''.join(lines)
    # Only scene IDs/metadata differ from frozen long runtime before bookkeeping wrapper.
    restored_source=changed.replace(metadata,oldmeta).replace("CASE = '"+CASE+"'","CASE = '"+oldcase+"'").replace("KIND = '"+KIND+"'","KIND = '"+oldkind+"'")
    check('runtime_reverse_scene_identity_exact',restored_source==source)
    changed+='''
# Plane-control applicability only. Numerical gates and time adaptation unchanged.
_flat_base_routed_load = routed_load

def routed_load(name,path):
    module = _flat_base_routed_load(name,path)
    path = Path(path)
    if path.resolve() == (NEURAL/'metrics.py').resolve():
        original = module.summarize
        def summarize(*args,**kwargs):
            value = original(*args,**kwargs)
            value['coverage']['obstacle_status'] = 'NOT_APPLICABLE_FLAT_CONTROL'
            value['rough_terrain_pass'] = None
            value['all_pass'] = None
            value['rough_gate_applicability'] = 'NOT_APPLICABLE_FLAT_CONTROL; original five gates remain reference diagnostics'
            return value
        module.summarize = summarize
    elif path.resolve() == (NEURAL/'runtime.py').resolve():
        # Original numerical conjunction remains exact, but obstacle-only flags are N/A.
        src = path.read_text()
        old = "('behavior_diagnostic_pass','rough_terrain_pass','all_pass')"
        if src.count(old)!=2:
            raise ValueError('Expected exactly two obstacle flag bookkeeping tuples')
        src = src.replace(old,"('behavior_diagnostic_pass',)")
        tree,_ = scheduled_tree(src,path)
        previous_upstream = module.upstream
        exec(compile(tree,str(path)+'::flat_applicability_only','exec'),module.__dict__)
        module.upstream = previous_upstream
        module.module = routed_load
    return module
'''
    put('runtime.py',changed)
    put('run.py',(ROUGH/'run.py').read_text())
    binding=read(ROUGH/'INPUT_BINDINGS.json');binding.update(kind=KIND,summary_kind=KIND+'_SUMMARY');save(HERE/'INPUT_BINDINGS.json',binding)
    spec=importlib.util.spec_from_file_location('runtime',HERE/'runtime.py');r=importlib.util.module_from_spec(spec);sys.modules['runtime']=r;spec.loader.exec_module(r)
    check('same_schedule',r.SCHEDULE==read(ROUGH/'READY.json')['schedule'])
    namespace,state=r.context(audit=True)
    check('correct_control_kind',namespace['KIND']==KIND and namespace['ROOT']==HERE)
    check('old336_policy_globals',str(namespace['load_policies'].__globals__['ROOT']).endswith('20261113T064313Z/sensory_body_repair'))
    camera=state['old'].camera_runtime();check('same_camera44200',camera['_capture'].__globals__['TOTAL_MS']==44200 and camera['CameraStream'].capture.__globals__['_capture'] is camera['_capture'])
    for name in ('runner.py','audit.py'):
        tree,rows,routes=r.loop_tree(name);compile(tree,str(r.LOOP/name),'exec');check('routed:'+name,routes['route']==1)
    check('parent_guard_source_exists',(HERE/'parent_zero_step_guard.py').is_file())
    for p in HERE.glob('*.py'): compile(p.read_text(),str(p),'exec')
    check('no_scientific_imports',not any(n.split('.')[0] in {'numpy','torch','mujoco','PIL','scipy','physics','student_wave','flygym'} for n in sys.modules))
    zero=dict(Core_constructions=0,NN_forwards=0,array_payload_loads=0,physics_steps=0,render_calls=0)
    save(HERE/'CODECHECK.json',dict(kind=KIND,all_passed=all(checks.values()),checks=checks,numeric_duration_changes_from_long38=0,numerical_gate_changes=0,scene_identity_reverse_exact=True,**zero))
    save(HERE/'PREFLIGHT.json',dict(kind=KIND,all_passed=True,actual_Core_constructions=0,actual_native_steps=0,parent_compiled_guard='NOT_RUN_REQUIRED_BEFORE_TRIAL',**zero))
    for p in HERE.rglob('*'):
        if p.is_file() and p.name not in {'SOURCE_LEDGER.json','FROZEN.json','READY.json'}: hashes[str(p)]=sha(p)
    save(HERE/'SOURCE_LEDGER.json',dict(kind=KIND,source_hashes=hashes,old_sources_immutable=True,opaque_binary_hash_only=True))
    hashes[str(HERE/'SOURCE_LEDGER.json')]=sha(HERE/'SOURCE_LEDGER.json')
    save(HERE/'FROZEN.json',dict(kind=KIND,source_hashes=hashes))
    save(HERE/'READY.json',dict(kind=KIND,case=CASE,status='READY_ONLYCODE_PARENT_COMPILED_GUARD_AND_TRIAL_NOT_RUN',freeze_sha256=sha(HERE/'FROZEN.json'),schedule=r.SCHEDULE,one_future_trial=1,results='NOT_RUN',**zero))
    print(HERE/'READY.json')
if __name__=='__main__': main()
