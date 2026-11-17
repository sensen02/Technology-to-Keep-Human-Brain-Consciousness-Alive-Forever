"""ONLYCODE reproducible source preparation/check; stdlib/AST/JSON/XML only."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import textwrap
import xml.etree.ElementTree as ET
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V1 = HERE.parent/'rough_terrain_long_window'
V2 = V1/'REPAIR_CAPTURE_HORIZON_ONLY_V2'
CASE = 'mild_blocks_seed17_start_repair_v1_long38p4'
KIND = 'SOURCE_NECK_ROUGH_TERRAIN_LONG38P4_V1'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')

def put(name, text):
    with (HERE/name).open('x') as stream:
        stream.write(text)

def main():
    if (HERE/'READY.json').exists():
        frozen = read(HERE/'FROZEN.json')
        for path,digest in frozen['source_hashes'].items():
            if sha(path)!=digest:
                raise ValueError('Frozen drift '+path)
        if sha(HERE/'FROZEN.json')!=read(HERE/'READY.json')['freeze_sha256']:
            raise ValueError('Freeze identity drift')
        print(HERE/'READY.json')
        return
    checks = {}
    def check(name,value):
        checks[name] = bool(value)
        if not value:
            raise ValueError(name)
    # Read JSON/XML and source text; archives are hashed only as opaque bytes.
    hashes = dict(read(V2/'FROZEN.json')['source_hashes'])
    for path,digest in hashes.items():
        check('frozen:'+path,sha(path)==digest)
    for root in (V1,V2):
        for name in ('FROZEN.json','READY.json','SOURCE_LEDGER.json','PREREG.json','SUMMARY.json','JOBS.json'):
            path = root/name
            if path.exists():
                hashes[str(path)] = sha(path)
        for path in (root/'trials').rglob('*'):
            if path.is_file():
                hashes[str(path)] = sha(path)
    source = (V1/'runtime.py').read_text()
    substitutions = [
        ("CASE = 'mild_blocks_seed17_start_repair_v1_long19p2'",f"CASE = '{CASE}'"),
        ("KIND = 'SOURCE_NECK_ROUGH_TERRAIN_LONG19P2_V1'",f"KIND = '{KIND}'"),
        ('ANN_MS = 19200','ANN_MS = 38400'),
        ("'autonomous window 9.6 to 19.2 seconds'","'autonomous window 19.2 to 38.4 seconds'"),
        ('::long19p2','::long38p4')]
    source_ledger = []
    for old,new in substitutions:
        check('unique_runtime:'+old,source.count(old)==1 if old!='::long19p2' else source.count(old)==2)
        source = source.replace(old,new)
        source_ledger.append(dict(file='runtime.py',old=old,new=new))
    # Reverse the entire inherited runtime exactly, before adding the camera wrapper.
    restored = source
    for old,new in reversed(substitutions):
        restored = restored.replace(new,old)
    check('runtime_reverse_AST',ast.dump(ast.parse(restored),include_attributes=True)==ast.dump(ast.parse((V1/'runtime.py').read_text()),include_attributes=True))
    camera_tree = ast.parse((V2/'runtime.py').read_text())
    camera_node = next(n for n in camera_tree.body if isinstance(n,ast.FunctionDef) and n.name=='camera_namespace')
    camera = ast.get_source_segment((V2/'runtime.py').read_text(),camera_node)
    camera = camera.replace("g['scheduled_tree']",'scheduled_tree').replace("g['SCHEDULE']",'SCHEDULE').replace('capture_horizon_only_v2','capture_horizon_38p4')
    source += '\n\n_duration_adapted_upstream = adapted_upstream\n\n'+camera+'\n\ndef adapted_upstream(old):\n    namespace = vars(_duration_adapted_upstream(old)).copy()\n    original_camera_runtime = namespace["camera_runtime"]\n    namespace["camera_runtime"] = lambda: camera_namespace(original_camera_runtime)\n    return types.SimpleNamespace(**namespace)\n'
    put('runtime.py',source)
    for name in ('run.py','orchestrate.py','render.py'):
        original = (V1/name).read_text()
        text = original
        replacements = [('long19p2','long38p4'),('LONG19P2','LONG38P4'),('Full25000ms/1536000','Full44200ms/3072000'),('/25.000s','/44.200s'),('window19.2s','window38.4s'),('Full25second1250frame','Full44p2second2210frame')]
        if name=='render.py':
            replacements += [('-25.)','-44.2)'),('seconds=25.,','seconds=44.2,')]
        used=[]
        for old,new in replacements:
            if old in text:
                text=text.replace(old,new)
                used.append((old,new))
                source_ledger.append(dict(file=name,old=old,new=new))
        reverse=text
        for old,new in reversed(used):
            reverse=reverse.replace(new,old)
        check('reverse_full_source:'+name,reverse==original)
        put(name,text)
    spec=importlib.util.spec_from_file_location('runtime',HERE/'runtime.py')
    r=importlib.util.module_from_spec(spec)
    sys.modules['runtime']=r
    spec.loader.exec_module(r)
    check('exact_schedule',(r.ANN_MS,r.TOTAL_MS,r.DECISIONS,r.FRAMES,r.POST_NATIVE,r.TOTAL_NATIVE)==(38400,44200,8840,2210,3072000,3536000))
    prepared=HERE/'PREPARED'
    prepared.mkdir()
    ledger=read(V2/'PREPARED/MODEL_LEDGER.json')
    old=ledger['cases'][r.OLD_CASE+'_long19p2']
    xml=Path(old['xml'])
    tree=ET.fromstring(xml.read_bytes())
    blocks=[x for x in old['terrain'] if x['name'].startswith('terrain_block_')]
    check('six_blocks_heights_front',len(blocks)==6 and sorted(set(x['top_z'] for x in blocks))==[.02,.04] and abs(min(x['footprint'][0] for x in blocks)-1.35)<1e-12)
    newxml=prepared/(CASE+'.xml')
    with newxml.open('xb') as stream:
        stream.write(xml.read_bytes())
    check('XML_all_physics_and_visual_bytes_identical',sha(newxml)==old['xml_sha256']==sha(xml))
    entry=copy.deepcopy(old)
    entry.update(xml=str(newxml),xml_sha256=sha(newxml))
    ledger['cases']={CASE:entry}
    save(prepared/'MODEL_LEDGER.json',ledger)
    binding=read(V2/'INPUT_BINDINGS.json')
    binding.update(kind=KIND,summary_kind=KIND+'_SUMMARY')
    save(HERE/'INPUT_BINDINGS.json',binding)
    rows=[]
    for path in (r.LOOP/'runner.py',r.LOOP/'audit.py',r.NEURAL/'runtime.py',r.NEURAL/'metrics.py'):
        adapted,changes=r.scheduled_tree(path.read_text(),path)
        compile(adapted,str(path)+'::static','exec')
        rows.extend(changes)
    for name in ('runner.py','audit.py'):
        adapted,changes,routes=r.loop_tree(name)
        compile(adapted,str(r.LOOP/name)+'::static','exec')
        check('actual_routed_loop:'+name,bool(changes) and routes['route']==1)
    namespace,state=r.context(audit=True)
    check('old336_policy_globals',str(namespace['load_policies'].__globals__['ROOT']).endswith('20261113T064313Z/sensory_body_repair'))
    for function in (state['old'].stable,state['old'].BrainNeckLoop.finalize):
        path=Path(function.__code__.co_filename.split('::')[0])
        parsed=ast.parse(path.read_text())
        nodes=[n for n in ast.walk(parsed) if isinstance(n,ast.FunctionDef) and n.name==function.__name__]
        check('unique_method:'+function.__name__,len(nodes)==1)
        segment=textwrap.dedent(ast.get_source_segment(path.read_text(),nodes[0]))
        _,changes=r.scheduled_tree(segment,path)
        for row in changes:
            row['line']+=nodes[0].lineno-1
        rows.extend(changes)
    camera=state['old'].camera_runtime()
    capture=camera['_capture']
    check('actual_nested_capture_globals',camera['CameraStream'].capture.__globals__['_capture'] is capture and capture.__globals__['TOTAL_MS']==44200)
    path=Path(capture.__code__.co_filename.split('::')[0])
    camera_source=path.read_text()
    cls=next(n for n in ast.parse(camera_source).body if isinstance(n,ast.ClassDef) and n.name=='CameraStream')
    func=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='capture')
    segment=textwrap.dedent(ast.get_source_segment(camera_source,func))
    adapted,changes=r.scheduled_tree(segment,path)
    check('only_camera_duration_constant',len(changes)==1 and changes[0]['old_value']==15400 and changes[0]['new_expression']=='TOTAL_MS')
    guard=next(n.test for n in adapted.body[0].body if isinstance(n,ast.If))
    code=compile(ast.Expression(guard),str(path)+'::guard_only','eval')
    decisions={str(ms):bool(eval(code,dict(r.SCHEDULE,ms=ms))) for ms in (-20,0,15400,24980,25000,44180,44200,44220,44181,1)}
    check('camera_boundaries',decisions=={'-20':True,'0':False,'15400':False,'24980':False,'25000':False,'44180':False,'44200':True,'44220':True,'44181':True,'1':True})
    camera_v2=copy.deepcopy(adapted)
    camera_new=copy.deepcopy(adapted)
    class Bind(ast.NodeTransformer):
        def __init__(self,value): self.value=value
        def visit_Name(self,node):
            return ast.copy_location(ast.Constant(self.value),node) if node.id=='TOTAL_MS' else node
    bound_v2=Bind(25000).visit(camera_v2)
    bound_new=Bind(44200).visit(camera_new)
    class Reverse(ast.NodeTransformer):
        def visit_Constant(self,node):
            return ast.copy_location(ast.Constant(25000),node) if type(node.value) is int and node.value==44200 else node
    check('camera_V2_25000_to_44200_reverse_exact',ast.dump(bound_v2,include_attributes=True)==ast.dump(Reverse().visit(bound_new),include_attributes=True))
    hashes[str(path)]=sha(path)
    for row in changes:
        row['line']+=func.lineno-1
    rows.extend(changes)
    check('expected_all_37_duration_sites',len(rows)==37)
    check('video_mapping_all_frames', 'range(0,r.TOTAL_MS,20)' in (HERE/'render.py').read_text() and 'seconds=44.2' in (HERE/'render.py').read_text() and '-44.2)' in (HERE/'render.py').read_text())
    forbidden={'numpy','torch','mujoco','flygym','physics','student_wave','PIL','scipy'}
    check('zero_scientific_imports',not any(n.split('.')[0] in forbidden for n in sys.modules))
    compiled={str(p):bool(compile(p.read_text(),str(p),'exec')) for p in HERE.glob('*.py')}
    check('all_code_compiles',all(compiled.values()))
    counters=dict(Core_constructions=0,NN_forwards=0,array_payload_loads=0,physics_steps=0,render_calls=0,training_updates=0)
    save(HERE/'DURATION_AST_LEDGER.json',dict(inverse_full_AST_exact=True,replacements=rows,schedule=r.SCHEDULE,numeric_threshold_changes=0,coverage=['runner decisions','trace/geometry/terrain shapes','neural/capture shapes','native recorder','cycles/metrics','fresh336 audit','summary','video frame mapping']))
    save(HERE/'CAPTURE_HORIZON_PROOF.json',dict(source=str(path),effective_V2_bound=25000,new_bound=44200,original_capture_AST_value=15400,symbol='TOTAL_MS',reverse_full_AST_exact=True,guard_invalid_by_ms=decisions,actual_nested_capture_routed=True,**counters))
    save(HERE/'CODECHECK.json',dict(kind=KIND,all_passed=all(checks.values()),checks=checks,compiled=compiled,**counters))
    save(HERE/'PREFLIGHT.json',dict(kind=KIND,all_passed=True,actual_Core_constructions=0,actual_native_steps=0,NN_forwards=0,array_payload_loads=0,schedule=r.SCHEDULE,checks=checks))
    save(HERE/'SOURCE_SUBSTITUTIONS.json',dict(source_substitutions=source_ledger,reverse_full_source_or_AST_exact=True,camera_wrapper_source=str(V2/'runtime.py'),camera_wrapper_other_science_changes=0))
    for p in HERE.rglob('*'):
        if p.is_file() and p.name not in {'SOURCE_LEDGER.json','FROZEN.json','READY.json'}:
            hashes[str(p)]=sha(p)
    save(HERE/'SOURCE_LEDGER.json',dict(kind=KIND,source_hashes=hashes,V1_V2_immutable=True,opaque_archive_hash_only=True))
    hashes[str(HERE/'SOURCE_LEDGER.json')]=sha(HERE/'SOURCE_LEDGER.json')
    save(HERE/'FROZEN.json',dict(kind=KIND,source_hashes=hashes))
    save(HERE/'READY.json',dict(kind=KIND,case=CASE,status='READY_CODE_ONLY_ZERO_CORE',freeze_sha256=sha(HERE/'FROZEN.json'),new_trial_count=1,schedule=r.SCHEDULE,scientific_results='NOT_RUN',V1_V2_immutable=True,**counters))
    print(HERE/'READY.json')

if __name__=='__main__':
    main()
