"""Stdlib, zero-Core preparation/static preflight; never opens array payloads."""
import ast
import copy
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import runtime as r

def main():
    if (r.HERE/'READY.json').exists():
        r.verify()
        print(str(r.HERE/'READY.json'))
        return
    checks = {}
    def check(name,value):
        checks[name] = bool(value)
        if not value:
            raise ValueError(name)
    oldfreeze = r.read(r.REPAIR/'FROZEN.json')
    for path,digest in oldfreeze['source_hashes'].items():
        check('source:'+path,r.sha(path)==digest)
    resultpath = r.REPAIR/'trials'/r.OLD_CASE/'RESULT.json'
    result = r.read(resultpath)
    auditpath = resultpath.with_name('VERIFY.json')
    audit = r.read(auditpath)
    check('old_rough_integrity',audit['integrity_all_pass'] and all(audit['checks'].values()) and audit['result_sha256']==r.sha(resultpath))
    oldledgerpath = r.REPAIR/'PREPARED/MODEL_LEDGER.json'
    old = r.read(oldledgerpath)['cases'][r.OLD_CASE]
    xml = Path(old['xml'])
    check('source_XML_SHA',r.sha(xml)==old['xml_sha256']==result['actual_xml_sha256'])
    tree = ET.fromstring(xml.read_bytes())
    geoms = {node.get('name'):node for node in tree.findall('./worldbody/geom')}
    blocks = [item for item in old['terrain'] if item['name'].startswith('terrain_block_')]
    check('exact_six_blocks',len(blocks)==6)
    check('front_1p35',abs(min(item['footprint'][0] for item in blocks)-1.35)<1e-12)
    check('real_heights',sorted(set(item['top_z'] for item in blocks))==[.02,.04])
    camera = []
    for item in blocks:
        geom = geoms[item['name']]
        pos = list(map(float,geom.get('pos').split()))
        size = list(map(float,geom.get('size').split()))
        check('actual_top:'+item['name'],abs(pos[2]+size[2]-item['top_z'])<1e-12)
        check('actual_front:'+item['name'],abs(pos[0]-size[0]-item['footprint'][0])<1e-12)
        camera.append(dict(name=item['name'],XML_attributes=dict(geom.attrib),ledger=item))
    flat = r.read(r.SOURCE/'terrain_body_baseline/PREPARED/MODEL_LEDGER.json')['cases']['flat_box_control']
    check('flat_box_intentionally_flush',all(item['top_z']==0 for item in flat['terrain']))
    check('schedule_exact',r.TOTAL_MS==25000 and r.TOTAL_NATIVE==2000000 and r.POST_NATIVE==1536000 and r.DECISIONS==5000 and r.FRAMES==1250)
    ledger = r.read(oldledgerpath)
    entry = copy.deepcopy(old)
    prepared = r.HERE/'PREPARED'
    prepared.mkdir(exist_ok=False)
    newxml = prepared/(r.CASE+'.xml')
    with newxml.open('xb') as stream:
        stream.write(xml.read_bytes())
    entry.update(xml=str(newxml),xml_sha256=r.sha(newxml))
    ledger['cases'] = {r.CASE:entry}
    r.save(prepared/'MODEL_LEDGER.json',ledger)
    check('new_XML_byte_identical',xml.read_bytes()==newxml.read_bytes())
    binding = r.read(r.REPAIR/'INPUT_BINDINGS.json')
    binding.update(kind=r.KIND,summary_kind=r.KIND+'_SUMMARY')
    r.save(r.HERE/'INPUT_BINDINGS.json',binding)
    changes = []
    for path in (r.LOOP/'runner.py',r.LOOP/'audit.py',r.NEURAL/'runtime.py',r.NEURAL/'metrics.py'):
        _,rows = r.scheduled_tree(path.read_text(),path)
        changes.extend(rows)
    for name in ('runner.py','audit.py'):
        tree,rows,route = r.loop_tree(name)
        compile(tree,str(r.LOOP/name)+'::static','exec')
        check('loop_compiles:'+name,bool(rows) and route['route']==1)
    # Namespace construction is stdlib only; pin and load_policies are not called.
    namespace,state = r.context(audit=True)
    check('policy_load_globals_original',str(namespace['load_policies'].__globals__['ROOT']).endswith('20261113T064313Z/sensory_body_repair'))
    upstream = state['old']
    for function in (upstream.stable,upstream.BrainNeckLoop.finalize):
        # Recover actual source methods for complete actual-line ledger.
        sourcepath = Path(function.__code__.co_filename.split('::')[0])
        original_name = function.__name__
        source_tree = ast.parse(sourcepath.read_text())
        candidates = [n for n in ast.walk(source_tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==original_name]
        check('method_source_unique:'+original_name,len(candidates)==1)
        n = candidates[0]
        for node in ast.walk(n):
            if isinstance(node,ast.Constant) and type(node.value) is int and node.value in r.SYMBOLS:
                symbol = r.SYMBOLS[node.value]
                changes.append(dict(source=str(sourcepath),line=node.lineno,column=node.col_offset,old_value=node.value,new_expression=symbol,new_value=r.SCHEDULE[symbol]))
    check('no_heavy_imports',not any(name.split('.')[0] in {'numpy','torch','mujoco','flygym','physics','student_wave'} for name in sys.modules))
    r.save(r.HERE/'DURATION_AST_LEDGER.json',dict(inverse_full_AST_exact=True,replacements=changes,schedule=r.SCHEDULE,numeric_threshold_changes=0))
    r.save(r.HERE/'CAMERA_AUDIT.json',dict(flat_box_control='top_z=0 measurement control; intentionally no raised obstacle silhouette',rough_actual_geometry=camera,first_front=1.35,physical_XML_byte_identical=True,view='oblique low side camera plus real RGB inset; full71 mesh containment and block segmentation audited on parent render',camera_changes_physics=False,height_exaggeration=False,parent_actual_visual_review='PENDING'))
    r.save(r.HERE/'PREFLIGHT.json',dict(kind=r.KIND,all_passed=all(checks.values()),checks=checks,actual_Core_constructions=0,actual_native_steps=0,NN_forwards=0,array_payload_loads=0))
    hashes = dict(oldfreeze['source_hashes'])
    extra = [resultpath,auditpath,oldledgerpath,xml,newxml,prepared/'MODEL_LEDGER.json']
    extra += list(r.HERE.glob('*.py'))+list(r.HERE.glob('*.json'))+list(r.HERE.glob('*.md'))
    hashes.update({str(path):r.sha(path) for path in extra})
    r.save(r.HERE/'SOURCE_LEDGER.json',dict(kind=r.KIND,source_hashes=hashes,old_sources_changed=False))
    hashes[str(r.HERE/'SOURCE_LEDGER.json')] = r.sha(r.HERE/'SOURCE_LEDGER.json')
    r.save(r.HERE/'FROZEN.json',dict(kind=r.KIND,source_hashes=hashes))
    r.save(r.HERE/'READY.json',dict(kind=r.KIND,status='READY_CODE_ONLY_ZERO_CORE',preflight=str(r.HERE/'PREFLIGHT.json'),freeze_sha256=r.sha(r.HERE/'FROZEN.json'),new_trial_count=1,NN_training_updates=0,actual_Core_constructions=0,actual_native_steps=0))
    print(str(r.HERE/'READY.json'))

if __name__=='__main__':
    main()
