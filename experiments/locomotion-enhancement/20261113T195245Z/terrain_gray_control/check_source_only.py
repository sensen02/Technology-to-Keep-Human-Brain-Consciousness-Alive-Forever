"""Parent stdlib-only gray-control source/namespace check, no model execution."""
from pathlib import Path
import ast
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    names = ('runtime.py','prepare.py','preflight.py','run.py','orchestrate.py','matched.py','check_source_only.py')
    checks = {}
    for name in names:
        tree = ast.parse((HERE/name).read_text(),filename=str(HERE/name))
        compile(tree,str(HERE/name),'exec')
        imports = [a.name for n in tree.body if isinstance(n,ast.Import) for a in n.names]
        imports += [n.module for n in tree.body if isinstance(n,ast.ImportFrom)]
        checks[name+'_no_module_level_scientific_import'] = not bool(set(imports)&{'numpy','torch','mujoco','physics','student_wave','flygym'})
    import runtime
    functions = runtime.neural_functions()
    prepared,metrics,core = functions['terrain_modules']()
    checks['new_neural_ROOT_case_kind'] = functions['HERE']==HERE and functions['TERRAIN_ROOT']==HERE and functions['CASES']==(runtime.CASE,) and functions['KIND']==runtime.KIND
    checks['new_core_closure_ROOT'] = core.core_type.__globals__['HERE']==HERE
    checks['new_core_prepared_verifier'] = core.core_type.__globals__['verify_prepared'] is runtime.verify_prepared
    old = runtime.load('_gray_source_old_neural',runtime.PARENT/'terrain_neural_baseline/runtime.py')
    checks['numerical_context_code_unchanged'] = functions['context'].__code__.co_code==old.context.__code__.co_code
    checks['old_upstream_policy_ROOT_preserved'] = functions['UPSTREAM_ROOT']==old.UPSTREAM_ROOT
    checks['old_module_globals_unchanged'] = old.HERE==runtime.PARENT/'terrain_neural_baseline' and old.CASES==('plane_control','flat_box_control','mild_blocks_seed17')
    ledger = runtime.read(runtime.BROWN/'PREPARED/MODEL_LEDGER.json')['cases'][runtime.BROWN_CASE]
    tree = ET.parse(ledger['xml']).getroot()
    blocks = [g for g in tree.iter('geom') if g.get('name','').startswith('terrain_block_')]
    checks['exact_six_registered_source_blocks'] = len(blocks)==6
    checks['no_material_color_override'] = all('material' not in g.attrib for g in blocks) and all('material' not in g.attrib for d in tree.findall('.//default') for g in d.findall('geom'))
    checks['source_RGB_alpha_explicit'] = all(len(g.attrib['rgba'].split())==4 for g in blocks)
    checks['nonzero_brown_opponent_is_source_color_fact'] = all(float(g.attrib['rgba'].split()[0])-.5*float(g.attrib['rgba'].split()[1])-.5*float(g.attrib['rgba'].split()[2])>0 for g in blocks)
    checks['neutral_mean_metadata_only'] = all(abs(math.fsum(map(float,g.attrib['rgba'].split()[:3]))/3.-.4)<1e-12 for g in blocks)
    reference = runtime.read(runtime.BROWN_FOLDER/'RESULT.json')
    audit = runtime.read(runtime.BROWN_FOLDER/'VERIFY.json')
    checks['qualified_brown_integrity_even_behavior_FAIL'] = audit['integrity_all_pass'] and all(audit['checks'].values()) and audit['result_sha256']==sha(runtime.BROWN_FOLDER/'RESULT.json') and reference['physical_steps']==1232000
    checks['brown_rough_FAIL_retained'] = reference['terrain_summary']['rough_terrain_pass'] is False and reference['autonomous']['slip_ratio_max']>.5
    prereg = runtime.read(HERE/'PREREG.json')
    checks['new_kind'] = prereg['kind']==runtime.KIND
    binding = runtime.read(runtime.BROWN/'INPUT_BINDINGS.json')
    for key in ('brain_sha256','odor_sha256','visual_sha256','motor_sha256','neck_sha256'):
        checks['existing_'+key+'_preserved'] = prereg['existing_policy_ids'][key]==binding[key]
    checks['no_scientific_imports_or_model_calls'] = not any(name in sys.modules for name in ('numpy','torch','mujoco','physics','student_wave','flygym'))
    paths = [HERE/name for name in names]+[HERE/'PREREG.json',runtime.BROWN/'runtime.py',runtime.BROWN/'PREPARED/MODEL_LEDGER.json',Path(ledger['xml']),runtime.BROWN_FOLDER/'RESULT.json',runtime.BROWN_FOLDER/'VERIFY.json']
    value = dict(kind=runtime.KIND,all_passed=all(checks.values()),checks=checks,
        source_hashes={str(path):sha(path) for path in paths},Core_constructions=0,NN_forwards=0,new_physics_steps=0,
        stored_array_payloads_loaded=0,gray_XML_created=False)
    with (HERE/'SOURCE_ONLY_CHECK.json').open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(value))
    if not value['all_passed']:
        raise SystemExit(2)


if __name__=='__main__':
    main()
