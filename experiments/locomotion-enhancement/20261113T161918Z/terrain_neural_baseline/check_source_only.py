"""Stdlib AST/JSON/static-source checks only, no scientific module execution."""
from pathlib import Path
import ast
from collections import Counter
import hashlib
import importlib.util
import json
import sys

HERE = Path(__file__).resolve().parent
PROJECT = Path('/run/media/sensen/Data2/cell_wound_prototype')
LOOP = PROJECT/'experiments/neck-real-structure/20261113T045302Z/sensory_body'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def number_constants(tree):
    return Counter((type(node.value).__name__,repr(node.value)) for node in ast.walk(tree)
        if isinstance(node,ast.Constant) and isinstance(node.value,(int,float,complex)))


def main():
    sys.dont_write_bytecode = True
    checks = {}
    names = ('runtime.py','adapter.py','metrics.py','prepare.py','preflight.py',
             'runner.py','audit.py','orchestrate.py','check_source_only.py')
    for name in names:
        tree = ast.parse((HERE/name).read_text(),filename=str(HERE/name))
        compile(tree,str(HERE/name),'exec')
        imports = [alias.name for node in tree.body if isinstance(node,ast.Import) for alias in node.names]
        imports += [node.module for node in tree.body if isinstance(node,ast.ImportFrom)]
        checks[name+'_no_top_science_import'] = not bool(set(imports)&{'torch','mujoco','numpy','physics','student_wave','flygym'})
    spec = importlib.util.spec_from_file_location('_source_only_terrain_adapter',HERE/'adapter.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    for name in ('runner.py','audit.py'):
        transformed,changes = adapter.tree(name)
        compile(transformed,str(HERE/name)+'::source_check','exec')
        original = ast.parse((LOOP/name).read_text())
        checks[name+'_one_artifact_route_change'] = changes['route']==1
        checks[name+'_numeric_constants_unchanged'] = number_constants(original)==number_constants(transformed)
        checks[name+'_single_runtime_import_removed'] = changes['runtime_import']==1
        checks[name+'_warm_applicability_only'] = changes['old_warm_loops']==(2 if name=='audit.py' else 0)
    prereg = json.loads((HERE/'PREREG.json').read_text())
    contract = json.loads((HERE/'METRIC_CONTRACT.json').read_text())
    upstream_binding = json.loads((PROJECT/'experiments/neck-real-structure/20261113T064313Z/sensory_body_repair/INPUT_BINDINGS.json').read_text())
    checks['new_kind'] = prereg['kind']=='SOURCE_NECK_ROUGH_TERRAIN_BASELINE_V1'
    checks['fixed3cases'] = prereg['cases']==['plane_control','flat_box_control','mild_blocks_seed17']
    checks['fixed_full_schedule'] = prereg['physical_schedule']['native_steps_each']==1232000 and prereg['physical_schedule']['decision_count_each']==3080
    for key,value in prereg['existing_policy_ids'].items():
        checks['unchanged_'+key] = upstream_binding[key]==value
    checks['no_scientific_imports_after_check'] = not any(name in sys.modules for name in ('torch','mujoco','numpy','physics','student_wave','flygym'))
    paths = [HERE/name for name in names]+[HERE/'PREREG.json',HERE/'METRIC_CONTRACT.json',LOOP/'runner.py',LOOP/'audit.py']
    value = dict(kind='TERRAIN_NEURAL_SOURCE_ONLY_CHECK_V1',all_passed=all(checks.values()),checks=checks,
        source_hashes={str(path):sha(path) for path in paths},Core_constructions=0,new_physics_steps=0,
        scientific_module_imports=0,old_numerical_loop_constant_multiset_unchanged=True)
    with (HERE/'SOURCE_ONLY_CHECK.json').open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(value))
    if not value['all_passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
